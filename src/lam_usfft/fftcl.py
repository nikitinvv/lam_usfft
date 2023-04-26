import cupy as cp
import numpy as np
from lam_usfft.usfft1d import usfft1d
from lam_usfft.usfft2d import usfft2d
from lam_usfft.fft2d import fft2d
from lam_usfft import utils
import time


class FFTCL():
    def __init__(self, n0, n1, n2, detw, deth, ntheta, n1c=None, dethc=None, nthetac=None):
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.detw = detw
        self.deth = deth
        self.ntheta = ntheta

        if n1c == None:
            self.n1c = n1
        else:
            self.n1c = n1c

        if dethc == None:
            self.dethc = deth
        else:
            self.dethc = dethc

        if nthetac == None:
            self.nthetac = ntheta
        else:
            self.nthetac = nthetac

        self.cl_usfft1d = usfft1d(self.n0, self.n1c, self.n2, self.deth)
        self.cl_usfft2d = usfft2d(
            self.dethc, self.n1, self.n2, self.ntheta, self.detw, self.dethc)  # n0 becomes deth
        self.cl_fft2d = fft2d(self.nthetac, self.detw, self.deth)
        
        pinned_block_size = max(self.n1*self.n0*self.n2, self.n1*self.deth*self.n2, self.ntheta*self.deth*self.detw)
        gpu_block_size_size = max(self.n1c*self.n0*self.n2, self.n1c*self.deth*self.n2, self.n1*self.dethc*self.n2,self.dethc*self.ntheta*self.detw,self.nthetac*self.deth*self.detw)
        
        # reusable pinned memory blocks
        self.pab0 = utils.pinned_array(np.empty(pinned_block_size,dtype='complex64'))
        self.pab1 = utils.pinned_array(np.empty(pinned_block_size,dtype='complex64'))
        # pointers (no memory allocation)
        self.pa0 =  self.pab0[:self.n1*self.n0*self.n2].reshape(self.n1, self.n0, self.n2)
        self.pa1 =  self.pab1[:self.n1*self.deth*self.n2].reshape(self.n1,self.deth,self.n2)
        self.pa2 =  self.pab0[:self.ntheta*self.deth*self.detw].reshape(self.ntheta,self.deth,self.detw)
        self.pa3 =  self.pab1[:self.ntheta*self.deth*self.detw].reshape(self.ntheta,self.deth,self.detw)
        
        # reusable gpu memory blocks
        self.gb0 = cp.empty(2*gpu_block_size_size,dtype='complex64')
        self.gb1 = cp.empty(2*gpu_block_size_size,dtype='complex64')
        # pointers (no memory allocation)
        self.ga0 = self.gb0[:2*self.n1c*self.n0*self.n2].reshape(2,self.n1c,self.n0,self.n2)
        self.ga1 = self.gb1[:2*self.n1c*self.deth*self.n2].reshape(2,self.n1c,self.deth,self.n2)
        self.ga2 = self.gb0[:2*self.n1*self.dethc*self.n2].reshape(2,self.n1,self.dethc,self.n2)
        self.ga3 = self.gb1[:2*self.dethc*self.ntheta*self.detw].reshape(2,self.ntheta,self.dethc,self.detw)
        self.ga4 = self.gb0[:2*self.nthetac*self.deth*self.detw].reshape(2,self.nthetac,self.deth,self.detw)
        self.ga5 = self.gb1[:2*self.nthetac*self.deth*self.detw].reshape(2,self.nthetac,self.deth,self.detw)

        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.cl_usfft1d.free()
        self.cl_usfft2d.free()

    # @profile
    def usfft1d_chunks(self, out_t, inp_t, out_gpu, inp_gpu, phi, direction='fwd'):               
        nchunk = self.n1//self.n1c
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    if direction == 'fwd':
                        self.cl_usfft1d.fwd(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
                    else:
                        self.cl_usfft1d.adj(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    out_gpu[(k-2)%2].get(out=out_t[(k-2)*self.n1c:(k-1)*self.n1c])# contiguous copy, fast                            
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    inp_gpu[k%2].set(inp_t[k*self.n1c:(k+1)*self.n1c])# contiguous copy, fast
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()
            
    # @profile
    def usfft2d_chunks(self, out, inp, out_gpu, inp_gpu, theta, phi, direction='fwd'):
        theta = cp.array(theta)        
        nchunk = self.deth//self.dethc
        
        for k in range(nchunk+2):            
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    if direction == 'fwd':
                        self.cl_usfft2d.fwd(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr,theta.data.ptr, phi, k-1, self.deth, self.stream2.ptr)
                    else:
                        self.cl_usfft2d.adj(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr,theta.data.ptr, phi, k-1, self.deth, self.stream2.ptr)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    for j in range(out.shape[0]):# non-contiguous copy, slow but comparable with gpu computations
                        out_gpu[(k-2)%2,j].get(out=out[j,(k-2)*self.dethc:(k-1)*self.dethc])                    
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy           
                    for j in range(inp.shape[0]):
                        inp_gpu[k%2,j].set(inp[j,k*self.dethc:(k+1)*self.dethc])# non-contiguous copy, slow but comparable with gpu computations)
                        
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()                                    
            
    def fft2_chunks(self, out, inp, out_gpu, inp_gpu, direction='fwd'):
        nchunk = self.ntheta//self.nthetac
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    if direction == 'fwd':
                        self.cl_fft2d.fwd(out_gpu[(k-1)%2].data.ptr,inp_gpu[(k-1)%2].data.ptr,self.stream2.ptr)
                    else:
                        self.cl_fft2d.adj(out_gpu[(k-1)%2].data.ptr,inp_gpu[(k-1)%2].data.ptr,self.stream2.ptr)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy        
                    out_gpu[(k-2)%2].get(out=out[(k-2)*self.nthetac:(k-1)*self.nthetac])# contiguous copy, fast                                        
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    inp_gpu[k%2].set(inp[k*self.nthetac:(k+1)*self.nthetac])# contiguous copy, fast
                    
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()
            
    # @profile    
    def fwd_lam(self, u, theta, phi):
        
        self.pa0[:] = u
        # step 1: 1d batch usffts in the z direction to the grid ku*sin(phi)
        # input [self.n1, self.n0, self.n2], output [self.n1, self.deth, self.n2]
        self.usfft1d_chunks(self.pa1,self.pa0,self.ga1,self.ga0, phi, direction='fwd')                        
        # step 2: 2d batch usffts in [x,y] direction to the grid ku*cos(theta)+kv * sin(theta)*cos(phi)
        # input [self.n1, self.deth, self.n2], output [self.ntheta, self.deth, self.detw]
        
        self.usfft2d_chunks(self.pa2, self.pa1, self.ga3, self.ga2, theta, phi, direction='fwd')
        # step 3: 2d batch fft in [det x,det y] direction
        # input [self.ntheta, self.deth, self.detw], output [self.ntheta, self.deth, self.detw]        
        self.fft2_chunks(self.pa3, self.pa2, self.ga5, self.ga4, direction='adj')

        data = self.pa3.copy()
       # c = np.float32(1/np.sqrt(self.ntheta*self.n0))
        return data#*c
    
    def adj_lam(self, data, theta, phi):
        
        self.pa3[:] = data    
        #steps 1,2,3 of the fwd operator but in reverse order
        self.fft2_chunks(self.pa2, self.pa3, self.ga4, self.ga5, direction='fwd')
        self.usfft2d_chunks(self.pa1, self.pa2, self.ga2, self.ga3, theta, phi, direction='adj')
        self.usfft1d_chunks(self.pa0,self.pa1,self.ga0,self.ga1, phi, direction='adj')
        
        u = self.pa0.copy()                                
        #c = np.float32(1/np.sqrt(self.ntheta*self.n0))
        return u#*c

    ##FUNCTIONS FOR ITERATIVE SCHEMES        
    def line_search(self, minf, gamma, Lu, Ld):
        """Line search for the step sizes gamma"""
        while(minf(Lu)-minf(Lu+gamma*Ld) < 0):
            gamma *= 0.5
        return gamma

    def line_search_ext(self, minf, gamma, Lu, Ld, gu, gd):
        """Line search for the step sizes gamma"""
        while(minf(Lu, gu)-minf(Lu+gamma*Ld, gu+gamma*gd) < 0):
            gamma *= 0.5
            if(gamma < 1e-8):
                gamma = 0
                break
        return gamma
    
    def cg_lam(self, data, u, theta, phi, titer, dbg=False):
        """CG solver for ||Lu-data||_2"""
        
        # minimization functional
        def minf(Lu):
            f = np.linalg.norm(Lu-data)**2
            return f
        for i in range(titer):
            Lu = self.fwd_lam(u, theta, phi)
            grad = self.adj_lam(Lu-data, theta, phi)
            
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            #Ld = self.fwd_lam(d, theta, phi)
            gamma = 1#0.5*self.line_search(minf, 8, Lu, Ld)
            grad0 = grad.copy()
            # update step
            u = u + gamma*d
            # check convergence
            if (dbg == True):
                print("%4d, gamma %.3e, fidelity %.7e" %
                        (i, gamma, minf(Lu)))
        return u


    def fwd_reg(self, u):
        """Forward operator for regularization"""
        res = np.zeros([3, *u.shape], dtype='complex64')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        res*=1/np.sqrt(3)
        return res


    def adj_reg(self, gr):
        """Adjoint operator for regularization"""
        res = np.zeros(gr.shape[1:], dtype='complex64')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        res *= -1/np.sqrt(3)  # normalization
        return res


    def solve_reg(self, u, lamd, rho, alpha):
        """ Regularizer problem"""
        z = self.fwd_reg(u)+lamd/rho
        # Soft-thresholding
        za = np.sqrt(np.real(np.sum(z*np.conj(z), 0)))
        z[:, za <= alpha/rho] = 0
        z[:, za > alpha/rho] -= alpha/rho * \
            z[:, za > alpha/rho]/(za[za > alpha/rho])
        return z

    def update_penalty(self, psi, h, h0, rho):
        """Update rhofor a faster convergence"""
        # rho
        r = cp.linalg.norm(psi - h)**2
        s = cp.linalg.norm(rho*(h-h0))**2
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        return rho
    
    @profile
    def cg_lam_ext(self, data, g, init, theta, phi, rho, titer, dbg=True):
        """extended CG solver for ||Lu-data||_2+rho||gu-g||_2"""
        # minimization functional
        def minf(Lu, gu):
            return np.linalg.norm(Lu-data)**2+rho*np.linalg.norm(gu-g)**2
        u = init.copy()
        
        for i in range(titer):
            Lu = self.fwd_lam(u,theta, phi)
            gu = self.fwd_reg(u)
            grad = self.adj_lam(Lu-data,theta, phi) + \
                rho*self.adj_reg(gu-g)
            # Dai-Yuan direction            
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d                
            grad0 = grad.copy()
            Ld = self.fwd_lam(d,theta, phi)
            gd = self.fwd_reg(d)
            # line search
            gamma = 1
            # gamma = 0.5*self.line_search_ext(minf, 4, Lu, Ld,gu,gd)
            # update step
            u = u + gamma*d
            # check convergence
            if dbg:
                print("%4d, gamma %.3e, fidelity %.7e" %
                        (i, gamma, minf(Lu,gu)))
        return u

    @profile
    def admm(self, data, h, psi, lamd, u, theta, phi, alpha, titer, niter, dbg=False):
        """ ADMM for laminography problem with TV regularization"""
        rho = 0.5
        for m in range(niter):
            # keep previous iteration for penalty updates
            h0 = h.copy()
            # laminography problem
            u = self.cg_lam_ext(data, psi-lamd/rho, u, theta, phi, rho, titer)            
            # regularizer problem
            psi = self.solve_reg(u, lamd, rho, alpha)
            # h updates
            h = self.fwd_reg(u)
            # lambda update
            lamd = lamd + rho * (h-psi)
            # update rho for a faster convergence
            rho = self.update_penalty(psi, h, h0, rho)
            # Lagrangians difference between two iterations
            if dbg:
                lagr = self.take_lagr(
                    u, psi, data, h, lamd,theta, phi, alpha,rho)
                print("%d/%d) rho=%.2e, Lagrangian terms:   %.2e %.2e %.2e %.2e, Sum: %.2e" %
                        (m, niter, rho, *lagr))
        return u


    def take_lagr(self, u, psi, data, h, lamd, theta, phi, alpha, rho):
        """ Lagrangian terms for monitoring convergence"""
        lagr = np.zeros(5, dtype="float32")
        Lu = self.fwd_lam(u,theta, phi)
        lagr[0] += np.linalg.norm(Lu-data)**2
        lagr[1] = alpha*np.sum(np.sqrt(np.real(np.sum(psi*np.conj(psi), 0))))        
        lagr[2] = np.sum(np.real(np.conj(lamd)*(h-psi)))        
        lagr[3] = rho*np.linalg.norm(h-psi)**2
        lagr[4] = np.sum(lagr[:4])
        return lagr
