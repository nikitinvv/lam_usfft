import cupy as cp
import numpy as np
from lam_usfft.usfft1d import usfft1d
from lam_usfft.usfft2d import usfft2d
from lam_usfft.fft2d import fft2d
from lam_usfft import utils
from threading import Thread
import time
import numexpr

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
        gpu_block_size = max(self.n1c*self.n0*self.n2, self.n1c*self.deth*self.n2, self.n1*self.dethc*self.n2,self.dethc*self.ntheta*self.detw,self.nthetac*self.deth*self.detw)
        
        # reusable pinned memory blocks
        self.pab0 = utils.pinned_array(np.empty(pinned_block_size,dtype='complex64'))
        self.pab1 = utils.pinned_array(np.empty(pinned_block_size,dtype='complex64'))
        self.pab2 = utils.pinned_array(np.empty(pinned_block_size,dtype='complex64'))
        # pointers (no memory allocation)
        self.pa0 =  self.pab0[:self.n1*self.n0*self.n2].reshape(self.n1, self.n0, self.n2)
        self.pa1 =  self.pab1[:self.n1*self.deth*self.n2].reshape(self.n1,self.deth,self.n2)
        self.pa2 =  self.pab0[:self.ntheta*self.deth*self.detw].reshape(self.ntheta,self.deth,self.detw)
        self.pa3 =  self.pab1[:self.ntheta*self.deth*self.detw].reshape(self.ntheta,self.deth,self.detw)
        
        # reusable gpu memory blocks
        self.gb0 = cp.empty(2*gpu_block_size,dtype='complex64')
        self.gb1 = cp.empty(2*gpu_block_size,dtype='complex64')
        self.gb2 = cp.empty(2*gpu_block_size,dtype='complex64')
        
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
        # nchunk = self.n1//self.n1c
        nchunk = int(np.ceil(self.n1/self.n1c))
        
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    if direction == 'fwd':
                        self.cl_usfft1d.fwd(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
                    else:
                        self.cl_usfft1d.adj(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    st, end = (k-2)*self.n1c, min(self.n1,(k-1)*self.n1c)
                    s = end-st
                    out_gpu[(k-2)%2,:s].get(out=out_t[st:end])# contiguous copy, fast                            
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    st, end = k*self.n1c, min(self.n1,(k+1)*self.n1c)
                    s = end-st
                    inp_gpu[k%2,:s].set(inp_t[st:end])# contiguous copy, fast
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()
            
    # @profile
    def usfft2d_chunks(self, out, inp, out_gpu, inp_gpu, theta, phi, direction='fwd'):
        theta = cp.array(theta)        
        # nchunk = self.deth//self.dethc
        nchunk = int(np.ceil(self.deth/self.dethc))
        
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
                        st, end = (k-2)*self.dethc, min(self.deth,(k-1)*self.dethc)
                        s = end-st
                        out_gpu[(k-2)%2,j,:s].get(out=out[j,st:end])   
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy           
                    for j in range(inp.shape[0]):
                        st, end = k*self.dethc, min(self.deth,(k+1)*self.dethc)
                        s = end-st
                        inp_gpu[k%2,j,:s].set(inp[j,st:end])# non-contiguous copy, slow but comparable with gpu computations)                    
                        
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()                                    
            
    def fft2_chunks(self, out, inp, out_gpu, inp_gpu, direction='fwd'):
        # nchunk = self.ntheta//self.nthetac
        # print(np.linalg.norm(inp[-1]))
        nchunk = int(np.ceil(self.ntheta/self.nthetac))
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    if direction == 'fwd':
                        self.cl_fft2d.fwd(out_gpu[(k-1)%2].data.ptr,inp_gpu[(k-1)%2].data.ptr,self.stream2.ptr)
                    else:
                        self.cl_fft2d.adj(out_gpu[(k-1)%2].data.ptr,inp_gpu[(k-1)%2].data.ptr,self.stream2.ptr)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy        
                    st, end = (k-2)*self.nthetac, min(self.ntheta,(k-1)*self.nthetac)
                    s = end-st
                    out_gpu[(k-2)%2, :s].get(out=out[st:end])# contiguous copy, fast                                        
                    
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    st, end = k*self.nthetac, min(self.ntheta,(k+1)*self.nthetac)
                    s = end-st
                    inp_gpu[k%2, :s].set(inp[st:end])# contiguous copy, fast
                    
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()
            
    # @profile    
    def fwd_lam(self, u, theta, phi):
        
        utils.copy(u,self.pa0)
        # step 1: 1d batch usffts in the z direction to the grid ku*sin(phi)
        # input [self.n1, self.n0, self.n2], output [self.n1, self.deth, self.n2]
        self.usfft1d_chunks(self.pa1,self.pa0,self.ga1,self.ga0, phi, direction='fwd')                        
        # step 2: 2d batch usffts in [x,y] direction to the grid ku*cos(theta)+kv * sin(theta)*cos(phi)
        # input [self.n1, self.deth, self.n2], output [self.ntheta, self.deth, self.detw]
        
        self.usfft2d_chunks(self.pa2, self.pa1, self.ga3, self.ga2, theta, phi, direction='fwd')
        # step 3: 2d batch fft in [det x,det y] direction
        # input [self.ntheta, self.deth, self.detw], output [self.ntheta, self.deth, self.detw]        
        self.fft2_chunks(self.pa3, self.pa2, self.ga5, self.ga4, direction='adj')
        data = utils.copy(self.pa3)
        
        return data
    
    # @profile
    def adj_lam(self, data, theta, phi):
        
        utils.copy(data,self.pa3)
        #steps 1,2,3 of the fwd operator but in reverse order
        self.fft2_chunks(self.pa2, self.pa3, self.ga4, self.ga5, direction='fwd')
        self.usfft2d_chunks(self.pa1, self.pa2, self.ga2, self.ga3, theta, phi, direction='adj')
        self.usfft1d_chunks(self.pa0,self.pa1,self.ga0,self.ga1, phi, direction='adj')        
        u = utils.copy(self.pa0)
        return u

    def _linear_operation_axis0(self,out,x,y,a,b,st,end):
        out[st:end] = a*x[st:end]+b*y[st:end]        
    
    def _linear_operation_axis1(self,out,x,y,a,b,st,end):
        out[:,st:end] = a*x[:,st:end]+b*y[:,st:end]                
        
    def linear_operation(self,x,y,a,b,axis=0,nthreads=16,out=None,dbg=False):
        """ out = ax+by"""
        if out is None:
            out = np.empty_like(x)
        mthreads = []
        # nchunk = x.shape[axis]//nthreads
        nchunk = int(np.ceil(x.shape[axis]/nthreads))
        
        if axis==0:
            fun = self._linear_operation_axis0
        elif axis==1:
            fun = self._linear_operation_axis1
        for k in range(nthreads):
            
            th = Thread(target=fun,args=(out,x,y,a,b,k*nchunk,min(x.shape[axis],(k+1)*nchunk)))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()
        
        return out
          
    def dai_yuan(self,grad,grad0,d):
        #alpha = np.linalg.norm(grad)**2 / \
                    #(np.sum(np.conj(d)*(grad-grad0))+1e-32)     

        # take parts of preallocated memory
        pa0 = self.pab0[:self.n1*self.n0*self.n2].reshape(self.n1,self.n0,self.n2)
        pa1 = self.pab1[:self.n1*self.n0*self.n2].reshape(self.n1,self.n0,self.n2)
        pa2 = self.pab2[:self.n1*self.n0*self.n2].reshape(self.n1,self.n0,self.n2)
        ga0 = self.gb0[:2*self.n1c*self.n0*self.n2].reshape(2,self.n1c,self.n0,self.n2)
        ga1 = self.gb1[:2*self.n1c*self.n0*self.n2].reshape(2,self.n1c,self.n0,self.n2)
        ga2 = self.gb2[:2*self.n1c*self.n0*self.n2].reshape(2,self.n1c,self.n0,self.n2)

        utils.copy(grad,pa0)
        utils.copy(grad0,pa1)
        utils.copy(d,pa2)

        dividend = 0
        # nchunk = self.n1//self.n1c
        nchunk = int(np.ceil(self.n1/self.n1c))
        
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    dividend += cp.sum(ga0[(k-1)%2]*cp.conj(ga0[(k-1)%2]))
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    st, end = k*self.n1c, min(self.n1,(k+1)*self.n1c)
                    s = end-st
                    ga0[k%2,:s].set(pa0[st:end])# contiguous copy, fast
                    
            self.stream1.synchronize()
            self.stream2.synchronize()

        divisor = 0               
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    divisor += cp.sum(cp.conj(ga2[(k-1)%2])*(ga0[(k-1)%2]-ga1[(k-1)%2]))
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    st, end = k*self.n1c, min(self.n1,(k+1)*self.n1c)
                    s = end-st
                    ga0[k%2,:s].set(pa0[st:end])# contiguous copy, fast
                    ga1[k%2,:s].set(pa1[st:end])
                    ga2[k%2,:s].set(pa2[st:end])

                    
            self.stream1.synchronize()
            self.stream2.synchronize()
        
        alpha = dividend/(divisor+1e-32)                
        
        return alpha                    
                    
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
    
    def cg_lam(self, data, u, theta, phi, titer, dbg=False,dbg_step=1):
        """CG solver for ||Lu-data||_2"""
        
        # minimization functional
        def minf(Lu):
            f = np.linalg.norm(Lu-data)**2
            return f
        for i in range(titer):            
            ##Slow version:
            # Lu = self.fwd_lam(u, theta, phi)
            # Ludata = Lu-data            
            # grad = self.adj_lam(Ludata, theta, phi)
            ## Fast version:
            grad = np.empty_like(u)
            self.gradL(grad,u,data,theta, phi)
            self.linear_operation(grad,grad,-1,0,out=grad)            
            # Dai-Yuan direction            
            if i == 0:                
                d = utils.copy(grad)
            else:
                ## Slow version:
                # alpha1 = np.linalg.norm(grad)**2 / (np.sum(np.conj(d)*(-grad+grad0))+1e-32)
                # d = -grad + alpha*d                
                ## Fast version:
                alpha = self.dai_yuan_alpha(grad,grad0,d)
                # print(alpha)
                self.linear_operation(grad,d,1,alpha,out=d)
            grad0 = utils.copy(grad)                        
            # line search            
            # Ld = self.fwd_lam(d,theta, phi)
            # gd = self.fwd_reg(d)            
            # gamma = 0.5*self.line_search_ext(minf, 4, Lu, Ld,gu,gd)
            gamma = 1# seems gamma=1 works nicely, no need to do line search
            # update step
            ##Slow version:
            # u = u + gamma*d
            ##Fast version:
            self.linear_operation(u,d,1,gamma,out=u)
            # check convergence
            if dbg and i%dbg_step==0:
                Lu = self.fwd_lam(u,theta, phi)
                print("%4d, gamma %.3e, fidelity %.7e" %
                        (i, gamma, minf(Lu)))
        return u

    # def fwd_reg(self, u):
    ##Slow version:
    #     """Forward operator for regularization"""
    #     res = np.zeros([3, *u.shape], dtype='complex64')
    #     res[0, :, :, :-1] = numexpr.evaluate('u[:, :, 1:]-u[:, :, :-1]')
    #     res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
    #     res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
    #     res*=1/np.sqrt(3)
    #     return res   

    def _fwd_reg(self, res, u, st, end):                
        res[0, st:end, :, :-1] = u[st:end, :, 1:]-u[st:end, :, :-1]
        res[1, st:end, :-1, :] = u[st:end, 1:, :]-u[st:end, :-1, :]
        end0 = min(u.shape[0]-1,end)
        res[2, st:end0, :, :] = u[1+st:1+end0, :, :]-u[st:end0, :, :]
        res[:,st:end] *=1/np.sqrt(3)
    
    def fwd_reg(self, u, nthreads=16):
        ##Fast version:
        res = np.zeros([3, *u.shape], dtype='complex64')
        nchunk = int(np.ceil(u.shape[0]/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self._fwd_reg,args=(res,u,k*nchunk,min((k+1)*nchunk,u.shape[0])))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()
        return res

   # def adj_reg(self, gr):
    #     """Adjoint operator for regularization"""
    ##Slow version:
    #     res = np.zeros(gr.shape[1:], dtype='complex64')
    #     res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
    #     res[:, :, 0] = gr[0, :, :, 0]
    #     res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
    #     res[:, 0, :] += gr[1, :, 0, :]
    #     res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
    #     res[0, :, :] += gr[2, 0, :, :]
    #     res *= -1/np.sqrt(3)  # normalization
    #     return res
    
    ####Parallel version
    def _adj_reg0(self, res, gr, st, end):
        res[st:end, :, 1:] = gr[0, st:end, :, 1:]-gr[0, st:end, :, :-1]
        res[st:end, :, 0] = gr[0, st:end, :, 0]
    
    def _adj_reg1(self, res, gr, st, end):        
        res[st:end, 1:, :] += gr[1, st:end, 1:, :]-gr[1, st:end, :-1, :]
        res[st:end, 0, :] += gr[1, st:end, 0, :]
    
    def _adj_reg2(self, res, gr, st, end):                
        end0 = min(gr.shape[1]-1,end)
        res[1+st:1+end0, :, :] += gr[2, 1+st:1+end0, :, :]-gr[2, st:end0, :, :]        
        res[1+st:1+end0] *= -1/np.sqrt(3)  # normalization
        if st==0:
            res[0, :, :] += gr[2, 0, :, :]
            res[0, :, :] *= -1/np.sqrt(3)  # normalization
    
    def adj_reg(self, gr, nthreads=16):
        ##Fast version:
        res = np.zeros(gr.shape[1:], dtype='complex64')
        nchunk = int(np.ceil(gr.shape[1]/nthreads))
        mthreads = []
        for fun in [self._adj_reg0,self._adj_reg1,self._adj_reg2]:
            for k in range(nthreads):
                th = Thread(target=fun,args=(res,gr,k*nchunk,min((k+1)*nchunk,gr.shape[1])))
                mthreads.append(th)
                th.start()
            for th in mthreads:
                th.join()
        return res
    
    def _soft_thresholding(self,z,alpha,rho,st,end):
        za = np.sqrt(np.real(np.sum(z[:,st:end]*np.conj(z[:,st:end]), 0)))
        cond = (za > alpha/rho)
        z[:,st:end][:, ~cond] = 0
        z[:,st:end][:, cond] -= alpha/rho * \
            z[:,st:end][:, cond]/(za[cond])
        
    def soft_thresholding(self,z,alpha,rho,nthreads=16):
        
        # nchunk = self.n1//nthreads
        nchunk = int(np.ceil(self.n1/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self._soft_thresholding,args=(z,alpha,rho,k*nchunk,min(self.n1,(k+1)*nchunk)))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()
        return z
        
    # @profile
    def solve_reg(self, u, lamd, rho, alpha):
        """ Regularizer problem"""
        ##Slow version:
        # z = self.fwd_reg(u)+lamd/rho        
        # za = np.sqrt(np.real(np.sum(z*np.conj(z), 0)))        
        # z[:, za <= alpha/rho] = 0
        # z[:, za > alpha/rho] -= alpha/rho * \
        #     z[:, za > alpha/rho]/(za[za > alpha/rho])
        ##Fast version:
        z = self.fwd_reg(u)
        self.linear_operation(z,lamd,1,1.0/rho,axis=1,out=z)        
        z = self.soft_thresholding(z,alpha,rho)        
        return z

    def _update_penalty(self, rres, sres, psi, h, h0, rho, st, end, id):
        """Update rho for a faster convergence"""
        # rho
        tmp = psi[st:end] - h[st:end]
        rres[id] += np.real(np.sum(tmp*np.conj(tmp)))
        tmp = rho*(h[st:end]-h0[st:end])
        sres[id] += np.real(np.sum(tmp*np.conj(tmp)))
            
    def update_penalty(self, psi, h, h0, rho, nthreads=16):
        """Update rhofor a faster convergence"""
        ##Slow version:
        # r = np.linalg.norm(psi - h)**2
        # s = np.linalg.norm(rho*(h-h0))**2
        ##Fast version:
        rres = np.zeros(nthreads,dtype='float64')
        sres = np.zeros(nthreads,dtype='float64')
        mthreads = []
        # nchunk = self.n1//nthreads
        nchunk = int(np.ceil(self.n1/nthreads))
        
        for j in range(3):
            for k in range(nthreads):
                th = Thread(target=self._update_penalty,args=(rres,sres,psi[j], h[j],h0[j],rho,k*nchunk,min(self.n1,(k+1)*nchunk),k))
                th.start()
                mthreads.append(th)
            for th in mthreads:
                th.join()
        r = np.sum(rres)            
        s = np.sum(sres)         
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        return rho
    
    def gradL(self,grad,u,data,theta,phi):
        grad[:]=self.adj_lam(self.fwd_lam(u,theta, phi)-data,theta, phi)
            
    def gradG(self,gradG,u,g):
        gradG[:]=self.adj_reg(self.fwd_reg(u)-g)

    def _dai_yuan_dividend(self,res,grad,st,end,id):
        res[id] = np.sum(grad[st:end]*np.conj(grad[st:end]))
    
    def _dai_yuan_divisor(self,res,grad,grad0,d,st,end,id):
        res[id] = np.sum(np.conj(d[st:end])*(-grad[st:end]+grad0[st:end]))
        
    def dai_yuan_alpha(self,grad,grad0,d,nthreads=16):        
        res = np.zeros(nthreads,dtype='complex64')
        mthreads = []
        # nchunk = grad.shape[0]//nthreads
        nchunk = int(np.ceil(grad.shape[0]/nthreads))
        for k in range(nthreads):
            th = Thread(target=self._dai_yuan_dividend,args=(res,grad,k*nchunk,min(grad.shape[0],(k+1)*nchunk),k))
            th.start()
            mthreads.append(th)
        
        for th in mthreads:
            th.join()
        dividend = np.sum(res)
        
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self._dai_yuan_divisor,args=(res,grad,grad0,d,k*nchunk,min(grad.shape[0],(k+1)*nchunk),k))
            th.start()
            mthreads.append(th)
        
        for th in mthreads:
            th.join()
        divisor = np.sum(res)
        return dividend/(divisor+1e-32)
        
    # @profile
    def cg_lam_ext(self, data, g, init, theta, phi, rho, titer, dbg=False, dbg_step=1):
        """extended CG solver for ||Lu-data||_2+rho||gu-g||_2"""
        # minimization functional
        def minf(Lu, gu):
            return np.linalg.norm(Lu-data)**2+rho*np.linalg.norm(gu-g)**2
        u = utils.copy(init)
        grad = np.empty_like(u)
        gradG = np.empty_like(u)
        for i in range(titer):
            # Compute gradient 
            ## Slow version:
            # Lu = self.fwd_lam(u,theta, phi)
            # gu = self.fwd_reg(u)
            # grad = self.adj_lam(Lu-data,theta, phi) + \
            #     rho*self.adj_reg(gu-g)
            # grad = -grad  #NOTE switched to -grad since it is easier to work with
            ## Fast version:
            grad_thread = Thread(target=self.gradL,args = (grad,u,data,theta, phi))
            gradG_thread = Thread(target=self.gradG,args = (gradG,u,g))
            grad_thread.start()
            gradG_thread.start()
            grad_thread.join()
            gradG_thread.join() 
            self.linear_operation(grad,gradG,-1,-rho,out=grad)            
            # Dai-Yuan direction            
            if i == 0:                
                d = utils.copy(grad)
            else:
                ## Slow version:
                # alpha = np.linalg.norm(grad)**2 / (np.sum(np.conj(d)*(-grad+grad0))+1e-32)
                # d = -grad + alpha*d                
                ## Fast version:
                alpha = self.dai_yuan_alpha(grad,grad0,d)
                self.linear_operation(grad,d,1,alpha,out=d)
            grad0 = utils.copy(grad)                        
            # line search            
            # Ld = self.fwd_lam(d,theta, phi)
            # gd = self.fwd_reg(d)            
            # gamma = 0.5*self.line_search_ext(minf, 4, Lu, Ld,gu,gd)
            gamma = 1# seems gamma=1 works nicely, no need to do line search
            # update step
            ##Slow version:
            # u = u + gamma*d
            ##Fast version:
            self.linear_operation(u,d,1,gamma,out=u)
            # check convergence
            if dbg and i%dbg_step==0:
                Lu = self.fwd_lam(u,theta, phi)
                gu = self.fwd_reg(u)
                print("%4d, gamma %.3e, fidelity %.7e" %
                        (i, gamma, minf(Lu,gu)))
        return u

    # @profile
    def admm(self, data, h, psi, lamd, u, theta, phi, alpha, titer, niter, dbg=False, dbg_step=1):
        """ ADMM for laminography problem with TV regularization"""
        rho = 0.5
        for m in range(niter):
            # keep previous iteration for penalty updates
            h0 = utils.copy(h)
            # laminography problem
            u = self.cg_lam_ext(data, psi-lamd/rho, u, theta, phi, rho, titer, False)            
            # regularizer problem
            psi = self.solve_reg(u, lamd, rho, alpha)
            # h updates
            h = self.fwd_reg(u)
            # lambda update
            ##Slow version:
            # lamd = lamd + rho * (h-psi)
            ##Fast version:
            
            self.linear_operation(lamd,h,1,rho,axis=1,out=lamd)
            self.linear_operation(lamd,psi,1,-rho,axis=1,out=lamd)
            # update rho for a faster convergence
            rho = self.update_penalty(psi, h, h0, rho)
            
            # Lagrangians difference between two iterations
            if dbg and m%dbg_step==0:
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
