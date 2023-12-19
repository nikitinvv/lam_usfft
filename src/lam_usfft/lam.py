import cupy as cp
import numpy as np
from threading import Thread

from lam_usfft.usfft1d import usfft1d
from lam_usfft.usfft2d import usfft2d
from lam_usfft.fft2d import fft2d
from lam_usfft import utils
from lam_usfft import logging
import psutil
import gc
# logging.setup_custom_logger('logging', level="INFO")
log = logging.getLogger(__name__)

class LAM():

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
        
        ################################
        s3 = [self.ntheta,self.deth,self.detw]
        s2 = [self.ntheta,self.deth,(self.detw//2+1)]
        s1 = [self.n1,(self.deth//2+1),self.n2]
        s0 = [self.n1,self.n0,self.n2]
        
        s5c = [self.nthetac,self.deth,self.detw]
        s4c = [self.nthetac,self.deth,self.detw//2+1]
        s3c = [2*self.ntheta,self.dethc,self.detw//2+1]        
        s2c = [self.n1,self.dethc,self.n2]
        s1c = [self.n1c,self.deth//2+1,self.n2]
        s0c = [self.n1c,self.n0,self.n2]
               
        global_block_size0 = max(np.prod(s1)*2,np.prod(s3))
        global_block_size1 = max(np.prod(s0),np.prod(s2)*2)
        gpu_block_size = max(np.prod(s0c),np.prod(s1c)*2,np.prod(s2c)*2,np.prod(s3c)*2,np.prod(s4c)*2,np.prod(s5c))
        
        self.pab0 = np.empty(global_block_size0,dtype='float32')
        self.pab1 = np.empty(global_block_size1,dtype='float32')
        
        self.pa33 =  self.pab0[:np.prod(s3)].reshape(s3)
        self.pa22 =  self.pab1[:np.prod(s2)*2].view('complex64').reshape(s2)        
        self.pa11 =  self.pab0[:np.prod(s1)*2].view('complex64').reshape(s1)        
        self.pa00 =  self.pab1[:np.prod(s0)].reshape(s0)
        
        
        self.gab0 = cp.empty(2*gpu_block_size,dtype='float32')
        self.gab1 = cp.empty(2*gpu_block_size,dtype='float32')
        self.gpab0 = utils.pinned_array(np.empty(gpu_block_size,dtype='float32'))
        self.gpab1 = utils.pinned_array(np.empty(gpu_block_size,dtype='float32'))
        
        self.ga55 = self.gab0[:2*np.prod(s5c)].reshape(2,*s5c)
        self.ga44 = self.gab1[:2*np.prod(s4c)*2].view('complex64').reshape(2,*s4c)
        self.ga33 = self.gab0[:2*np.prod(s3c)*2].view('complex64').reshape(2,*s3c)
        self.ga22 = self.gab1[:2*np.prod(s2c)*2].view('complex64').reshape(2,*s2c)
        self.ga11 = self.gab0[:2*np.prod(s1c)*2].view('complex64').reshape(2,*s1c)
        self.ga00 = self.gab1[:2*np.prod(s0c)].reshape(2,*s0c)        
        
        self.gpa55 = self.gpab0[:np.prod(s5c)].reshape(s5c)
        self.gpa44 = self.gpab1[:np.prod(s4c)*2].view('complex64').reshape(s4c)
        self.gpa33 = self.gpab0[:np.prod(s3c)*2].view('complex64').reshape(s3c)
        self.gpa22 = self.gpab1[:np.prod(s2c)*2].view('complex64').reshape(s2c)
        self.gpa11 = self.gpab0[:np.prod(s1c)*2].view('complex64').reshape(s1c)
        self.gpa00 = self.gpab1[:np.prod(s0c)].reshape(s0c)        
        ################################
              
        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)
    
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        del self.cl_usfft1d
        del self.cl_usfft2d
        del self.cl_fft2d
        del self.pab0
        del self.pab1
        gc.collect()
    
    
    def usfft1d_chunks(self, out_t, inp_t, out_gpu, inp_gpu, out_p, inp_p, phi, direction='fwd'):
        # log.info("usfft1d by chunks.")               
        nchunk = int(np.ceil(self.n1/self.n1c))
        
        for k in range(nchunk+2):
            # utils.printProgressBar(
                # k, nchunk+1, nchunk-k+1, length=40)
            if(k > 0 and k < nchunk+1):
                with self.stream2:# gpu computations
                    if direction == 'fwd':
                        self.cl_usfft1d.fwd(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
                    else:
                        self.cl_usfft1d.adj(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
            if(k > 1):
                with self.stream3: # gpu->cpu pinned copy
                    out_gpu[(k-2)%2].get(out=out_p)# contiguous copy, fast  # not swapaxes
                    
            if(k<nchunk):
                st, end = k*self.n1c, min(self.n1,(k+1)*self.n1c)
                s = end-st
                # inp_p[:s] = inp_t[st:end]
                utils.copy(inp_t[st:end],inp_p)
                with self.stream1:  
                    inp_gpu[k%2].set(inp_p)
                  
            self.stream3.synchronize()      
            
            if(k > 1):
                # cpu pinned->cpu copy    
                st, end = (k-2)*self.n1c, min(self.n1,(k-1)*self.n1c)
                s = end-st                    
                # out_t[st:end] = out_p[:s]
                utils.copy(out_p[:s],out_t[st:end])
                
            self.stream1.synchronize()
            self.stream2.synchronize()
    
    def usfft2d_chunks(self, out, inp, out_gpu, inp_gpu, out_p, inp_p, theta, phi, direction='fwd'):
        # log.info("usfft2d by chunks.")    
        theta = cp.array(theta)        
        nchunk = int(np.ceil((self.deth//2+1)/self.dethc))
        for k in range(nchunk+2):    
            # utils.printProgressBar(
                # k, nchunk+1, nchunk-k+1, length=40)                
            if(k > 0 and k < nchunk+1):
                with self.stream2: # gpu computations       
                    if direction == 'fwd':                        
                        self.cl_usfft2d.fwd(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr,theta.data.ptr, phi, k-1, self.deth, self.stream2.ptr)
                    else:
                        self.cl_usfft2d.adj(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr,theta.data.ptr, phi, k-1, self.deth, self.stream2.ptr)
            if(k > 1):
                with self.stream3: # gpu->cpu copy
                    out_gpu[(k-2)%2].get(out=out_p)                    
                    
            if(k<nchunk):
                # cpu -> cpu pinned copy
                st, end = k*self.dethc, min(self.deth//2+1,(k+1)*self.dethc)                                        
                s = end-st            
                                
                # copy the flipped part of the array for handling r2c FFT
                if direction == 'fwd':                                            
                    utils.copy(inp[:,st:end],inp_p[:,:s])                     
                else:
                    utils.copy(inp[:,st:end],inp_p[:self.ntheta,:s])                
                    if k==0:                    
                        utils.copy(inp[:,self.deth-end+1:self.deth-st+1],inp_p[self.ntheta:,-s:-1])
                        utils.copy(inp[:,0],inp_p[self.ntheta:,-1])
                    else:
                        utils.copy(inp[:,self.deth-end+1:self.deth-st+1],inp_p[self.ntheta:,-s:])
                                                
                with self.stream1:  # cpu pinned->gpu copy                   
                    inp_gpu[k%2].set(inp_p)
                
            self.stream3.synchronize()                                                                        
            if (k > 1):
                # cpu pinned->cpu copy
                st, end = (k-2)*self.dethc, min(self.deth//2+1,(k-1)*self.dethc)
                s = end-st
                if direction == 'fwd':  
                    utils.copy(out_p[:self.ntheta,:s],out[:,st:end])                     
                  
                  
                    # proper filling frequencies for r2c
                    if k==2 and k==nchunk+1:
                        utils.copy(out_p[self.ntheta:,-s:-1,1:-1],out[:,self.deth-end+1:self.deth-st,1:-1])
                    elif k==2:                  
                        utils.copy(out_p[self.ntheta:,-s:-1],out[:,self.deth-end+1:self.deth-st])
                    elif k==nchunk+1:
                        utils.copy(out_p[self.ntheta:,-s:,1:-1],out[:,self.deth-end+1:self.deth-st+1,1:-1])
                    else:
                        utils.copy(out_p[self.ntheta:,-s:],out[:,self.deth-end+1:self.deth-st+1])
                    
                else:                                        
                    utils.copy(out_p[:,:s],out[:,st:end])                     
            
            self.stream1.synchronize()
            self.stream2.synchronize()
        
        if direction == 'fwd':  #maybe not needed
            out[:,self.deth//2+1:,-1] = np.conj(out[:,1:self.deth//2,-1])[:,::-1]                        
            out[:,self.deth//2+1:,0] = np.conj(out[:,1:self.deth//2,0])[:,::-1]                        
                        
    def fft2_chunks(self, out, inp, out_gpu, inp_gpu, out_p, inp_p, direction='fwd'):
        # log.info("fft2 by chunks.")
        
        nchunk = int(np.ceil(self.ntheta/self.nthetac))
        for k in range(nchunk+2):
            # utils.printProgressBar(
                # k, nchunk+1, nchunk-k+1, length=40)
            if(k > 0 and k < nchunk+1):
                with self.stream2: # gpu computations
                    if direction == 'fwd':
                        self.cl_fft2d.fwd(out_gpu[(k-1)%2].data.ptr,inp_gpu[(k-1)%2].data.ptr,self.stream2.ptr)                    
                    else:                                                
                        self.cl_fft2d.adj(out_gpu[(k-1)%2].data.ptr,inp_gpu[(k-1)%2].data.ptr,self.stream2.ptr)                    
            if(k > 1):
                with self.stream3:  # gpu->cpu pinned copy                            
                    out_gpu[(k-2)%2].get(out=out_p)
                                                                    
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    st, end = k*self.nthetac, min(self.ntheta,(k+1)*self.nthetac)
                    s = end-st
                    utils.copy(inp[st:end],inp_p[:s])
                    inp_gpu[k%2].set(inp_p)

            self.stream3.synchronize()                                        
            if(k > 1):
                # cpu pinned ->cpu copy                
                st, end = (k-2)*self.nthetac, min(self.ntheta,(k-1)*self.nthetac)
                s = end-st
                utils.copy(out_p[:s],out[st:end])                
            self.stream1.synchronize()
            self.stream2.synchronize()
                 
    def fwd_lam(self, u, theta, phi, res=None):        
        utils.copy(u,self.pa00)        
        self.usfft1d_chunks(self.pa11,self.pa00,self.ga11,self.ga00,self.gpa11,self.gpa00, phi, 'fwd')         
        # c1r = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/c1r.tiff')
        # c1c = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/c1c.tiff')                
        # print(np.linalg.norm((c1r+1j*c1c)[:,:65]-self.pa11))
        
        self.usfft2d_chunks(self.pa22, self.pa11, self.ga33, self.ga22, self.gpa33, self.gpa22, theta, phi, 'fwd')                
        # c2r = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/c2r.tiff')
        # c2c = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/c2c.tiff')                
        # print('!',np.linalg.norm(c2r[:,:,0:65]-self.pa22.real[:,:,0:65]),np.linalg.norm(c2r[:,:,:65]))
        # print('!',np.linalg.norm(c2c[:,:,0:65]-self.pa22.imag[:,:,0:65]),np.linalg.norm(c2c[:,:,:65]))
        
        self.fft2_chunks(self.pa33, self.pa22, self.ga55, self.ga44,self.gpa55, self.gpa44, 'adj')                
        
        # c3r = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/c3r.tiff')
        # print(np.linalg.norm(c3r-self.pa33),np.linalg.norm(c3r))
                
        if res is None:
            res = np.empty_like(self.pa33)
        utils.copy(self.pa33,res)
        
        return res
    
    def adj_lam(self, data, theta, phi, res=None):        
        utils.copy(data,self.pa33)
        self.fft2_chunks(self.pa22, self.pa33, self.ga44, self.ga55,self.gpa44, self.gpa55, 'fwd')   
        # c3r = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/a3r.tiff')
        # c3c = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/a3c.tiff')                
        # print(np.linalg.norm(c3r[:,:,:65]-self.pa22.real),np.linalg.norm(c3r[:,:,:65]))
        # print(np.linalg.norm(c3c[:,:,:65]-self.pa22.imag),np.linalg.norm(c3c[:,:,:65]))        
        
        self.usfft2d_chunks(self.pa11, self.pa22, self.ga22, self.ga33, self.gpa22, self.gpa33, theta, phi, 'adj')
        # c2r = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/a2r.tiff')
        # c2c = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/a2c.tiff')                
        # print('!!',np.linalg.norm(c2r[:,:65,:]-self.pa11.real),np.linalg.norm(c2r[:,:65]))
        # print('!!',np.linalg.norm(c2c[:,:65,:]-self.pa11.imag),np.linalg.norm(c2c[:,:65]))        
                
        self.usfft1d_chunks(self.pa00,self.pa11,self.ga00,self.ga11,self.gpa00,self.gpa11, phi, 'adj') 
        # a1r = dxchange.read_tiff('/home/beams/TOMO/vnikitin/lam_usfft/tests/res/a1r.tiff')
                
        # print(np.linalg.norm(a1r-self.pa00.real),np.linalg.norm(a1r))
        
        if res is None:
            res = np.empty_like(self.pa00)
        utils.copy(self.pa00,res)
        return res
    
    def _linear_operation_axis0(self,out,x,y,a,b,st,end):
        out[st:end] = a*x[st:end]+b*y[st:end]        
    
    def _linear_operation_axis1(self,out,x,y,a,b,st,end):
        out[:,st:end] = a*x[:,st:end]+b*y[:,st:end]                
        
    def linear_operation(self,x,y,a,b,axis=0,nthreads=8,out=None,dbg=False):
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
    
    def cg_lam(self, data, u, theta, phi, titer,gamma=1, dbg=False,dbg_step=1):
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
            # gamma = 0.5*self.line_search(minf, 4, Lu, Ld)
            #gamma = gamma# seems gamma=1 works nicely, no need to do line search
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
    #     res = np.zeros([3, *u.shape], dtype='float32')
    #     res[0, :, :, :-1] = numexpr.evaluate('u[:, :, 1:]-u[:, :, :-1]')
    #     res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
    #     res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
    #     res*=1/np.sqrt(3)
    #     return res   

    def _fwd_reg(self, res, u, st, end, ax):                
        if ax == 0:
            res[st:end, :, :-1] = u[st:end, :, 1:]-u[st:end, :, :-1]
        if ax == 1:
            res[st:end, :-1, :] = u[st:end, 1:, :]-u[st:end, :-1, :]
        if ax == 2:
            end0 = min(u.shape[0]-1,end)
            res[st:end0, :, :] = u[1+st:1+end0, :, :]-u[st:end0, :, :]            
        res[st:end] *=1/np.sqrt(3)
    
    def fwd_reg(self, u, res=None, nthreads=8, axes=[0,1,2]):
        ##Fast version:
        axes = [axes] if isinstance(axes, int) else axes
        if res is None:
            res = np.empty([len(axes), *u.shape], dtype='float32')
        nchunk = int(np.ceil(u.shape[0]/nthreads))
        mthreads = []
        for ax in range(len(axes)):
            for k in range(nthreads):
                th = Thread(target=self._fwd_reg,args=(res[ax],u,k*nchunk,min((k+1)*nchunk,u.shape[0]),axes[ax]))
                mthreads.append(th)
                th.start()
            for th in mthreads:
                th.join()
        return res

   # def adj_reg(self, gr):
    #     """Adjoint operator for regularization"""
    ##Slow version:
    #     res = np.zeros(gr.shape[1:], dtype='float32')
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
        res[st:end, :, 1:] = (gr[st:end, :, 1:]-gr[st:end, :, :-1])*(-1/np.sqrt(3))
        res[st:end, :, 0] = (gr[st:end, :, 0])*(-1/np.sqrt(3))
    
    def _adj_reg1(self, res, gr, st, end):        
        res[st:end, 1:, :] += (gr[st:end, 1:, :]-gr[st:end, :-1, :])*(-1/np.sqrt(3))
        res[st:end, 0, :] += (gr[st:end, 0, :])*(-1/np.sqrt(3))
    
    def _adj_reg2(self, res, gr, st, end):                
        end0 = min(gr.shape[0]-1,end)
        res[1+st:1+end0, :, :] += (gr[1+st:1+end0, :, :]-gr[st:end0, :, :])*(-1/np.sqrt(3))
        if st==0:
            res[0, :, :] += gr[0, :, :]*(-1/np.sqrt(3))
        
    def adj_reg(self, gr, res=None, nthreads=8,axes=[0,1,2]):
        ##Fast version:
        axes = [axes] if isinstance(axes, int) else axes
        if res is None:
            res = np.empty(gr.shape[1:], dtype='float32')
        nchunk = int(np.ceil(gr.shape[1]/nthreads))
        mthreads = []
        funs = [self._adj_reg0,self._adj_reg1,self._adj_reg2]
        
        for ifun in range(len(axes)):#[self._adj_reg0,self._adj_reg1,self._adj_reg2]:
            fun = funs[axes[ifun]]            
            for k in range(nthreads):
                th = Thread(target=fun,args=(res,gr[ifun],k*nchunk,min((k+1)*nchunk,gr.shape[1])))
                mthreads.append(th)
                th.start()
            for th in mthreads:
                th.join()
        return res
        
    def gradL(self,grad,u,data,theta,phi):
        t = self.fwd_lam(u,theta, phi)
        self.linear_operation(t,data,1,-1,1,out=t)
        self.adj_lam(t,theta, phi, grad)
            
    def gradG(self,gradG,u,g):
        # np.random.seed(0)
        # u = np.random.random(u.shape).astype('float32')
        # g = np.random.random([3,*u.shape]).astype('float32')
        # t = self.fwd_reg(u)
        # self.linear_operation(t,g,1,-1,1,out=t)
        # self.adj_reg(t,gradG,axes=(0,1,2))
        
        
        t = np.empty([1,*u.shape],dtype='float32')
        self.fwd_reg(u,t,axes=0)         
        self.linear_operation(t[0],g[0],1,-1,1,out=t[0])
        self.adj_reg(t,gradG,axes=0)
                
        self.fwd_reg(u,t,axes=1)         
        self.linear_operation(t[0],g[1],1,-1,1,out=t[0])
        self.adj_reg(t,gradG,axes=1)
        
        self.fwd_reg(u,t,axes=2)         
        self.linear_operation(t[0],g[2],1,-1,1,out=t[0])
        self.adj_reg(t,gradG,axes=2)                

    def _dai_yuan_dividend(self,res,grad,st,end,id):
        res[id] = np.sum(grad[st:end]*np.conj(grad[st:end]))
    
    def _dai_yuan_divisor(self,res,grad,grad0,d,st,end,id):
        res[id] = np.sum(np.conj(d[st:end])*(-grad[st:end]+grad0[st:end]))
        
    def dai_yuan_alpha(self,grad,grad0,d,nthreads=8):        
        res = np.zeros(nthreads,dtype='float32')
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
    def cg_lam_ext(self, data, g, u, grads, theta, phi, rho, titer, gamma=1, dbg=False, dbg_step=1):
        """extended CG solver for ||Lu-data||_2+rho||gu-g||_2"""
        # minimization functional
        def minf(Lu, gu):
            return np.linalg.norm(Lu-data)**2+rho*np.linalg.norm(gu-g)**2
        # u = utils.copy(init)
        
        grad = grads[0]
        gradG = grads[1]
        grad0 = grads[2]
        d = np.empty_like(u)
        for i in range(titer):
            # Compute gradient 
            ## Slow version:
            # Lu = self.fwd_lam(u,theta, phi)
            # gu = self.fwd_reg(u)
            # grad = self.adj_lam(Lu-data,theta, phi) + \
            #     rho*self.adj_reg(gu-g)
            # grad = -grad  #NOTE switched to -grad since it is easier to work with
            ## Fast version:
            log.info(f'Lstep0, {psutil.virtual_memory()[3]/1000000000})')
            # self.gradL(grad,u,data,theta, phi)
            # log.info(f'Lstep1, {psutil.virtual_memory()[3]/1000000000})')
            # self.gradG(gradG,u,g)            
            grad_thread = Thread(target=self.gradL,args = (grad,u,data,theta, phi))
            gradG_thread = Thread(target=self.gradG,args = (gradG,u,g))
            grad_thread.start()
            gradG_thread.start()
            grad_thread.join()
            gradG_thread.join() 
            log.info(f'Lstep1, {psutil.virtual_memory()[3]/1000000000})')
            self.linear_operation(grad,gradG,-1,-rho,out=grad)            
            log.info(f'Lstep2, {psutil.virtual_memory()[3]/1000000000})')
            
            # Dai-Yuan direction            
            if i == 0:                
                utils.copy(grad,d)
            else:
                ## Slow version:
                # alpha = np.linalg.norm(grad)**2 / (np.sum(np.conj(d)*(-grad+grad0))+1e-32)
                # d = -grad + alpha*d                
                ## Fast version:
                alpha = self.dai_yuan_alpha(grad,grad0,d)
                self.linear_operation(grad,d,1,alpha,out=d)
            log.info(f'Lstep3, {psutil.virtual_memory()[3]/1000000000})')
            utils.copy(grad,grad0)                        
            # line search            
            # Ld = self.fwd_lam(d,theta, phi)
            # gd = self.fwd_reg(d)            
            # gamma = 0.5*self.line_search_ext(minf, 4, Lu, Ld,gu,gd)
            # gamma = 1# seems gamma=1 works nicely, no need to do line search
            # update step
            ##Slow version:
            # u = u + gamma*d
            ##Fast version:
            log.info(f'Lstep4, {psutil.virtual_memory()[3]/1000000000})')
            self.linear_operation(u,d,1,gamma,out=u)
            # check convergence
            if dbg and i%dbg_step==0:
                Lu = self.fwd_lam(u,theta, phi)
                gu = self.fwd_reg(u)
                print("%4d, gamma %.3e, fidelity %.7e" %
                        (i, gamma, minf(Lu,gu)))
        # return u
