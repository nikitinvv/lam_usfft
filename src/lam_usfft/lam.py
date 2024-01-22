import cupy as cp
import numpy as np
from lam_usfft.usfft1d import usfft1d
from lam_usfft.usfft2d import usfft2d
from lam_usfft.fft2d import fft2d
from lam_usfft import utils

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
        
        # init            
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
        
        # cpu memory blocks
        self.pab0 = np.empty(global_block_size0,dtype='float32')
        self.pab1 = np.empty(global_block_size1,dtype='float32')
        
        # reusing cpu memory (pointers)
        self.pa33 =  self.pab0[:np.prod(s3)].reshape(s3)
        self.pa22 =  self.pab1[:np.prod(s2)*2].view('complex64').reshape(s2)        
        self.pa11 =  self.pab0[:np.prod(s1)*2].view('complex64').reshape(s1)        
        self.pa00 =  self.pab1[:np.prod(s0)].reshape(s0)
        
        # gpu memory blocks
        self.gab0 = cp.empty(2*gpu_block_size,dtype='float32')
        self.gab1 = cp.empty(2*gpu_block_size,dtype='float32')
        
        # reusing gpu memory (pointers)
        self.ga55 = self.gab0[:2*np.prod(s5c)].reshape(2,*s5c)
        self.ga44 = self.gab1[:2*np.prod(s4c)*2].view('complex64').reshape(2,*s4c)
        self.ga33 = self.gab0[:2*np.prod(s3c)*2].view('complex64').reshape(2,*s3c)
        self.ga22 = self.gab1[:2*np.prod(s2c)*2].view('complex64').reshape(2,*s2c)
        self.ga11 = self.gab0[:2*np.prod(s1c)*2].view('complex64').reshape(2,*s1c)
        self.ga00 = self.gab1[:2*np.prod(s0c)].reshape(2,*s0c)        
        
        # pinned memory blocks
        self.gpab0 = utils.pinned_array(np.empty(gpu_block_size,dtype='float32'))
        self.gpab1 = utils.pinned_array(np.empty(gpu_block_size,dtype='float32'))        
        
        # reusing pinned memory (pointers)
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
    
    
    def usfft1d_chunks(self, out_t, inp_t, out_gpu, inp_gpu, out_p, inp_p, phi, direction='fwd'):
        nchunk = int(np.ceil(self.n1/self.n1c))
        
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:# gpu computations
                    if direction == 'fwd':
                        self.cl_usfft1d.fwd(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
                    else:
                        self.cl_usfft1d.adj(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
            if(k > 1):
                with self.stream3: # gpu->cpu pinned copy
                    out_gpu[(k-2)%2].get(out=out_p)
                    
            if(k<nchunk):
                st, end = k*self.n1c, min(self.n1,(k+1)*self.n1c)
                s = end-st
                utils.copy(inp_t[st:end],inp_p)# cpu->cpu pinned copy
                with self.stream1:  # cpu->gpu copy
                    inp_gpu[k%2].set(inp_p)
                  
            self.stream3.synchronize()      
            
            if(k > 1):
                # cpu pinned->cpu copy    
                st, end = (k-2)*self.n1c, min(self.n1,(k-1)*self.n1c)
                s = end-st                    
                utils.copy(out_p[:s],out_t[st:end])
                
            self.stream1.synchronize()
            self.stream2.synchronize()
    
    def usfft2d_chunks(self, out, inp, out_gpu, inp_gpu, out_p, inp_p, theta, phi, direction='fwd'):
        theta = cp.array(theta)        
        nchunk = int(np.ceil((self.deth//2+1)/self.dethc))
        for k in range(nchunk+2):    
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
        nchunk = int(np.ceil(self.ntheta/self.nthetac))
        for k in range(nchunk+2):
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
        self.usfft2d_chunks(self.pa22, self.pa11, self.ga33, self.ga22, self.gpa33, self.gpa22, theta, phi, 'fwd')                        
        self.fft2_chunks(self.pa33, self.pa22, self.ga55, self.ga44,self.gpa55, self.gpa44, 'adj')                        
        if res is None:
            res = np.empty_like(self.pa33)
        utils.copy(self.pa33,res)
        
        return res
    
    def adj_lam(self, data, theta, phi, res=None):        
        utils.copy(data,self.pa33)
        self.fft2_chunks(self.pa22, self.pa33, self.ga44, self.ga55,self.gpa44, self.gpa55, 'fwd')           
        self.usfft2d_chunks(self.pa11, self.pa22, self.ga22, self.ga33, self.gpa22, self.gpa33, theta, phi, 'adj')
        self.usfft1d_chunks(self.pa00,self.pa11,self.ga00,self.ga11,self.gpa00,self.gpa11, phi, 'adj') 
        if res is None:
            res = np.empty_like(self.pa00)
        utils.copy(self.pa00,res)
        return res
    
    