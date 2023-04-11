"""Module for tomography."""

import cupy as cp
import numpy as np
from ffttests.usfft1d import usfft1d
from ffttests.usfft2d import usfft2d
from ffttests.fft2d import fft2d
from ffttests import utils
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
        return data

    def adj_lam(self, data, theta, phi):
        
        self.pa3[:] = data    
        #steps 1,2,3 of the fwd operator but in reverse order
        self.fft2_chunks(self.pa2, self.pa3, self.ga4, self.ga5, direction='fwd')
        self.usfft2d_chunks(self.pa1, self.pa2, self.ga2, self.ga3, theta, phi, direction='adj')
        self.usfft1d_chunks(self.pa0,self.pa1,self.ga0,self.ga1, phi, direction='adj')
        
        u = self.pa0.copy()                                
        return u