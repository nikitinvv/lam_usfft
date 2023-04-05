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
        self.pa1 =  self.pab1[:self.n1*self.deth*self.detw].reshape(self.n1,self.deth,self.detw)
        self.pa2 =  self.pab0[:self.ntheta*self.deth*self.detw].reshape(self.ntheta,self.deth,self.detw)
        self.pa3 =  self.pab1[:self.ntheta*self.deth*self.detw].reshape(self.ntheta,self.deth,self.detw)
        
        # reusable gpu memory blocks
        self.gb0 = cp.empty(gpu_block_size_size,dtype='complex64')
        self.gb1 = cp.empty(gpu_block_size_size,dtype='complex64')
        # pointers (no memory allocation)
        self.ga0 = self.gb0[:self.n1c*self.n0*self.n2].reshape(self.n1c,self.n0,self.n2)
        self.ga1 = self.gb1[:self.n1c*self.deth*self.n2].reshape(self.n1c,self.deth,self.n2)
        self.ga2 = self.gb0[:self.n1*self.dethc*self.n2].reshape(self.n1,self.dethc,self.n2)
        self.ga3 = self.gb1[:self.dethc*self.ntheta*self.detw].reshape(self.dethc,self.ntheta,self.detw)
        self.ga4 = self.gb0[:self.nthetac*self.deth*self.detw].reshape(self.nthetac,self.deth,self.detw)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.cl_usfft1d.free()
        self.cl_usfft2d.free()

    @profile
    def fwd_usfft1d_chunks(self, out_t, inp_t, out_gpu, inp_gpu, phi):                
        for k in range(self.n1//self.n1c):
            inp_gpu.set(inp_t[k*self.n1c:(k+1)*self.n1c])# contiguous copy, fast
            
            self.cl_usfft1d.fwd(out_gpu.data.ptr, inp_gpu.data.ptr, phi)
            cp.cuda.Device(0).synchronize()# for performance tests        
            
            out_gpu.get(out=out_t[k*self.n1c:(k+1)*self.n1c])# contiguous copy, fast                            
        
    @profile
    def fwd_usfft2d_chunks(self, out, inp, out_gpu, inp_gpu, theta, phi):
        theta = cp.array(theta)        
        for k in range(self.deth//self.dethc):
            for j in range(inp.shape[0]):
                inp_gpu[j].set(inp[j,k*self.dethc:(k+1)*self.dethc])# non-contiguous copy, slow but comparable with gpu computations)
        
            self.cl_usfft2d.fwd(out_gpu.data.ptr, inp_gpu.data.ptr,theta.data.ptr, phi, k, self.deth)
            cp.cuda.Device(0).synchronize()# for performance tests        
        
            for j in range(out.shape[0]):# non-contiguous copy, slow but comparable with gpu computations
                out_gpu[:,j].get(out=out[j,k*self.dethc:(k+1)*self.dethc])
        
    @profile
    def fwd_fft2_chunks(self, out, inp, inp_gpu):
        for k in range(self.ntheta//self.nthetac):
            inp_gpu.set(inp[k*self.nthetac:(k+1)*self.nthetac])# contiguous copy, fast
            
            self.cl_fft2d.adj(inp_gpu.data.ptr)
            cp.cuda.Device(0).synchronize()# for performance tests          
            
            inp_gpu.get(out=out[k*self.nthetac:(k+1)*self.nthetac])# contiguous copy, fast                    
        
    def fwd_lam(self, u, theta, phi):
        
        self.pa0[:] = u.swapaxes(0,1)        

        # step 1: 1d batch usffts in the z direction to the grid ku*sin(phi)
        # input [self.n1, self.n0, self.n2], output [self.n1, self.deth, self.n2]
        self.fwd_usfft1d_chunks(self.pa1,self.pa0,self.ga1,self.ga0, phi)                        
        # step 2: 2d batch usffts in [x,y] direction to the grid ku*cos(theta)+kv * sin(theta)*cos(phi)
        # input [self.deth, self.n1, self.n2], output [self.ntheta, self.deth, self.detw]
        
        self.fwd_usfft2d_chunks(self.pa2, self.pa1, self.ga3, self.ga2, theta, phi)
        # step 3: 2d batch fft in [det x,det y] direction
        # input [self.ntheta, self.deth, self.detw], output [self.ntheta, self.deth, self.detw]        
        
        self.fwd_fft2_chunks(self.pa3, self.pa2, self.ga4)
    
        return self.pa3

