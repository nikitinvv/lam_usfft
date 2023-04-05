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
        
         # threads for filling the resulting array
        self.write_threads = []
        for k in range(16):#16 is probably enough but can be changed
            self.write_threads.append(utils.WRThread())
                

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.cl_usfft1d.free()
        self.cl_usfft2d.free()

    @profile
    def fwd_usfft1d_chunks(self, out_t, inp_t, out_gpu, inp_gpu,phi):
        ku = (cp.arange(-self.deth//2, self.deth//2) /
              self.deth*cp.sin(phi)).astype('float32')
                
        for k in range(self.n1//self.n1c):
            inp_gpu.set(inp_t[k*self.n1c:(k+1)*self.n1c])# contiguous copy, fast
            self.cl_usfft1d.fwd(out_gpu.data.ptr, inp_gpu.data.ptr, ku.data.ptr)
            cp.cuda.Device(0).synchronize()# for performance tests        
            out_gpu.get(out=out_t[k*self.n1c:(k+1)*self.n1c])# contiguous copy, fast                            
        
    @profile
    def fwd_usfft2d_chunks(self, out, inp, out_gpu, inp_gpu, x_gpu, y_gpu, theta, phi):
        stream1 = cp.cuda.Stream(non_blocking=False)
        ku0 = (cp.arange(-self.deth//2, self.deth//2)/self.deth).astype('float32')        
        for k in range(self.deth//self.dethc):
            for j in range(inp.shape[0]):
                inp_gpu[j].set(inp[j,k*self.dethc:(k+1)*self.dethc])# non-contiguous copy, slow but comparable with gpu computations)
            [ku, kv] = np.meshgrid(cp.arange(-self.detw//2, self.detw//2).astype('float32') /
                                   self.detw, ku0[k*self.dethc:(k+1)*self.dethc])
            for itheta in range(self.ntheta):
                x_gpu[:,itheta] = ku*cp.cos(theta[itheta])+kv * \
                    cp.sin(theta[itheta])*cp.cos(phi)
                y_gpu[:,itheta] = ku*np.sin(theta[itheta])-kv * \
                    cp.cos(theta[itheta])*cp.cos(phi)
            x_gpu[x_gpu >= 0.5] = 0.5 - 1e-5
            x_gpu[x_gpu < -0.5] = -0.5 + 1e-5
            y_gpu[y_gpu >= 0.5] = 0.5 - 1e-5
            y_gpu[y_gpu < -0.5] = -0.5 + 1e-5

            self.cl_usfft2d.fwd(out_gpu.data.ptr, inp_gpu.data.ptr, x_gpu.data.ptr, y_gpu.data.ptr)
            cp.cuda.Device(0).synchronize()# for performance tests        
        
            #with stream1:
            for j in range(out.shape[0]):# non-contiguous copy, slow but comparable with gpu computations
                out_gpu[:,j].get(out=out[j,k*self.dethc:(k+1)*self.dethc])
        
                # out[j,k*self.dethc:(k+1)*self.dethc]=out_gpu[:,j].get()
    
    @profile
    def fwd_fft2_chunks(self, out, inp, inp_gpu):
        for k in range(self.ntheta//self.nthetac):
            inp_gpu.set(inp[k*self.nthetac:(k+1)*self.nthetac])# contiguous copy, fast
            self.cl_fft2d.adj(inp_gpu.data.ptr)
            cp.cuda.Device(0).synchronize()# for performance tests          
            inp_gpu.get(out=out[k*self.nthetac:(k+1)*self.nthetac])# contiguous copy, fast                    
    
    def fwd_lam(self, u, theta, phi):
        pa0 =  utils.pinned_array(np.ones([ self.n1, self.n0, self.n2], dtype='complex64'))        
        pa1 =  utils.pinned_array(np.zeros([self.n1, self.deth,  self.n2], dtype='complex64'))
        pa2 =  utils.pinned_array(np.zeros([self.ntheta,self.deth,  self.detw], dtype='complex64'))
        pa3 =  utils.pinned_array(np.zeros([self.ntheta, self.deth, self.detw], dtype='complex64'))
        
        ga0 = cp.zeros([self.n1c, self.n0,   self.n2], dtype='complex64')
        ga1 = cp.zeros([self.n1c, self.deth, self.n2], dtype='complex64')
        ga2 = cp.zeros([self.n1, self.dethc,  self.n2], dtype='complex64')
        ga3 = cp.zeros([self.dethc, self.ntheta, self.detw], dtype='complex64')
        ga4 = cp.zeros([self.nthetac, self.deth,  self.detw], dtype='complex64')
        x_gpu = cp.zeros([ self.dethc, self.ntheta,  self.detw], dtype='float32')
        y_gpu = cp.zeros([ self.dethc, self.ntheta, self.detw], dtype='float32')

        pa0[:] = u.swapaxes(0,1)
        
        # step 1: 1d batch usffts in the z direction to the grid ku*sin(phi)
        # input [self.n1, self.n0, self.n2], output [self.n1, self.deth, self.n2]
        self.fwd_usfft1d_chunks(pa1,pa0,ga1,ga0, phi)                        
        # step 2: 2d batch usffts in [x,y] direction to the grid ku*cos(theta)+kv * sin(theta)*cos(phi)
        # input [self.deth, self.n1, self.n2], output [self.ntheta, self.deth, self.detw]
        self.fwd_usfft2d_chunks(pa2, pa1, ga3, ga2, x_gpu, y_gpu, theta, phi)
        # step 3: 2d batch fft in [det x,det y] direction
        # input [self.ntheta, self.deth, self.detw], output [self.ntheta, self.deth, self.detw]        
        self.fwd_fft2_chunks(pa3, pa2, ga4)
    
        return pa3

