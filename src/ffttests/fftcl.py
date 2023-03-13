"""Module for tomography."""

import cupy as cp
import numpy as np
from ffttests.usfft1d import usfft1d
from ffttests.usfft2d import usfft2d
from ffttests.fft2d import fft2d
from ffttests import utils
import time

def eq2us1d(x, f, eps, N):
    # parameters for the USFFT transform
    N0 = N
    mu0 = -np.log(eps)/(2*N0**2)
    Te1 = 1/np.pi*np.sqrt(-mu0*np.log(eps)+(mu0*N0)**2/4)
    M0 = np.int32(np.ceil(2*N0*Te1))

    # smearing kernel (ker)
    ker = np.zeros([2*N0, 1, 1])
    xeq0 = np.arange(-N0//2, N0//2)
    ker[N0//2:N0//2+N0] = np.exp(-mu0*xeq0**2)[:, np.newaxis, np.newaxis]

    # FFT and compesantion for smearing
    fe = np.zeros([2*N0, f.shape[1], f.shape[2]], dtype=complex)
    fe[N0//2:N0//2+N0, :] = f/(2*N0)/(ker[N0//2:N0//2+N0])
    Fe0 = np.fft.fftshift(np.fft.fft(
        np.fft.fftshift(fe, axes=0), axis=0), axes=0)

    # wrapping array Fe0
    idx = np.arange(-M0, 2*N0+M0)
    idx0 = np.mod(idx+2*N0, 2*N0)
    Fe = np.zeros([2*N0+2*M0, *Fe0.shape[1:]], dtype=complex)
    Fe[idx+M0] = Fe0[idx0]

    # smearing operation (Fe=f*theta)
    F = np.zeros([x.shape[0], *f.shape[1:]], dtype=complex)
    for k in range(x.shape[0]):
        F[k] = 0
        ell0 = np.int32(np.floor(2*N0*x[k]))
        for i0 in range(2*M0+1):
            F[k] += Fe[N0+ell0+i0] * \
                np.sqrt(np.pi/mu0) * \
                (np.exp(-np.pi**2/mu0*((ell0-M0+i0)/(2*N0)-x[k])**2))
    return F


def eq2us2d(x, y, s, f, eps, N):
    # parameters for the USFFT transform
    [N0, N1] = N
    mu0 = -np.log(eps)/(2*N0**2)
    mu1 = -np.log(eps)/(2*N1**2)
    Te1 = 1/np.pi*np.sqrt(-mu0*np.log(eps)+(mu0*N0)**2/4)
    Te2 = 1/np.pi*np.sqrt(-mu1*np.log(eps)+(mu1*N1)**2/4)
    M0 = np.int32(np.ceil(2*N0*Te1))
    M1 = np.int32(np.ceil(2*N1*Te2))

    # smearing kernel (ker)
    ker = np.zeros((2*N0, 2*N1))
    [xeq0, xeq1] = np.mgrid[-N0//2:N0//2, -N1//2:N1//2]
    ker[N0//2:N0//2+N0, N1//2:N1//2+N1] = np.exp(-mu0*xeq0**2-mu1*xeq1**2)
    # FFT and compesantion for smearing
    fe = np.zeros([f.shape[0], 2*N0, 2*N1], dtype=complex)
    fe[:, N0//2:N0//2+N0, N1//2:N1//2+N1] = f / \
        (2*N0*2*N1)/ker[N0//2:N0//2+N0, N1//2:N1//2+N1]
    Fe0 = np.fft.fftshift(np.fft.fft2(
        np.fft.fftshift(fe, axes=[1, 2])), axes=[1, 2])

    # wrapping array Fe0
    [idx, idy] = np.mgrid[-M0:2*N0+M0, -M1:2*N1+M1]
    idx0 = np.mod(idx+2*N0, 2*N0)
    idy0 = np.mod(idy+2*N1, 2*N1)
    Fe = np.zeros([f.shape[0], 2*N0+2*M0, 2*N1+2*M1], dtype=complex)
    Fe[:, idx+M0, idy+M1] = Fe0[:, idx0, idy0]

    # smearing operation (Fe=f*theta)
    F = np.zeros([x.shape[0]], dtype=complex)
    for k in range(x.shape[0]):
        F[k] = 0
        ell0 = np.int32(np.floor(2*N0*y[k]))
        ell1 = np.int32(np.floor(2*N1*x[k]))
        for i0 in range(2*M0+1):
            for i1 in range(2*M1+1):
                F[k] += Fe[s[k], N0+ell0+i0, N1+ell1+i1] * np.pi/np.sqrt(mu0*mu1)*(np.exp(-np.pi**2/mu0*(
                    (ell0-M0+i0)/(2*N0)-y[k])**2-np.pi**2/mu1*((ell1-M1+i1)/(2*N1)-x[k])**2))
    return F


class FFTCL():
    def __init__(self, n0, n1, n2, detw, deth, ntheta, n1c=None, dethc=None, nthetac=None):
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
        self.read_threads = []
        for k in range(16):#16 is probably enough but can be changed
            self.write_threads.append(utils.WRThread())
            self.read_threads.append(utils.WRThread())            
        
        self.out_t =  utils.pinned_array(np.zeros([16, self.deth, self.n1c, self.n2], dtype='complex64'))
        self.inp_t =  utils.pinned_array(np.zeros([16, self.n0, self.n1c, self.n2], dtype='complex64'))

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.cl_usfft1d.free()
        self.cl_usfft2d.free()

    # def fwd_usfft1d(self, u, phi):
    #     res = cp.zeros([self.deth, self.n1, self.n2], dtype='complex64')
    #     x = (cp.arange(-self.deth//2, self.deth//2) /
    #          self.deth*cp.sin(phi)).astype('float32')
    #     u_gpu = cp.array(u)
    #     self.cl_usfft1d.fwd(res.data.ptr, u_gpu.data.ptr, x.data.ptr)
    #     # ff = eq2us1d(x.get(), u, 1e-3, self.n0).astype('complex64')
    #     # print(f'norm ff :{np.linalg.norm(ff)}')
    #     # print(f'norm res: {np.linalg.norm(res.get())}')
    #     # print(f'error: {np.linalg.norm(ff-res.get())}')
    #     return res.get()

    # def fwd_usfft2d(self, u, theta, phi):
    #     res = cp.zeros([self.ntheta, self.deth, self.detw], dtype='complex64')
    #     # res = cp.zeros([self.n0, (2*self.n1+2*4), (2*self.n2+2*4)], dtype='complex64')

    #     u_gpu = cp.array(u)
    #     x = cp.zeros([self.ntheta, self.deth * self.detw], dtype='float32')
    #     y = cp.zeros([self.ntheta, self.deth * self.detw], dtype='float32')

    #     [ku, kv] = cp.meshgrid(cp.arange(-self.detw//2, self.detw//2) /
    #                            self.detw, cp.arange(-self.deth//2, self.deth//2)/self.deth)
    #     ku = ku.flatten()
    #     kv = kv.flatten()

    #     for itheta in range(self.ntheta):
    #         x[itheta] = ku*cp.cos(theta[itheta])+kv * \
    #             cp.sin(theta[itheta])*cp.cos(phi)
    #         y[itheta] = ku*cp.sin(theta[itheta])-kv * \
    #             cp.cos(theta[itheta])*cp.cos(phi)
    #     x[x >= 0.5] = 0.5 - 1e-5
    #     x[x < -0.5] = -0.5 + 1e-5
    #     y[y >= 0.5] = 0.5 - 1e-5
    #     y[y < -0.5] = -0.5 + 1e-5
    #     self.cl_usfft2d.fwd(res.data.ptr, u_gpu.data.ptr,
    #                         x.data.ptr, y.data.ptr)
    #     # s = np.tile(np.arange(self.detw*self.deth)//self.detw, self.ntheta)
    #     # ff = eq2us2d(x.flatten().get(), y.flatten().get(), s, u, 1e-3,
    #     #              [self.n1, self.n2]).astype('complex64').reshape([self.ntheta, self.deth, self.detw])
    #     # print(f'norm ff :{np.linalg.norm(ff)}')
    #     # print(f'norm res: {np.linalg.norm(res.get())}')
    #     # print(f'error: {np.linalg.norm(ff-res.get())}')
    #     return res.get()

    # def adj_fft2d(self, u):
    #     u_gpu = cp.array(u)
    #     self.cl_fft2d.adj(u_gpu.data.ptr)
    #     return u_gpu.get()

    @profile
    def fwd_usfft1d_chunks(self, inp, phi):
        out = np.zeros([self.deth, self.n1, self.n2], dtype='complex64')
        inp_gpu = cp.zeros([self.n0, self.n1c, self.n2], dtype='complex64')
        out_gpu = cp.zeros([self.deth, self.n1c, self.n2], dtype='complex64')
        
        ku = (cp.arange(-self.deth//2, self.deth//2) /
              self.deth*cp.sin(phi)).astype('float32')
                
        t = time.time()
        for k in range(self.n1//self.n1c):
            st = k*self.n1c
            end = (k+1)*self.n1c            
            
            ithread = utils.find_free_thread(self.read_threads) 
            self.inp_t[ithread] = inp[:, k*self.n1c:(k+1)*self.n1c]
            # self.read_threads[ithread].run(utils.read_array, (self.inp_t[ithread], inp, st, end))            
            
            inp_gpu.set(self.inp_t[ithread])
            self.cl_usfft1d.fwd(out_gpu.data.ptr, inp_gpu.data.ptr, ku.data.ptr)
            cp.cuda.Device(0).synchronize()
            ithread = utils.find_free_thread(self.write_threads) 
            out_gpu.get(out=self.out_t[ithread])                    
            
            self.write_threads[ithread].run(utils.write_array, (out, self.out_t[ithread], st, end))            
        print(time.time()-t)
        
        for t in self.read_threads:
            t.join()
        for t in self.write_threads:
            t.join()

        return out
    
    def fwd_usfft2d_chunks(self, inp, theta, phi):
        out = np.zeros([self.ntheta, self.deth, self.detw], dtype='complex64')
        inp_gpu = cp.zeros([self.dethc, self.n1, self.n2], dtype='complex64')
        out_gpu = cp.zeros(
            [self.ntheta, self.dethc, self.detw], dtype='complex64')

        x_gpu = cp.zeros([self.ntheta, self.dethc, self.detw], dtype='float32')
        y_gpu = cp.zeros([self.ntheta, self.dethc, self.detw], dtype='float32')

        ku0 = (cp.arange(-self.deth//2, self.deth//2) /
               self.deth).astype('float32')
        
        t= time.time()
        for k in range(self.deth//self.dethc):
            # inp_gpu = cp.array(inp[k*self.dethc:(k+1)*self.dethc])
            inp_gpu.set(inp[k*self.dethc:(k+1)*self.dethc])
            [ku, kv] = np.meshgrid(cp.arange(-self.detw//2, self.detw//2).astype('float32') /
                                   self.detw, ku0[k*self.dethc:(k+1)*self.dethc])
            for itheta in range(self.ntheta):
                x_gpu[itheta] = ku*cp.cos(theta[itheta])+kv * \
                    cp.sin(theta[itheta])*cp.cos(phi)
                y_gpu[itheta] = ku*np.sin(theta[itheta])-kv * \
                    cp.cos(theta[itheta])*cp.cos(phi)
            x_gpu[x_gpu >= 0.5] = 0.5 - 1e-5
            x_gpu[x_gpu < -0.5] = -0.5 + 1e-5
            y_gpu[y_gpu >= 0.5] = 0.5 - 1e-5
            y_gpu[y_gpu < -0.5] = -0.5 + 1e-5

            self.cl_usfft2d.fwd(
                out_gpu.data.ptr, inp_gpu.data.ptr, x_gpu.data.ptr, y_gpu.data.ptr)
            out[:, k*self.dethc:(k+1)*self.dethc] = out_gpu.get()
        print(time.time()-t)
        
        return out
    
    def fwd_fft2_chunks(self, inp):
        t= time.time()
        inp_gpu = cp.zeros(
            [self.nthetac, self.deth, self.detw], dtype='complex64')
        for k in range(self.ntheta//self.nthetac):
            inp_gpu.set(inp[k*self.nthetac:(k+1)*self.nthetac])
            self.cl_fft2d.adj(inp_gpu.data.ptr)
            inp[k*self.nthetac:(k+1)*self.nthetac] = inp_gpu.get()
        print(time.time()-t)
        return inp   
    
    def fwd_lam(self, u, theta, phi):

        # step 1: 1d batch usffts in the z direction to the grid ku*sin(phi)
        # input [self.n0, self.n1, self.n2], output [self.deth, self.n1, self.n2]
        u = self.fwd_usfft1d_chunks(u, phi)        
        
        # step 2: 2d batch usffts in [x,y] direction to the grid ku*cos(theta)+kv * sin(theta)*cos(phi)
        # input [self.deth, self.n1, self.n2], output [self.ntheta, self.deth, self.detw]
        u = self.fwd_usfft2d_chunks(u, theta, phi)
        
        # step 3: 2d batch fft in [det x,det y] direction
        # input [self.ntheta, self.deth, self.detw], output [self.ntheta, self.deth, self.detw]
        u = self.fwd_fft2_chunks(u)
    
        return u

