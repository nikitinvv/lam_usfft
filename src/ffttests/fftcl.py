"""Module for tomography."""

import cupy as cp
import numpy as np
from ffttests.usfft1d import usfft1d
from ffttests.usfft2d import usfft2d


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
    def __init__(self, n0, n1, n2, detw, deth, ntheta):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.detw = detw
        self.deth = deth
        self.ntheta = ntheta

        self.cl_usfft1d = usfft1d(n0, n1, n2, deth)
        self.cl_usfft2d = usfft2d(n0, n1, n2, self.ntheta, detw, deth)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.cl_usfft1d.free()
        self.cl_usfft2d.free()
        

    def fwd_fft1d(self, u, phi):
        res = cp.zeros([self.deth, self.n1, self.n2], dtype='complex64')
        x = (cp.arange(-self.deth//2, self.deth//2) / self.deth*cp.sin(phi)).astype('float32')
        u_gpu = cp.array(u)
        self.cl_usfft1d.fwd(res.data.ptr, u_gpu.data.ptr, x.data.ptr)
        ff = eq2us1d(x.get(), u, 1e-3, self.n0).astype('complex64')
        print(f'norm ff :{np.linalg.norm(ff)}')
        print(f'norm res: {np.linalg.norm(res.get())}')
        print(f'error: {np.linalg.norm(ff-res.get())}')
        return res.get()

    def fwd_fft2d(self, u, theta, phi):
        res = cp.zeros([self.ntheta, self.deth, self.detw], dtype='complex64')
        # res = cp.zeros([self.n0, (2*self.n1+2*4), (2*self.n2+2*4)], dtype='complex64')

        u_gpu = cp.array(u)
        x = cp.zeros([self.ntheta, self.deth * self.detw], dtype='float32')
        y = cp.zeros([self.ntheta, self.deth * self.detw], dtype='float32')

        [ku, kv] = cp.meshgrid(cp.arange(-self.detw//2, self.detw//2) /
                               self.detw, cp.arange(-self.deth//2, self.deth//2)/self.deth)
        ku = ku.flatten()
        kv = kv.flatten()

        for itheta in range(self.ntheta):
            x[itheta] = ku*cp.cos(theta[itheta])+kv * \
                cp.sin(theta[itheta])*cp.cos(phi)
            y[itheta] = ku*cp.sin(theta[itheta])-kv * \
                cp.cos(theta[itheta])*cp.cos(phi)
        x[x >= 0.5] = 0.5 - 1e-5
        x[x < -0.5] = -0.5 + 1e-5
        y[y >= 0.5] = 0.5 - 1e-5
        y[y < -0.5] = -0.5 + 1e-5
        self.cl_usfft2d.fwd(res.data.ptr, u_gpu.data.ptr,
                            x.data.ptr, y.data.ptr)
        s = np.tile(np.arange(self.detw*self.deth)//self.detw, self.ntheta)
        ff = eq2us2d(x.flatten().get(), y.flatten().get(), s, u, 1e-3,
                     [self.n1, self.n2]).astype('complex64').reshape([self.ntheta, self.deth, self.detw])
        print(f'norm ff :{np.linalg.norm(ff)}')
        print(f'norm res: {np.linalg.norm(res.get())}')
        print(f'error: {np.linalg.norm(ff-res.get())}')
        return res.get()
