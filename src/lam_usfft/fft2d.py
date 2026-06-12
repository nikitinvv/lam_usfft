import numpy as np
import cupy as cp
from cupy.cuda import cufft

from lam_usfft.cuda_kernels import fftshiftc2d_kernel, mulc_kernel


def _ceil_div(a, b):
    return -(-a // b)


class fft2d:
    """Batched 2D FFT over (deth, detw) for ntheta batches."""

    def __init__(self, ntheta, detw, deth):
        self.ntheta = ntheta
        self.detw   = detw
        self.deth   = deth
        self.inv_n  = np.float32(1.0 / (deth * detw))

        self.plan = cufft.PlanNd(
            (deth, detw),
            (deth, detw), 1, deth * detw,
            (deth, detw), 1, deth * detw,
            cufft.CUFFT_C2C, ntheta,
            "C", -1, None,
        )

        self.bs = (32, 32, 1)
        self.gs = (_ceil_div(detw, self.bs[0]),
                   _ceil_div(deth, self.bs[1]),
                   _ceil_div(ntheta, self.bs[2]))

    def fwd(self, g, f):
        """Forward batched 2D FFT.

        Parameters
        ----------
        g : cupy.ndarray, complex64, output, shape (ntheta, deth, detw)
        f : cupy.ndarray, complex64, input,  shape (ntheta, deth, detw)
        """
        fftshiftc2d_kernel(self.gs, self.bs,
                           (f, self.detw, self.deth, self.ntheta))
        self.plan.fft(f, g, cufft.CUFFT_FORWARD)
        fftshiftc2d_kernel(self.gs, self.bs,
                           (g, self.detw, self.deth, self.ntheta))
        mulc_kernel(self.gs, self.bs,
                    (g, self.detw, self.deth, self.ntheta, self.inv_n))

    def adj(self, g, f):
        """Adjoint (inverse) batched 2D FFT.

        Parameters
        ----------
        g : cupy.ndarray, complex64, output, shape (ntheta, deth, detw)
        f : cupy.ndarray, complex64, input,  shape (ntheta, deth, detw)
        """
        fftshiftc2d_kernel(self.gs, self.bs,
                           (f, self.detw, self.deth, self.ntheta))
        self.plan.fft(f, g, cufft.CUFFT_INVERSE)
        fftshiftc2d_kernel(self.gs, self.bs,
                           (g, self.detw, self.deth, self.ntheta))
        mulc_kernel(self.gs, self.bs,
                    (g, self.detw, self.deth, self.ntheta, self.inv_n))
