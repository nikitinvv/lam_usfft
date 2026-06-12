import math
import numpy as np
import cupy as cp
from cupy.cuda import cufft

from lam_usfft.cuda_kernels import (
    divker2d_kernel,
    fftshiftc2d_kernel,
    gather2d_kernel,
)


EPS = 1e-3


def _ceil_div(a, b):
    return -(-a // b)


class usfft2d:
    """2D unequally-spaced FFT batched over n0 along the (n1, n2) NUFFT axes.

    The caller supplies precomputed ``x, y`` cupy arrays of shape
    ``(ntheta, n0, detw)`` holding the gather coordinates for the current
    chunk; they are computed once per (theta, phi) on the Rec side (see
    ``Rec.make_geometry``) and sliced along the deth axis per chunk by the
    surrounding ``gpu_batch`` decorator.

    Internal NUFFT grid is the unpadded buffer ``fdee2d`` of shape
    ``(n0, 2*n1, 2*n2)``. Wrap-around at the grid boundaries is folded into
    gather2d via modulo arithmetic (matching holotomocupy_mpi/tomo.py).

    Python attributes use geometric labels (n0 = batch = dethc, n1 / n2 are
    the two NUFFT axes); kernel calls reverse the order — (n2, n1, n0) for
    the size triple, (m2, m1) for stencil widths, (mu2, mu1) for the
    NUFFT damping — so the kernel's tx/ty/tz threads see (innermost,
    outermost-bound, middle) as expected by its C-order indexing.
    """

    def __init__(self, n0, n1, n2, ntheta, detw, deth):
        # No axis reorder — geometric labels pass straight through.
        self.n0     = n0
        self.n1     = n1
        self.n2     = n2
        self.ntheta = ntheta
        self.detw   = detw
        self.deth   = deth

        mu1 = -math.log(EPS) / (2 * n1 * n1)
        mu2 = -math.log(EPS) / (2 * n2 * n2)
        self.mu1 = np.float32(mu1)
        self.mu2 = np.float32(mu2)
        self.m1  = math.ceil(
            2 * n1 / math.pi
            * math.sqrt(-mu1 * math.log(EPS) + (mu1 * n1) ** 2 / 4)
        )
        self.m2  = math.ceil(
            2 * n2 / math.pi
            * math.sqrt(-mu2 * math.log(EPS) + (mu2 * n2) ** 2 / 4)
        )
        self.n1p = 2 * n1
        self.n2p = 2 * n2

        # unpadded NUFFT grid: (n0=batch, 2*n1, 2*n2)
        self.fdee2d = cp.empty((n0, self.n1p, self.n2p), dtype="complex64")

        # Batched 2D FFT (rank-2, batch=n0) over the (2*n1, 2*n2) plane.
        self.plan = cufft.PlanNd(
            (self.n1p, self.n2p),
            (self.n1p, self.n2p), 1, self.n1p * self.n2p,
            (self.n1p, self.n2p), 1, self.n1p * self.n2p,
            cufft.CUFFT_C2C, n0,
            "C", -1, None,
        )

        self.bs = (16, 8, 8)
        self.gs0 = (_ceil_div(n2, self.bs[0]),
                    _ceil_div(n1, self.bs[1]),
                    _ceil_div(n0, self.bs[2]))
        self.gs1 = (_ceil_div(self.n2p, self.bs[0]),
                    _ceil_div(self.n1p, self.bs[1]),
                    _ceil_div(n0, self.bs[2]))
        self.gs2 = (_ceil_div(detw, self.bs[0]),
                    _ceil_div(deth, self.bs[1]),
                    _ceil_div(ntheta, self.bs[2]))

    def fwd(self, g, f, x, y):
        """Forward 2D NUFFT.

        Parameters
        ----------
        g : cupy.ndarray, complex64, output, shape (ntheta, n0 (=dethc), detw)
        f : cupy.ndarray, complex64, input,  shape (n1,     n0 (=dethc), n2)
        x, y : cupy.ndarray, float32, shape (ntheta, n0 (=dethc), detw) — gather coords
        """
        self.fdee2d.fill(0)
        divker2d_kernel(self.gs0, self.bs,
                        (self.fdee2d, f, self.n2, self.n1, self.n0,
                         self.mu2, self.mu1, False))
        fftshiftc2d_kernel(self.gs1, self.bs,
                           (self.fdee2d, self.n2p, self.n1p, self.n0))
        self.plan.fft(self.fdee2d, self.fdee2d, cufft.CUFFT_FORWARD)
        fftshiftc2d_kernel(self.gs1, self.bs,
                           (self.fdee2d, self.n2p, self.n1p, self.n0))
        gather2d_kernel(self.gs2, self.bs,
                        (g, self.fdee2d, x, y,
                         self.m2, self.m1, self.mu2, self.mu1,
                         self.n2, self.n1, self.n0,
                         self.detw, self.deth, self.ntheta, False))

    def adj(self, f, g, x, y):
        """Adjoint 2D NUFFT.

        Parameters
        ----------
        f : cupy.ndarray, complex64, output, shape (n1,     n0 (=dethc), n2)
        g : cupy.ndarray, complex64, input,  shape (ntheta, n0 (=dethc), detw)
        x, y : cupy.ndarray, float32, shape (ntheta, n0 (=dethc), detw) — gather coords
        """
        self.fdee2d.fill(0)
        gather2d_kernel(self.gs2, self.bs,
                        (g, self.fdee2d, x, y,
                         self.m2, self.m1, self.mu2, self.mu1,
                         self.n2, self.n1, self.n0,
                         self.detw, self.deth, self.ntheta, True))
        fftshiftc2d_kernel(self.gs1, self.bs,
                           (self.fdee2d, self.n2p, self.n1p, self.n0))
        self.plan.fft(self.fdee2d, self.fdee2d, cufft.CUFFT_INVERSE)
        fftshiftc2d_kernel(self.gs1, self.bs,
                           (self.fdee2d, self.n2p, self.n1p, self.n0))
        divker2d_kernel(self.gs0, self.bs,
                        (self.fdee2d, f, self.n2, self.n1, self.n0,
                         self.mu2, self.mu1, True))
