import math
import numpy as np
import cupy as cp

from lam_usfft.cuda_kernels import (
    divker1d_kernel,
    fftshiftc1d_kernel,
    gather1d_kernel,
)


EPS = 1e-3


def _ceil_div(a, b):
    return -(-a // b)


class usfft1d:
    """1D unequally-spaced FFT batched over [n1, n2] along the n0 axis.

    Internal NUFFT grid is the unpadded buffer ``fdee1d`` of shape
    ``(2*n0, n1, n2)``. Gather/scatter wrap-around at the grid boundary is
    handled by modulo arithmetic inside gather1d itself (same pattern as
    holotomocupy_mpi/tomo.py). ``phi`` is fixed at construction so the gather
    z-coords can be precomputed once.

    Python attributes use geometric labels (n0 = NUFFT axis, n1 = chunk
    axis, n2 = transverse axis) — matching the array shapes the caller
    supplies. Kernel calls reverse the order to (n2, n1, n0) so the
    kernel's tx/ty/tz threads see (innermost, outermost-bound, middle)
    as expected by its C-order indexing.
    """

    def __init__(self, n0, n1, n2, deth, phi):
        # No axis reorder — geometric labels pass straight through.
        self.n0, self.n1, self.n2 = n0, n1, n2
        self.deth = deth
        self.phi  = float(phi)

        mu0 = -math.log(EPS) / (2 * n0 * n0)
        self.mu0 = np.float32(mu0)
        self.m0  = math.ceil(
            2 * n0 / math.pi
            * math.sqrt(-mu0 * math.log(EPS) + (mu0 * n0) ** 2 / 4)
        )
        self.nz = 2 * n0    # FFT length / fftshift bound

        # unpadded NUFFT grid: outermost axis is the NUFFT axis (size 2*n0).
        self.fdee1d = cp.empty((self.nz, n1, n2), dtype="complex64")

        # Precomputed gather z-coords (was take_x_1d kernel).
        self.x = cp.asarray(
            (np.arange(deth, dtype="float32") - deth / 2.0)
            / deth * np.float32(np.sin(self.phi))
        )

        # NOTE: No explicit cuFFT plan here. A `cufft.get_fft_plan(...,
        # axes=(0,))` does build correctly, but cupy's `with plan:` →
        # `cufft.fft(arr, axis=0)` dispatcher does NOT honor user-supplied
        # plans for non-trailing axes (it only does for axes=(-1,) / (-2,-1)).
        # So we fall back to plain `cp.fft.fft(arr, axis=0)`, which builds
        # and caches its own plan internally per (shape, dtype, axes).

        # Kernel block + grid sizes — bounds use the reversed (n2, n1, n0) order.
        self.bs = (16, 8, 8)
        self.gs0 = (_ceil_div(n2, self.bs[0]),
                    _ceil_div(n1, self.bs[1]),
                    _ceil_div(n0, self.bs[2]))
        self.gs1 = (_ceil_div(n2, self.bs[0]),
                    _ceil_div(n1, self.bs[1]),
                    _ceil_div(self.nz, self.bs[2]))
        self.gs2 = (_ceil_div(n2, self.bs[0]),
                    _ceil_div(n1, self.bs[1]),
                    _ceil_div(deth, self.bs[2]))

    def fwd(self, g, f):
        """Forward 1D NUFFT.

        Parameters
        ----------
        g : cupy.ndarray, complex64, output, shape (n1, deth, n2)
        f : cupy.ndarray, complex64, input,  shape (n1, n0,   n2)
        """
        self.fdee1d.fill(0)
        divker1d_kernel(self.gs0, self.bs,
                        (self.fdee1d, f, self.n2, self.n1, self.n0,
                         self.mu0, False))
        fftshiftc1d_kernel(self.gs1, self.bs,
                           (self.fdee1d, self.n2, self.n1, self.nz))
        self.fdee1d[:] = cp.fft.fft(self.fdee1d, axis=0)
        fftshiftc1d_kernel(self.gs1, self.bs,
                           (self.fdee1d, self.n2, self.n1, self.nz))
        gather1d_kernel(self.gs2, self.bs,
                        (g, self.fdee1d, self.x, self.m0, self.mu0,
                         self.n2, self.n1, self.n0, self.deth, False))

    def adj(self, f, g):
        """Adjoint 1D NUFFT.

        Parameters
        ----------
        f : cupy.ndarray, complex64, output, shape (n1, n0,   n2)
        g : cupy.ndarray, complex64, input,  shape (n1, deth, n2)
        """
        self.fdee1d.fill(0)
        gather1d_kernel(self.gs2, self.bs,
                        (g, self.fdee1d, self.x, self.m0, self.mu0,
                         self.n2, self.n1, self.n0, self.deth, True))
        fftshiftc1d_kernel(self.gs1, self.bs,
                           (self.fdee1d, self.n2, self.n1, self.nz))
        # CUFFT_INVERSE: no normalization → norm="forward" cancels cupy's 1/N
        self.fdee1d[:] = cp.fft.ifft(self.fdee1d, axis=0, norm="forward")
        fftshiftc1d_kernel(self.gs1, self.bs,
                           (self.fdee1d, self.n2, self.n1, self.nz))
        divker1d_kernel(self.gs0, self.bs,
                        (self.fdee1d, f, self.n2, self.n1, self.n0,
                         self.mu0, True))
