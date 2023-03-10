"""Module for tomography."""

import cupy as cp
import numpy as np
from ffttests.fft import fft


class FFTCL(fft):
    """Base class for laminography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a laminography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    n0 : int
        Object size in x.
    n1 : int
        Object size in y.
    n2 : int
        Object size in z.
    """

    def __init__(self, n0, n1, n2, m):
        # create class for the tomo transform associated with first gpu
        super().__init__(n0, n1, n2, m)  # reorder sizes?

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_fft1d(self, u):
        """Laminography transform (L)"""
        res = cp.zeros([2*self.n0+2*self.m, self.n1, self.n2], dtype='complex64')

        u_gpu = cp.array(u)
        
        # C++ wrapper, send pointers to GPU arrays
        self.fwd1d(res.data.ptr, u_gpu.data.ptr)
        return res.get()

    def fwd_fft2d(self, u):
        """Laminography transform (L)"""
        res = cp.zeros([self.n0, 2*self.n1+2*self.m, 2*self.n2+2*self.m], dtype='complex64')

        u_gpu = cp.array(u)
        
        # C++ wrapper, send pointers to GPU arrays
        self.fwd2d(res.data.ptr, u_gpu.data.ptr)
        return res.get()
