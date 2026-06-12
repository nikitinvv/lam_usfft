import numpy as np
import cupy as cp


def pinned_array(array):
    """Allocate pinned host memory and associate it with a numpy array."""
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


def pinned_empty(shape, dtype):
    """Allocate uninitialized pinned host memory of the given shape/dtype.

    Unlike pinned_array(np.empty(...)), this skips both the numpy allocation
    AND the copy into pinned memory — use for scratch buffers that are fully
    written before they're read. At n=1024 (8 GiB complex64 buffers) each
    avoided memset+copy is ~3 GiB of pointless host bandwidth.
    """
    dt = np.dtype(dtype)
    n  = int(np.prod(shape))
    mem = cp.cuda.alloc_pinned_memory(n * dt.itemsize)
    return np.frombuffer(mem, dt, n).reshape(shape)


def redot(a, b):
    """Real-part inner product <a, b>_R = sum(re(a)*re(b) + im(a)*im(b))."""
    return cp.vdot(a.view("float32"), b.view("float32"))


def lap(a, b, c):
    """Discrete 3-D Laplacian stencil; b is the centre slice, a/c are its z-neighbours.

    Returns ∇²b ≈ a + c + b[..,y±1] + b[..,x±1] − 6·b.
    """
    return (a + c
            + cp.roll(b, -1, axis=1) + cp.roll(b, 1, axis=1)
            + cp.roll(b, -1, axis=2) + cp.roll(b, 1, axis=2)
            - 6 * b)


def paddata(data, ne):
    """Pad tomography projections along the last axis."""
    n = data.shape[-1]
    return np.pad(data, ((0, 0), (0, 0), (ne // 2 - n // 2, ne // 2 - n // 2)), "edge")


def unpaddata(data, n):
    """Unpad tomography projections."""
    ne = data.shape[-1]
    return data[:, :, ne // 2 - n // 2 : ne // 2 + n // 2]


def unpadobject(f, n):
    """Unpad a 3-D object on the last two axes."""
    ne = f.shape[-1]
    return f[:, ne // 2 - n // 2 : ne // 2 + n // 2, ne // 2 - n // 2 : ne // 2 + n // 2]
