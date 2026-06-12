"""Sinogram preprocessing for laminography: Fourier-Wavelet ring removal.

Pipeline position
-----------------
Apply on the *post-flat-field*, *post-negative-log* sinogram, before BH-CG
reconstruction. The standard tomography chain (and same here) is::

      raw counts I, dark D, flat F
              │
              ▼  I/I0 := (I − D) / (F − D)         (flat-field correction)
              │
              ▼  d := − log(I/I0)                 (Beer-Lambert)         ← minus_log_inplace
              │
              ▼  remove rings from d              (FW filter)            ← remove_stripe_fw_inplace
              │
              ▼  rec.BH(d)                        (reconstruction)

Rings come from per-pixel detector response (gain/offset drifts) that survive
flat-fielding and look additive in log-space — which is why FW is applied
*after* the log, not before. Applying before log gives stripes the wrong
multiplicative weighting and the filter under-corrects.

Algorithm and core routines (DWTForward, DWTInverse, afb1d, sfb1d, _reflect,
_mypad, remove_stripe_fw) are vendored from tomocupy
(https://github.com/tomography/tomocupy, see
src/tomocupy/processing/remove_stripe.py — UChicago Argonne, LLC, BSD-3).
Kept self-contained here so lam_usfft has no runtime dependency on tomocupy.
Local additions:
  * ``minus_log_inplace`` — Beer-Lambert transform with a safety clamp.
  * ``remove_stripe_fw_inplace`` — pinned-host wrapper that chunks the GPU FW
    call along deth so the full 4-D wavelet tower never needs to fit on the
    GPU at once (~4× the data size in transient memory inside FW).

For laminography data of shape ``(ntheta, deth, detw)`` rings appear as
stripes along ``ntheta`` at fixed ``(deth, detw)`` — the algorithm is
identical to the standard tomographic case (ntheta=nproj, deth=nz, detw=ni).
"""

from __future__ import annotations

import cupy as cp
import numpy as np

try:
    import pywt
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lam_usfft.remove_stripe requires PyWavelets (pip install pywavelets). "
    ) from e


__all__ = [
    "DWTForward", "DWTInverse",
    "minus_log_inplace",
    "remove_stripe_fw", "remove_stripe_fw_inplace",
]


# =============================================================================
# Vendored DWT primitives — verbatim from tomocupy/processing/remove_stripe.py
# (small reformatting only; semantics unchanged).
# =============================================================================

def _reflect(x, minx, maxx):
    """Reflect values in *x* about scalar bounds *minx* and *maxx*."""
    x = cp.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = cp.fmod(x - minx, rng_by_2)
    normed_mod = cp.where(mod < 0, mod + rng_by_2, mod)
    out = cp.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return cp.array(out, dtype=x.dtype)


def _mypad(x, pad, value=0):
    """numpy-like 2-D padding via reflect index gather (vertical OR horizontal)."""
    if pad[0] == 0 and pad[1] == 0:
        m1, m2 = pad[2], pad[3]
        l = x.shape[-2]
        xe = _reflect(cp.arange(-m1, l + m2, dtype="int32"), -0.5, l - 0.5)
        return x[:, :, xe]
    elif pad[2] == 0 and pad[3] == 0:
        m1, m2 = pad[0], pad[1]
        l = x.shape[-1]
        xe = _reflect(cp.arange(-m1, l + m2, dtype="int32"), -0.5, l - 0.5)
        return x[:, :, :, xe]


def afb1d(x, h0, h1="zero", dim=-1):
    """1-D analysis filter bank: stride-2 convolution along ``dim``.

    Output channels interleaved [lo_0, hi_0, lo_1, hi_1, ...] to match the
    grouped-convolution ordering expected by ``DWTForward``.
    """
    C = x.shape[1]
    d = dim % 4
    N = x.shape[d]
    h0f = h0.flatten()
    h1f = h1.flatten()
    L = h0f.size
    outsize = pywt.dwt_coeff_len(N, L, mode="symmetric")
    p = 2 * (outsize - 1) - N + L
    pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
    x = _mypad(x, pad=pad)
    B = x.shape[0]
    if d == 3:
        H = x.shape[2]
        out = cp.empty((B, C, 2, H, outsize), dtype="float32")
        sl0 = x[:, :, :, 0:2 * outsize:2]
        out[:, :, 0] = h0f[0] * sl0
        out[:, :, 1] = h1f[0] * sl0
        for j in range(1, L):
            sl = x[:, :, :, j:j + 2 * outsize:2]
            out[:, :, 0] += h0f[j] * sl
            out[:, :, 1] += h1f[j] * sl
    else:
        W = x.shape[3]
        out = cp.empty((B, C, 2, outsize, W), dtype="float32")
        sl0 = x[:, :, 0:2 * outsize:2, :]
        out[:, :, 0] = h0f[0] * sl0
        out[:, :, 1] = h1f[0] * sl0
        for i in range(1, L):
            sl = x[:, :, i:i + 2 * outsize:2, :]
            out[:, :, 0] += h0f[i] * sl
            out[:, :, 1] += h1f[i] * sl
    return out.reshape(B, 2 * C, *out.shape[3:])


def sfb1d(lo, hi, g0, g1="zero", dim=-1):
    """1-D synthesis filter bank: scatter-add (upsampled transposed conv)."""
    C = lo.shape[1]
    d = dim % 4
    g0f = g0.flatten()
    g1f = g1.flatten()
    L = g0f.size
    B = lo.shape[0]
    if d == 3:
        H, W = lo.shape[2], lo.shape[3]
        wi = (W - 1) * 2 + L
        out = cp.zeros((B, C, H, wi), dtype="float32")
        for j in range(L):
            out[:, :, :, j:j + 2 * W:2] += g0f[j] * lo + g1f[j] * hi
        return out[:, :, :, (L - 2):wi - (L - 2)]
    else:
        H, W = lo.shape[2], lo.shape[3]
        hi_size = (H - 1) * 2 + L
        out = cp.zeros((B, C, hi_size, W), dtype="float32")
        for i in range(L):
            out[:, :, i:i + 2 * H:2, :] += g0f[i] * lo + g1f[i] * hi
        return out[:, :, (L - 2):hi_size - (L - 2), :]


class DWTForward:
    """2-D DWT forward decomposition."""

    def __init__(self, wave="db1"):
        wave = pywt.Wavelet(wave)
        h0_col, h1_col = wave.dec_lo, wave.dec_hi
        h0_row, h1_row = h0_col, h1_col
        self.h0_col = cp.array(h0_col).astype("float32")[::-1].reshape((1, 1, -1, 1))
        self.h1_col = cp.array(h1_col).astype("float32")[::-1].reshape((1, 1, -1, 1))
        self.h0_row = cp.array(h0_row).astype("float32")[::-1].reshape((1, 1, 1, -1))
        self.h1_row = cp.array(h1_row).astype("float32")[::-1].reshape((1, 1, 1, -1))

    def apply(self, x):
        lohi = afb1d(x, self.h0_row, self.h1_row, dim=3)
        y = afb1d(lohi, self.h0_col, self.h1_col, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        x = cp.ascontiguousarray(y[:, :, 0])
        yh = cp.ascontiguousarray(y[:, :, 1:])
        return x, yh


class DWTInverse:
    """2-D DWT inverse reconstruction."""

    def __init__(self, wave="db1"):
        wave = pywt.Wavelet(wave)
        g0_col, g1_col = wave.rec_lo, wave.rec_hi
        g0_row, g1_row = g0_col, g1_col
        self.g0_col = cp.array(g0_col).astype("float32").reshape((1, 1, -1, 1))
        self.g1_col = cp.array(g1_col).astype("float32").reshape((1, 1, -1, 1))
        self.g0_row = cp.array(g0_row).astype("float32").reshape((1, 1, 1, -1))
        self.g1_row = cp.array(g1_row).astype("float32").reshape((1, 1, 1, -1))

    def apply(self, coeffs):
        yl, yh = coeffs
        # Two independent sfb1d(dim=2) batched as a single C=2 call.
        lo_hi = sfb1d(
            cp.concatenate([yl, yh[:, :, 1]], axis=1),
            cp.concatenate([yh[:, :, 0], yh[:, :, 2]], axis=1),
            self.g0_col, self.g1_col, dim=2,
        )
        yl = sfb1d(lo_hi[:, :1], lo_hi[:, 1:], self.g0_row, self.g1_row, dim=3)
        return yl


def remove_stripe_fw(data, sigma=3.0, wname="sym16", level=7):
    """Münch-Trtik-Hassan-Vogel-Stampanoni Fourier-Wavelet stripe removal.

    Parameters
    ----------
    data : cupy.ndarray (nproj, nz, ni) — sinogram block, on the GPU.
    sigma : Gaussian damping width along the projection-axis frequency.
    wname : PyWavelets wavelet name (e.g. 'sym16', 'db5').
    level : number of DWT decomposition levels.

    Returns the corrected sinogram (cupy array, same shape and dtype as ``data``).
    """
    nproj, nz, ni = data.shape
    nproj_pad = nproj + nproj // 8

    xfm = DWTForward(wave=wname)
    ifm = DWTInverse(wave=wname)

    cc = []
    sli = cp.zeros([nz, 1, nproj_pad, ni], dtype="float32")
    sli[:, 0,
        (nproj_pad - nproj) // 2:(nproj_pad + nproj) // 2] = (
            data.astype("float32").swapaxes(0, 1))

    for k in range(level):
        sli, c = xfm.apply(sli)
        cc.append(c)
        # Vertical bandpass (sensitive to horizontal-stripe artefacts).
        band = cc[k][:, 0, 1]
        _, my, mx = band.shape
        # rfft along the projection axis: half-spectrum is enough for real input.
        fcV = cp.fft.rfft(band, axis=1)
        myr = my // 2 + 1
        y_hat = cp.fft.ifftshift((cp.arange(-my, my, 2) + 1) / 2)[:myr]
        damp = -cp.expm1(-y_hat ** 2 / (2 * sigma ** 2))
        fcV *= damp[:, None]
        cc[k][:, 0, 1] = cp.fft.irfft(fcV, my, axis=1)

    for k in range(level)[::-1]:
        shape0 = cc[k][0, 0, 1].shape
        sli = sli[:, :, :shape0[0], :shape0[1]]
        sli = ifm.apply((sli, cc[k]))

    out = sli[:, 0,
              (nproj_pad - nproj) // 2:(nproj_pad + nproj) // 2,
              :ni].astype(data.dtype)
    return out.swapaxes(0, 1)


# =============================================================================
# Preprocessing helpers — pinned-host, chunked along deth.
# =============================================================================

def _chunk_iter(nz, dethc):
    """Yield (start, end) deth-axis slabs of width up to dethc."""
    dethc = min(int(dethc), nz)
    for st in range(0, nz, dethc):
        yield st, min(st + dethc, nz)


def minus_log_inplace(data, eps=1e-6, dethc=64):
    """In-place Beer-Lambert transform: data := -log(max(data, eps)).

    Applied to flat-field-corrected I/I0 to produce the sinogram for
    reconstruction. The ``eps`` clamp guards against zero/negative samples
    (dead pixels, over-corrected flatfields) that would otherwise produce
    +inf/NaN entries.

    Chunked along deth so the full (ntheta, deth, detw) volume need not
    reside on the GPU. Each chunk: H2D → maximum(eps) → log → negate → D2H.
    """
    if data.ndim != 3:
        raise ValueError(f"data must be 3-D (ntheta, deth, detw); got shape {data.shape}")
    if data.dtype != np.float32:
        raise ValueError(f"data must be float32; got {data.dtype}")
    eps_f32 = np.float32(eps)
    _, nz, _ = data.shape
    for st, end in _chunk_iter(nz, dethc):
        chunk = cp.asarray(data[:, st:end, :])           # H2D
        cp.maximum(chunk, eps_f32, out=chunk)
        cp.log(chunk, out=chunk)
        cp.negative(chunk, out=chunk)
        cp.asnumpy(chunk, out=data[:, st:end, :])        # D2H
    return data


def remove_stripe_fw_inplace(data, sigma=3.0, wname="sym16", level=7,
                              dethc=64):
    """In-place chunked FW ring removal on a pinned-host (ntheta, deth, detw)
    float32 array.

    The core ``remove_stripe_fw`` allocates ~4× the input as transient wavelet
    coefficients on the GPU. At n=1024 the full 4 GiB sinogram would balloon
    to ~16 GiB GPU — won't fit alongside Rec's chunking pools. Since the FW
    algorithm is independent across ``deth`` (each z-slice is filtered alone
    along the projection axis), chunking along that axis is loss-free.

    Each chunk: H2D one ``(ntheta, dethc, detw)`` slab, run FW on the GPU,
    D2H back into the original pinned array (overwrites the input slab).
    """
    if data.ndim != 3:
        raise ValueError(f"data must be 3-D (ntheta, deth, detw); got shape {data.shape}")
    if data.dtype != np.float32:
        raise ValueError(f"data must be float32; got {data.dtype}")
    _, nz, _ = data.shape
    for st, end in _chunk_iter(nz, dethc):
        chunk = cp.asarray(data[:, st:end, :])           # H2D
        chunk = remove_stripe_fw(chunk, sigma, wname, level)
        cp.asnumpy(chunk, out=data[:, st:end, :])        # D2H, overwrites in place
    return data
