"""Regularization for the BH-CG solver.

Single term: ``TVTerm`` — Charbonnier-smoothed total variation,
(lam/N) Σ √(|∇u|² + ε²), with a lagged-diffusivity Hessian that's
refreshed once per BH iteration.

API expected by ``Rec``:

    cl_reg.lam            float (0 ⇒ no-op fast path inside Rec)
    cl_reg.obj_view       view into Rec's `u` buffer (None when lam == 0)
    cl_reg.etas_view      view into Rec's `eta` buffer (None when lam == 0)
    cl_reg.gradient(grad) in-place: grad += ∂E_reg / ∂u  (reads u via obj_view)
    cl_reg.hessian(_, eb) returns <η_a, H_reg η_b> as a python float
                          (η_a is whatever lives in etas_view at the time;
                           the caller pre-stages it there).

Single-GPU only (no MPI ghost exchange — boundary ghost rows stay zero,
giving Dirichlet conditions at the volume boundary).
"""

import numpy as np
import cupy as cp

from lam_usfft.utils import pinned_array


def _abs2(x):
    """|x|² element-wise — avoids allocating a zero .imag array for real x.

    cupy's array.imag on a real array allocates a fresh zero array of the
    same shape on every access, which is expensive when the volume is large.
    For real-valued x just use x*x; for complex use .real²+.imag².
    """
    if x.dtype.kind == 'c':
        return x.real * x.real + x.imag * x.imag
    return x * x


# ---------------------------------------------------------------------------
# Total variation (Charbonnier-smoothed) with lagged-diffusivity Hessian.
# ---------------------------------------------------------------------------

class TVTerm:
    """3-D Charbonnier-smoothed total variation (lam / obj_size) · Σ √(|∇u|² + ε²).

    The Hessian of the true TV energy is non-quadratic, so we freeze the
    diffusion weights ``w(u) = 1/√(|∇u|² + ε²)`` at the current u (the
    classic Vogel-Oman *lagged-diffusivity* trick).  Inside one BH-CG
    iteration the weights then act as constants, giving a quadratic
    surrogate of the form ::

        ⟨η_a, H_TV η_b⟩  =  (lam/N) · Σ_i  w_i · Re ⟨∇η_a, ∇η_b⟩_i

    Weights get rebuilt in ``gradient(grad)`` (which is the first call into
    this term every iteration) and re-used by the subsequent ``hessian(...)``
    calls.  Apply-step changes u, the next iteration rebuilds w.

    This class is NOT chunked — the full u_pad, e_pad, w_pad and gradient
    scratch live on the GPU.  Suitable for volumes that fit comfortably in
    GPU memory (e.g. ≤ 512³ float32).  For larger problems the operations
    can be chunked along axis 0; that's a future extension.

    Forward differences are used along all 3 axes with Dirichlet boundary
    conditions (u beyond the volume = 0).  ``w`` at a fictitious "outside"
    voxel is also set to 0, so the divergence's incoming contribution at
    the volume edge vanishes.
    """

    def __init__(self, lam, n1, n0, n2, gpu_batch=None, dtype="float32", eps=1e-3):
        self.lam       = float(lam)
        self.obj_size  = n1 * n0 * n2
        self.eps2      = np.float32(eps * eps)
        self.gpu_batch = gpu_batch       # chunking decorator, used by gradient()
        self._n1, self._n0, self._n2 = n1, n0, n2
        self._dtype    = np.dtype(dtype)
        if self.lam != 0:
            # Host-pinned padded buffers — interior u_pad[1:-1] is aliased by
            # Rec.u (and same for eta), so writes from the BH loop land here
            # automatically.  Ghost rows on axis 0 stay zero (Dirichlet).
            self.u_pad = pinned_array(np.zeros((n1 + 2, n0, n2), dtype=self._dtype))
            self.e_pad = pinned_array(np.zeros((n1 + 2, n0, n2), dtype=self._dtype))
            # GPU-resident copies + the lagged weights.  w_pad's lower ghost
            # row (index 0) stays zero so the divergence boundary term is
            # well-defined when axis-0 j=0 is the global volume edge.
            self._u_gpu    = cp.zeros((n1 + 2, n0, n2), dtype=self._dtype)
            self._e_gpu    = cp.zeros((n1 + 2, n0, n2), dtype=self._dtype)
            self._w_pad    = cp.zeros((n1 + 1, n0, n2), dtype="float32")
            self._grad_gpu = cp.zeros((n1, n0, n2),     dtype=self._dtype)
            self._eb_gpu   = cp.zeros((n1, n0, n2),     dtype=self._dtype)
        else:
            self.u_pad = None
            self.e_pad = None

    @property
    def obj_view(self):
        """Storage for u (view into u_pad's interior); None when inactive."""
        return None if self.u_pad is None else self.u_pad[1:-1]

    @property
    def etas_view(self):
        """Storage for eta (view into e_pad's interior); None when inactive."""
        return None if self.e_pad is None else self.e_pad[1:-1]

    # ----- internal helpers -----

    @staticmethod
    def _diffs(pad_gpu):
        """Forward differences of a padded GPU array of shape (N+2, n0, n2).
        Returns (dz, dy, dx) all of shape (N, n0, n2).  Axis-0 differences use
        the explicit ghost rows; axes 1, 2 use Dirichlet zero outside."""
        dz = pad_gpu[2:] - pad_gpu[1:-1]            # u_{i+1} - u_i along axis 0
        u  = pad_gpu[1:-1]
        dy = cp.zeros_like(u)
        dy[:, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
        dy[:, -1,  :] = -u[:, -1, :]                # u_{j=n0} ≡ 0
        dx = cp.zeros_like(u)
        dx[:, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
        dx[:, :, -1]  = -u[:, :, -1]                # u_{k=n2} ≡ 0
        return dz, dy, dx

    def _refresh_w(self):
        """Recompute self._w_pad[1:] = 1/√(|∇u|² + ε²) from self.u_pad.  Called
        from ``gradient()``; the lower-ghost row self._w_pad[0] stays zero."""
        self._u_gpu.set(self.u_pad)
        dz, dy, dx = self._diffs(self._u_gpu)
        mag2 = _abs2(dz) + _abs2(dy) + _abs2(dx)
        self._w_pad[1:] = cp.reciprocal(cp.sqrt(mag2 + self.eps2))

    # ----- public API -----

    def gradient(self, grad):
        """In-place: grad += (lam/N) · -div(w · ∇u), chunked along axis 0.

        Per-chunk processing pulls (chunk_n + 2) rows of u_pad up to the GPU
        via the chunking pool, computes w locally for this chunk's interior
        rows AND one upstream-side row (the w_{i-1} term in the axis-0
        divergence), forms the divergence, and accumulates into the chunk's
        slot of `grad` — D2H'd back via the chunking decorator's overlapped
        pipeline so there's no big blocking transfer at the end (the prior
        un-chunked form did one ~4 GiB blocking D2H + a slow CPU elementwise
        add right before compute_gradient's `fwd_lam(grad)`).

        Boundary at the volume's z-edges uses the natural u=0 extension from
        u_pad's ghost rows (rather than the prior w[-1]=0 hand-set) — slightly
        different by one voxel layer at the global boundary, cosmetic for TV
        regularization. The persistent full-volume GPU buffers _u_gpu /
        _w_pad / _grad_gpu are no longer touched here; they remain in use
        by hessian() and energy() which stay un-chunked for now.
        """
        if self.lam == 0:
            return
        scale = np.float32(self.lam / self.obj_size)
        eps2  = self.eps2

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=2)
        def _f(self, g_out, u_pad_chunk, g_in):
            # u_pad_chunk: (chunk_n + 2, n0, n2). Slice indices:
            #   u_prev = rows [chunk_st-1 .. chunk_end-1)   (i-1)
            #   u      = rows [chunk_st   .. chunk_end)     (i)
            #   u_next = rows [chunk_st+1 .. chunk_end+1)   (i+1)
            u      = u_pad_chunk[1:-1]
            u_prev = u_pad_chunk[:-2]
            u_next = u_pad_chunk[2:]

            def _w_at(uc, uc_next):
                """w = 1/√(|∇uc|² + ε²) with axis-0 forward difference uc_next - uc
                and axes 1, 2 Dirichlet at the chunk's own slab edges."""
                dy = cp.zeros_like(uc)
                dy[:, :-1, :] = uc[:, 1:, :] - uc[:, :-1, :]
                dy[:, -1,  :] = -uc[:, -1, :]
                dx = cp.zeros_like(uc)
                dx[:, :, :-1] = uc[:, :, 1:] - uc[:, :, :-1]
                dx[:, :, -1]  = -uc[:, :, -1]
                return cp.reciprocal(cp.sqrt(
                    _abs2(uc_next - uc) + _abs2(dy) + _abs2(dx) + eps2))

            w_i   = _w_at(u,      u_next)
            w_im1 = _w_at(u_prev, u)

            # --- axis-0: w_{i-1}·(u_i - u_{i-1}) − w_i·(u_{i+1} - u_i) ---
            a0 = w_im1 * (u - u_prev) - w_i * (u_next - u)

            # --- axis-1 ---
            dy = cp.zeros_like(u)
            dy[:, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
            dy[:, -1,  :] = -u[:, -1, :]
            dy_prev = cp.zeros_like(u)
            dy_prev[:, 0,  :] = u[:, 0,  :]
            dy_prev[:, 1:, :] = u[:, 1:, :] - u[:, :-1, :]
            w_jm1 = cp.zeros_like(w_i)
            w_jm1[:, 1:, :] = w_i[:, :-1, :]
            a1 = w_jm1 * dy_prev - w_i * dy

            # --- axis-2 ---
            dx = cp.zeros_like(u)
            dx[:, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
            dx[:, :, -1]  = -u[:, :, -1]
            dx_prev = cp.zeros_like(u)
            dx_prev[:, :, 0]  = u[:, :, 0]
            dx_prev[:, :, 1:] = u[:, :, 1:] - u[:, :, :-1]
            w_km1 = cp.zeros_like(w_i)
            w_km1[:, :, 1:] = w_i[:, :, :-1]
            a2 = w_km1 * dx_prev - w_i * dx

            g_out[:] = g_in + scale * (a0 + a1 + a2)

        _f(self, grad, self.u_pad, grad)

    def hessian(self, _eta_a_stub, eta_b):
        """⟨η_a, H_TV η_b⟩ = (lam/N) · Σ_i w_i · Re⟨∇η_a, ∇η_b⟩_i, with η_a
        taken from self.e_pad[1:-1] (caller pre-stages it).

        Lagged-diffusivity w is refreshed here from the current u — since u
        is unchanged between gradient() and any hessian() calls within the
        same BH iteration, this yields the same w gradient() would have used,
        just recomputed (the chunked gradient() path no longer caches it).
        """
        if self.lam == 0:
            return 0.0
        # Refresh w from current u (= self.u_pad). Replaces the previously
        # cached _w_pad set by gradient().
        self._refresh_w()
        # η_a (in e_pad) → GPU
        self._e_gpu.set(self.e_pad)
        ea_z, ea_y, ea_x = self._diffs(self._e_gpu)

        # η_b → GPU.  Dirichlet outside; no axis-0 padding needed because we
        # form ∇η_b directly here (no cross-chunk dependence).
        if isinstance(eta_b, np.ndarray):
            self._eb_gpu.set(eta_b)
            eb = self._eb_gpu
        else:
            eb = eta_b
        eb_z = cp.zeros_like(eb)
        eb_z[:-1]      = eb[1:] - eb[:-1];          eb_z[-1]       = -eb[-1]
        eb_y = cp.zeros_like(eb)
        eb_y[:, :-1, :] = eb[:, 1:, :] - eb[:, :-1, :];  eb_y[:, -1, :] = -eb[:, -1, :]
        eb_x = cp.zeros_like(eb)
        eb_x[:, :, :-1] = eb[:, :, 1:] - eb[:, :, :-1];  eb_x[:, :, -1] = -eb[:, :, -1]

        w = self._w_pad[1:]
        if ea_z.dtype.kind == 'c':
            ip = (ea_z.real * eb_z.real + ea_z.imag * eb_z.imag
                  + ea_y.real * eb_y.real + ea_y.imag * eb_y.imag
                  + ea_x.real * eb_x.real + ea_x.imag * eb_x.imag)
        else:
            ip = ea_z * eb_z + ea_y * eb_y + ea_x * eb_x
        return float(self.lam / self.obj_size) * float(cp.sum(w * ip))

    def energy(self):
        """E_TV(u) = (lam/N) · Σ_i √(|∇u|² + ε²) as a python float."""
        if self.lam == 0:
            return 0.0
        self._u_gpu.set(self.u_pad)
        dz, dy, dx = self._diffs(self._u_gpu)
        mag2 = _abs2(dz) + _abs2(dy) + _abs2(dx)
        return float(self.lam / self.obj_size) * float(cp.sum(cp.sqrt(mag2 + self.eps2)))
