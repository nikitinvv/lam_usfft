"""Bilinear-Hessian CG reconstruction for the linear laminography forward
operator L = fft2d ∘ usfft2d ∘ usfft1d.

Specialisation of holotomocupy_mpi/rec_mpi.py Rec:
    - L is purely linear, so the holography cascade chain rule collapses to a
      single fwd_lam / adj_lam pair (H η = LᵀL η + reg.hessian(η)).
    - Single variable `u` (no coupled prb / pos / proj), so vars/grads/etas
      flatten to plain pinned-numpy arrays held on the Rec instance.
    - Single GPU / single rank (no MPI).

User-facing arrays are **real-valued float32**: u, eta, grad on the volume
side; d, Lu, Leta, Lgrad on the projection side. The internal FFT pipeline
(usfft1d → usfft2d → fft2d) still operates on complex64 — only the small
pa1, pa2 scratch buffers stay complex. Real↔complex conversion happens
per chunk on the GPU at the pipeline boundaries (inside usfft1d_batch and
fft2_batch), so host memory and PCIe bandwidth see real arrays only.

All CPU↔GPU pipelining (the three chunk drivers + the elementwise reductions)
goes through `Chunking.gpu_batch`. Rec owns three Chunking instances, one per
operator chunk size (n1c / dethc / nthetac), each with its own pre-allocated
double-buffered GPU pool and three async streams.

The acquisition geometry (theta, phi) is fixed at construction. We keep only
the 1-D factors (cos θ, sin θ, ku, kv) on the GPU; the per-chunk (ntheta,
dethc, detw) gather coordinates x, y are built inside usfft2d_batch on the
fly — materialising the full (ntheta, deth, detw) x, y grids would cost
~8 GiB of GPU memory at n=1024 to no benefit.
"""

import os

import numpy as np
import cupy as cp
import psutil
import tifffile


from lam_usfft.logger_config import logger as _logger

from lam_usfft.usfft1d import usfft1d
from lam_usfft.usfft2d import usfft2d
from lam_usfft.fft2d import fft2d
from lam_usfft.chunking import Chunking
from lam_usfft.extra_terms import TVTerm
from lam_usfft.utils import pinned_empty


_process = psutil.Process()


def _mem_stats():
    """Return (process RSS in GiB, GPU used in GiB) for debug reporting."""
    rss = _process.memory_info().rss / 1024**3
    free, total = cp.cuda.runtime.memGetInfo()
    return rss, (total - free) / 1024**3


def _pool_bytes(*shapes, double_buffer=True, slack=1.1, dtype="float32"):
    """Bytes needed to hold the listed per-chunk arrays (all of one dtype) in a pool."""
    item = np.dtype(dtype).itemsize
    total = sum(int(np.prod(s)) for s in shapes) * item
    if double_buffer:
        total *= 2
    return int(slack * total)


def _pool_bytes_mixed(*pairs, double_buffer=True, slack=1.1):
    """Bytes needed for a mixed-dtype pool. Each pair is (shape, dtype)."""
    total = 0
    for shape, dtype in pairs:
        total += int(np.prod(shape)) * np.dtype(dtype).itemsize
    if double_buffer:
        total *= 2
    return int(slack * total)


class Rec:
    """BH-CG solver for ½ ||L u − d||² + (lam/N) Σ √(|∇u|² + ε²).

    The only regularizer is Charbonnier-smoothed total variation (lagged-
    diffusivity Hessian). Setting lam = 0 disables it.

    All user-facing arrays (u, eta, grad, d, Lu, Leta) are float32; the
    FFT pipeline scratch (pa1, pa2) is complex64 (the underlying NUFFT /
    cuFFT kernels operate on complex64).
    """

    def __init__(self, n0, n1, n2, detw, deth, ntheta, theta, phi,
                 n1c=None, dethc=None, nthetac=None,
                 lam=0.0, niter=64, start_iter=0,
                 tv_eps=1e-3, axis=None):
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

        self.n0, self.n1, self.n2 = n0, n1, n2
        self.detw, self.deth, self.ntheta = detw, deth, ntheta
        self.n1c     = n1c     if n1c     is not None else n1
        self.dethc   = dethc   if dethc   is not None else deth
        self.nthetac = nthetac if nthetac is not None else ntheta
        self.lam, self.niter, self.start_iter = float(lam), int(niter), int(start_iter)
        self.theta = np.asarray(theta, dtype="float32")
        self.phi   = np.float32(phi)

        # Rotation-axis position (horizontal pixel index in detw coords). The
        # NUFFT operators assume the axis projects to detw/2; if it's elsewhere
        # we apply a Fourier-shift phase to the freq projections between fft2d
        # and usfft2d, which is exactly equivalent to shifting the real-space
        # data by (axis - detw/2) pixels but costs nothing extra (no edge pad,
        # no resampling).
        self.axis = float(axis) if axis is not None else detw / 2.0
        axis_offset = self.axis - detw / 2.0       # in pixels (can be fractional)
        if axis_offset != 0:
            ku = (cp.arange(detw, dtype="float32") - detw / 2.0) / detw
            # fwd_lam: shift simulated projections by +axis_offset pixels
            # ⇒ multiply freq by exp(-2πi · ku · axis_offset). Shape (1,1,detw)
            # so it broadcasts across (ntheta, dethc) when applied per-chunk.
            phase_fwd = cp.exp((-2j * cp.pi * np.float32(axis_offset)) * ku).astype("complex64")
            self.axis_phase_fwd = phase_fwd[None, None, :]
            self.axis_phase_adj = cp.conj(phase_fwd)[None, None, :]
        else:
            self.axis_phase_fwd = None
            self.axis_phase_adj = None

        # --- low-level operator instances ---
        self.cl_usfft1d = usfft1d(self.n0, self.n1c, self.n2, self.deth, self.phi)
        self.cl_usfft2d = usfft2d(self.dethc, self.n1, self.n2,
                                  self.ntheta, self.detw, self.dethc)
        self.cl_fft2d   = fft2d(self.nthetac, self.detw, self.deth)

        # --- Chunking instances: one per operator chunk size ---
        # n1c-chunked along axis 0: usfft1d wrap + all elementwise/reduction ops
        # on (n1, n0, n2)- or (ntheta, deth, detw)-shaped arrays.
        # The volume side (u/eta/grad) and projection side (Lu/Leta/Lgrad) are
        # float32; pa1 (kernel I/O across the usfft1d boundary) is complex64.
        obj_in_r  = (self.n1c, self.n0,   self.n2)        # float32 (u chunk)
        obj_out_c = (self.n1c, self.deth, self.n2)        # complex64 (pa1 chunk)
        proj_r    = (self.n1c, self.deth, self.detw)      # float32 (Lu chunk)
        cands_n1 = [
            # usfft1d: float32 obj_in + complex64 pa1 (fwd) ≡ same totals for adj
            _pool_bytes_mixed((obj_in_r, "float32"), (obj_out_c, "complex64")),
            # linear_batch on volume side: 3 × float32 obj
            _pool_bytes(obj_in_r, obj_in_r, obj_in_r, dtype="float32"),
            # linear_batch on projection side: 3 × float32 proj
            _pool_bytes(proj_r, proj_r, proj_r, dtype="float32"),
        ]
        if self.lam != 0:
            # Chunked TV gradient: g_out + u_pad_chunk (n1c+2 rows) + g_in.
            obj_pad_r = (self.n1c + 2, self.n0, self.n2)
            cands_n1.append(_pool_bytes(obj_pad_r, obj_in_r, obj_in_r, dtype="float32"))
        self.cl_chunking = Chunking(max(cands_n1), self.n1c)

        # dethc-chunked along axis 1: usfft2d wrap.
        # pa1 → pa2 is complex64 → complex64 (no boundary conversion). The
        # per-chunk x, y gather coords are built on the fly from 1-D factors
        # inside usfft2d_batch; only the dethc-slice of the 3-D kv array
        # (a 1×dethc×1 strided buffer, a few KB total) is passed through
        # the decorator as a proper input.
        u2_in_c  = (self.n1,     self.dethc, self.n2)     # complex64 (pa1 chunk)
        u2_out_c = (self.ntheta, self.dethc, self.detw)   # complex64 (pa2 chunk)
        u2_kv    = (1,           self.dethc, 1)           # float32 kv slice
        self.cl_chunking_deth = Chunking(
            _pool_bytes(u2_in_c, u2_out_c, dtype="complex64")
            + _pool_bytes(u2_kv, slack=1.0, dtype="float32"),
            self.dethc,
        )

        # nthetac-chunked along axis 0: fft2d wrap.
        # fwd dir (used in adj_lam): float32 d → complex64 pa2.
        # adj dir (used in fwd_lam): complex64 pa2 → float32 Lu (or Lgrad).
        # Both directions need slots for one float32 + one complex64 chunk.
        f2_shape = (self.nthetac, self.deth, self.detw)
        self.cl_chunking_theta = Chunking(
            _pool_bytes_mixed((f2_shape, "float32"), (f2_shape, "complex64")),
            self.nthetac,
        )

        # --- short-hand helpers from the obj-axis Chunking ---
        self.linear_batch       = self.cl_chunking.linear_batch
        self.linear_redot_batch = self.cl_chunking.linear_redot_batch
        self.redot_batch        = self.cl_chunking.redot_batch
        self.mulc_batch         = self.cl_chunking.mulc_batch
        self.gpu_batch          = self.cl_chunking.gpu_batch

        # --- regularizer term (TV; no-op when lam == 0) ---
        self.cl_reg_term = TVTerm(self.lam, self.n1, self.n0, self.n2,
                                  self.gpu_batch, dtype="float32", eps=tv_eps)

        # --- persistent solver state (pre-allocated pinned arrays) ---
        # User-facing volumes and projections are float32. Scratch buffers
        # that are written before being read use pinned_empty to avoid the
        # gratuitous memset+copy that pinned_array(np.zeros/np.empty) would
        # incur on construction (≈3 GiB host bandwidth per buffer at n=1024).
        obj_shape  = (self.n1, self.n0, self.n2)
        proj_shape = (self.ntheta, self.deth, self.detw)

        # `u` aliases the reg-term's pad interior when lam > 0 (avoids an extra
        # copy in the term's gradient); otherwise it's a plain pinned array.
        # u and eta need an explicit zero — both BH() and the documented
        # caller contract assume them initialised.
        ov = self.cl_reg_term.obj_view
        if ov is not None:
            self.u = ov
        else:
            self.u = pinned_empty(obj_shape, dtype="float32")
            self.u[:] = 0
        ev = self.cl_reg_term.etas_view
        if ev is not None:
            self.eta = ev
        else:
            self.eta = pinned_empty(obj_shape, dtype="float32")
            self.eta[:] = 0
        # grad, pa1, pa2 — written before they're read.
        self.grad = pinned_empty(obj_shape, dtype="float32")
        self.pa1  = pinned_empty((self.n1, self.deth, self.n2), dtype="complex64")
        self.pa2  = pinned_empty(proj_shape, dtype="complex64")
        # Linearity cache. L is linear, so we maintain (Lu, Leta) in lock-step
        # with (u, eta) using only proj-space linear_batch's. The BH-CG inner
        # loop then needs ZERO fwd_lam for its Hessian inner products — they're
        # just redot's over cached L-arrays. Lgrad is a transient that holds
        # (Lu−d) during compute_gradient, then is overwritten with L·grad for
        # use in compute_beta/compute_alpha. Per BH iter we go from
        # ~5 fwd_lam + 1 adj_lam down to exactly 1 fwd_lam (on grad) + 1 adj_lam.
        self.Lu       = pinned_empty(proj_shape, dtype="float32")  # cached: L·u
        self.Leta     = pinned_empty(proj_shape, dtype="float32")  # cached: L·eta
        self.Lgrad = pinned_empty(proj_shape, dtype="float32")  # transient: Lr → Lgrad

        # Pre-step residual norm, stashed by compute_gradient so the BH dbg
        # branch can print it without redoing a full fwd_lam.
        self._last_resid = float("nan")

        # --- precompute 1-D geometry factors on GPU. ---
        self._init_geom_1d()

    # =========================================================================
    # Geometry: 1-D factors only. The full (ntheta, deth, detw) gather grids
    # x, y are NOT materialised — they're constructed per chunk inside
    # usfft2d_batch from these factors plus the chunk's kv slice. Holding
    # them resident would cost ~8 GiB GPU memory at n=1024 for no gain.
    # =========================================================================
    def _init_geom_1d(self):
        th = cp.asarray(self.theta)
        self._cos_t = cp.cos(th)[:, None, None].astype("float32")  # (ntheta, 1, 1)
        self._sin_t = cp.sin(th)[:, None, None].astype("float32")  # (ntheta, 1, 1)
        self._cos_p = np.float32(np.cos(self.phi))
        self._ku    = (cp.arange(self.detw, dtype="float32") - self.detw / 2.0) / self.detw
        kv          = (cp.arange(self.deth, dtype="float32") - self.deth / 2.0) / self.deth
        # 3-D shape (1, deth, 1) so the cl_chunking_deth decorator treats kv
        # as a proper axis-1-chunked input — each chunk sees its own (1, dethc, 1) slice.
        self._kv_3d = cp.ascontiguousarray(kv[None, :, None])

    # =========================================================================
    # Chunked drivers: each wraps the corresponding operator in a @gpu_batch
    # decorator from its dedicated Chunking instance. Real↔complex boundary
    # conversions happen inside usfft1d_batch and fft2_batch (the two
    # pipeline ends that touch user-facing float32 arrays).
    # =========================================================================
    def usfft1d_batch(self, out, inp, direction):
        """Chunked usfft1d wrap; chunked along axis 0 by self.cl_chunking.

        direction='fwd': inp float32 (n1,n0,n2)   → out complex64 (n1,deth,n2)  [pa1].
        direction='adj': inp complex64 (n1,deth,n2)[pa1] → out float32 (n1,n0,n2).
        The kernel itself is complex64-only; per-chunk promotion (.astype) /
        demotion (.real) happens on the GPU here.
        """
        @self.cl_chunking.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _f(rec, g, f):
            if direction == "fwd":
                f_c = f.astype("complex64")              # real → complex, one kernel
                rec.cl_usfft1d.fwd(g, f_c)
            else:
                g_c = cp.empty(g.shape, dtype="complex64")
                rec.cl_usfft1d.adj(g_c, f)
                g[:] = g_c.real                          # strided complex→real
        _f(self, out, inp)

    def usfft2d_batch(self, out, inp, direction):
        """Chunked usfft2d wrap; chunked along axis 1 by cl_chunking_deth.

        Both `inp` (pa1) and `out` (pa2) are complex64 — usfft2d sits in the
        middle of the FFT pipeline, no boundary conversion. Per-chunk gather
        coords x, y are built on the fly from the 1-D geometry factors.

        Rotation-axis off-center: after the fwd gather (or before the adj
        scatter), the projection-side freq chunk is multiplied by the
        precomputed 1-D Fourier-shift phase. Mathematically equivalent to
        shifting the real-space data by (axis - detw/2) pixels along detw,
        but with no edge padding and no resampling.
        """
        @self.cl_chunking_deth.gpu_batch(axis_out=1, axis_inp=1, nout=1)
        def _f(rec, g, f, kv_chunk):
            # kv_chunk: (1, dethc_actual, 1) — chunk slice of rec._kv_3d.
            ku    = rec._ku[None, None, :]
            cos_t = rec._cos_t
            sin_t = rec._sin_t
            cos_p = rec._cos_p
            x = ku * cos_t + kv_chunk * sin_t * cos_p
            y = ku * sin_t - kv_chunk * cos_t * cos_p
            cp.clip(x, -0.5 + 1e-5, 0.5 - 1e-5, out=x)
            cp.clip(y, -0.5 + 1e-5, 0.5 - 1e-5, out=y)
            if direction == "fwd":
                rec.cl_usfft2d.fwd(g, f, x, y)
                if rec.axis_phase_fwd is not None:
                    g *= rec.axis_phase_fwd       # (1,1,detw) broadcasts over (ntheta, dethc)
            else:
                if rec.axis_phase_adj is not None:
                    f *= rec.axis_phase_adj       # f is the proj-side input chunk
                rec.cl_usfft2d.adj(g, f, x, y)
        _f(self, out, inp, self._kv_3d)

    def fft2_batch(self, out, inp, direction):
        """Chunked fft2d wrap; chunked along axis 0 by cl_chunking_theta.

        direction='fwd' (adj_lam path): inp float32 d → out complex64 pa2.
        direction='adj' (fwd_lam path): inp complex64 pa2 → out float32 Lu.
        Per-chunk real↔complex conversion on the GPU; the kernel itself is
        complex64-only.
        """
        @self.cl_chunking_theta.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _f(rec, g, f):
            if direction == "fwd":
                f_c = f.astype("complex64")
                rec.cl_fft2d.fwd(g, f_c)
            else:
                g_c = cp.empty(f.shape, dtype="complex64")
                rec.cl_fft2d.adj(g_c, f)
                g[:] = g_c.real
        _f(self, out, inp)

    # =========================================================================
    # Linear laminography operator and its adjoint
    # =========================================================================
    def fwd_lam(self, u, out=None):
        """L u: forward laminography on real volume u.

        Parameters
        ----------
        u   : pinned float32 array of shape (n1, n0, n2).
        out : pinned float32 array of shape (ntheta, deth, detw); allocated if None.

        Returns `out`.
        """
        if out is None:
            out = pinned_empty(self.Lu.shape, dtype="float32")
        self.usfft1d_batch(self.pa1, u, "fwd")           # u(f32) → pa1(c64)
        self.usfft2d_batch(self.pa2, self.pa1, "fwd")    # pa1(c64) → pa2(c64)
        self.fft2_batch(out, self.pa2, "adj")            # pa2(c64) → out(f32)
        return out

    def adj_lam(self, data, out=None):
        """Lᵀ data: adjoint laminography on real data.

        Parameters
        ----------
        data : pinned float32 array of shape (ntheta, deth, detw).
        out  : pinned float32 array of shape (n1, n0, n2); allocated if None.

        Returns `out`.
        """
        if out is None:
            out = pinned_empty(self.u.shape, dtype="float32")
        self.fft2_batch(self.pa2, data, "fwd")           # data(f32) → pa2(c64)
        self.usfft2d_batch(self.pa1, self.pa2, "adj")    # pa2(c64) → pa1(c64)
        self.usfft1d_batch(out, self.pa1, "adj")         # pa1(c64) → out(f32)
        return out

    # =========================================================================
    # BH-CG primitives (linear-L specialisation)
    # =========================================================================
    def compute_gradient(self, d):
        """self.grad := Lᵀ(Lu − d) + ∂R/∂u; refresh self.Lgrad := L·grad.

        Uses the cached self.Lu (kept in sync by BH/apply_step), so this method
        does NO fwd_lam on u. The single fwd_lam(grad) at the end is the one
        operator call per BH iteration that the linearity cache can't avoid —
        the gradient is structurally `Lᵀ(Lu−d) + reg_grad`, not a linear combo
        of cached quantities, so its L-image must be freshly computed for the
        Hessian inner products in compute_beta / compute_alpha.

        Side-effect: stashes the start-of-iter residual norm |Lu − d| in
        self._last_resid for the BH dbg branch to print.
        """
        # Residual into Lgrad — Lu must be preserved for the apply_step sync.
        self.linear_batch(self.Lu, d, 1, -1, out=self.Lgrad)   # Lgrad := Lu − d
        # ||Lu − d|| via the on-GPU chunked reduction.
        self._last_resid = float(self.redot_batch(self.Lgrad, self.Lgrad)) ** 0.5
        self.adj_lam(self.Lgrad, out=self.grad)                # grad := Lᵀ(Lu − d)
        self.cl_reg_term.gradient(self.grad)                   # += reg contribution
        # Refresh L·grad into Lgrad — the residual has been consumed.
        self.fwd_lam(self.grad, out=self.Lgrad)                # Lgrad := L·grad

    def _hessian(self, La, Lb, eta_a, eta_b):
        """<eta_a, H eta_b> = <La, Lb> + <η_a, H_reg η_b>, given precomputed
        La = L·η_a and Lb = L·η_b. Zero fwd_lam calls — that's the whole point
        of the linearity cache."""
        if La is Lb:
            main = float(self.redot_batch(La, La))
        else:
            main = float(self.redot_batch(La, Lb))
        if self.lam == 0:
            return main
        ev = self.cl_reg_term.etas_view  # alias of e_pad interior
        if eta_b is not ev:
            ev[:] = eta_b
        return main + self.cl_reg_term.hessian(None, eta_a)

    def _two_redots(self, A, B):
        """In one chunked streaming pass over (A, B): returns (<A, B>, <B, B>).

        Compared to two back-to-back redot_batch calls (the natural way to
        write compute_beta's two _hessian's), this halves the projection-side
        host→GPU traffic in compute_beta: B (= Leta, ~4 GiB at n=1024) is
        streamed up once instead of twice. A still streams once. Two scalar
        accumulators in a single (2,) GPU buffer.
        """
        acc = cp.zeros(2, dtype="float32")

        @self.cl_chunking.gpu_batch(axis_out=0, axis_inp=0)
        def _f(_self, acc_out, a, b):
            acc_out[0] += cp.vdot(a.ravel(), b.ravel())
            acc_out[1] += cp.vdot(b.ravel(), b.ravel())

        _f(self, acc, A, B)
        return float(acc[0]), float(acc[1])

    def compute_beta(self, i):
        """β_i = <grad, H η_prev> / <η_prev, H η_prev>, with β=0 at start_iter.

        Uses cached L·grad (= self.Lgrad, set by compute_gradient) and
        L·eta (= self.Leta, kept in sync by compute_alpha). No fwd_lam.
        The two data-term inner products <Lgrad, Leta> and <Leta, Leta>
        are computed in a single fused pass over the projection arrays
        (Leta streamed once instead of twice).
        """
        if i == self.start_iter:
            return 0.0
        # Fused data-term Hessian inner products: one streaming pass.
        main_top, main_bot = self._two_redots(self.Lgrad, self.Leta)
        top, bottom = main_top, main_bot
        if self.lam != 0:
            # eta is already aliased to cl_reg_term.etas_view, so no staging
            # needed for either reg-side hessian call (e_pad's interior IS eta).
            top    += self.cl_reg_term.hessian(None, self.grad)
            bottom += self.cl_reg_term.hessian(None, self.eta)
        # Guard <eta, H eta> ≈ 0: eta is in (or near) H's null space — typically
        # only happens once CG has effectively converged. Return β=0 so the
        # next eta := -grad starts a fresh search direction.
        if bottom <= 0:
            _logger.warning("compute_beta: <eta,H eta>=%.3e ≤ 0 → restarting CG (β=0)", bottom)
            return 0.0
        return top / bottom

    def compute_alpha(self, beta):
        """In-place: eta := β·eta − grad AND Leta := β·Leta − Lgrad (lin combo of
        cached L-arrays — no fwd_lam). Returns α = −<grad, η_new> / <η_new, H η_new>.
        """
        # Sync Leta with eta first (uses cached L·grad = Lgrad).
        self.linear_batch(self.Leta, self.Lgrad, beta, -1)     # Leta := β·Leta − Lgrad
        # Fused eta update + <grad, eta_new> in one GPU sweep.
        top    = -float(self.linear_redot_batch(self.eta, self.grad, beta, -1))
        bottom = self._hessian(self.Leta, self.Leta, self.eta, self.eta)
        # Guard against <eta_new, H eta_new> = 0: eta is in (or near) the
        # Hessian's null space — return zero step instead of dividing by zero.
        # The next iteration's compute_beta picks up where this leaves off.
        if bottom <= 0:
            _logger.warning("compute_alpha: <eta,H eta>=%.3e ≤ 0 → α=0 (skip step)", bottom)
            return 0.0
        return top / bottom

    def apply_step(self, alpha):
        """self.u := self.u + α·self.eta; self.Lu := self.Lu + α·self.Leta.

        The proj-side sync is one extra linear_batch per iter — trivially cheap
        compared to the fwd_lam it replaces.
        """
        self.linear_batch(self.u,  self.eta,  1, alpha)
        self.linear_batch(self.Lu, self.Leta, 1, alpha)

    # =========================================================================
    # Per-iteration visualisation (mid-slices written to TIFF)
    # =========================================================================
    def write_vis_slices(self, i, vis_dir):
        """Checkpoint dump on each vis tick:
          • Mid-z (xy) slice  → ``vis_dir/xy_{i:04}.tiff``
          • Mid-y (xz) slice  → ``vis_dir/xz_{i:04}.tiff``
          • Full volume       → ``vis_dir/vol_{i:04}.tiff``   (multi-page TIFF)

        The full volume is a single multi-page TIFF (one page per n1 slice) so
        it loads cleanly into Fiji / Avizo / Dragonfly without per-iter
        sub-directories piling up. Sizing reference: at BIN=8 the Chawla chip
        volume is ~46 MB per checkpoint; at BIN=1 it grows to ~16 GiB per
        checkpoint — choose ``vis_step`` accordingly to keep disk usage sane.
        """
        os.makedirs(vis_dir, exist_ok=True)
        u_re   = np.asarray(self.u, dtype="float32")
        xy_mid = u_re[self.n1 // 2,    :,            :]   # horizontal (n0, n2)
        xz_mid = u_re[:,               self.n0 // 2, :]   # vertical   (n1, n2)
        xy_path  = os.path.join(vis_dir, f"xy_{i:04}.tiff")
        xz_path  = os.path.join(vis_dir, f"xz_{i:04}.tiff")
        vol_path = os.path.join(vis_dir, f"vol_{i:04}.tiff")
        tifffile.imwrite(xy_path, xy_mid)
        tifffile.imwrite(xz_path, xz_mid)
        # Multi-page TIFF of the full (n1, n0, n2) volume. BigTIFF kicks in
        # automatically for volumes > 4 GiB.
        tifffile.imwrite(vol_path, u_re, bigtiff=(u_re.nbytes > 2**31))
        vol_gib = u_re.nbytes / 1024**3
        _logger.info(
            "      vis: wrote mid-slices → %s, %s  +  full volume %s (%.2f GiB)",
            xy_path, xz_path, vol_path, vol_gib,
        )

    # =========================================================================
    # Top-level CG driver
    # =========================================================================
    def BH(self, d, dbg=False, dbg_step=1, vis_step=-1, vis_dir="vis"):
        """Run BH-CG for `niter` iterations starting at `start_iter`.

        Parameters
        ----------
        d : pinned float32 array of shape (ntheta, deth, detw) — data.
        dbg, dbg_step : log residual + RAM/GPU every dbg_step iters when dbg=True.
            Output goes through the ``lam_usfft.rec`` logger at INFO level; a
            stderr handler is attached on first dbg call if none is configured.
            The printed `|Lu-d|` is the START-of-iteration residual cached by
            compute_gradient — equivalent to the end of iter i-1 for CG. The
            first print (i=start_iter) therefore shows |d| when u starts at 0.
        vis_step : every vis_step iters, dump mid-slice TIFFs AND the full
            volume as a multi-page TIFF (vis_dir/{xy,xz,vol}_{i:04}.tiff).
            ≤ 0 disables. Mind the disk: at BIN=1 the volume is ~16 GiB per
            checkpoint, so pick vis_step generously for full-res runs.
        vis_dir  : output directory for the vis tiffs (created on demand).

        Reads/writes `self.u` (the reconstruction). Returns `self.u`.
        Caller is responsible for initialising `self.u` before invocation.
        """
        # Linearity-cache initialisation:
        #   - eta and Leta start at zero (initial search direction is zero).
        #   - Lu := L·u via the one fwd_lam this method performs. Amortised
        #     across niter iterations the inner loop pays just 1 fwd_lam +
        #     1 adj_lam per iter.
        self.eta[:]  = 0
        self.Leta[:] = 0
        self.fwd_lam(self.u, out=self.Lu)
        for i in range(self.start_iter, self.niter):
            self.compute_gradient(d)
            beta  = self.compute_beta(i)
            alpha = self.compute_alpha(beta)
            self.apply_step(alpha)
            if dbg and i % dbg_step == 0:
                resid = self._last_resid                  # cached by compute_gradient
                rss, gpu = _mem_stats()
                # Full BH-CG objective: ½||Lu−d||² + R(u). The residual is the
                # start-of-iter ||Lu−d|| (cached free by compute_gradient); R is
                # evaluated on the current u via the reg term's energy().
                F = 0.5 * resid * resid + self.cl_reg_term.energy()
                _logger.info(
                    "%4d  alpha=%.3e  beta=%.3e  RAM=%.2fGB  GPU=%.2fGB  F=%.4e",
                    i, alpha, beta, rss, gpu, F,
                )
            if vis_step > 0 and i % vis_step == 0:
                self.write_vis_slices(i, vis_dir)
        return self.u
