"""Smoke tests for the BH-CG laminography solver."""

import numpy as np
from lam_usfft.rec import Rec
from lam_usfft.utils import pinned_empty


def _alloc_d(rec):
    """Pinned-host float32 buffer matching one fwd_lam output."""
    return pinned_empty((rec.ntheta, rec.deth, rec.detw), dtype="float32")


def _theta_phi(ntheta):
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")
    phi   = np.pi / 2 + 20 / 180 * np.pi
    return theta, phi


def test_adjoint_identity():
    """Low-level regression: <L u, d> ≈ <u, Lᵀ d> to float32 precision."""
    n0 = n1 = n2 = 64; detw = deth = 64; ntheta = 32
    theta, phi = _theta_phi(ntheta)
    rng = np.random.default_rng(0)

    rec = Rec(n0, n1, n2, detw, deth, ntheta, theta, phi, lam=0.0, niter=0)
    rec.u[:] = rng.standard_normal(rec.u.shape).astype("float32")
    Lu = _alloc_d(rec)
    rec.fwd_lam(rec.u, out=Lu)
    d  = _alloc_d(rec)
    d[:] = rng.standard_normal(Lu.shape).astype("float32")
    Lt_d = rec.adj_lam(d)
    lhs = float(np.vdot(Lu, d))
    rhs = float(np.vdot(rec.u, Lt_d))
    rel = abs(lhs - rhs) / abs(lhs)
    print(f"<Lu, d>  = {lhs}")
    print(f"<u, Ltd> = {rhs}")
    print(f"rel diff = {rel:.3e}")
    assert rel < 1e-5, f"adjoint identity broken: rel diff = {rel}"


def test_bh_chunked_vs_unchunked():
    """Chunked vs unchunked BH must produce identical reconstructions."""
    n0 = n1 = n2 = 64; detw = deth = 64; ntheta = 32
    theta, phi = _theta_phi(ntheta)
    rng = np.random.default_rng(0)
    u_seed = rng.standard_normal((n1, n0, n2)).astype("float32")

    rec_ref = Rec(n0, n1, n2, detw, deth, ntheta, theta, phi, lam=0.0, niter=8)
    rec_ref.u[:] = u_seed
    d = _alloc_d(rec_ref)
    rec_ref.fwd_lam(rec_ref.u, out=d)
    rec_ref.u[:] = 0
    rec_ref.BH(d)
    u_ref = np.array(rec_ref.u, copy=True)

    rec_ch = Rec(n0, n1, n2, detw, deth, ntheta, theta, phi,
                 n1c=16, dethc=16, nthetac=8,
                 lam=0.0, niter=8)
    rec_ch.u[:] = 0
    rec_ch.BH(d)
    u_ch = np.array(rec_ch.u, copy=True)

    diff = float(np.linalg.norm(u_ch - u_ref) / np.linalg.norm(u_ref))
    print(f"chunked vs unchunked rel diff = {diff:.3e}")
    assert diff < 1e-4, f"chunked path diverged from unchunked: {diff}"


def test_bh_quadratic_convergence():
    """BH-CG on a recoverable problem (u_true ∈ range(Lᵀ)) drives both the
    residual and the object error down."""
    n0 = n1 = n2 = 32; detw = deth = 32; ntheta = 64
    theta, phi = _theta_phi(ntheta)
    rng = np.random.default_rng(0)

    rec = Rec(n0, n1, n2, detw, deth, ntheta, theta, phi, lam=0.0, niter=128)
    d_seed = _alloc_d(rec)
    d_seed[:] = rng.standard_normal(d_seed.shape).astype("float32")
    # u_true = Lᵀ(d_seed) so it lives in range(Lᵀ) — no null-space contamination.
    rec.u[:] = rec.adj_lam(d_seed)
    u_true = np.array(rec.u, copy=True)

    d = _alloc_d(rec)
    rec.fwd_lam(rec.u, out=d)
    d0_norm = float(np.linalg.norm(d))

    rec.u[:] = 0
    rec.BH(d, dbg=True, dbg_step=32)

    Lu = rec.fwd_lam(rec.u)
    resid = float(np.linalg.norm(Lu - d) / d0_norm)
    obj   = float(np.linalg.norm(rec.u - u_true) / np.linalg.norm(u_true))
    print(f"BH-CG (lam=0):  |Lu-d|/|d| = {resid:.3e}   |u-u_true|/|u_true| = {obj:.3e}")
    assert resid < 1e-3, f"residual did not converge: {resid}"
    assert obj   < 1e-1, f"object error too large: {obj}"


def test_bh_with_regularizer_runs():
    """With lam > 0 and noisy data, BH-CG with TV regularization produces a
    measurably smoother result (lower Laplacian-norm of the reconstruction)."""
    n0 = n1 = n2 = 64; detw = deth = 64; ntheta = 32
    theta, phi = _theta_phi(ntheta)
    rng = np.random.default_rng(1)
    u_true = rng.standard_normal((n1, n0, n2)).astype("float32")

    norms = {}
    for lam in (0.0, 10000.0):
        rec = Rec(n0, n1, n2, detw, deth, ntheta, theta, phi, lam=lam, niter=32)
        rec.u[:] = u_true
        d = _alloc_d(rec)
        rec.fwd_lam(rec.u, out=d)
        d += (0.01 * rng.standard_normal(d.shape)).astype("float32")
        rec.u[:] = 0
        rec.BH(d)

        l = (np.roll(rec.u,  1, axis=0) + np.roll(rec.u, -1, axis=0)
             + np.roll(rec.u,  1, axis=1) + np.roll(rec.u, -1, axis=1)
             + np.roll(rec.u,  1, axis=2) + np.roll(rec.u, -1, axis=2)
             - 6 * rec.u)
        norms[lam] = float(np.linalg.norm(l))
        print(f"  lam={lam:.0e}: ||Δu_rec|| = {norms[lam]:.4e}")

    assert norms[10000.0] < 0.8 * norms[0.0], f"Regularization did not smooth the result: {norms}"


if __name__ == "__main__":
    test_adjoint_identity()
    test_bh_chunked_vs_unchunked()
    test_bh_quadratic_convergence()
    test_bh_with_regularizer_runs()
    print("\nALL TESTS PASSED")
