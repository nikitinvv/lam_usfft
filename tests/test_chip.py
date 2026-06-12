"""Reconstruct 128 middle slices of the chip delta volume with BH-CG.

Loads `delta-chip-256.tiff` (the synthetic 256³ chip phantom of refractive-index
δ values), crops to 128 z-slices around the middle, forward-projects through the
laminography operator L, and runs BH-CG to recover u from the simulated data.
"""

import os
import numpy as np
import dxchange

from lam_usfft.rec import Rec
from lam_usfft.utils import pinned_empty


HERE = os.path.dirname(__file__)
TIFF = os.path.join(HERE, "delta-chip-256.tiff")


def test_chip_bh():
    # Load 256³ chip phantom. TIFF axis 0 = z slices; transpose to put z on
    # axis 1 (our `n0` convention — the axis usfft1d transforms into deth).
    vol = dxchange.read_tiff(TIFF).astype("float32")    # shape (256, 256, 256)
    # z0  = (vol.shape[0] - 128) // 2
    # vol = vol[z0 : z0 + 128]                            # (128, 256, 256)
    # Zero the top and bottom quarters along the z axis — only the central
    # half of the slab contains material (simulates a thin sample embedded in
    # vacuum, which is what laminography typically images).
    q = vol.shape[0] // 4 + 24                              # 32
    vol[:q]  = 0
    vol[-q:] = 0
    u_true = vol.swapaxes(0, 1).astype('float32')       # (256, 128, 256) = (n1, n0, n2)

    n1, n0, n2 = u_true.shape                           # 256, 128, 256
    detw, deth = n2, n1                                 # detector matches transverse extent
    ntheta = 128
    n1c, dethc, nthetac = 32, 32, 32
    phi   = np.pi / 2 - 20 / 180 * np.pi                # 30° tilt from beam direction
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")

    # Charbonnier-smoothed TV regularization. With chip δ ~ 1e-5, typical
    # |∇u| ~ 1e-5 too, so pick eps well below that to stay in the TV regime
    # (eps ≫ |∇u| would make it behave like Sobolev/Tikhonov instead).
    rec = Rec(n0, n1, n2, detw, deth, ntheta, theta, phi,
              n1c=n1c, dethc=dethc, nthetac=nthetac,
              lam=0.05, tv_eps=1e-7, niter=513)

    # Simulate data from the true volume, then reconstruct from zero.
    # d is pinned-host so the per-chunk H2D inside compute_gradient is async-fast.
    rec.u[:] = u_true
    d = pinned_empty((ntheta, deth, detw), dtype="float32")
    rec.fwd_lam(rec.u, out=d)
    d0 = float(np.linalg.norm(d))
    print(f"chip:  volume shape (n1,n0,n2) = ({n1},{n0},{n2}),  "
          f"data shape (ntheta,deth,detw) = ({ntheta},{deth},{detw}),  "
          f"|d|={d0:.3e}")

    # Save the simulated data as a per-angle TIFF stack (one file per 2-D angle).
    import tifffile
    out_root = "/data2/vnikitin/tmp/lam/"
    data_dir = os.path.join(out_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    for k in range(ntheta):
        tifffile.imwrite(os.path.join(data_dir, f"d_{k:05d}.tiff"), d[k])
    print(f"chip:  wrote {ntheta} per-angle data tiffs to {data_dir}/d_*.tiff")

    rec.u[:] = 0
    rec.BH(d, dbg=True, dbg_step=4, vis_step=8, vis_dir=out_root)

    Lu = rec.fwd_lam(rec.u)
    resid = float(np.linalg.norm(Lu - d) / d0)
    obj   = float(np.linalg.norm(rec.u - u_true) / np.linalg.norm(u_true))
    print(f"chip final:  |Lu-d|/|d| = {resid:.3e}   |u-u_true|/|u_true| = {obj:.3e}")

    # The residual must drop substantially from its starting value (which equals
    # |d| because u starts at 0, so |L·0 - d| = |d|). Object error is bounded
    # below by the null-space contribution of L (missing-wedge laminography)
    # so we only assert a generous bound there.
    assert resid < 1e-1, f"residual barely moved: {resid}"
    assert obj   < 1.0,  f"obj error not finite-bounded: {obj}"


if __name__ == "__main__":
    test_chip_bh()
    print("\nchip test PASSED")
