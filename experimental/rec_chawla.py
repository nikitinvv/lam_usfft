"""BH-CG laminography reconstruction of the Chawla intel-FOV dataset.

Dataset
    /data2/vnikitin/Chawla/lamino_intelFOV_lam20_dist50_energy_40_148.h5

Geometry
    Laminography tilt = 20° from the no-tilt position (rotation axis 20°
    off vertical, 70° off the beam). In Rec's convention this maps to::

        phi = π/2 − 20·π/180   ≈ 1.2217  rad

Pipeline
    1. Load raw projections + flats + darks from /exchange/* (APS layout).
    2. Average flats and darks; produce log-corrected projections
       p = −log( (data − dark) / (flat − dark) )   (clipped to avoid log of <=0).
    3. Pad the detector-width axis (1/4 on each side) so the laminographic
       reconstruction has enough room laterally.
    4. Configure Rec, run BH-CG with TV regularization; dump mid-slice TIFFs
       periodically (via vis_step) and write a full real-part TIFF stack
       at the end.

Run this on tomo2 — /data2 lives there.
"""

import os
import sys

import numpy as np
import h5py
import tifffile

from lam_usfft.rec import Rec
from lam_usfft.utils import pinned_empty
from lam_usfft.logger_config import logger, add_file_handler, set_log_level


# ============================================================================
# Paths & geometry
# ============================================================================

DATA_FILE = "/data2/vnikitin/Chawla/lamino_intelFOV_lam20_dist50_energy_40_148.h5"
OUT_BASE  = "/data2/vnikitin/tmp/lam/chawla"     # run dirs are OUT_BASE_lam<LAM_TV>

LAM_DEG = -20.0                                  # laminography tilt angle
PHI     = np.pi / 2 - LAM_DEG / 180.0 * np.pi   # Rec convention

# Rotation-axis position on the detector (horizontal pixel index) BEFORE
# binning.  Divided by BIN below to get the binned-coord position and passed
# to Rec, which absorbs the offset as a Fourier-shift phase on the projection
# side of the NUFFT (no extra padding / no resampling).
AXIS = 1647.0

# Reconstruction knobs (tune for your problem / GPU memory).
BIN          = 2          # 1 = no binning; 2 = ½ each axis; 4 = ¼ each axis.
NITER        = 1025
LAM_TV       = 1e8          # TV strength (set to 0.0 to disable; tune after a first run)
TV_EPS       = 1e-7         # Charbonnier ε — keep < typical |∇u| for TV regime.
DBG_STEP     = 8
VIS_STEP     = 32
LATERAL_PAD  = None            # pad detw by detw // LATERAL_PAD on each side (None to skip)

# Output paths — folder name encodes LAM_TV so successive runs at different
# regularization strengths don't clobber each other. `:g` gives a tidy string
# (0 → "0", 1e4 → "10000", 1.5e6 → "1.5e+06").
OUT_ROOT = f"{OUT_BASE}_lam{LAM_TV:g}"
VIS_DIR  = os.path.join(OUT_ROOT, "vis")
REC_DIR  = os.path.join(OUT_ROOT, "rec")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REC_DIR, exist_ok=True)

# Logging: colored stdout + tee to a run log inside OUT_ROOT.
set_log_level("INFO")
add_file_handler(os.path.join(OUT_ROOT, "rec_chawla.log"))


# ============================================================================
# 1.  Load + flat/dark correction
# ============================================================================


def load_projections(fname, bin_factor=1):
    """Read raw projection stack + flats + darks from an APS/exchange-format
    HDF5 file. Returns (proj_log, theta_rad)."""
    with h5py.File(fname, "r") as f:
        if "/exchange/data" not in f:
            raise KeyError(f"{fname} has no /exchange/data — see layout above; "
                           "adapt the dataset paths in load_projections().")
        # Raw counts.  Subsample (not averaged) along all three axes —
        # bin_factor decimates angles, detector height, and detector width.
        if bin_factor == 1:
            data  = f["/exchange/data"][:].astype("float32")
        else:
            data  = f["/exchange/data"][::bin_factor,
                                       ::bin_factor,
                                       ::bin_factor].astype("float32")
        # Flat / dark fields don't have a theta axis — bin axes 1 and 2 only.
        flat = (f["/exchange/data_white"][:].astype("float32")
                if "/exchange/data_white" in f else None)
        dark = (f["/exchange/data_dark"][:].astype("float32")
                if "/exchange/data_dark"  in f else None)

        if bin_factor > 1:
            if flat is not None: flat = flat[:, ::bin_factor, ::bin_factor]
            if dark is not None: dark = dark[:, ::bin_factor, ::bin_factor]
        # Crop axis 1 (detector height) from the bottom so its length is a
        # multiple of 4 — needed by downstream chunking / FFT sizing.
        crop_h = (data.shape[1] // 4) * 4
        if crop_h != data.shape[1]:
            data = data[:, :crop_h]
            if flat is not None: flat = flat[:, :crop_h]
            if dark is not None: dark = dark[:, :crop_h]

        # Angles — match the data-side angle subsampling.
        # Sniff degrees vs radians below and convert.
        theta = f["/exchange/theta"][::bin_factor].astype("float32")

    theta = theta * (np.pi / 180.0)

    # Flat/dark correction.
    logger.info(f"raw data:   shape={data.shape}  dtype={data.dtype}  "
                f"range=[{data.min():.3e}, {data.max():.3e}]")
    if flat is not None:
        flat_avg = flat.mean(axis=0, dtype="float32")
        logger.info(f"flat field: shape={flat.shape}  mean={flat_avg.mean():.3e}")
    else:
        flat_avg = np.ones(data.shape[1:], dtype="float32")
        logger.warning("flat field: missing — using all-ones")
    if dark is not None:
        dark_avg = dark.mean(axis=0, dtype="float32")
        logger.info(f"dark field: shape={dark.shape}  mean={dark_avg.mean():.3e}")
    else:
        dark_avg = np.zeros(data.shape[1:], dtype="float32")
        logger.warning("dark field: missing — using zeros")

    # p = -log( max((data - dark) / (flat - dark), eps) )
    eps = np.float32(1e-6)
    denom = np.maximum(flat_avg - dark_avg, eps)
    proj  = np.maximum((data - dark_avg) / denom, eps)
    print(np.mean(proj))
    np.log(proj, out=proj)
    print(np.mean(proj))
    proj *= -1.0
    logger.info(f"after -log: shape={proj.shape}  "
                f"range=[{proj.min():.3e}, {proj.max():.3e}]  mean={proj.mean():.3e}")
    return proj, theta


def pad_lateral(proj, pad_div):
    """Edge-pad the detector-width axis by detw//pad_div on each side. Returns
    a fresh contiguous float32 array."""
    if pad_div is None or pad_div <= 0:
        return proj
    detw = proj.shape[2]
    pad  = detw // pad_div
    logger.info(f"padding detw {detw} → {detw + 2*pad} (edge, {pad} each side)")
    return np.ascontiguousarray(
        np.pad(proj, ((0, 0), (0, 0), (pad, pad)), mode="edge").astype("float32"))


# ============================================================================
# 2.  Reconstruct
# ============================================================================

def main():
    proj, theta = load_projections(DATA_FILE, bin_factor=BIN)
    # Rotation-axis pixel index in the current (binned) detw coords. pad_lateral
    # adds symmetric padding which shifts every pixel — including the axis —
    # right by the per-side pad amount; account for that before applying it.
    axis_binned = AXIS / BIN
    if LATERAL_PAD is not None and LATERAL_PAD > 0:
        axis_binned += proj.shape[2] // LATERAL_PAD
    proj = pad_lateral(proj, LATERAL_PAD)
    logger.info(f"rotation axis: {AXIS:.1f} (pre-bin) → {axis_binned:.3f} "
                f"(in final detw={proj.shape[2]})")

    ntheta, deth, detw = proj.shape
    # Volume shape: same n0=deth (the laminographic z dimension), and n1=n2=detw
    # to give the reconstruction enough lateral extent. This matches the
    # convention used by the deleted test_admm_brain.py / test_cg_brain.py.
    n0, n1, n2 = deth, detw, detw
    # n0 = deth // 2
    logger.info("shapes:")
    logger.info(f"  projections (ntheta, deth, detw) = ({ntheta}, {deth}, {detw})")
    logger.info(f"  volume      (n1, n0, n2)         = ({n1}, {n0}, {n2})")
    logger.info(f"  phi = {PHI:.4f} rad  ({LAM_DEG}° tilt from vertical, {90-LAM_DEG}° from beam)")

    # Conservative default chunk sizes — tuned later for the actual GPU.
    n1c     = 1 if n1     >= 32 else n1
    dethc   = 1 if deth   >= 32 else deth
    nthetac = 1 if ntheta >= 32 else ntheta


    rec = Rec(n0, n1, n2, detw, deth, ntheta, theta, PHI,
              n1c=n1c, dethc=dethc, nthetac=nthetac,
              lam=LAM_TV, tv_eps=TV_EPS, niter=NITER,
              axis=axis_binned)

    # Pinned-host data buffer matching the Rec contract (float32, (ntheta, deth, detw)).
    d = pinned_empty((ntheta, deth, detw), dtype="float32")
    d[:] = proj
    del proj   # release the original numpy copy

    # Initialize u to zero and run BH-CG.
    rec.u[:] = 0
    logger.info(f"running BH-CG: niter={NITER}, lam_TV={LAM_TV}, "
                f"tv_eps={TV_EPS}, vis every {VIS_STEP}")
    rec.u[:] = 1.18
    d-=np.mean(rec.fwd_lam(rec.u))
    rec.u[:] = 0
    rec.BH(d, dbg=True, dbg_step=DBG_STEP, vis_step=VIS_STEP, vis_dir=VIS_DIR)

    # ----- save reconstruction -----
    logger.info(f"writing reconstruction tiff stack to {REC_DIR}/u_{{slice:05d}}.tiff ...")
    u = np.asarray(rec.u)
    # Crop back the lateral padding so the output volume matches the original
    # detector field of view.
    if LATERAL_PAD:
        pad = detw // LATERAL_PAD
        u   = u[:, pad:-pad, pad:-pad] if pad else u
    for k in range(u.shape[0]):
        tifffile.imwrite(os.path.join(REC_DIR, f"u_{k:05d}.tiff"), u[k])
    logger.info(f"done — wrote {u.shape[0]} slices of shape {u.shape[1:]}")


if __name__ == "__main__":
    main()
