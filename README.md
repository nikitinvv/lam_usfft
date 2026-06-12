# lam_usfft

GPU-accelerated iterative reconstruction for parallel-beam X-ray laminography.

Solves

$$
\min_u\; \tfrac{1}{2}\,\lVert L u - d \rVert^2 \;+\; \tfrac{\lambda}{N} \sum_i \sqrt{\lvert \nabla u \rvert_i^2 + \varepsilon^2}
$$

with bilinear-Hessian conjugate gradient (BH-CG) and Charbonnier-smoothed total variation.
The forward operator `L` is factored as `fft2d ∘ usfft2d ∘ usfft1d` (USFFT, O(N³ log N)) and
chunked across three CUDA streams so projection and volume arrays larger than VRAM live on
pinned host memory and stream through the GPU asynchronously.

Companion to [TomocuPy](https://tomocupy.readthedocs.io/) — same operator factorisation, same
chunking pattern, focused on the iterative-reconstruction-with-regularization path that
TomocuPy's filtered back-projection mode does not cover.

## Install

The implementation is pure Python + [cupy](https://cupy.dev) calling raw CUDA C kernels at
runtime. No `scikit-build`, `swig`, or `nvcc` build step needed:

```bash
conda create -n lam_usfft -c conda-forge \
    cupy tifffile h5py pywavelets psutil dxchange
conda activate lam_usfft
pip install .
```

Tested with cupy 12+, CUDA 11.8 / 12.x, NVIDIA Tesla A100 and V100.

## Quick start

```python
import numpy as np
from lam_usfft import (
    Rec,
    minus_log_inplace,
    remove_stripe_fw_inplace,
    pinned_empty,
)

# --- problem geometry ---------------------------------------------------------
n0, n1, n2     = 256, 256, 256          # volume (z, y, x) — laminographic axis is n0
detw, deth     = 256, 256               # detector
ntheta         = 128
phi            = np.pi / 2 - 20 * np.pi / 180   # 20° laminography tilt
theta          = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype("float32")

# --- solver -------------------------------------------------------------------
rec = Rec(n0, n1, n2, detw, deth, ntheta, theta, phi,
          n1c=32, dethc=32, nthetac=32,    # chunk sizes
          lam=1e4, tv_eps=1e-7,            # TV strength + Charbonnier ε
          niter=257,
          axis=None,                       # rotation-axis pixel (None = detw/2)
          support=None)                    # optional (start, end) along n0

# --- preprocessing on pinned host (in-place, chunked along deth) --------------
d = pinned_empty((ntheta, deth, detw), dtype="float32")
d[:] = projections_after_flatfield        # I/I0, shape (ntheta, deth, detw)
minus_log_inplace(d, eps=1e-6)             # Beer-Lambert
remove_stripe_fw_inplace(d, sigma=3.0, wname="sym16", level=7)   # Münch FW

# --- reconstruct --------------------------------------------------------------
rec.u[:] = 0
rec.BH(d, dbg=True, dbg_step=4, vis_step=8, vis_dir="vis/")
```

`rec.u` holds the final volume. With `vis_step` set, every checkpoint dumps a mid-z slice,
a mid-y slice, and a full multi-page TIFF of the volume into `vis_dir/`.

## Key features

| Area | What it gives you |
|---|---|
| **Operator** | Linear `L u = fft2d ∘ usfft2d ∘ usfft1d`; `L*` adjoint; chunked along three different axes for pipelining |
| **Solver** | BH-CG with Charbonnier-TV — outer CG, lagged-diffusivity Hessian |
| **Linearity cache** | `(Lu, Lη)` updated alongside `(u, η)` by cheap proj-space linear combinations → 1 fwd + 1 adj per iter (was ~5 fwd + 1 adj) |
| **Memory** | `float32` everywhere user-facing; `complex64` only inside FFT scratch; per-chunk real↔complex conversion on GPU |
| **Vertical support mask** | `support=(start, end)` along `n0` — projected gradient onto the known sample extent |
| **Off-center rotation axis** | `axis=…` absorbed as a per-chunk Fourier-shift phase — no edge padding, no resampling |
| **Preprocessing** | `minus_log_inplace`, `remove_stripe_fw_inplace` (FW vendored from TomocuPy) — both chunked along deth |
| **Logging** | per-iter `α`, `β`, RAM, GPU, objective `F` — goes through Python `logging` (see `lam_usfft/logger_config.py`) |
| **Visualisation** | mid-slice TIFFs + full-volume multi-page TIFF per `vis_step` |

## Tests

```bash
cd tests
python test_bh.py        # adjoint identity, chunked vs unchunked,
                         # quadratic convergence, regularizer smoothing
python test_chip.py      # synthetic chip phantom with TV regularization
python test_perf.py      # warm-up + timed fwd_lam / adj_lam at n=1024
```

## Project layout

```
src/lam_usfft/
├── rec.py             Rec class + BH-CG driver
├── chunking.py        @gpu_batch decorator + pre-allocated GPU pool
├── extra_terms.py     TVTerm (Charbonnier TV with lagged-diffusivity Hessian)
├── usfft1d.py         1-D NUFFT (batched)
├── usfft2d.py         2-D NUFFT (gather/scatter in Fourier space)
├── fft2d.py           cuFFT 2-D wrapper
├── remove_stripe.py   FW ring removal + minus_log preprocessing (vendored from TomocuPy)
├── cuda_kernels.py    raw CUDA C kernel source (runtime-compiled via cupy)
├── logger_config.py   coloured stdout + optional file handler
└── utils.py           pinned_array / pinned_empty / redot / lap

experimental/
├── rec_chawla.py      end-to-end reconstruction of the Chawla intel-FOV IC dataset
└── rec_chip.py        synthetic chip phantom driver

tests/
├── test_bh.py         smoke tests for the solver
├── test_chip.py       chip phantom reconstruction
└── test_perf.py       n=1024 timing harness
```

## Algorithm notes

The Fourier-slice theorem extends cleanly from tomography (rotation axis ⟂ beam) to
laminography (axis tilted by `φ`): only the unequally-spaced 3-D Fourier-space sampling
grid changes,

$$
\xi_1 = k_u \cos\theta + k_v \sin\theta \sin\varphi,\quad
\xi_2 = k_u \sin\theta - k_v \cos\theta \sin\varphi,\quad
\xi_3 = k_v \cos\varphi.
$$

The resulting *missing cone* along the rotation axis (half-angle `φ`) is the source of
"tail" streaks under high-contrast features — exactly the artefacts TV regularization
suppresses. See [`fourier_slice_laminography.tex`](fourier_slice_laminography.tex) for the
formula derivation.

## References

- V. Nikitin et al., *Laminography as a tool for imaging large-size samples with high
  resolution*. **J. Synchrotron Rad. 31, 851–866 (2024).** [doi:10.1107/S1600577524002923](https://doi.org/10.1107/S1600577524002923)
- V. Nikitin et al., *Nano-laminography with a transmission X-ray microscope*.
  **J. Synchrotron Rad. 32, 1452–1462 (2025).**
- V. Nikitin, *TomocuPy*. **J. Synchrotron Rad. 30, 179–191 (2023).**

## License

Same license as the parent repository (see [LICENSE](LICENSE) if present).
FW ring removal vendored from TomocuPy (UChicago Argonne, LLC — BSD-3).
