import numpy as np
import cupy as cp
import time

from lam_usfft.rec import Rec
from lam_usfft.utils import pinned_empty

n = 1024
n0 = n
n1 = n
n2 = n
detw = n
deth = n
ntheta = n

n1c = n1 // 8
dethc = deth // 8
nthetac = ntheta // 8
phi = np.pi / 2 - 30 / 180 * np.pi
theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=True).astype("float32")

rec = Rec(n0, n1, n2, detw, deth, ntheta, theta, phi,
          n1c=n1c, dethc=dethc, nthetac=nthetac)

# Pinned-host float32 input for async H2D inside the chunked pipeline.
f = pinned_empty((n1, n0, n2), dtype="float32")
f[:] = 0
f[n1 // 4:3 * n1 // 4, n0 // 4:3 * n0 // 4, n2 // 4:3 * n2 // 4] = 1

# Warm-up: trigger JIT/plan caches so they don't pollute the timing.
data = rec.fwd_lam(f)
_   = rec.adj_lam(data)
cp.cuda.Device().synchronize()

t = time.time()
data = rec.fwd_lam(f)
cp.cuda.Device().synchronize()
print(time.time() - t)

t = time.time()
fr = rec.adj_lam(data)
cp.cuda.Device().synchronize()
print(time.time() - t)
