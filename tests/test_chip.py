import numpy as np
from ffttests.fftcl import FFTCL
import dxchange
import time

n = 256
n0 = n
n1 = n
n2 = n
detw = n
deth = n
ntheta = n

n1c = n1//4
dethc = deth//4
nthetac = ntheta//4
phi = np.pi/2-30/180*np.pi
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:
    f = -dxchange.read_tiff('delta-chip-256.tiff')[128-n0//2:128+n0//2,128-n1//2:128+n1//2,128-n2//2:128+n2//2]+1j*dxchange.read_tiff('delta-chip-256.tiff')[128-n0//2:128+n0//2,128-n1//2:128+n1//2,128-n2//2:128+n2//2]

    for k in range(1):    
        t = time.time()
        res = slv.fwd_lam(f, theta, phi)
        print(time.time()-t)
    dxchange.write_tiff(res.real, 'res/datare.tiff', overwrite=True)
    dxchange.write_tiff(res.imag, 'res/dataim.tiff', overwrite=True)
    print(np.linalg.norm(res))
