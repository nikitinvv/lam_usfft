import numpy as np
from ffttests.fftcl import FFTCL
import dxchange
import time
import matplotlib.pyplot as plt


n0 = 256
n1 = 256
n2 = 256
detw = 256
deth = 128
ntheta = 8

n1c = n1//4
dethc = deth//4
nthetac = ntheta//4
phi = np.pi/2-30/180*np.pi
theta = np.linspace(0, 360, ntheta, endpoint=True).astype('float32')
f = -dxchange.read_tiff('delta-chip-256.tiff')+1j*0
f = f[f.shape[0]//2-n0//2:f.shape[0]//2+n0//2, f.shape[1]//2-n1 //
      2:f.shape[1]//2+n1//2, f.shape[2]//2-n2//2:f.shape[2]//2+n2//2]
# f = np.zeros([n0,n1,n2]).astype('complex64')
# f = np.random.random([n0,n1,n2])+1j*np.random.random([n0,n1,n2])

f = f.astype('complex64')
with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:

    # data = slv.fwd_usfft1d(f, phi)
    # ddata = slv.fwd_usfft2d(data, theta, phi)
    # res = slv.adj_fft2d(ddata)
    # dxchange.write_tiff(res.real,'res/data.tiff',overwrite=True)

    res = slv.fwd_lam(f, theta, phi)
    dxchange.write_tiff(res.real, 'res/data.tiff', overwrite=True)
    print(np.linalg.norm(res))
