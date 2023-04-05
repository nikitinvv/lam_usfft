import numpy as np
from ffttests.fftcl import FFTCL
import dxchange
import time

n = 2048
n0 = n
n1 = n
n2 = n
detw = n
deth = n
ntheta = n

n1c = n1//32
dethc = deth//32
nthetac = ntheta//32
phi = np.pi/2-30/180*np.pi
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

f = np.zeros([n0,n1,n2]).astype('complex64')
f[n0//4:3*n0//4,n1//4:3*n1//4,n2//4:3*n2//4] = 1
f = f.astype('complex64')
with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:
    t = time.time()
    res = slv.fwd_lam(f, theta, phi)
    print(time.time()-t)
