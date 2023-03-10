import numpy as np
from ffttests.fftcl import FFTCL
import dxchange
import time
n0 = 18
n1 = 20
n2 = 20
m = 4
f = np.random.random([n0,n1,n2]).astype('complex64')
# f = np.ones([n0,n1,n2]).astype('complex64')
fe = np.zeros([2*n0,n1,n2]).astype('complex64')
fe[n0//2:-n0//2] = f
with FFTCL(n0, n1, n2, m) as slv:
    t = time.time()
    data = slv.fwd_fft1d(f)
    print(time.time()-t)
    np.fft.fftshift(np.fft.fft(np.fft.fftshift(data[n0//2+m:-n0//2-m],axes=0),axis=0),axes=0)
    ddata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(fe,axes=0),axis=0),axes=0)
    print(np.linalg.norm(data))
    print(np.linalg.norm(ddata))
    print(np.linalg.norm(data[m:-m]-ddata))
    