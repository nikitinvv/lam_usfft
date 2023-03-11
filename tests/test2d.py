import numpy as np
from ffttests.fftcl import FFTCL
import dxchange
import time
n0 = 16
n1 = 20
n2 = 36
m = 5
f = np.random.random([n0,n1,n2]).astype('complex64')
f = np.ones([n0,n1,n2]).astype('complex64')
fe = np.zeros([n0,2*n1,2*n2]).astype('complex64')
fe[:,n1//2:-n1//2,n2//2:-n2//2] = f
with FFTCL(n0, n1, n2, m) as slv:
    t = time.time()
    data = slv.fwd_fft2d(f)
    print(np.linalg.norm(data))
    print(np.linalg.norm(data[:,m:-m,m:-m]))
    print(time.time()-t)
    ddata = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fe,axes=(1,2)),axes=(1,2)),axes=(1,2))
    # ddata = np.fft.fft2(fe,axes=(1,2))
    print(np.linalg.norm(data))
    print(np.linalg.norm(ddata))
    print(np.linalg.norm(data[:,m:-m,m:-m]-ddata))
    