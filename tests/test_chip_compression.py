import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import time
import sys
import base64
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
phi = np.pi/2-30/180*np.pi# 30 deg w.r.t. the beam direction
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

import zlib

with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:
    f = dxchange.read_tiff('delta-chip-256.tiff')[128-n0//2:128+n0//2,128-n1//2:128+n1//2,128-n2//2:128+n2//2].swapaxes(0,1)# shape [n1,n0,n2] is more optimal for computations
    f = f-1j*0.5*f
    f = np.ascontiguousarray(f)+np.random.random(f.shape).astype('complex64')+1j*np.random.random(f.shape).astype('complex64')
    fc = zlib.compress(f)
    print(f'compression {sys.getsizeof(fc)/sys.getsizeof(f)}')
    f1 = zlib.decompress(fc)
    f1 = np.frombuffer(f1, dtype=np.complex64).reshape(f.shape)
    print(np.linalg.norm(f-f1))
    
    