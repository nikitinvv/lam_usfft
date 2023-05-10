import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import time
import scipy.ndimage as ndimage

n = 512
n0 = n
n1 = n
n2 = n
detw = n
deth = n
ntheta = n

n1c = n1//8
dethc = deth//8
nthetac = ntheta//8
phi = np.pi/2-30/180*np.pi# 30 deg w.r.t. the beam direction
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

u = dxchange.read_tiff('delta-chip-256.tiff').swapaxes(0,1)# shape [n1,n0,n2] is more optimal for computations
u = ndimage.zoom(u,n//256,order=1).astype('complex64')
with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    data = slv.fwd_lam(u, theta, phi)    
    
    u = np.zeros(u.shape,dtype='complex64')
    u = slv.cg_lam(data, u, theta, phi, 4,dbg=False)
    
    dxchange.write_tiff(u.real, 'res/ure.tiff', overwrite=True)
    dxchange.write_tiff(u.imag, 'res/uim.tiff', overwrite=True)
    
    print(np.linalg.norm(u))
