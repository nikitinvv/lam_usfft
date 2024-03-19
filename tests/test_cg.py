import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import time
import scipy.ndimage as ndimage

n = 256
n0 = n//4
n1 = n
n2 = n
detw = n
deth = n
ntheta = 200

n1c = n1//4
dethc = deth//4
nthetac = ntheta//4
phi = np.pi/2-20/180*np.pi# 30 deg w.r.t. the beam direction
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

u = np.ones([n1,n0,n2],dtype='float32')
# u = dxchange.read_tiff('delta-chip-256.tiff')[128-n0//2:128+n0//2,128-n1//2:128+n1//2,128-n2//2:128+n2//2].swapaxes(0,1)# shape [n1,n0,n2] is more optimal for computations    
with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    data = slv.fwd_lam(u, theta, phi)    
    u = np.zeros(u.shape,dtype='float32')
    u = slv.cg_lam(data, u, theta, phi, 64, 1, dbg=True,dbg_step=1)
    
    dxchange.write_tiff(u, 'res/ure.tiff', overwrite=True)    
    
    print(np.linalg.norm(u))
