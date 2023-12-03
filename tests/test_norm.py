import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange

n = 128
n0 = n
n1 = n
n2 = n
detw = n
deth = n
ntheta = n

n1c = n1//4
dethc = deth//4
nthetac = ntheta//4
phi = np.pi/2-20/180*np.pi
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:
    f = dxchange.read_tiff('delta-chip-256.tiff')[128-n0//2:128+n0//2,128-n1//2:128+n1//2,128-n2//2:128+n2//2].swapaxes(0,1)# shape [n1,n0,n2] is more optimal for computations    
    data = slv.fwd_lam(f, theta, phi)
    fr = slv.adj_lam(data, theta, phi)
    ddata = slv.fwd_lam(fr, theta, phi)
        
    print(np.sum(f*np.conj(fr)),np.sum(data*np.conj(data)))    
    
    print(np.sum(data*np.conj(ddata))/np.sum(ddata*np.conj(ddata)))    
    print(np.sum(f*np.conj(fr))/np.sum(fr*np.conj(fr)))    
    print('')
    dxchange.write_tiff(data.real, 'res/datare.tiff', overwrite=True)
    dxchange.write_tiff(data.imag, 'res/dataim.tiff', overwrite=True)
    # 
    # print(np.linalg.norm(data))
