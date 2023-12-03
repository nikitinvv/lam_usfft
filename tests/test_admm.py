import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import scipy.ndimage as ndimage

n = 256
n0 = 128
n1 = n
n2 = n
detw = n
deth = n
ntheta = 384

n1c = n1//8
dethc = deth//8
nthetac = ntheta//8
phi = np.pi/2-30/180*np.pi
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

u = dxchange.read_tiff('delta-chip-256.tiff').swapaxes(0,1)-0.5*1j*dxchange.read_tiff('delta-chip-256.tiff').swapaxes(0,1)
u = u[:,32:32+128]
#u = ndimage.zoom(u,n//256,order=1).astype('complex64')

with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    data = slv.fwd_lam(u, theta, phi)    
    u = np.zeros(u.shape,dtype='complex64')
    u = slv.cg_lam(data, u, theta, phi, 32,dbg=True)
    dxchange.write_tiff(u.real, 'res/ure_cg.tiff', overwrite=True)
    dxchange.write_tiff(u.imag, 'res/uim_cg.tiff', overwrite=True)
    
    u = np.zeros(u.shape,dtype='complex64')
    psi = np.zeros([3,*u.shape],dtype='complex64')
    h = np.zeros([3,*u.shape],dtype='complex64')    
    lamd = np.zeros([3,*u.shape],dtype='complex64')    
    niter = 257
    liter = 4
    alpha = 1.5e-8
    u = slv.admm(data, h, psi, lamd, u, theta, phi, alpha, liter, niter, dbg=True,dbg_step=4)
    
    dxchange.write_tiff(u.real, 'res/ure_admm.tiff', overwrite=True)
    dxchange.write_tiff(u.imag, 'res/uim_admm.tiff', overwrite=True)
    
    print(np.linalg.norm(u))
