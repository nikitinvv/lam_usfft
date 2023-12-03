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

theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

u = -dxchange.read_tiff('delta-chip-256.tiff').swapaxes(0,1)
u = u[:,32:32+128]

phia = [0,10,20,30,40,50]
for k in range(len(phia)):
    phi = np.pi/2-phia[k]/180*np.pi
    with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
        data = slv.fwd_lam(u, theta, phi)    
        # u = np.zeros(u.shape,dtype='complex64')
        # u = slv.cg_lam(data, u, theta, phi, 100,dbg=True)
        # dxchange.write_tiff(u.real, f'res/ure_cg_{phia[k]}.tiff', overwrite=True)
        # dxchange.write_tiff(u[128].real, f'res/ure_cg0_{phia[k]}.tiff', overwrite=True)
        
        u = np.zeros(u.shape,dtype='complex64')
        psi = np.zeros([3,*u.shape],dtype='complex64')
        h = np.zeros([3,*u.shape],dtype='complex64')    
        lamd = np.zeros([3,*u.shape],dtype='complex64')    
        niter = 257
        liter = 4
        alpha = 1.5e-8
        u = slv.admm(data, h, psi, lamd, u, theta, phi, alpha, liter, niter, dbg=True,dbg_step=4)

        dxchange.write_tiff(u.real, f'res/ure_admm_{phia[k]}.tiff', overwrite=True)
        dxchange.write_tiff(u[128].real, f'res/ure_admm0_{phia[k]}.tiff', overwrite=True)
