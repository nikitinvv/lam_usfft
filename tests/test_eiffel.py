import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import scipy.ndimage as ndimage

n0 = 112
n1 = 2020
n2 = 2020
detw = 2816
deth = 256
ntheta = 64

n1c = 101
dethc = 32
nthetac = 8
phi = np.pi/2-2/180*np.pi
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

u = np.zeros([n0,n1,n2],dtype='complex64')
u0 = np.load('/local/tomodata1/vnikitin/8id/Eiffel_in_Air_4b4um.npy').swapaxes(0,1)
u[:u0.shape[0],:u0.shape[1],:u0.shape[2]] = u0
u = u.swapaxes(0,1)
with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    data = slv.fwd_lam(u, theta, phi)    
    dxchange.write_tiff(data.real, 'res/data_eiffel_re.tiff', overwrite=True)
    dxchange.write_tiff(u.real, 'res/u_eiffel.tiff', overwrite=True)
    dxchange.write_tiff(data.imag, 'res/data_eiffel_im.tiff', overwrite=True)
    dxchange.write_tiff(u.imag, 'res/u_eiffel_im.tiff', overwrite=True)
    
    u = np.zeros(u.shape,dtype='complex64')
    u = slv.cg_lam(data, u, theta, phi, 32,dbg=True)
    
    dxchange.write_tiff(u.real, 'res/ucg_eiffel_re.tiff', overwrite=True)
    dxchange.write_tiff(u.imag, 'res/ucg_eiffel_im.tiff', overwrite=True)
    # exit()
    psi = np.zeros([3,*u.shape],dtype='complex64')
    h = np.zeros([3,*u.shape],dtype='complex64')    
    lamd = np.zeros([3,*u.shape],dtype='complex64')    
    niter = 32
    liter = 4
    alpha = 3e-8
    u = np.zeros(u.shape,dtype='complex64')
    u = slv.admm(data, h, psi, lamd, u, theta, phi, alpha, liter, niter, dbg=True)
    
    dxchange.write_tiff(u.real, 'res/utv_eiffel_re.tiff', overwrite=True)
    dxchange.write_tiff(u.imag, 'res/utv_eiffel_im.tiff', overwrite=True)
    
    print(np.linalg.norm(u))
