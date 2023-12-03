import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import scipy.ndimage as ndimage

n0 = 128# work for now with multiples of 16..
n1 = 2048# work for now with multiples of 16..
n2 = 2048
detw = 2816# work for now with multiples of 16..
deth = 256
ntheta = 128

n1c = 128
dethc = 32
nthetac = 16
phi = np.pi/2-2/180*np.pi
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

u = np.zeros([n0,n1,n2],dtype='complex64')
u0 = np.load('/local/Eiffel_in_Air_4b4um.npy').swapaxes(0,1)
u[:u0.shape[0],:u0.shape[1],:u0.shape[2]] = u0
# u=u[::2,::2,::2]
# [n0,n1,n2]=u.shape

u = u.swapaxes(0,1)
with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    data = slv.fwd_lam(u, theta, phi)    
    # dxchange.write_tiff(data.real, '/local/res/data_eiffel_re.tiff', overwrite=True)
    # dxchange.write_tiff(u.real, '/local/res/u_eiffel.tiff', overwrite=True)
    # dxchange.write_tiff(data.imag, '/local/res/data_eiffel_im.tiff', overwrite=True)
    # dxchange.write_tiff(u.imag, '/local/res/u_eiffel_im.tiff', overwrite=True)
    
    # u = np.zeros(u.shape,dtype='complex64')
    # u = slv.cg_lam(data, u, theta, phi, 32,dbg=True,dbg_step=4)
    
    # dxchange.write_tiff(u.real, '/local/res/ucg_eiffel_re.tiff', overwrite=True)
    # dxchange.write_tiff(u.imag, '/local/res/ucg_eiffel_im.tiff', overwrite=True)
    # # exit()
    for alpha in [1e-8,4e-8,8e-8,1.2e-7]:
        psi = np.zeros([3,*u.shape],dtype='complex64')
        h = np.zeros([3,*u.shape],dtype='complex64')    
        lamd = np.zeros([3,*u.shape],dtype='complex64')    
        niter = 65
        liter = 4
        u = np.zeros(u.shape,dtype='complex64')
        u = slv.admm(data, h, psi, lamd, u, theta, phi, alpha, liter, niter,dbg=True,dbg_step=4)
        
        dxchange.write_tiff(u.real, f'/local/res/utv_eiffel{alpha}_re.tiff', overwrite=True)
        dxchange.write_tiff(u.imag, f'/local/res/utv_eiffel{alpha}_im.tiff', overwrite=True)
        
    