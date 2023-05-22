import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import sys

fname = sys.argv[1]
n0 = 128# work for now with multiples of 16..
n1 = 2020# work for now with multiples of 16..
n2 = 2020
detw = 2800# work for now with multiples of 16..
deth = 200
ntheta = 100

n1c = 140
dethc = 32
nthetac = 16
phi = np.pi/2-2/180*np.pi
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

u = np.zeros([n0,n1,n2],dtype='complex64')
u0 = np.load(f'/local/{fname}.npy').swapaxes(0,1)
u[:u0.shape[0],:u0.shape[1],:u0.shape[2]] = u0

u = u.swapaxes(0,1)
with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    data = slv.fwd_lam(u, theta, phi)    
    
    # dxchange.write_tiff(data.real, f'/local/res/data_{fname}_re.tiff', overwrite=True)
    # dxchange.write_tiff(u.real, f'/local/res/u_{fname}_re.tiff', overwrite=True)
    # dxchange.write_tiff(data.imag, f'/local/res/data_{fname}_im.tiff', overwrite=True)
    # dxchange.write_tiff(u.imag, f'/local/res/u_{fname}_im.tiff', overwrite=True)
    # u = np.zeros(u.shape,dtype='complex64')
    # u = slv.cg_lam(data, u, theta, phi, 64,dbg=True,dbg_step=4)
    
    # dxchange.write_tiff(u.real, f'/local/res/ucg_{fname}_re.tiff', overwrite=True)
    # dxchange.write_tiff(u.imag, f'/local/res/ucg_{fname}_im.tiff', overwrite=True)
    
    # exit()
    for alpha in [1e-8,4e-8,8e-8,1.2e-7]:
        psi = np.zeros([3,*u.shape],dtype='complex64')
        h = np.zeros([3,*u.shape],dtype='complex64')    
        lamd = np.zeros([3,*u.shape],dtype='complex64')    
        niter = 65
        liter = 4
        u = np.zeros(u.shape,dtype='complex64')
        u = slv.admm(data, h, psi, lamd, u, theta, phi, alpha, liter, niter,dbg=True,dbg_step=4)
        
        dxchange.write_tiff(u.real, f'/local/res/utv_{fname}{alpha}_re.tiff', overwrite=True)
        dxchange.write_tiff(u.imag, f'/local/res/utv_{fname}{alpha}_im.tiff', overwrite=True)
        
    