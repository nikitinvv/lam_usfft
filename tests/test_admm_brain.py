import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange

# *npy data files can be downloaded from box: https://anl.app.box.com/folder/207370334443

bin = 4 # 2

data_file = f'/data/2023-04/Nikitin_rec/data_brain_bin{bin}x{bin}.npy'
rec_folder = f'/data/2023-04/Nikitin_rec/rec_admm_bin{bin}x{bin}'
data = np.load(data_file).astype('complex64')

data = np.pad(data,((0,0),(0,0),(data.shape[2]//4,data.shape[2]//4)),'edge')
n0 = data.shape[1]
n1 = data.shape[2]
n2 = data.shape[2]
detw = data.shape[2]
deth = data.shape[1]
ntheta = data.shape[0]

print(f'data size (ntheta,deth,detw) = ({ntheta},{deth},{detw}) ')
print(f'reconstruction size (n0,n1,n2) = ({n0},{n1},{n2}) ')

n1c = 32
dethc = 32
nthetac = 50
phi = np.pi/2+20/180*np.pi# 20 deg w.r.t. the beam direction
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')
gamma = 2


with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    u = np.zeros([n1,n0,n2],dtype='complex64')#reshaped
    psi = np.zeros([3,*u.shape],dtype='complex64')
    h = np.zeros([3,*u.shape],dtype='complex64')    
    lamd = np.zeros([3,*u.shape],dtype='complex64')    
    niter = 33
    liter = 4
    alpha = 2e-9
    u = slv.admm(data, h, psi, lamd, u, theta, phi, alpha, liter, niter, gamma, dbg=True,dbg_step=4)
    
    u = u.swapaxes(0,1)
    u = u[:,u.shape[1]//6:-u.shape[1]//6,u.shape[1]//6:-u.shape[1]//6]
    dxchange.write_tiff_stack(u.real, f'{rec_folder}/u.tiff', overwrite=True)
    
    print(np.linalg.norm(u))
