import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import sys

bin = int(sys.argv[1]) # 2
alpha = float(sys.argv[2])
# pref = sys.argv[3]
data_file = f'/data/2023-04-Nikitin/np/brain_lam_20deg_middle_421_bin{bin}.npy'
rec_folder = f'/data/2023-04-Nikitin/np/brain_lam_20deg_middle_421_bin{bin}_{alpha}rec/'
print('load data')
data = np.load(data_file)
data = np.pad(data,((0,0),(0,0),(data.shape[2]//4,data.shape[2]//4)),'edge')
print('pad data')
n0 = data.shape[1]
n1 = data.shape[2]
n2 = data.shape[2]
detw = data.shape[2]
deth = data.shape[1]
ntheta = data.shape[0]

print(f'data size (ntheta,deth,detw) = ({ntheta},{deth},{detw}) ')
print(f'reconstruction size (n0,n1,n2) = ({n0},{n1},{n2}) ')

n1c = 8
dethc = 8
nthetac = 5
phi = np.pi/2+20/180*np.pi# 20 deg w.r.t. the beam direction
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')


with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    # u = np.zeros([n1,n0,n2],dtype='float32')
    # u = slv.cg_lam(data, u, theta, phi, 64, gamma=2,dbg=True,dbg_step=4)
    # u = u.swapaxes(0,1)
    # u = u[:,u.shape[1]//6:-u.shape[1]//6,u.shape[1]//6:-u.shape[1]//6]
    # dxchange.write_tiff_stack(u, f'{rec_folder}/u.tiff', overwrite=True)
    # exit()
    u = np.zeros([n1,n0,n2],dtype='float32')#reshaped        
    psi = np.zeros([3,*u.shape],dtype='float32')
    h = np.zeros([3,*u.shape],dtype='float32')    
    lamd = np.zeros([3,*u.shape],dtype='float32')    

    niter = 64
    liter = 4
    alpha = float(sys.argv[2])
    
    u = slv.admm(data, h, psi, lamd, u, theta, phi, alpha, liter, niter, dbg=False,dbg_step=4,rec_folder=rec_folder)
    u = u.swapaxes(0,1)
    u = u[:,u.shape[1]//6:-u.shape[1]//6,u.shape[1]//6:-u.shape[1]//6]
    print(np.linalg.norm(u))
    dxchange.write_tiff_stack(u, f'{rec_folder}/u.tiff', overwrite=True)
    