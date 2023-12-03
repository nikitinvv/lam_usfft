import numpy as np
from lam_usfft.fftcl import FFTCL
import dxchange
import sys
import os
import gc
# *npy data files can be downloaded from box: https://anl.app.box.com/folder/207370334443

bin = int(sys.argv[1]) # 2
alpha = float(sys.argv[2])
# pref = sys.argv[3]
data_file = f'/data/2023-04-Nikitin/np/brain_lam_20deg_middle_421_bin{bin}.npy'
rec_folder = f'/data/2023-04-Nikitin/np/brain_lam_20deg_middle_421_bin{bin}_{alpha}rec/'
print('load data')
data = np.load(data_file)#[::2]
# data = data[::2,:,(3200-2048-512)//2//2**bin:(3200+2048+512)//2//2**bin]
# data = np.pad(data,((0,0),(0,0),((4096//2**bin-data.shape[2])//2,(4096//2**bin-data.shape[2])//2)),'edge')
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
nthetac = 10
phi = np.pi/2+20/180*np.pi# 20 deg w.r.t. the beam direction
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')
gamma = 2


with FFTCL(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:    
    
    # u = np.memmap('/data/tmp/u.dat', dtype=np.complex64,
    #           mode='w+', shape=(n1,n0,n2))
    
    # psi = np.memmap('/data/tmp/psi.dat', dtype=np.complex64,
    #           mode='w+', shape=(3,*u.shape))
    # h = np.memmap('/data/tmp/h.dat', dtype=np.complex64,
    #           mode='w+', shape=(3,*u.shape))
    # lamd = np.memmap('/data/tmp/h.dat', dtype=np.complex64,
    #           mode='w+', shape=(3,*u.shape))
    # for k in {'h','u','psi','data','lamd'}:
    #     fname = f'{pref}/{k}' 
    #     if not os.path.exists(fname):
    #         os.makedirs(fname)
    
    print((n0*n1*n2*17+ntheta*deth*detw*2)*8/1024**3)
        
    u = np.zeros([n1,n0,n2],dtype='float32')#reshaped        
    psi = np.zeros([3,n1,n0,n2],dtype='float32')    
    h = np.zeros([3,n1,n0,n2],dtype='float32')        
    lamd = np.zeros([3,n1,n0,n2],dtype='float32') 
    grad = np.zeros([n1,n0,n2],dtype='float32')#reshaped        
    grad0 = np.zeros([n1,n0,n2],dtype='float32')#reshaped        
    gradG = np.zeros([n1,n0,n2],dtype='float32')#reshaped       
    d = np.zeros([n1,n0,n2],dtype='float32')#reshaped         
    bb = max(ntheta*deth*detw,3*n0*n1*n2)
    tmp1 = np.zeros([ntheta,deth,detw],dtype='float32')#reshaped  
    tmp2 = np.zeros([3,n1,n0,n2],dtype='float32')        
    # tmp = np.zeros(bb,dtype='float32')#reshaped  
    # tmp1 = tmp[:ntheta*deth*detw].reshape(ntheta,deth,detw)
    # tmp2 = tmp[:3*n0*n1*n2].reshape(3,n1,n0,n2)
    # exit()
    niter = 100
    liter = 4
    alpha = float(sys.argv[2])
    
    u = slv.admm(u, psi, h, lamd, data,grad,grad0,gradG, tmp1,tmp2,d, theta, phi, alpha, liter, niter, gamma, dbg=True,dbg_step=4)
    u = u.swapaxes(0,1)
    u = u[:,u.shape[1]//6:-u.shape[1]//6,u.shape[1]//6:-u.shape[1]//6]
    print(np.linalg.norm(u))
    dxchange.write_tiff_stack(u.real, f'{rec_folder}/u.tiff', overwrite=True)
    