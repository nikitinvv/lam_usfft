import numpy as np
from ffttests.fftcl import FFTCL
import dxchange
import time
import matplotlib.pyplot as plt


n0 = 16
n1 = 10
n2 = 6
detw = 8
deth = 16
ntheta = 8
phi = np.pi/2-30/180*np.pi
theta = np.linspace(0, 360, ntheta, endpoint=True).astype('float32')
# f = -dxchange.read_tiff('delta-chip-256.tiff')+1j*0

f = np.zeros([n0,n1,n2]).astype('complex64')
f = np.random.random([n0,n1,n2])+1j*np.random.random([n0,n1,n2])

f = f.astype('complex64')
with FFTCL(n0, n1, n2, detw, deth, ntheta) as slv:
    t= time.time()
    data = slv.fwd_fft1d(f, phi)
    print(time.time()-t)
    t= time.time()
    ddata = slv.fwd_fft2d(data, theta, phi)
    print(time.time()-t)
    
    # data0 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(f,axes=0),axis=0),axes=0).astype('complex64')
    # print(np.linalg.norm(data))
    # print(np.linalg.norm(data0))
    # dxchange.write_tiff(data.real, 'data/r', overwrite=True)
    # dxchange.write_tiff(data0.real, 'data/r0', overwrite=True)
    # plt.subplot(1,2,1)
    # plt.imshow(data[deth//2].real)
    # plt.colorbar()
    # plt.subplot(1,2,2)
    # plt.imshow(data0[deth//2].real)
    # plt.colorbar()
    # plt.show()
    
    
    # 