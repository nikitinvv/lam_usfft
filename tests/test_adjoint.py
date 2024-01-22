import numpy as np
import time
from lam_usfft import LAM

n = 1024 

# reconstructed volume dimensions
n0 = n
n1 = n
n2 = n

# data dimensions
detw = n # detector width
deth = n # detector height
ntheta = n # number of projections

# chunk dimensions (to fit gpu memory)
n1c = n1//32
dethc = deth//32
nthetac = ntheta//32

# lamino angle
phi = np.pi/2-20/180*np.pi

# projection angles
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=True).astype('float32')

with LAM(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac) as slv:
    # generate some volume
    f = np.zeros([n1,n0,n2],dtype='float32')
    f[n1//4:3*n1//4,n0//4:3*n0//4,n2//4:3*n2//4] = 1
    
    # model data (forward transform)
    t = time.time()
    data = slv.fwd_lam(f, theta, phi)    
    print(f'fwd_lam time {time.time()-t}s')
    
    # adjoint transform
    t = time.time()
    fr = slv.adj_lam(data, theta, phi)    
    print(f'adj_lam time {time.time()-t}s')
    
    # check the operators work correctly:
    print(f'Checking result (can be commented for performance tests):') 
    print(f'{np.sum(f*np.conj(fr)):.4e}=={np.sum(data*np.conj(data)):.4e}')    
    
