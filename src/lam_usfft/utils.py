
import numpy as np
import cupy as cp
import os
from threading import Thread
import time

def pinned_array(array):
    """Allocate pinned memory and associate it with numpy array"""

    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(
        mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


def signal_handler(sig, frame):
    """Calls abort_scan when ^C or ^Z is typed"""

    print('Abort')
    os.system('kill -9 $PPID')

def write_array(res,res0,st,end):             
    res[:,st:end] = res0[:,:end-st]
    
def read_array(res0,res,st,end):             
    res0[:,:end-st] = res[:,st:end]

def _copy(res, u, st, end):
    res[st:end] = u[st:end]
    
def copy(u, res=[], nthreads=8):
    if res==[]:
        res = np.empty_like(u)
    nchunk = int(np.ceil(u.shape[0]//nthreads))
    mthreads = []
    for k in range(nthreads):
        th = Thread(target=_copy,args=(res,u,k*nchunk,min((k+1)*nchunk,u.shape[0])))
        mthreads.append(th)
        th.start()
    for th in mthreads:
        th.join()
    return res
    