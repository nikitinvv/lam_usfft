#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
#                               DISCLAIMER                                    #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS           #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT    #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
# *************************************************************************** #

import numpy as np
import cupy as cp
from threading import Thread
import sys

# Print iterations progress
def printProgressBar(iteration, total, qsize, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(
        f'\rqueue size {qsize:03d} | {prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

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
    sys.exit(1)

def _copy(res, u, st, end):
        res[st:end] = u[st:end]
        
def copy(u, res=None, nthreads=16):
    if res is None:
        res = np.empty([u.shape[0],u.shape[1],u.shape[2]],dtype=u.dtype)
    nchunk = int(np.ceil(u.shape[0]/nthreads))
    mthreads = []
    for k in range(nthreads):
        th = Thread(target=_copy,args=(res,u,k*nchunk,min((k+1)*nchunk,u.shape[0])))
        mthreads.append(th)
        th.start()
    for th in mthreads:
        th.join()
    return res

def _copyTransposed(res, u, st, end):
    res[st:end] = u[:,st:end].swapaxes(0,1)        
    
def copyTransposed(u, res=None, nthreads=16):
    if res is None:
        res = np.empty([u.shape[1],u.shape[0],u.shape[2]],dtype=u.dtype)
    nchunk = int(np.ceil(u.shape[1]/nthreads))
    mthreads = []
    for k in range(nthreads):
        th = Thread(target=_copyTransposed,args=(res,u,k*nchunk,min((k+1)*nchunk,u.shape[1])))
        mthreads.append(th)
        th.start()
    for th in mthreads:
        th.join()        
    return res