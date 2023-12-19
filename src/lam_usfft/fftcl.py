import cupy as cp
import numpy as np
from threading import Thread
import psutil
from lam_usfft.lam import LAM
from lam_usfft import utils
from lam_usfft import logging
import dxchange
from pathlib import Path
import gc
logging.setup_custom_logger('logging', level="INFO")
log = logging.getLogger(__name__)

class FFTCL():

    def __init__(self, n0, n1, n2, detw, deth, ntheta, n1c=None, dethc=None, nthetac=None):
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.detw = detw
        self.deth = deth
        self.ntheta = ntheta
        
        if n1c == None:
            self.n1c = n1
        else:
            self.n1c = n1c

        if dethc == None:
            self.dethc = deth
        else:
            self.dethc = dethc

        if nthetac == None:
            self.nthetac = ntheta
        else:
            self.nthetac = nthetac
                       
        self.lam_cl = LAM(n0, n1, n2, detw, deth, ntheta, n1c, dethc, nthetac)      
        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)
    
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.lam_cl = []
    
    def _linear_operation_axis0(self,out,x,y,a,b,st,end):
        out[st:end] = a*x[st:end]+b*y[st:end]        
    
    def _linear_operation_axis1(self,out,x,y,a,b,st,end):
        out[:,st:end] = a*x[:,st:end]+b*y[:,st:end]                
        
    def linear_operation(self,x,y,a,b,axis=0,nthreads=8,out=None,dbg=False):
        """ out = ax+by"""
        if out is None:
            out = np.empty_like(x)
        mthreads = []
        # nchunk = x.shape[axis]//nthreads
        nchunk = int(np.ceil(x.shape[axis]/nthreads))
        
        if axis==0:
            fun = self._linear_operation_axis0
        elif axis==1:
            fun = self._linear_operation_axis1
        for k in range(nthreads):
            
            th = Thread(target=fun,args=(out,x,y,a,b,k*nchunk,min(x.shape[axis],(k+1)*nchunk)))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()
        
        return out
          
    def _fwd_reg(self, res, u, st, end):                
        res[0, st:end, :, :-1] = u[st:end, :, 1:]-u[st:end, :, :-1]
        res[1, st:end, :-1, :] = u[st:end, 1:, :]-u[st:end, :-1, :]
        end0 = min(u.shape[0]-1,end)
        res[2, st:end0, :, :] = u[1+st:1+end0, :, :]-u[st:end0, :, :]
        res[:,st:end] *=1/np.sqrt(3)
    
    def fwd_reg(self, u, res=None, nthreads=8):
        ##Fast version:
        if res is None:
            res = np.zeros([3, *u.shape], dtype='float32')
        nchunk = int(np.ceil(u.shape[0]/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self._fwd_reg,args=(res,u,k*nchunk,min((k+1)*nchunk,u.shape[0])))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()
        return res

    ####Parallel version
    def _adj_reg0(self, res, gr, st, end):
        res[st:end, :, 1:] = gr[0, st:end, :, 1:]-gr[0, st:end, :, :-1]
        res[st:end, :, 0] = gr[0, st:end, :, 0]
    
    def _adj_reg1(self, res, gr, st, end):        
        res[st:end, 1:, :] += gr[1, st:end, 1:, :]-gr[1, st:end, :-1, :]
        res[st:end, 0, :] += gr[1, st:end, 0, :]
    
    def _adj_reg2(self, res, gr, st, end):                
        end0 = min(gr.shape[1]-1,end)
        res[1+st:1+end0, :, :] += gr[2, 1+st:1+end0, :, :]-gr[2, st:end0, :, :]        
        res[1+st:1+end0] *= -1/np.sqrt(3)  # normalization
        if st==0:
            res[0, :, :] += gr[2, 0, :, :]
            res[0, :, :] *= -1/np.sqrt(3)  # normalization
    
    def adj_reg(self, gr, res=None, nthreads=8):
        ##Fast version:
        if res is None:
            res = np.zeros(gr.shape[1:], dtype='float32')
        nchunk = int(np.ceil(gr.shape[1]/nthreads))
        mthreads = []
        for fun in [self._adj_reg0,self._adj_reg1,self._adj_reg2]:
            for k in range(nthreads):
                th = Thread(target=fun,args=(res,gr,k*nchunk,min((k+1)*nchunk,gr.shape[1])))
                mthreads.append(th)
                th.start()
            for th in mthreads:
                th.join()
        return res
    
    def _soft_thresholding(self,z,alpha,rho,st,end):
        za = np.sqrt(np.real(np.sum(z[:,st:end]*np.conj(z[:,st:end]), 0)))
        cond = (za > alpha/rho)
        z[:,st:end][:, ~cond] = 0
        z[:,st:end][:, cond] -= alpha/rho * \
            z[:,st:end][:, cond]/(za[cond])
        
    def soft_thresholding(self,z,alpha,rho,nthreads=8):
        
        # nchunk = self.n1//nthreads
        nchunk = int(np.ceil(self.n1/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self._soft_thresholding,args=(z,alpha,rho,k*nchunk,min(self.n1,(k+1)*nchunk)))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()        
        
    # @profile
    def solve_reg(self, u, lamd, rho, alpha, res=None):
        """ Regularizer problem"""
        ##Slow version:
        # z = self.fwd_reg(u)+lamd/rho        
        # za = np.sqrt(np.real(np.sum(z*np.conj(z), 0)))        
        # z[:, za <= alpha/rho] = 0
        # z[:, za > alpha/rho] -= alpha/rho * \
        #     z[:, za > alpha/rho]/(za[za > alpha/rho])
        ##Fast version:
        if res is None:
            res = np.zeros([3, *u.shape], dtype='float32')
        self.fwd_reg(u,res)
        self.linear_operation(res,lamd,1,1.0/rho,axis=1,out=res)        
        self.soft_thresholding(res,alpha,rho)        
        # return z

    def _update_penalty(self, rres, sres, psi, h, h0, rho, st, end, id):
        """Update rho for a faster convergence"""
        # rho
        tmp = psi[st:end] - h[st:end]
        rres[id] += np.real(np.sum(tmp*np.conj(tmp)))
        tmp = rho*(h[st:end]-h0[st:end])
        sres[id] += np.real(np.sum(tmp*np.conj(tmp)))
            
    def update_penalty(self, psi, h, h0, rho, nthreads=8):
        """Update rhofor a faster convergence"""
        ##Slow version:
        # r = np.linalg.norm(psi - h)**2
        # s = np.linalg.norm(rho*(h-h0))**2
        ##Fast version:
        rres = np.zeros(nthreads,dtype='float64')
        sres = np.zeros(nthreads,dtype='float64')
        mthreads = []
        # nchunk = self.n1//nthreads
        nchunk = int(np.ceil(self.n1/nthreads))
        
        for j in range(3):
            for k in range(nthreads):
                th = Thread(target=self._update_penalty,args=(rres,sres,psi[j], h[j],h0[j],rho,k*nchunk,min(self.n1,(k+1)*nchunk),k))
                th.start()
                mthreads.append(th)
            for th in mthreads:
                th.join()
        r = np.sum(rres)            
        s = np.sum(sres)         
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        return rho
    
    def save0(self,path,data,st,end):        
        np.save(path,data[:,st:end])
    
    def load0(self,path,data,st,end):
        data[:,st:end] = np.load(path)
        
    def save_parallel(self,path,data,nthreads=8):        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        nchunk = int(np.ceil(data.shape[1]/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self.save0,args=(f'{path}{k}',data,k*nchunk,min(data.shape[1],(k+1)*nchunk)))
            th.start()
            mthreads.append(th)
        for th in mthreads:
            th.join()
    
    def load_parallel(self,path,shape,data=None,nthreads=8):
        if data is None:
            data = np.empty(shape,dtype='float32')
        nchunk = int(np.ceil(data.shape[1]/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self.load0,args=(f'{path}{k}.npy',data,k*nchunk,min(data.shape[1],(k+1)*nchunk)))
            th.start()
            mthreads.append(th)
        for th in mthreads:
            th.join()
        return data
    
    # @profile
    def admm(self, data, psi, lamd, u, theta, phi, alpha, titer, niter, gamma=1, dbg=False, dbg_step=1,rec_folder='/data/tmp'):
        """ ADMM for laminography problem with TV regularization"""
        rho = 0.5
        for m in range(niter):
            log.info(f'step0, {psutil.virtual_memory()[3]/1000000000})')                        
            self.linear_operation(psi,lamd,1,-1/rho,axis=1,out=psi)
            self.save_parallel(f'{rec_folder}/lamd',lamd)
            
            log.info(f'step1, {psutil.virtual_memory()[3]/1000000000})')            
            grads = lamd # reuse memory
            with LAM(self.n0, self.n1, self.n2, self.detw, self.deth, self.ntheta, self.n1c, self.dethc, self.nthetac) as slv:    
                slv.cg_lam_ext(data, psi, u, grads, theta, phi, rho, titer, gamma, False)                        
            
            log.info(f'step2, {psutil.virtual_memory()[3]/1000000000})')
            
            # regularizer problem
            self.load_parallel(f'{rec_folder}/lamd',[3,self.n1,self.n0,self.n2],lamd)
            self.solve_reg(u, lamd, rho, alpha,psi)
            self.save_parallel(f'{rec_folder}/psi',psi)
            
            log.info(f'step3, {psutil.virtual_memory()[3]/1000000000})')
            h = psi #reuse memory
            self.fwd_reg(u,h)
            # lambda update   
            log.info(f'step4, {psutil.virtual_memory()[3]/1000000000})')                   
            self.linear_operation(lamd,h,1,rho,axis=1,out=lamd)#lamd = lamd + rho * (h-psi)            
            
            psi = h #reuse memory            
            self.load_parallel(f'{rec_folder}/psi',[3,self.n1,self.n0,self.n2],psi)
            self.linear_operation(lamd,psi,1,-rho,axis=1,out=lamd)
            # update rho for a faster convergence
            # rho = self.update_penalty(psi, h, h0, rho)
            
            # Lagrangians difference between two iterations
            # if dbg and m%dbg_step==0:
            #     lagr = self.take_lagr(
            #         u, psi, data, h, lamd,theta, phi, alpha,rho)
            #     print("%d/%d) rho=%.2e, Lagrangian terms:   %.2e %.2e %.2e %.2e, Sum: %.2e" %
            #             (m, niter, rho, *lagr))
            log.info(f'save {rec_folder}/iters/t{m:03}.tiff')            
            dxchange.write_tiff(u[:,u.shape[1]//2],f'{rec_folder}/iters/t{m:03}.tiff',overwrite=True)            
            dxchange.write_tiff(u[u.shape[0]//2],f'{rec_folder}/iters2/t{m:03}.tiff',overwrite=True)            
            if m%8==0:
                dxchange.write_tiff_stack(u,f'{rec_folder}_tmp/t.tiff',overwrite=True)            
                
        return u

    def take_lagr(self, u, psi, data, h, lamd, theta, phi, alpha, rho):
        """ Lagrangian terms for monitoring convergence"""
        lagr = np.zeros(5, dtype="float32")
        Lu = self.fwd_lam(u,theta, phi)
        lagr[0] += np.linalg.norm(Lu-data)**2
        lagr[1] = alpha*np.sum(np.sqrt(np.real(np.sum(psi*np.conj(psi), 0))))        
        lagr[2] = np.sum(np.real(np.conj(lamd)*(h-psi)))        
        lagr[3] = rho*np.linalg.norm(h-psi)**2
        lagr[4] = np.sum(lagr[:4])
        return lagr
