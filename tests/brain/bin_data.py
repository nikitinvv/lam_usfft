import h5py
import numpy as np
import sys
bin=int(sys.argv[1])

with h5py.File('/data/2023-04-Nikitin/brain_lam_20deg_middle_421.h5','r') as fid:
    # with h5py.File('/data/2023-04-Nikitin/brain_lam_20deg_middle_421_bin1.h5','w') as fidout:
    data = fid['exchange/data']
    data = data[:3000:2**bin,:,6+3232//2-3200//2:6+3232//2+3200//2].astype('float32')
    for k in range(bin):
        data = 0.5*(data[:,::2]+data[:,::2])
        data = 0.5*(data[:,:,::2]+data[:,:,::2])
    # fidout.create_dataset('exchange/data',data=data)    
    
    data_white = fid['exchange/data_white']
    data_white = data_white[:,:,6+3232//2-3200//2:6+3232//2+3200//2].astype('float32')
    
    for k in range(bin):
        data_white = 0.5*(data_white[:,::2]+data_white[:,::2])
        data_white = 0.5*(data_white[:,:,::2]+data_white[:,:,::2])
    # fidout.create_dataset('exchange/data_white',data=data)
    data_dark = fid['exchange/data_dark']
    data_dark = data_dark[:,:,6+3232//2-3200//2:6+3232//2+3200//2].astype('float32')
    for k in range(bin):        
        data_dark = 0.5*(data_dark[:,::2]+data_dark[:,::2])
        data_dark = 0.5*(data_dark[:,:,::2]+data_dark[:,:,::2])
    # fidout.create_dataset('exchange/data_dark',data=data)
    
    data = (data-np.mean(data_dark,axis=0))/(np.mean(data_white,axis=0)-np.mean(data_dark,axis=0))
    data = -np.log(data)
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    
    np.save(f'/data/2023-04-Nikitin/np/brain_lam_20deg_middle_421_bin{bin}.npy',data)
    # import dxchange
    # dxchange.write_tiff_stack(data,'/data/tmp/t')
    #theta = fid['exchange/theta'][::2]
    # fidout.create_dataset('exchange/theta',data=theta)
    
    
    
    