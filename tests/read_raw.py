import struct
import numpy as np
import dxchange
fid = open('file', 'rb')
tomo = np.float32(struct.unpack(2*256*256*520*'f', fid.read(2*256*256*520*4)))
tomo = (tomo[::2]+1j*tomo[1::2]).reshape(256,520,256)
# tomo=np.fft.fftshift(tomo,axes=(1,))

dxchange.write_tiff(tomo.real,'t',overwrite=True)



fid = open('fileo', 'rb')
tomo2 = np.float32(struct.unpack(2*256*256*520*'f', fid.read(2*256*256*520*4)))
tomo2 = (tomo2[::2]+1j*tomo2[1::2]).reshape(256,520,256)
dxchange.write_tiff(tomo2.real,'t1',overwrite=True)


