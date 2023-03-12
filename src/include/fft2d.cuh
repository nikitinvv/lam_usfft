#ifndef fft_CUH
#define fft_CUH

#include <cufft.h>


class fft2d {
  bool is_free = false;
  
  float2 *f;
  cufftHandle plan2dchunk;
  
  dim3 BS2d, GS2d0;
  
  size_t ntheta,detw,deth;
  
public:  
  fft2d(size_t ntheta, size_t detw, size_t deth);  
  ~fft2d();  
  void adj(size_t f_);
  void free();
};

#endif
