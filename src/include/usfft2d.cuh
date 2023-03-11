#ifndef fft_CUH
#define fft_CUH

#include <cufft.h>


class usfft2d {
  bool is_free = false;
  
  
  float2 *f;
  float2 *g;
  
  float2 *fdee2d;
  float* x;
  float* y;
  cufftHandle plan2dchunk;
  
  dim3 BS2d, GS2d0, GS2d1, GS2d2;
  
  size_t n0,n1,n2;  
  size_t ntheta,detw,deth;
  size_t m0,m1;
  float mu0;float mu1;
public:  
  usfft2d(size_t n0, size_t n1, size_t n2, size_t ntheta, size_t detw, size_t deth);  
  ~usfft2d();  
  void fwd(size_t g_, size_t f_, size_t x_, size_t y_);
  void free();
};

#endif
