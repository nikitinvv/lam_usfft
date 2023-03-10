#ifndef fft_CUH
#define fft_CUH

#include <cufft.h>


class fft {
  bool is_free = false;
  
  
  float2 *f;
  float2 *g;
  
  float2 *fdee1d;
  float2 *fdee2d;

  cufftHandle plan1dchunk;
  cufftHandle plan2dchunk;
  
  dim3 BS1d, GS1d;
  dim3 BS2d, GS2d;
  dim3 BS3d, GS3d;

public:
  size_t n0,n1,n2,m;  
  
  fft(size_t n0, size_t n1, size_t n2, size_t m);
  ~fft();  
  void fwd1d(size_t g_, size_t f_);
  void fwd2d(size_t g_, size_t f_);
  void free();
};

#endif
