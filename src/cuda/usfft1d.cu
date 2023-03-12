#include "kernels_usfft1d.cu"
#include "usfft1d.cuh"
#include <stdio.h>
#define EPS 1e-3

usfft1d::usfft1d(size_t n0_, size_t n1_, size_t n2_, size_t deth_) {

  n0 = n2_; // reorder from python
  n1 = n1_;
  n2 = n0_;
  deth = deth_;

  mu2 = -log(EPS) / (2 * n2 * n2);
  m2 = ceil(2 * n2 * 1 / PI * sqrt(-mu2 * log(EPS) + (mu2 * n2) * (mu2 * n2) / 4));
  int ffts[1];
  int idist;
  int istride;
  int inembed[1];

  // usfft1d 1d
  ffts[0] = 2 * n2;
  idist = 1; //(2*n0 + 2*m) * n2 * n1;
  istride = n1 * n0;
  inembed[0] = n1 * n0;

  cudaMalloc((void **)&fdee1d, n1 * n0 * (2 * n2 + 2 * m2) * sizeof(float2));
  cufftPlanMany(&plan1dchunk, 1, ffts, inembed, istride, idist, inembed, istride, idist, CUFFT_C2C, n1 * n0);
  
  BS1d = dim3(16, 16, 4);
  GS1d0 = dim3(ceil(n0 / (float)BS1d.x), ceil(n1 / (float)BS1d.y), ceil(n2 / (float)BS1d.z));
  GS1d1 = dim3(ceil(n0 / (float)BS1d.x), ceil(n1 / (float)BS1d.y), ceil((2 * n2 + 2 * m2) / (float)BS1d.z));
  GS1d2 = dim3(ceil(n0 / (float)BS1d.x), ceil(n1 / (float)BS1d.y), ceil(deth / (float)BS1d.z));
}

// destructor, memory deallocation
usfft1d::~usfft1d() { free(); }

void usfft1d::free() {
  if (!is_free) {
    cudaFree(fdee1d);
    cufftDestroy(plan1dchunk);
    is_free = true;
  }
}

void usfft1d::fwd(size_t g_, size_t f_, size_t x_) {

  f = (float2 *)f_;
  g = (float2 *)g_;
  x = (float *)x_;
  cudaMemset(fdee1d, 0, n0 * n1 * (2 * n2 + 2 * m2) * sizeof(float2));
  divker1d<<<GS1d0, BS1d>>>(fdee1d, f, n0, n1, n2, m2, mu2);
  fftshiftc1d<<<GS1d1, BS1d>>>(fdee1d, n0, n1, 2 * n2 + 2 * m2);
  cufftExecC2C(plan1dchunk, (cufftComplex *)&fdee1d[m2 * n0 * n1].x,
               (cufftComplex *)&fdee1d[m2 * n0 * n1].x, CUFFT_FORWARD);
  fftshiftc1d<<<GS1d1, BS1d>>>(fdee1d, n0, n1, 2 * n2 + 2 * m2);
  wrap1d<<<GS1d1, BS1d>>>(fdee1d, n0, n1, n2, m2);
  cudaMemcpy(g, fdee1d, n0 * n1 * (2 * n2 + 2 * m2) * 8, cudaMemcpyDefault);
  gather1d<<<GS1d2, BS1d>>>(g, fdee1d, x, m2, mu2, n0, n1, n2, deth);
}
