#include <stdio.h>

#include "fft.cuh"
#include "shift.cu"

fft::fft(size_t n0, size_t n1, size_t n2, size_t m)
: n0(n0), n1(n1), n2(n2), m(m) {
  
  int ffts[2];
  int idist;
  int istride;
  int inembed[2];
  
  // fft 1d
  ffts[0] = 2*n0;
  idist = 1;//(2*n2 + 2*m) * n0 * n1;
  istride = n1*n2;
  inembed[0] = n1*n2; 

  cudaMalloc((void **)&fdee1d, n1 * n2*(2 * n0 + 2 * m) * sizeof(float2));
  cufftPlanMany(&plan1dchunk, 1, ffts, inembed, istride, idist, inembed, istride, idist,CUFFT_C2C,n1*n2);
  BS1d.x = 16;
  BS1d.y = 16;
  BS1d.z = 4;
  GS1d.x = ceil(n2/16.);
  GS1d.y = ceil(n1/16.);
  GS1d.z = ceil((2*n0+2*m)/4.);


  // fft 2d
  ffts[0] = 2*n1;
  ffts[1] = 2*n2;
  idist = (2*n2 + 2*m) * (2*n1 + 2*m);
  inembed[0] = (2*n1 + 2*m);
  inembed[1] = (2*n2 + 2*m);

  cudaMalloc((void **)&fdee2d, n0 * (2 * n1 + 2 * m)*(2 * n2 + 2 * m) * sizeof(float2));
  cufftPlanMany(&plan2dchunk, 2, ffts, inembed, 1, idist, inembed, 1, idist,CUFFT_C2C,n0);
  BS2d.x = 16;
  BS2d.y = 16;
  BS2d.z = 4;
  GS2d.x = ceil((2 * n2 + 2 * m)/16.);
  GS2d.y = ceil((2 * n1 + 2 * m)/16.);
  GS2d.z = ceil(n0/4.);


  BS3d.x = 16;
  BS3d.y = 16;
  BS3d.z = 4;
  GS3d.x = ceil(n2/16.);
  GS3d.y = ceil(n1/16.);
  GS3d.z = ceil(n0/4.);
}

// destructor, memory deallocation
fft::~fft() { free(); }

void fft::free() {
  if (!is_free) {
    cudaFree(fdee1d);
    cudaFree(fdee2d);
    cufftDestroy(plan1dchunk);
    cufftDestroy(plan2dchunk);
    is_free = true;
  }
}

void fft::fwd1d(size_t g_, size_t f_) {
  
  f = (float2*)f_;
  g = (float2*)g_;
  cudaMemset(fdee1d, 0, n2 * n1 * (2 * n0 + 2 * m) * sizeof(float2));  
  cudaMemcpy(&fdee1d[n2*n1*(n0/2+m)],f, n0*n1*n2*8,cudaMemcpyDefault);
  fftshiftc1d <<<GS1d, BS1d>>> (fdee1d, n2, n1,  2*n0+2*m);  
  cufftExecC2C(plan1dchunk, (cufftComplex *)&fdee1d[m * n2*n1].x,(cufftComplex *)&fdee1d[m * n2*n1].x, CUFFT_FORWARD);  
  fftshiftc1d <<<GS1d, BS1d>>> (fdee1d, n2, n1,  2*n0+2*m);  
  cudaMemcpy(g,fdee1d, n1*n2*(2*n0+2*m)*8,cudaMemcpyDefault);
}

void fft::fwd2d(size_t g_, size_t f_) {
  
  f = (float2*)f_;
  g = (float2*)g_;
  cudaMemset(fdee2d, 0, n0 * (2 * n1 + 2 * m)*(2 * n2 + 2 * m) * sizeof(float2));  
  // cudaMemcpy(&fdee2d[m+n2/2+(m+n1/2)*(2 * n2 + 2 * m)],f, n0*n1*n2*8,cudaMemcpyDefault);
  setfdee2d<<<GS3d, BS3d>>>(fdee2d,f,n2,n1,n0,m);
  fftshiftc2d <<<GS2d, BS2d>>> (fdee2d, (2 * n2 + 2 * m), (2 * n1 + 2 * m),  n0);  
  cufftExecC2C(plan2dchunk, (cufftComplex *)&fdee2d[m+m*(2 * n2 + 2 * m)].x,(cufftComplex *)&fdee2d[m+m*(2 * n2 + 2 * m)].x, CUFFT_FORWARD);  
  fftshiftc2d <<<GS2d, BS2d>>> (fdee2d, (2 * n2 + 2 * m), (2 * n1 + 2 * m),  n0);  
  cudaMemcpy(g,fdee2d, n0 * (2 * n1 + 2 * m)*(2 * n2 + 2 * m)*8,cudaMemcpyDefault);
}