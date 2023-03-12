#include "kernels_usfft2d.cu"
#include "usfft2d.cuh"
#define EPS 1e-3

usfft2d::usfft2d(size_t n0_, size_t n1_, size_t n2_, size_t ntheta_, size_t detw_, size_t deth_) {

  n0 = n2_; // reorder from python
  n1 = n1_;
  n2 = n0_;
  ntheta = ntheta_;
  detw = detw_;
  deth = deth_;

  mu0 = -log(EPS) / (2 * n0 * n0);
  mu1 = -log(EPS) / (2 * n1 * n1);
  m0 = ceil(2 * n0 * 1 / PI * sqrt(-mu0 * log(EPS) + (mu0 * n0) * (mu0 * n0) / 4));
  m1 = ceil(2 * n1 * 1 / PI * sqrt(-mu1 * log(EPS) + (mu1 * n1) * (mu1 * n1) / 4));

  int ffts[2];
  int idist;
  int inembed[2];

  // usfft2d 2d
  ffts[0] = 2 * n1;
  ffts[1] = 2 * n0;
  idist = (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1);
  inembed[0] = (2 * n1 + 2 * m1);
  inembed[1] = (2 * n0 + 2 * m0);

  cudaMalloc((void **)&fdee2d, n2 * (2 * n1 + 2 * m1) * (2 * n0 + 2 * m0) * sizeof(float2));
  cufftPlanMany(&plan2dchunk, 2, ffts, inembed, 1, idist, inembed, 1, idist, CUFFT_C2C, n2);

  BS2d = dim3(32, 32, 1);
  GS2d0 = dim3(ceil(n0 / (float)BS2d.x), ceil(n1 / (float)BS2d.y), ceil(n2 / (float)BS2d.z));
  GS2d1 = dim3(ceil((2 * n0 + 2 * m0) / (float)BS2d.x), ceil((2 * n1 + 2 * m1) / (float)BS2d.y), ceil(n2 / (float)BS2d.z));
  GS2d2 = dim3(ceil(detw / (float)BS2d.x), ceil(deth / (float)BS2d.y), ceil(ntheta / (float)BS2d.z));
}

// destructor, memory deallocation
usfft2d::~usfft2d() { free(); }

void usfft2d::free() {
  if (!is_free) {
    cudaFree(fdee2d);
    cufftDestroy(plan2dchunk);
    is_free = true;
  }
}

void usfft2d::fwd(size_t g_, size_t f_, size_t x_, size_t y_) {

  f = (float2 *)f_;
  g = (float2 *)g_;
  x = (float *)x_;
  y = (float *)y_;
  cudaMemset(fdee2d, 0, n2 * (2 * n1 + 2 * m1) * (2 * n0 + 2 * m0) * sizeof(float2));
  divker2d<<<GS2d0, BS2d>>>(fdee2d, f, n0, n1, n2, m0, m1, mu0, mu1);
  fftshiftc2d<<<GS2d1, BS2d>>>(fdee2d, (2 * n0 + 2 * m0), (2 * n1 + 2 * m1), n2);
  cufftExecC2C(plan2dchunk, (cufftComplex *)&fdee2d[m0 + m1 * (2 * n0 + 2 * m0)].x, (cufftComplex *)&fdee2d[m0 + m1 * (2 * n0 + 2 * m0)].x, CUFFT_FORWARD);
  fftshiftc2d<<<GS2d1, BS2d>>>(fdee2d, (2 * n0 + 2 * m0), (2 * n1 + 2 * m1), n2);
  wrap2d<<<GS2d1, BS2d>>>(fdee2d, n0, n1, n2, m0, m1);
  gather2d<<<GS2d2, BS2d>>>(g, fdee2d, x, y, m0, m1, mu0, mu1, n0, n1, n2, detw, deth, ntheta);
  // cudaMemcpy(g,fdee2d,n2 * (2 * n1 + 2 * m1)*(2 * n0 + 2 *
  // m0)*8,cudaMemcpyDefault)
}
