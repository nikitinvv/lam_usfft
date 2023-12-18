#include "kernels_fft2d.cu"
#include "fft2d.cuh"

fft2d::fft2d(size_t ntheta_, size_t detw_, size_t deth_) {

  ntheta = ntheta_;
  detw = detw_;
  deth = deth_;

  long long ffts[]={deth,detw};
  long long idist = deth*detw;
  long long odist = deth*(detw/2+1);
  long long inembed[] = {deth,detw};
  long long onembed[] = {deth,detw/2+1};

//  cufftPlanMany(&plan2dchunk, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, ntheta);
  size_t workSize = 0;
  cufftCreate(&plan2dchunk_fwd);
  cufftCreate(&plan2dchunk_inv);
  cufftXtMakePlanMany(plan2dchunk_fwd, 
      2, ffts, 
      inembed, 1, idist, CUDA_R_32F, 
      onembed, 1, odist, CUDA_C_32F, 
      ntheta, &workSize, CUDA_C_32F);  
    
  cufftXtMakePlanMany(plan2dchunk_inv, 
        2, ffts, 
        onembed, 1, odist, CUDA_C_32F, 
        inembed, 1, idist, CUDA_R_32F, 
        ntheta, &workSize, CUDA_C_32F);  

  BS2d = dim3(32, 32, 1);
  GS2d0 = dim3(ceil(detw / (float)BS2d.x), ceil(deth / (float)BS2d.y), ceil(ntheta / (float)BS2d.z));
}

// destructor, memory deallocation
fft2d::~fft2d() { free(); }

void fft2d::free() {
  if (!is_free) {
    cufftDestroy(plan2dchunk_fwd);
    cufftDestroy(plan2dchunk_inv);
    is_free = true;
  }
}

void fft2d::fwd(size_t g_, size_t f_, size_t stream_) {

  f = (float *)f_;
  g = (float2 *)g_;
  stream = (cudaStream_t)stream_;    
  
  cufftSetStream(plan2dchunk_fwd, stream);
  rfftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(f, detw, deth, ntheta);
  cufftXtExec(plan2dchunk_fwd, f,g,CUFFT_FORWARD);        
  irfftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(g, detw/2+1, deth, ntheta);  
  mulc<<<GS2d0, BS2d, 0, stream>>>(g, detw/2+1, deth, ntheta, 1.f/(deth*detw));  
}

void fft2d::adj(size_t f_, size_t g_, size_t stream_) {

  f = (float *)f_;
  g = (float2 *)g_;
  stream = (cudaStream_t)stream_;    
  
  cufftSetStream(plan2dchunk_inv, stream);  
  mulc<<<GS2d0, BS2d, 0, stream>>>(g, detw/2+1, deth, ntheta, 1.f/(deth*detw));  
  irfftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(g, detw/2+1, deth, ntheta);  
  cufftXtExec(plan2dchunk_inv, g,f,CUFFT_INVERSE);    
  rfftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(f, detw, deth, ntheta);  
}

