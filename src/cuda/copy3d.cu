#include "copy3d.cuh"

copy3d::copy3d() {

 
}

// destructor, memory deallocation
copy3d::~copy3d() { free(); }

void copy3d::free() {
  if (!is_free) {
    is_free = true;
  }
}

void copy3d:copyh2d(size_t df0, size_t f0, int st, int end, int n0, int n2)
{
  float2* f = (float2*)f0;
  float2* df = (float2*)df0;
	cudaMemcpy3DParms param = { 0 };
	param.srcPtr   = make_cudaPitchedPtr((void*)&f[st*n0].x, n2*sizeof(float2), n2, n0*(end-st));
	param.dstPtr = (void*)df;
	param.kind = cudaMemcpyHostToDevice;
	param.extent = ext;
	return cudaMemcpy3D(&param);
}