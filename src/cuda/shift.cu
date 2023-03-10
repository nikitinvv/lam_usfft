void __global__ fftshiftc1d(float2 *f, int n0, int n1, int n2) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  int g = (1 - 2 * ((tz + 1) % 2));
  int f_ind = tx + ty * n0  + tz * n0 * n1;
  f[f_ind].x *= g;
  f[f_ind].y *= g;
}

void __global__ fftshiftc2d(float2 *f, int detw, int deth, int ntheta) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= detw || ty >= deth || tz >= ntheta)
    return;
  int g = (1 - 2 * ((tx + 1) % 2))*(1 - 2 * ((ty + 1) % 2));
  int f_ind = tx + ty * detw  + tz * detw * deth;
  f[f_ind].x *= g;
  f[f_ind].y *= g;
}

void __global__ fftshiftc3d(float2 *f, int n0, int n1, int n2) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  int g = (1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2))* (1 - 2 * ((tz + 1) % 2));
  f[tx + ty * n0 + tz * n0 * n1].x *= g;
  f[tx + ty * n0 + tz * n0 * n1].y *= g; 
}

void __global__ setfdee2d(float2 *g, float2 *f, int n0, int n1, int n2, int m)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  
  int f_ind = tx + ty * n0 + tz * n0 * n1;
  int g_ind = (tx + n0 / 2 + m) + (ty + n1 / 2 + m) * (2 * n0 + 2 * m) + tz * (2 * n1 + 2 * m) * (2 * n0 + 2 * m); 

  g[g_ind].x = f[f_ind].x;
  g[g_ind].y = f[f_ind].y;
}
