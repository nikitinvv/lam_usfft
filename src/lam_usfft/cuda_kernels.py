import cupy as cp


_PI = "#define PI 3.1415926535897932384626433\n"


# ---------------------------------------------------------------------------
# usfft1d kernels (gather z-coords are now precomputed in usfft1d.__init__,
# so the device-side take_x_1d kernel has been removed)
# ---------------------------------------------------------------------------

divker1d_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void divker1d(float2 *g, float2 *f, int n0, int n1, int n2, float mu2, bool direction) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n0 || ty >= n1 || tz >= n2) return;
    float ker = __expf(-mu2 * (tz - n2 / 2) * (tz - n2 / 2));
    int f_ind = tx + tz * n0 + ty * n0 * n2;
    // g lives in a (2*n2, n1, n0) buffer with the real signal placed in rows
    // [n2/2, n2/2 + n2); no extra m2-offset since wrap-around is handled at
    // gather time (modulo 2*n2 inside gather1d).
    int g_ind = tx + ty * n0 + (tz + n2 / 2) * n0 * n1;

    if (direction == 0) {
        g[g_ind].x = f[f_ind].x / ker / (2 * n2);
        g[g_ind].y = f[f_ind].y / ker / (2 * n2);
    } else {
        f[f_ind].x = g[g_ind].x / ker / (2 * n2);
        f[f_ind].y = g[g_ind].y / ker / (2 * n2);
    }
}
""",
    "divker1d",
)


fftshiftc1d_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void fftshiftc1d(float2 *f, int n0, int n1, int n2) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n0 || ty >= n1 || tz >= n2) return;
    int g = (1 - 2 * ((tz + 1) % 2));
    int f_ind = tx + ty * n0 + tz * n0 * n1;
    f[f_ind].x *= g;
    f[f_ind].y *= g;
}
""",
    "fftshiftc1d",
)


gather1d_kernel = cp.RawKernel(
    _PI
    + r"""
extern "C" __global__
void gather1d(float2 *g, float2 *f, float *z, int m2, float mu2,
              int n0, int n1, int n2, int deth, bool direction) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n0 || ty >= n1 || tz >= deth) return;

    const int twon2 = 2 * n2;
    float2 g0;
    float z0 = z[tz];
    int g_ind = tx + tz * n0 + ty * n0 * deth;

    if (direction == 0) {
        g0.x = 0.0f;
        g0.y = 0.0f;
    } else {
        g0.x = g[g_ind].x;
        g0.y = g[g_ind].y;
    }

    for (int i2 = 0; i2 < 2 * m2 + 1; i2++) {
        int ell2 = floorf(twon2 * z0) - m2 + i2;
        float w2 = ell2 / (float)twon2 - z0;
        float w = sqrtf(PI / (mu2*n0)) * __expf(-PI * PI / mu2 * (w2 * w2));
        // wrap (n2 + ell2) modulo 2*n2 → directly addresses the unpadded
        // (2*n2, n1, n0) buffer; replaces the separate wrap1d kernel.
        int f_indz = (n2 + ell2 + twon2) % twon2;
        int f_ind  = tx + ty * n0 + f_indz * n0 * n1;

        if (direction == 0) {
            g0.x += w * f[f_ind].x;
            g0.y += w * f[f_ind].y;
        } else {
            atomicAdd(&(f[f_ind].x), w * g0.x);
            atomicAdd(&(f[f_ind].y), w * g0.y);
        }
    }

    if (direction == 0) {
        g[g_ind].x = g0.x;
        g[g_ind].y = g0.y;
    }
}
""",
    "gather1d",
)


# ---------------------------------------------------------------------------
# usfft2d kernels (gather coordinates x, y are now computed on the Rec side,
# see Rec._make_geometry; the device-side take_x_2d kernel has been removed)
# ---------------------------------------------------------------------------

divker2d_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void divker2d(float2 *g, float2 *f, int n0, int n1, int n2,
              float mu0, float mu1, bool direction) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n0 || ty >= n1 || tz >= n2) return;
    float ker = __expf(-mu0 * (tx - n0 / 2) * (tx - n0 / 2) -
                       mu1 * (ty - n1 / 2) * (ty - n1 / 2));
    int f_ind = tx + tz * n0 + ty * n0 * n2;
    // g lives in a (n2, 2*n1, 2*n0) buffer; real signal placed at
    // [n1/2:n1/2+n1, n0/2:n0/2+n0]. No m0/m1 offset — wrap-around handled
    // inside gather2d via modulo (2*n0, 2*n1).
    int g_ind = (tx + n0 / 2) + (ty + n1 / 2) * (2 * n0) +
                tz * (2 * n0) * (2 * n1);

    if (direction == 0) {
        g[g_ind].x = f[f_ind].x / ker / (4 * n0 * n1);
        g[g_ind].y = f[f_ind].y / ker / (4 * n0 * n1);
    } else {
        f[f_ind].x = g[g_ind].x / ker / (4 * n0 * n1);
        f[f_ind].y = g[g_ind].y / ker / (4 * n0 * n1);
    }
}
""",
    "divker2d",
)


fftshiftc2d_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void fftshiftc2d(float2 *f, int n0, int n1, int n2) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n0 || ty >= n1 || tz >= n2) return;
    int g = (1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2));
    f[tx + ty * n0 + tz * n0 * n1].x *= g;
    f[tx + ty * n0 + tz * n0 * n1].y *= g;
}
""",
    "fftshiftc2d",
)


gather2d_kernel = cp.RawKernel(
    _PI
    + r"""
extern "C" __global__
void gather2d(float2 *g, float2 *f, float *x, float *y, int m0,
              int m1, float mu0, float mu1, int n0, int n1, int n2,
              int detw, int deth, int ntheta, bool direction) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= detw || ty >= deth || tz >= ntheta) return;

    const int twon0 = 2 * n0;
    const int twon1 = 2 * n1;
    const int slab_stride = twon0 * twon1;
    int g_ind = tx + ty * detw + tz * detw * deth;

    float x0 = x[g_ind];
    float y0 = y[g_ind];

    float2 g0;
    if (direction == 0) {
        g0.x = 0.0f;
        g0.y = 0.0f;
    } else {
        g0.x = g[g_ind].x;
        g0.y = g[g_ind].y;
    }
    for (int i1 = 0; i1 < 2 * m1 + 1; i1++) {
        int ell1 = floorf(twon1 * y0) - m1 + i1;
        int f_indy = (n1 + ell1 + twon1) % twon1;     // wrap → (0, 2*n1)
        for (int i0 = 0; i0 < 2 * m0 + 1; i0++) {
            int ell0 = floorf(twon0 * x0) - m0 + i0;
            int f_indx = (n0 + ell0 + twon0) % twon0; // wrap → (0, 2*n0)
            float w0 = ell0 / (float)twon0 - x0;
            float w1 = ell1 / (float)twon1 - y0;
            float w = PI / sqrtf(mu0 * mu1 * ntheta) *
                      __expf(-PI * PI / mu0 * (w0 * w0) - PI * PI / mu1 * (w1 * w1));
            // Directly addresses the unpadded (n2, 2*n1, 2*n0) buffer;
            // replaces the separate wrap2d kernel.
            int f_ind = f_indx + twon0 * f_indy + ty * slab_stride;
            if (direction == 0) {
                g0.x += w * f[f_ind].x;
                g0.y += w * f[f_ind].y;
            } else {
                atomicAdd(&(f[f_ind].x), w * g0.x);
                atomicAdd(&(f[f_ind].y), w * g0.y);
            }
        }
    }
    if (direction == 0) {
        g[g_ind].x = g0.x;
        g[g_ind].y = g0.y;
    }
}
""",
    "gather2d",
)


# ---------------------------------------------------------------------------
# fft2d kernels (shared fftshiftc2d above; mulc here)
# ---------------------------------------------------------------------------

mulc_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void mulc(float2 *f, int n0, int n1, int n2, float c) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n0 || ty >= n1 || tz >= n2) return;

    f[tx + ty * n0 + tz * n0 * n1].x *= c;
    f[tx + ty * n0 + tz * n0 * n1].y *= c;
}
""",
    "mulc",
)
