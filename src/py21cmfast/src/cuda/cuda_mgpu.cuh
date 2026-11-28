#ifndef P21_CUDA_MGPU_CUH
#define P21_CUDA_MGPU_CUH

#include <functional>

#include "cuda_common.cuh"

constexpr int kMgpuMaxDevices = 16;

struct P21MgpuGrid {
    int n_gpus = 0;
    int primary_dev = 0;
    int devs[kMgpuMaxDevices]{};
    cudaStream_t streams[kMgpuMaxDevices]{};
    bool peer_enabled[kMgpuMaxDevices][kMgpuMaxDevices]{};

    int nx = 0;
    int ny = 0;
    int nz = 0;
    int hz = 0;

    int x0[kMgpuMaxDevices]{};
    int sx[kMgpuMaxDevices]{};
    int y0[kMgpuMaxDevices]{};
    int sy[kMgpuMaxDevices]{};
    int max_sx = 0;
    int max_sy = 0;

    bool init(int nx_, int ny_, int nz_);
    void destroy(void);

    size_t real_elems_x(int g) const { return (size_t)sx[g] * (size_t)ny * (size_t)nz; }
    size_t cplx_elems_x(int g) const { return (size_t)sx[g] * (size_t)ny * (size_t)hz; }
    size_t cplx_elems_y(int g) const { return (size_t)sy[g] * (size_t)hz * (size_t)nx; }
    size_t total_cplx_elems(void) const { return (size_t)nx * (size_t)ny * (size_t)hz; }
};

int p21_mgpu_effective_ranks(void);
bool p21_mgpu_eor_enabled(size_t total_elems);

void *p21_mgpu_malloc(int dev, size_t nbytes, cudaStream_t stream);
void p21_mgpu_free(int dev, void *ptr, cudaStream_t stream);
size_t p21_mgpu_min_free_mem(const P21MgpuGrid &grid);

bool p21_mgpu_fft_r2c_slabs(const P21MgpuGrid &grid, float *const *d_real_x,
                            cufftComplex **d_cplx_y, void *h_stage_opt);

/* Low-VRAM streaming R2C from host (uses <= 128 MiB temporary device strip per GPU). */
bool p21_mgpu_fft_r2c_from_host(const P21MgpuGrid &grid, const float *h_real_in, double pre_scale,
                                cufftComplex **d_cplx_y, void *h_stage_opt);

/* Stream device-resident (nx, ny, nz) real field on primary_dev into distributed Y-slabs. */
bool p21_mgpu_r2c_from_device(const P21MgpuGrid &grid, const float *d_real_primary,
                              cudaStream_t primary_stream, double pre_scale,
                              cufftComplex **d_cplx_y, void *h_stage_opt);

/* Apply 21cmFAST k-space window filter across distributed Y-slabs (src and dst may alias). */
bool p21_mgpu_filter_yslabs(const P21MgpuGrid &grid, cufftComplex *const *d_src_y,
                            cufftComplex **d_dst_y, double box_len, double non_cubic_factor,
                            int filter_type, float R, float R_param, float R_star, float pre_scale);

/* Note: C2R consumes d_cplx_y as in-place working storage. */
bool p21_mgpu_fft_c2r_slabs(const P21MgpuGrid &grid, cufftComplex **d_cplx_y, float **d_real_x,
                            void *h_stage_opt);

/* Low-VRAM streaming C2R calling on_strip for each real X-strip (consuming d_cplx_y). */
using P21MgpuRealStripFn = std::function<bool(int g, int x_global_start, int cur_nx,
                                            float *d_real_strip, cudaStream_t stream)>;
bool p21_mgpu_fft_c2r_strips(const P21MgpuGrid &grid, cufftComplex **d_cplx_y, void *h_stage_opt,
                             const P21MgpuRealStripFn &on_strip);

/* Stream distributed Y-slab C2R back into a device-resident real grid on primary_dev. */
bool p21_mgpu_c2r_to_device(const P21MgpuGrid &grid, cufftComplex **d_cplx_y, float *d_real_primary,
                            cudaStream_t primary_stream, void *h_stage_opt);

bool p21_mgpu_enforce_real_planes_yslabs(const P21MgpuGrid &grid, cufftComplex **d_cplx_y);

#endif /* P21_CUDA_MGPU_CUH */
