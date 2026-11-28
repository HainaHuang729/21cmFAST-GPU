#include <algorithm>
#include <list>
#include <mutex>

#include "cuda_common.cuh"
#include "cuda_filters.cuh"

namespace {

struct PlanKey {
    int nx;
    int ny;
    int nz;
    int type;
    bool padded;

    bool operator==(const PlanKey &other) const {
        return nx == other.nx && ny == other.ny && nz == other.nz && type == other.type &&
               padded == other.padded;
    }
};

struct PlanEntry {
    PlanKey key;
    cufftHandle handle;
    size_t work_size = 0;
};

constexpr size_t kMaxCachedPlans = 16;
std::mutex g_fft_mu;
std::list<PlanEntry> g_plan_lru;

bool create_plan_3d(const PlanKey &k, cufftHandle *out_plan, size_t *out_work_size) {
    cufftHandle plan = 0;
    cufftResult res = cufftCreate(&plan);
    if (!P21_CUFFT_OK(res)) {
        return false;
    }
    res = cufftSetAutoAllocation(plan, 0);
    if (!P21_CUFFT_OK(res)) {
        cufftDestroy(plan);
        return false;
    }

    size_t work_size = 0;
    if (!k.padded) {
        res = cufftMakePlan3d(plan, k.nx, k.ny, k.nz, (cufftType)k.type, &work_size);
    } else {
        int n[3] = {k.nx, k.ny, k.nz};
        int r_last = 2 * (k.nz / 2 + 1);
        int c_last = k.nz / 2 + 1;
        int r_embed[3] = {k.nx, k.ny, r_last};
        int c_embed[3] = {k.nx, k.ny, c_last};
        int r_dist = k.nx * k.ny * r_last;
        int c_dist = k.nx * k.ny * c_last;

        int *inembed = (k.type == CUFFT_R2C) ? r_embed : c_embed;
        int *onembed = (k.type == CUFFT_R2C) ? c_embed : r_embed;
        int idist = (k.type == CUFFT_R2C) ? r_dist : c_dist;
        int odist = (k.type == CUFFT_R2C) ? c_dist : r_dist;

        res = cufftMakePlanMany(plan, 3, n, inembed, 1, idist, onembed, 1, odist,
                                (cufftType)k.type, 1, &work_size);
    }
    if (!P21_CUFFT_OK(res)) {
        cufftDestroy(plan);
        return false;
    }
    *out_plan = plan;
    *out_work_size = work_size;
    return true;
}

bool acquire_plan(int nx, int ny, int nz, cufftType type, bool padded, cudaStream_t stream,
                  cufftHandle *out_plan, size_t *out_work_size) {
    PlanKey key{nx, ny, nz, (int)type, padded};

    std::lock_guard<std::mutex> lock(g_fft_mu);
    for (auto it = g_plan_lru.begin(); it != g_plan_lru.end(); ++it) {
        if (it->key == key) {
            g_plan_lru.splice(g_plan_lru.begin(), g_plan_lru, it);
            if (!P21_CUFFT_OK(cufftSetStream(it->handle, stream))) {
                return false;
            }
            p21_cuda_stats_mut()->cufft_plan_hits++;
            *out_plan = it->handle;
            *out_work_size = it->work_size;
            return true;
        }
    }

    cufftHandle plan = 0;
    size_t work_size = 0;
    if (!create_plan_3d(key, &plan, &work_size)) {
        return false;
    }
    if (!P21_CUFFT_OK(cufftSetStream(plan, stream))) {
        cufftDestroy(plan);
        return false;
    }

    p21_cuda_stats_mut()->cufft_plan_misses++;
    while (g_plan_lru.size() >= kMaxCachedPlans) {
        cufftDestroy(g_plan_lru.back().handle);
        g_plan_lru.pop_back();
        p21_cuda_stats_mut()->cufft_plan_evictions++;
    }
    g_plan_lru.push_front(PlanEntry{key, plan, work_size});
    *out_plan = plan;
    *out_work_size = work_size;
    return true;
}

__global__ void copy_filter_box_kernel(const cufftComplex *src, cufftComplex *dst, int nx, int ny,
                                       int nz, double dk_x, double dk_y, double dk_z,
                                       int filter_type, float R, float R_param, double exp_term,
                                       double alpha_in, double beta_in, double alpha_out,
                                       double beta_out, float pre_scale) {
    int hz = nz / 2 + 1;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t total = (size_t)nx * ny * hz;

    for (size_t idx = tid; idx < total; idx += stride) {
        int z = (int)(idx % hz);
        size_t q = idx / hz;
        int y = (int)(q % ny);
        int x = (int)(q / ny);

        float kx = (x > nx / 2) ? (float)((x - nx) * dk_x) : (float)(x * dk_x);
        float ky = (y > ny / 2) ? (float)((y - ny) * dk_y) : (float)(y * dk_y);
        float kz = (float)(z * dk_z);
        float k_mag_sq = kx * kx + ky * ky + kz * kz;

        double w = p21_eval_filter_weight(filter_type, k_mag_sq, R, R_param, exp_term, alpha_in,
                                          beta_in, alpha_out, beta_out);

        cufftComplex v = src[idx];
        if (pre_scale != 1.0f) {
            v.x *= pre_scale;
            v.y *= pre_scale;
        }
        v.x = (float)((double)v.x * w);
        v.y = (float)((double)v.y * w);
        dst[idx] = v;
    }
}

}  // namespace

bool p21_cuda_fft_r2c(const float *d_real, cufftComplex *d_complex, int nx, int ny, int nz,
                      bool padded, cudaStream_t stream) {
    cufftHandle plan = 0;
    size_t work_size = 0;
    if (!acquire_plan(nx, ny, nz, CUFFT_R2C, padded, stream, &plan, &work_size)) {
        return false;
    }
    void *d_work = (work_size > 0) ? p21_cuda_pool_alloc(work_size, stream) : nullptr;
    if (work_size > 0 && !d_work) {
        return false;
    }
    bool ok = P21_CUFFT_OK(cufftSetWorkArea(plan, d_work)) &&
              P21_CUFFT_OK(cufftSetStream(plan, stream)) &&
              P21_CUFFT_OK(cufftExecR2C(plan, (cufftReal *)const_cast<float *>(d_real), d_complex));
    p21_cuda_pool_free(d_work, stream);
    if (!ok) {
        return false;
    }
    p21_cuda_stats_mut()->cufft_exec_r2c++;
    return true;
}

bool p21_cuda_fft_c2r(cufftComplex *d_complex_scratch, float *d_real, int nx, int ny, int nz,
                      bool padded, cudaStream_t stream) {
    cufftHandle plan = 0;
    size_t work_size = 0;
    if (!acquire_plan(nx, ny, nz, CUFFT_C2R, padded, stream, &plan, &work_size)) {
        return false;
    }
    void *d_work = (work_size > 0) ? p21_cuda_pool_alloc(work_size, stream) : nullptr;
    if (work_size > 0 && !d_work) {
        return false;
    }
    bool ok = P21_CUFFT_OK(cufftSetWorkArea(plan, d_work)) &&
              P21_CUFFT_OK(cufftSetStream(plan, stream)) &&
              P21_CUFFT_OK(cufftExecC2R(plan, d_complex_scratch, (cufftReal *)d_real));
    p21_cuda_pool_free(d_work, stream);
    if (!ok) {
        return false;
    }
    p21_cuda_stats_mut()->cufft_exec_c2r++;
    return true;
}

void p21_cuda_fft_clear_cache(void) {
    std::lock_guard<std::mutex> lock(g_fft_mu);
    for (auto &entry : g_plan_lru) {
        cufftDestroy(entry.handle);
    }
    g_plan_lru.clear();
}

bool p21_cuda_copy_and_filter_box(const cufftComplex *d_src, cufftComplex *d_dst, int nx, int ny,
                                  int nz, double box_len, double non_cubic_factor, int filter_type,
                                  float R, float R_param, float R_star, float pre_scale,
                                  cudaStream_t stream) {
    if (!d_src || !d_dst) {
        return false;
    }
    if (filter_type != FILTER_TOPHAT && filter_type != FILTER_SHARP_K &&
        filter_type != FILTER_GAUSSIAN && filter_type != FILTER_EXP_MFP &&
        filter_type != FILTER_SPHERICAL_SHELL_STRAIGHT_LINE &&
        filter_type != FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING) {
        return false;
    }

    double dk_x = 2.0 * M_PI / box_len;
    double dk_y = 2.0 * M_PI / box_len;
    double dk_z = 2.0 * M_PI / (box_len * non_cubic_factor);

    double exp_term = 0.0;
    double ab[4] = {0.0, 0.0, 0.0, 0.0};
    if (filter_type == FILTER_EXP_MFP) {
        exp_term = exp(-(double)R / (double)R_param);
    } else if (filter_type == FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING) {
        p21_get_ms_filter_params((double)R, (double)R_param, (double)R_star, ab);
    }

    size_t total = (size_t)nx * ny * (nz / 2 + 1);
    const int threads = 256;
    int blocks = (int)std::min((total + threads - 1) / threads, (size_t)65535);
    copy_filter_box_kernel<<<blocks, threads, 0, stream>>>(
        d_src, d_dst, nx, ny, nz, dk_x, dk_y, dk_z, filter_type, R, R_param, exp_term, ab[0], ab[1],
        ab[2], ab[3], pre_scale);
    return P21_CUDA_OK(cudaGetLastError());
}

bool p21_cuda_filter_box(cufftComplex *d_box, int nx, int ny, int nz, double box_len,
                         double non_cubic_factor, int filter_type, float R, float R_param,
                         float R_star, cudaStream_t stream) {
    return p21_cuda_copy_and_filter_box(d_box, d_box, nx, ny, nz, box_len, non_cubic_factor,
                                        filter_type, R, R_param, R_star, 1.0f, stream);
}
