#include <algorithm>
#include <math.h>
#include <mutex>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "cuda_common.cuh"
#include "cuda_mgpu.cuh"
#include "cuda_tables.cuh"

struct P21CudaIonCtx {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    size_t total = 0;
    size_t k_total = 0;
    int n_radii = 0;
    int use_ts_fluct = 0;
    int use_mcgs = 0;
    int use_reion_fb = 0;
    int vcb_flucts = 0;
    int recomb_model = 0;
    int cell_recomb = 0;
    int minimize_memory = 0;
    int ionise_entire_sphere = 0;
    int use_exp_filter = 0;
    int prev_first_z = 0;
    bool spill_kspace = false;

    /* Borrowed / owned inputs */
    const float *d_dens = nullptr;
    const float *d_prev_dens = nullptr;
    float *d_prev_dens_owned = nullptr;
    const float *d_vcb = nullptr;
    const float *d_ts_xe = nullptr;
    const float *d_ts_tk = nullptr;
    const float *d_ts_j21 = nullptr;
    const float *d_prev_zre = nullptr;
    float *d_prev_zre_owned = nullptr;
    const float *d_prev_gamma12 = nullptr;
    float *d_prev_gamma12_owned = nullptr;
    const float *d_prev_nrec = nullptr;
    float *d_prev_nrec_owned = nullptr;
    const float *d_prev_nion_acg = nullptr;
    const float *d_prev_nion_mcg = nullptr;
    const float *d_stars_in = nullptr;
    const float *d_sfr_in = nullptr;

    /* Unfiltered k-space grids (GPU or spilled host) */
    cufftComplex *d_k_deltax_unfilt = nullptr;
    cufftComplex *d_k_xe_unfilt = nullptr;
    cufftComplex *d_k_nrec_unfilt = nullptr;
    cufftComplex *d_k_mturn_acg_unfilt = nullptr;
    cufftComplex *d_k_prev_deltax_unfilt = nullptr;
    cufftComplex *d_k_mturn_mcg_unfilt = nullptr;
    cufftComplex *d_k_stars_unfilt = nullptr;
    cufftComplex *d_k_sfr_unfilt = nullptr;
    cufftComplex *h_k_deltax_unfilt = nullptr;
    cufftComplex *h_k_xe_unfilt = nullptr;
    cufftComplex *h_k_nrec_unfilt = nullptr;
    cufftComplex *h_k_mturn_acg_unfilt = nullptr;
    cufftComplex *h_k_prev_deltax_unfilt = nullptr;
    cufftComplex *h_k_mturn_mcg_unfilt = nullptr;
    cufftComplex *h_k_stars_unfilt = nullptr;
    cufftComplex *h_k_sfr_unfilt = nullptr;
    cufftComplex *d_k_work = nullptr;

    /* Distributed multi-GPU Y-slab state (Direction P EoR multi-GPU) */
    bool mgpu_mode = false;
    P21MgpuGrid mgpu_grid{};
    cufftComplex *y_deltax_unfilt[kMgpuMaxDevices]{};
    cufftComplex *y_xe_unfilt[kMgpuMaxDevices]{};
    cufftComplex *y_nrec_unfilt[kMgpuMaxDevices]{};
    cufftComplex *y_mturn_acg_unfilt[kMgpuMaxDevices]{};
    cufftComplex *y_prev_deltax_unfilt[kMgpuMaxDevices]{};
    cufftComplex *y_mturn_mcg_unfilt[kMgpuMaxDevices]{};
    cufftComplex *y_stars_unfilt[kMgpuMaxDevices]{};
    cufftComplex *y_sfr_unfilt[kMgpuMaxDevices]{};
    cufftComplex *y_work[kMgpuMaxDevices]{};

    /* Real-space work / filtered grids */
    float *d_real_work = nullptr;
    float *d_deltax_filt = nullptr;
    float *d_xe_filt = nullptr;
    float *d_nrec_filt = nullptr;
    float *d_mturn_acg_real = nullptr; /* unfiltered before R2C, filtered inside R loop */
    float *d_prev_deltax_filt = nullptr;
    float *d_mturn_mcg_real = nullptr; /* unfiltered before R2C, filtered inside R loop */
    float *d_stars_filt = nullptr;
    float *d_sfr_filt = nullptr;

    /* EDT scratch for IONISE_ENTIRE_SPHERE */
    uint8_t *d_sphere_mask = nullptr;
    int *d_edt_a = nullptr;
    int *d_edt_b = nullptr;
    int *d_n_centers = nullptr;

    /* Reduction scratch */
    double *d_blk0 = nullptr;
    double *d_blk1 = nullptr;
    double *d_scalar2 = nullptr;
    int *d_nonfinite = nullptr;
    int reduce_blocks = 0;

    /* Persistent output device buffers */
    float *d_z_reion = nullptr;
    float *d_neutral_fraction = nullptr;
    float *d_ionisation_rate_G12 = nullptr;
    float *d_mean_free_path = nullptr;
    float *d_kinetic_temperature = nullptr;
    float *d_cumulative_recombinations = nullptr;
    float *d_nion_cond_acg = nullptr;
    float *d_nion_cond_mcg = nullptr;
    size_t nion_cond_bytes = 0;

    cudaStream_t stream = nullptr;
};

namespace {

constexpr int kBlockThreads = 256;
constexpr int kEdtInf = 0x1fffffff;

std::mutex g_mhr_mu;
double *g_d_mhr_y = nullptr;
double *g_d_mhr_c = nullptr;
int g_mhr_nz = 0;
int g_mhr_ng = 0;
double g_mhr_del_z = 0.2;
double g_mhr_lng_min = -10.0;
double g_mhr_del_lng = 0.1;
int g_mhr_gen = -1;

__global__ void fill_float_kernel(float *arr, size_t n, float val) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = val;
    }
}

__global__ void clip_and_scale_kernel(const float *src, float *dst, size_t n, double scale,
                                      double lo, double hi) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double v = (double)src[idx] * scale;
        dst[idx] = (float)fmax(fmin(v, hi), lo);
    }
}

__global__ void scale_complex_kernel(cufftComplex *arr, size_t n, float inv_n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        cufftComplex v = arr[idx];
        v.x *= inv_n;
        v.y *= inv_n;
        arr[idx] = v;
    }
}

__global__ void ion_mcrit_kernel(size_t n, P21MturnParams mp, float vcb_const, int below_zheatmax,
                                 int vcb_flucts, const float *lowres_vcb, const float *j21_lw,
                                 const float *gamma12, const float *z_reion, float *mturn_acg_out,
                                 float *mturn_mcg_out, double *blk_acg, double *blk_mcg) {
    __shared__ double s_acg[kBlockThreads];
    __shared__ double s_mcg[kBlockThreads];

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double sum_acg = 0.0;
    double sum_mcg = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        float j21 = 0.0f, g12 = 0.0f, zre = 0.0f;
        float curr_vcb = vcb_const;
        if (vcb_flucts && mp.use_mcgs) {
            curr_vcb = lowres_vcb[i];
        }
        if (below_zheatmax) {
            if (mp.use_mcgs) j21 = j21_lw[i];
            if (mp.use_reion_feedback) {
                g12 = (gamma12 != nullptr) ? gamma12[i] : 0.0f;
                zre = (z_reion != nullptr) ? z_reion[i] : -1.0f;
            }
        }
        float mt_acg = 0.0f, mt_mcg = 0.0f;
        p21_compute_mturns_inhomogeneous(mp, j21, curr_vcb, g12, zre, &mt_acg, &mt_mcg);
        if (mp.use_reion_feedback) {
            double l10 = log10((double)mt_acg);
            mturn_acg_out[i] = (float)l10;
            sum_acg += l10;
        }
        if (mp.use_mcgs) {
            double l10 = log10((double)mt_mcg);
            mturn_mcg_out[i] = (float)l10;
            sum_mcg += l10;
        }
    }

    s_acg[tid] = sum_acg;
    s_mcg[tid] = sum_mcg;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            s_acg[tid] += s_acg[tid + s];
            s_mcg[tid] += s_mcg[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_acg[blockIdx.x] = s_acg[0];
        blk_mcg[blockIdx.x] = s_mcg[0];
    }
}

__global__ void finish_pair_sum_kernel(const double *b0, const double *b1, int nblocks,
                                         double *out2) {
    __shared__ double s0[kBlockThreads];
    __shared__ double s1[kBlockThreads];
    size_t tid = threadIdx.x;
    double a0 = 0.0, a1 = 0.0;
    for (int i = (int)tid; i < nblocks; i += kBlockThreads) {
        a0 += b0[i];
        a1 += b1[i];
    }
    s0[tid] = a0;
    s1[tid] = a1;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            s0[tid] += s0[tid + s];
            s1[tid] += s1[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out2[0] = s0[0];
        out2[1] = s1[0];
    }
}

__global__ void set_fully_neutral_kernel(size_t n, int use_ts_fluct, int minimize_memory,
                                         float global_xh_nofluct, double tk_nofluct,
                                         double adia_tk_term, const float *dens, const float *ts_xe,
                                         const float *ts_tk, float *neutral_out, float *tk_out,
                                         double *blk_xh) {
    __shared__ double sdata[kBlockThreads];
    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double sum_xh = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        float xh;
        if (use_ts_fluct) {
            xh = 1.0f - ts_xe[i];
            if (!minimize_memory) {
                tk_out[i] = ts_tk[i];
            }
        } else {
            xh = global_xh_nofluct;
            if (!minimize_memory) {
                tk_out[i] = (float)(tk_nofluct * (1.0 + adia_tk_term * (double)dens[i]));
            }
        }
        neutral_out[i] = xh;
        sum_xh += (double)xh;
    }

    sdata[tid] = sum_xh;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_xh[blockIdx.x] = sdata[0];
    }
}

/* Matches IonisationBox.c:clip_and_get_extrema: records unclipped min/max, stores clipped values. */
__global__ void clip_and_extrema_kernel(float *grid, size_t n, double lo, double hi,
                                        double *blk_min, double *blk_max) {
    __shared__ double smin[kBlockThreads];
    __shared__ double smax[kBlockThreads];

    double v0 = (double)grid[0];
    double lo_acc = v0;
    double hi_acc = v0;

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        float curr = grid[i];
        grid[i] = fmaxf((float)fmin((double)curr, hi), (float)lo);
        double dcurr = (double)curr;
        if (dcurr < lo_acc) lo_acc = dcurr;
        if (dcurr > hi_acc) hi_acc = dcurr;
    }

    smin[tid] = lo_acc;
    smax[tid] = hi_acc;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            if (smin[tid + s] < smin[tid]) smin[tid] = smin[tid + s];
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_min[blockIdx.x] = smin[0];
        blk_max[blockIdx.x] = smax[0];
    }
}

__global__ void finish_minmax_kernel(const double *bmin, const double *bmax, int nblocks,
                                     float init_val, double *out2) {
    __shared__ double smin[kBlockThreads];
    __shared__ double smax[kBlockThreads];
    size_t tid = threadIdx.x;
    double lo = (double)init_val, hi = (double)init_val;
    for (int i = (int)tid; i < nblocks; i += kBlockThreads) {
        if (bmin[i] < lo) lo = bmin[i];
        if (bmax[i] > hi) hi = bmax[i];
    }
    smin[tid] = lo;
    smax[tid] = hi;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            if (smin[tid + s] < smin[tid]) smin[tid] = smin[tid + s];
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out2[0] = smin[0];
        out2[1] = smax[0];
    }
}

__global__ void clip_simple_kernel(float *grid, size_t n, float lo, float hi) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grid[idx] = fmaxf(fminf(grid[idx], hi), lo);
    }
}

__global__ void calc_cond_nion_kernel(
    size_t n, int fc_r_idx, int lagrangian_source_grids, int need_mcgs_nion, int need_prev_ion,
    int use_reion_fb, int use_mcgs, int mass_dep_zeta, int filter_recombinations, int use_ts_fluct,
    int recomb_used, double mturn_acg_homog, double atomic_cooling_threshold,
    double nion_prefactor_acg, double nion_prefactor_mcg, float min_dens_clip,
    int prev_nion_is_zero, P21DevTable1Df nion_1d_acg, P21DevTable1Df nion_1d_acg_prev,
    P21DevTable2Df nion_2d_acg, P21DevTable2Df nion_2d_mcg, P21DevTable2Df nion_2d_acg_prev,
    P21DevTable2Df nion_2d_mcg_prev,
    float *deltax_filt, float *nrec_filt, float *xe_filt, float *stars_filt, float *sfr_filt,
    const float *mturn_acg_filt, const float *mturn_mcg_filt, const float *prev_deltax_filt,
    const float *prev_nion_acg_box, const float *prev_nion_mcg_box, float *box_nion_acg,
    float *box_nion_mcg, double *blk_acg, double *blk_mcg, int *nonfinite_flag) {
    __shared__ double s_acg[kBlockThreads];
    __shared__ double s_mcg[kBlockThreads];

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double sum_acg = 0.0;
    double sum_mcg = 0.0;
    int local_nf = 0;
    double l10_acg_hom = log10(mturn_acg_homog);

    for (size_t i = idx; i < n; i += stride) {
        float d = fmaxf(deltax_filt[i], min_dens_clip);
        deltax_filt[i] = d;

        if (filter_recombinations && nrec_filt != nullptr) {
            nrec_filt[i] = fmaxf(nrec_filt[i], 0.0f);
        }
        if (use_ts_fluct) {
            xe_filt[i] = fminf(fmaxf(xe_filt[i], 0.0f), 0.999f);
        }

        double nion_acg = 0.0, nion_mcg = 0.0;
        double prev_nion_acg = 0.0, prev_nion_mcg = 0.0;

        if (lagrangian_source_grids) {
            float st = fmaxf(stars_filt[i], 0.0f);
            stars_filt[i] = st;
            if (recomb_used) {
                sfr_filt[i] = fmaxf(sfr_filt[i], 0.0f);
            }
            nion_acg = (double)st;
        } else {
            double curr_dens = (double)d;
            double l10_acg = use_reion_fb ? (double)mturn_acg_filt[i] : l10_acg_hom;
            double l10_mcg = 0.0;
            bool allow_mcg = false;

            if (use_mcgs) {
                l10_mcg = (double)mturn_mcg_filt[i];
                allow_mcg = (pow(10.0, l10_acg) <= atomic_cooling_threshold * 1.001);
                if (allow_mcg) {
                    nion_mcg = exp(p21_eval_rg2d_f(curr_dens, l10_mcg, nion_2d_mcg));
                }
                if (need_prev_ion) {
                    double prev_dens =
                        (prev_deltax_filt != nullptr) ? (double)prev_deltax_filt[i] : -1.0;
                    prev_nion_acg =
                        (use_reion_fb && mass_dep_zeta)
                            ? exp(p21_eval_rg2d_f(prev_dens, l10_acg, nion_2d_acg_prev))
                            : exp(p21_eval_rg1d_f(prev_dens, nion_1d_acg_prev));
                    if (allow_mcg) {
                        prev_nion_mcg =
                            exp(p21_eval_rg2d_f(prev_dens, l10_mcg, nion_2d_mcg_prev));
                    }
                }
            }

            nion_acg = (use_reion_fb && mass_dep_zeta)
                           ? exp(p21_eval_rg2d_f(curr_dens, l10_acg, nion_2d_acg))
                           : exp(p21_eval_rg1d_f(curr_dens, nion_1d_acg));
        }

        if (need_mcgs_nion) {
            size_t out_idx = (size_t)fc_r_idx * n + i;
            if (nion_acg > nion_prefactor_acg) nion_acg = nion_prefactor_acg;
            if (nion_acg < 0.0) nion_acg = 1e-40;
            if (prev_nion_acg > nion_prefactor_acg) prev_nion_acg = nion_prefactor_acg;
            if (prev_nion_acg < 0.0) prev_nion_acg = 1e-40;

            double prev_box_a = prev_nion_is_zero ? 0.0 : (double)prev_nion_acg_box[out_idx];
            double val_a = prev_box_a + nion_acg - prev_nion_acg;
            if (val_a > nion_prefactor_acg) val_a = nion_prefactor_acg;
            float fa = (float)val_a;
            box_nion_acg[out_idx] = fa;
            if (!isfinite(fa)) local_nf = 1;
            sum_acg += (double)fa;

            if (nion_mcg > nion_prefactor_mcg) nion_mcg = nion_prefactor_mcg;
            if (nion_mcg < 0.0) nion_mcg = 1e-40;
            if (prev_nion_mcg > nion_prefactor_mcg) prev_nion_mcg = nion_prefactor_mcg;
            if (prev_nion_mcg < 0.0) prev_nion_mcg = 1e-40;

            double prev_box_m = prev_nion_is_zero ? 0.0 : (double)prev_nion_mcg_box[out_idx];
            double val_m = prev_box_m + nion_mcg - prev_nion_mcg;
            if (val_m > nion_prefactor_mcg) val_m = nion_prefactor_mcg;
            float fm = (float)val_m;
            box_nion_mcg[out_idx] = fm;
            if (!isfinite(fm)) local_nf = 1;
            sum_mcg += (double)fm;
        } else {
            if (!lagrangian_source_grids) {
                box_nion_acg[i] = (float)nion_acg;
            }
            sum_acg += nion_acg;
        }
    }

    if (local_nf) atomicOr(nonfinite_flag, 1);

    s_acg[tid] = sum_acg;
    s_mcg[tid] = sum_mcg;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            s_acg[tid] += s_acg[tid + s];
            s_mcg[tid] += s_mcg[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_acg[blockIdx.x] = s_acg[0];
        blk_mcg[blockIdx.x] = s_mcg[0];
    }
}

__device__ static inline float p21_compute_part_ion_temp(float T_HI, float res_xH, float T_re) {
    if (res_xH <= 0.0f) return T_re;
    if (res_xH >= 1.0f) return T_HI;
    return T_HI * res_xH + T_re * (1.0f - res_xH);
}

__device__ static inline float p21_compute_fully_ion_temp(float z_re, float z, float delta,
                                                          float T_re) {
    float result;
    if (fabsf(z - z_re) < 1e-4f) {
        result = 1.0f;
    } else {
        float delta_re = delta * (1.0f + z) / (1.0f + z_re);
        if (delta_re <= -1.0f) delta_re = -1.0f + 9e-8f;
        if (delta <= -1.0f) delta = -1.0f + 9e-8f;
        result = pow((1.0 + (double)delta) / (1.0 + (double)delta_re), 1.1333);
        result *= pow((1.0 + (double)z) / (1.0 + (double)z_re), 3.4);
        result *= expf(pow((1.0 + (double)z) / 7.1, 2.5) - pow((1.0 + (double)z_re) / 7.1, 2.5));
    }
    result *= pow((double)T_re, 1.7);
    result += pow(1e4 * ((1.0 + (double)z) / 4.0), 1.7) * (1.0 + (double)delta);
    result = pow((double)result, 0.5882);
    return result;
}

__global__ void find_ionised_regions_center_kernel(
    size_t n, int R_index, int fc_r_idx, int lagrangian_source_grids, int need_mcgs_nion,
    int mass_dep_zeta, int use_mcgs, int recomb_model, int cell_recomb, int use_ts_fluct,
    int minimize_memory, double R, double redshift, double photoncons_adj, double gamma_prefactor,
    double T_re, double TK_nofluct, double adia_TK_term, double mean_fix_acg, double mean_fix_mcg,
    double nion_limit_acg, double nion_limit_mcg, float prev_homo_nrec, double fract_float_err,
    double tiny_val, const float *dens_raw, const float *deltax_filt, const float *stars_filt,
    const float *sfr_filt, const float *box_nion_acg, const float *box_nion_mcg,
    const float *prev_nrec, const float *nrec_filt, const float *xe_filt, const float *ts_tk,
    const float *prev_z_reion, float *neutral_fraction, float *ionisation_rate_G12,
    float *mean_free_path, float *z_reion, float *kinetic_temperature) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    double raw_d = (double)dens_raw[idx];
    double curr_dens =
        (R_index == 0) ? (raw_d * photoncons_adj) : (double)deltax_filt[idx];

    double curr_nion_acg =
        lagrangian_source_grids ? (double)stars_filt[idx]
                                : (double)box_nion_acg[(size_t)fc_r_idx * n + idx];
    curr_nion_acg *= mean_fix_acg;
    if (lagrangian_source_grids) {
        curr_nion_acg /= (1.0 + curr_dens);
    }

    double curr_nion_mcg = 0.0;
    if (need_mcgs_nion) {
        curr_nion_mcg = mean_fix_mcg * (double)box_nion_mcg[(size_t)fc_r_idx * n + idx];
    }

    if (mass_dep_zeta) {
        if (curr_nion_acg < nion_limit_acg) curr_nion_acg = nion_limit_acg;
        if (use_mcgs && curr_nion_mcg < nion_limit_mcg) curr_nion_mcg = nion_limit_mcg;
    }

    double rec = 0.0;
    bool recomb_used = (recomb_model != RECOMB_MODEL_NO);
    if (recomb_used) {
        if (cell_recomb) {
            rec = (recomb_model == RECOMB_MODEL_INHOMOGENEOUS)
                      ? (prev_nrec != nullptr ? (double)prev_nrec[idx] : 0.0)
                      : (double)prev_homo_nrec;
        } else {
            rec = (nrec_filt != nullptr) ? (double)nrec_filt[idx] : 0.0;
        }
        rec /= (1.0 + curr_dens);
    }

    double xHII_xrays = use_ts_fluct ? (double)xe_filt[idx] : 0.0;

    if (curr_nion_acg + curr_nion_mcg > (1.0 - xHII_xrays) * (1.0 + rec)) {
        if (recomb_used && ((double)neutral_fraction[idx] > fract_float_err)) {
            if (lagrangian_source_grids) {
                ionisation_rate_G12[idx] =
                    (float)(R * gamma_prefactor / (1.0 + curr_dens) * (double)sfr_filt[idx]);
            } else {
                ionisation_rate_G12[idx] =
                    (float)(R * gamma_prefactor * (curr_nion_acg + curr_nion_mcg));
            }
            if (!minimize_memory) {
                mean_free_path[idx] = (float)R;
            }
        }
        float pz = (prev_z_reion != nullptr) ? prev_z_reion[idx] : -1.0f;
        z_reion[idx] = (pz < 0.0f) ? (float)redshift : pz;
        neutral_fraction[idx] = 0.0f;
    } else if (R_index == 0 && ((double)neutral_fraction[idx] > tiny_val)) {
        double res_xH = 1.0 - curr_nion_acg - curr_nion_mcg;
        if (!minimize_memory) {
            float T_hi = use_ts_fluct ? ts_tk[idx]
                                      : (float)(TK_nofluct * (1.0 + adia_TK_term * raw_d));
            kinetic_temperature[idx] = p21_compute_part_ion_temp(T_hi, (float)res_xH, (float)T_re);
        }
        res_xH -= xHII_xrays;
        if (res_xH < 0.0) res_xH = 0.0;
        else if (res_xH > 1.0) res_xH = 1.0;
        neutral_fraction[idx] = (float)res_xH;
    }
}

__global__ void find_ionised_centers_sphere_kernel(
    size_t n, int R_index, int fc_r_idx, int lagrangian_source_grids, int need_mcgs_nion,
    int mass_dep_zeta, int use_mcgs, int recomb_model, int cell_recomb, int use_ts_fluct,
    int minimize_memory, double R, double redshift, double photoncons_adj, double gamma_prefactor,
    double mean_fix_acg, double mean_fix_mcg, double nion_limit_acg, double nion_limit_mcg,
    float prev_homo_nrec, double fract_float_err, const float *dens_raw, const float *deltax_filt,
    const float *stars_filt, const float *sfr_filt, const float *box_nion_acg,
    const float *box_nion_mcg, const float *prev_nrec, const float *nrec_filt, const float *xe_filt,
    const float *prev_z_reion, const float *neutral_fraction, float *ionisation_rate_G12,
    float *mean_free_path, float *z_reion, int *edt_init, uint8_t *sphere_mask, int *n_centers) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    double raw_d = (double)dens_raw[idx];
    double curr_dens = (R_index == 0) ? (raw_d * photoncons_adj) : (double)deltax_filt[idx];

    double curr_nion_acg =
        lagrangian_source_grids ? (double)stars_filt[idx]
                                : (double)box_nion_acg[(size_t)fc_r_idx * n + idx];
    curr_nion_acg *= mean_fix_acg;
    if (lagrangian_source_grids) {
        curr_nion_acg /= (1.0 + curr_dens);
    }

    double curr_nion_mcg = 0.0;
    if (need_mcgs_nion) {
        curr_nion_mcg = mean_fix_mcg * (double)box_nion_mcg[(size_t)fc_r_idx * n + idx];
    }

    if (mass_dep_zeta) {
        if (curr_nion_acg < nion_limit_acg) curr_nion_acg = nion_limit_acg;
        if (use_mcgs && curr_nion_mcg < nion_limit_mcg) curr_nion_mcg = nion_limit_mcg;
    }

    double rec = 0.0;
    bool recomb_used = (recomb_model != RECOMB_MODEL_NO);
    if (recomb_used) {
        if (cell_recomb) {
            rec = (recomb_model == RECOMB_MODEL_INHOMOGENEOUS)
                      ? (prev_nrec != nullptr ? (double)prev_nrec[idx] : 0.0)
                      : (double)prev_homo_nrec;
        } else {
            rec = (nrec_filt != nullptr) ? (double)nrec_filt[idx] : 0.0;
        }
        rec /= (1.0 + curr_dens);
    }

    double xHII_xrays = use_ts_fluct ? (double)xe_filt[idx] : 0.0;
    if (curr_nion_acg + curr_nion_mcg > (1.0 - xHII_xrays) * (1.0 + rec)) {
        if (recomb_used && ((double)neutral_fraction[idx] > fract_float_err)) {
            if (lagrangian_source_grids) {
                ionisation_rate_G12[idx] =
                    (float)(R * gamma_prefactor / (1.0 + curr_dens) * (double)sfr_filt[idx]);
            } else {
                ionisation_rate_G12[idx] =
                    (float)(R * gamma_prefactor * (curr_nion_acg + curr_nion_mcg));
            }
            if (!minimize_memory) {
                mean_free_path[idx] = (float)R;
            }
        }
        float pz = (prev_z_reion != nullptr) ? prev_z_reion[idx] : -1.0f;
        z_reion[idx] = (pz < 0.0f) ? (float)redshift : pz;
        edt_init[idx] = 0;
        sphere_mask[idx] = 1;
        atomicAdd(n_centers, 1);
    } else {
        edt_init[idx] = kEdtInf;
        sphere_mask[idx] = 0;
    }
}

__global__ void edt_1d_periodic_shared_kernel(const int *in_box, int *out_box, int nx, int ny,
                                              int nz, int axis, int r_lim) {
    extern __shared__ int s_line[];
    int L = (axis == 0) ? nx : ((axis == 1) ? ny : nz);
    size_t line_id = blockIdx.x;
    size_t stride = 0;
    size_t base = 0;

    if (axis == 0) {
        int y = (int)(line_id / nz);
        int z = (int)(line_id % nz);
        if (y >= ny) return;
        base = (size_t)z + (size_t)nz * y;
        stride = (size_t)ny * (size_t)nz;
    } else if (axis == 1) {
        int x = (int)(line_id / nz);
        int z = (int)(line_id % nz);
        if (x >= nx) return;
        base = (size_t)z + (size_t)nz * (size_t)ny * (size_t)x;
        stride = (size_t)nz;
    } else {
        int x = (int)(line_id / ny);
        int y = (int)(line_id % ny);
        if (x >= nx) return;
        base = (size_t)nz * ((size_t)y + (size_t)ny * (size_t)x);
        stride = 1;
    }

    for (int u = threadIdx.x; u < L; u += blockDim.x) {
        s_line[u] = in_box[base + (size_t)u * stride];
    }
    __syncthreads();

    for (int u = threadIdx.x; u < L; u += blockDim.x) {
        int best = s_line[u];
        for (int d = -r_lim; d <= r_lim; ++d) {
            int v = u + d;
            if (v < 0) v += L;
            else if (v >= L) v -= L;
            int cand = s_line[v] + d * d;
            if (cand < best) best = cand;
        }
        out_box[base + (size_t)u * stride] = best;
    }
}

__global__ void edt_1d_periodic_global_kernel(const int *in_box, int *out_box, int nx, int ny,
                                              int nz, int axis, int r_lim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    if (idx >= total) return;

    int z = (int)(idx % nz);
    size_t q = idx / nz;
    int y = (int)(q % ny);
    int x = (int)(q / ny);

    int L = (axis == 0) ? nx : ((axis == 1) ? ny : nz);
    int u = (axis == 0) ? x : ((axis == 1) ? y : z);
    size_t stride = (axis == 0) ? ((size_t)ny * (size_t)nz) : ((axis == 1) ? (size_t)nz : 1ULL);
    size_t base = idx - (size_t)u * stride;

    int best = in_box[idx];
    for (int d = -r_lim; d <= r_lim; ++d) {
        int v = u + d;
        if (v < 0) v += L;
        else if (v >= L) v -= L;
        int cand = in_box[base + (size_t)v * stride] + d * d;
        if (cand < best) best = cand;
    }
    out_box[idx] = best;
}

__global__ void apply_sphere_edt_kernel(size_t n, const int *dist_sq, float rsq_curr_index,
                                        float *neutral_fraction) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && (float)dist_sq[idx] < rsq_curr_index) {
        neutral_fraction[idx] = 0.0f;
    }
}

__global__ void assign_partial_ionization_kernel(
    size_t n, int lagrangian_source_grids, int need_mcgs_nion, int mass_dep_zeta, int use_mcgs,
    int use_ts_fluct, int minimize_memory, double photoncons_adj, double T_re, double TK_nofluct,
    double adia_TK_term, double mean_fix_acg, double mean_fix_mcg, double nion_limit_acg,
    double nion_limit_mcg, double tiny_val, const float *dens_raw, const float *deltax_filt,
    const float *stars_filt, const float *box_nion_acg, const float *box_nion_mcg,
    const float *xe_filt, const float *ts_tk, float *neutral_fraction, float *kinetic_temperature) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if ((double)neutral_fraction[idx] <= tiny_val) {
        return;
    }

    double raw_d = (double)dens_raw[idx];
    double curr_dens = raw_d * photoncons_adj;

    double curr_nion_acg =
        lagrangian_source_grids ? (double)stars_filt[idx] : (double)box_nion_acg[idx];
    curr_nion_acg *= mean_fix_acg;
    if (lagrangian_source_grids) {
        curr_nion_acg /= (1.0 + curr_dens);
    }

    double curr_nion_mcg = need_mcgs_nion ? (mean_fix_mcg * (double)box_nion_mcg[idx]) : 0.0;
    if (mass_dep_zeta) {
        if (curr_nion_acg < nion_limit_acg) curr_nion_acg = nion_limit_acg;
        if (use_mcgs && curr_nion_mcg < nion_limit_mcg) curr_nion_mcg = nion_limit_mcg;
    }

    double xHII_xrays = use_ts_fluct ? (double)xe_filt[idx] : 0.0;
    double res_xH = 1.0 - curr_nion_acg - curr_nion_mcg;
    if (!minimize_memory) {
        float T_hi = use_ts_fluct ? ts_tk[idx] : (float)(TK_nofluct * (1.0 + adia_TK_term * raw_d));
        kinetic_temperature[idx] = p21_compute_part_ion_temp(T_hi, (float)res_xH, (float)T_re);
    }
    res_xH -= xHII_xrays;
    if (res_xH < 0.0) res_xH = 0.0;
    else if (res_xH > 1.0) res_xH = 1.0;
    neutral_fraction[idx] = (float)res_xH;
}

__global__ void set_ionized_temps_kernel(size_t n, int use_ts_fluct, float stored_redshift,
                                         float T_re, double TK_nofluct, double adia_tk_term,
                                         double tiny_val, const float *z_reion,
                                         const float *neutral_fraction, const float *dens,
                                         const float *ts_tk, float *kinetic_temperature,
                                         int *nonfinite_flag) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float zre = z_reion[idx];
    float nf = neutral_fraction[idx];
    float tk = kinetic_temperature[idx];

    if (zre > 0.0f && (double)nf < tiny_val) {
        float d = dens[idx];
        tk = p21_compute_fully_ion_temp(zre, stored_redshift, d, T_re);
        float floor_tk =
            use_ts_fluct ? ts_tk[idx] : (float)(TK_nofluct * (1.0 + adia_tk_term * (double)d));
        if (tk < floor_tk) {
            tk = floor_tk;
        }
        kinetic_temperature[idx] = tk;
    }
    if (!isfinite(tk)) {
        atomicOr(nonfinite_flag, 1);
    }
}

__global__ void ion_global_means_kernel(size_t n, int homo_recomb, const float *neutral_fraction,
                                        const float *gamma12, double *blk_xh, double *blk_g12) {
    __shared__ double s_xh[kBlockThreads];
    __shared__ double s_g[kBlockThreads];

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double xh = 0.0, g12 = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        xh += (double)neutral_fraction[i];
        if (homo_recomb) {
            g12 += (double)gamma12[i];
        }
    }

    s_xh[tid] = xh;
    s_g[tid] = g12;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            s_xh[tid] += s_xh[tid + s];
            s_g[tid] += s_g[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_xh[blockIdx.x] = s_xh[0];
        blk_g12[blockIdx.x] = s_g[0];
    }
}

__global__ void inhomo_recomb_kernel(size_t n, double stored_redshift, double fabs_dtdz, double dz,
                                     int mhr_nz, int mhr_ng, double mhr_del_z, double mhr_lng_min,
                                     double mhr_del_lng, double fract_float_err, const double *mhr_y,
                                     const double *mhr_c, const float *dens, const float *gamma12,
                                     const float *neutral_fraction, const float *prev_nrec,
                                     float *out_nrec, int *nonfinite_flag) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    double curr_dens = 1.0 + (double)dens[idx];
    double z_eff = cbrt(curr_dens) * (1.0 + stored_redshift) - 1.0;
    int z_ct = (int)(z_eff / mhr_del_z + 0.5);
    if (z_ct < 0) z_ct = 0;
    else if (z_ct >= mhr_nz) z_ct = mhr_nz - 1;

    double g12 = (double)gamma12[idx];
    double rr = 0.0;
    if (g12 > 0.0) {
        double lnGamma = log(g12);
        double max_lng = mhr_lng_min + mhr_del_lng * (double)(mhr_ng - 1);
        if (lnGamma >= mhr_lng_min) {
            if (lnGamma >= max_lng) {
                lnGamma = max_lng - fract_float_err;
            }
            double h = mhr_del_lng;
            int j = (int)floor((lnGamma - mhr_lng_min) / h);
            if (j < 0) j = 0;
            else if (j > mhr_ng - 2) j = mhr_ng - 2;

            double xj = mhr_lng_min + (double)j * h;
            double dx = lnGamma - xj;
            const double *row_y = mhr_y + (size_t)z_ct * mhr_ng;
            const double *row_c = mhr_c + (size_t)z_ct * mhr_ng;
            double yi = row_y[j];
            double yi1 = row_y[j + 1];
            double ci = row_c[j];
            double ci1 = row_c[j + 1];
            double bi = (yi1 - yi) / h - h / 3.0 * (ci1 + 2.0 * ci);
            double di = (ci1 - ci) / (3.0 * h);
            rr = yi + dx * (bi + dx * (ci + dx * di));
        }
    }

    double dNrec = rr * fabs_dtdz * dz * (1.0 - (double)neutral_fraction[idx]);
    if (!isfinite(dNrec)) {
        atomicOr(nonfinite_flag, 1);
    }
    double pn = (prev_nrec != nullptr) ? (double)prev_nrec[idx] : 0.0;
    out_nrec[idx] = (float)(pn + dNrec);
}

bool build_and_cache_mhr_spline(cudaStream_t stream, const double **d_y_out,
                                const double **d_c_out, int *nz_out, int *ng_out,
                                double *del_z_out, double *lng_min_out, double *del_lng_out) {
    const double *h_table = nullptr;
    int nz = 0, ng = 0, gen = 0;
    double del_z = 0.2, lng_min = -10.0, del_lng = 0.1;
    if (!p21_get_mhr_rr_table(&h_table, &nz, &ng, &del_z, &lng_min, &del_lng, &gen)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mhr_mu);
    if (!g_d_mhr_y || !g_d_mhr_c || g_mhr_gen != gen || g_mhr_nz != nz || g_mhr_ng != ng) {
        p21_cuda_pool_free(g_d_mhr_y, stream);
        p21_cuda_pool_free(g_d_mhr_c, stream);
        g_d_mhr_y = nullptr;
        g_d_mhr_c = nullptr;

        /* Solve GSL natural cubic spline c coefficients per redshift row via Thomas algorithm. */
        size_t total_elems = (size_t)nz * (size_t)ng;
        std::vector<double> c_host(total_elems, 0.0);
        std::vector<double> cp(ng, 0.0), dp(ng, 0.0);
        double h = del_lng;
        for (int z = 0; z < nz; ++z) {
            const double *y = h_table + (size_t)z * ng;
            double *c = c_host.data() + (size_t)z * ng;
            if (ng >= 3) {
                for (int i = 1; i < ng - 1; ++i) {
                    double a_i = h;
                    double b_i = 4.0 * h;
                    double c_i = (i + 1 < ng - 1) ? h : 0.0;
                    double r_i = 3.0 * ((y[i + 1] - y[i]) / h - (y[i] - y[i - 1]) / h);
                    double denom = (i == 1) ? b_i : (b_i - a_i * cp[i - 1]);
                    cp[i] = c_i / denom;
                    dp[i] = (i == 1) ? (r_i / denom) : ((r_i - a_i * dp[i - 1]) / denom);
                }
                c[ng - 2] = dp[ng - 2];
                for (int i = ng - 3; i >= 1; --i) {
                    c[i] = dp[i] - cp[i] * c[i + 1];
                }
            }
        }

        size_t bytes = total_elems * sizeof(double);
        g_d_mhr_y = (double *)p21_cuda_pool_alloc(bytes, stream);
        g_d_mhr_c = (double *)p21_cuda_pool_alloc(bytes, stream);
        if (!g_d_mhr_y || !g_d_mhr_c ||
            !P21_CUDA_OK(cudaMemcpyAsync(g_d_mhr_y, h_table, bytes, cudaMemcpyHostToDevice, stream)) ||
            !P21_CUDA_OK(
                cudaMemcpyAsync(g_d_mhr_c, c_host.data(), bytes, cudaMemcpyHostToDevice, stream))) {
            p21_cuda_pool_free(g_d_mhr_y, stream);
            p21_cuda_pool_free(g_d_mhr_c, stream);
            g_d_mhr_y = nullptr;
            g_d_mhr_c = nullptr;
            g_mhr_gen = -1;
            return false;
        }
        g_mhr_nz = nz;
        g_mhr_ng = ng;
        g_mhr_del_z = del_z;
        g_mhr_lng_min = lng_min;
        g_mhr_del_lng = del_lng;
        g_mhr_gen = gen;
    }

    *d_y_out = g_d_mhr_y;
    *d_c_out = g_d_mhr_c;
    *nz_out = g_mhr_nz;
    *ng_out = g_mhr_ng;
    *del_z_out = g_mhr_del_z;
    *lng_min_out = g_mhr_lng_min;
    *del_lng_out = g_mhr_del_lng;
    return true;
}

int read_ion_spill_kspace_env(void) {
    const char *env = getenv("P21_CUDA_ION_SPILL_KSPACE");
    if (!env || !*env) {
        return -1;
    }
    if (strcmp(env, "1") == 0) {
        return 1;
    }
    if (strcmp(env, "0") == 0) {
        return 0;
    }
    return -1;
}

void free_ion_rloop_scratch(P21CudaIonCtx *ctx) {
    if (!ctx) {
        return;
    }
    cudaStream_t stream = ctx->stream;
    auto pfree = [&](auto *&ptr) {
        if (ptr) {
            p21_cuda_pool_free(ptr, stream);
            ptr = nullptr;
        }
    };
    auto hfree = [&](cufftComplex *&ptr) {
        if (ptr) {
            free(ptr);
            ptr = nullptr;
        }
    };

    pfree(ctx->d_k_deltax_unfilt);
    pfree(ctx->d_k_xe_unfilt);
    pfree(ctx->d_k_nrec_unfilt);
    pfree(ctx->d_k_mturn_acg_unfilt);
    pfree(ctx->d_k_prev_deltax_unfilt);
    pfree(ctx->d_k_mturn_mcg_unfilt);
    pfree(ctx->d_k_stars_unfilt);
    pfree(ctx->d_k_sfr_unfilt);

    hfree(ctx->h_k_deltax_unfilt);
    hfree(ctx->h_k_xe_unfilt);
    hfree(ctx->h_k_nrec_unfilt);
    hfree(ctx->h_k_mturn_acg_unfilt);
    hfree(ctx->h_k_prev_deltax_unfilt);
    hfree(ctx->h_k_mturn_mcg_unfilt);
    hfree(ctx->h_k_stars_unfilt);
    hfree(ctx->h_k_sfr_unfilt);

    pfree(ctx->d_k_work);
    auto yfree = [&](cufftComplex *(&ys)[kMgpuMaxDevices]) {
        for (int g = 0; g < kMgpuMaxDevices; ++g) {
            if (ys[g]) {
                p21_mgpu_free(ctx->mgpu_grid.devs[g], ys[g], ctx->mgpu_grid.streams[g]);
                ys[g] = nullptr;
            }
        }
    };
    if (ctx->mgpu_mode) {
        yfree(ctx->y_deltax_unfilt);
        yfree(ctx->y_xe_unfilt);
        yfree(ctx->y_nrec_unfilt);
        yfree(ctx->y_mturn_acg_unfilt);
        yfree(ctx->y_prev_deltax_unfilt);
        yfree(ctx->y_mturn_mcg_unfilt);
        yfree(ctx->y_stars_unfilt);
        yfree(ctx->y_sfr_unfilt);
        yfree(ctx->y_work);
        ctx->mgpu_grid.destroy();
        ctx->mgpu_mode = false;
    }
    pfree(ctx->d_real_work);
    pfree(ctx->d_deltax_filt);
    pfree(ctx->d_xe_filt);
    pfree(ctx->d_nrec_filt);
    pfree(ctx->d_mturn_acg_real);
    pfree(ctx->d_prev_deltax_filt);
    pfree(ctx->d_mturn_mcg_real);
    pfree(ctx->d_stars_filt);
    pfree(ctx->d_sfr_filt);

    pfree(ctx->d_sphere_mask);
    pfree(ctx->d_edt_a);
    pfree(ctx->d_edt_b);
    pfree(ctx->d_n_centers);
}

}  // namespace

void p21_cuda_ionization_cleanup(void) {
    cudaStream_t stream = p21_cuda_stream();
    std::lock_guard<std::mutex> lock(g_mhr_mu);
    p21_cuda_pool_free(g_d_mhr_y, stream);
    p21_cuda_pool_free(g_d_mhr_c, stream);
    g_d_mhr_y = nullptr;
    g_d_mhr_c = nullptr;
    g_mhr_nz = 0;
    g_mhr_ng = 0;
    g_mhr_gen = -1;
}

extern "C" P21CudaIonCtx *p21_cuda_ion_begin(
    const P21IonBoxParams *params, int n_radii, PerturbedField *perturbed_field,
    PerturbedField *previous_perturbed_field, IonizedBox *previous_ionize_box, TsBox *spin_temp,
    EmissivityFields *emissivity_fields, InitialConditions *ini_boxes) {
    if (!params || !perturbed_field || !previous_ionize_box || n_radii <= 0 ||
        !p21_cuda_get_enabled()) {
        return nullptr;
    }
    if (simulation_options_global->HII_DIM <= 1 || !p21_cuda_ensure_init()) {
        return nullptr;
    }

    auto *ctx = new P21CudaIonCtx;
    ctx->nx = simulation_options_global->HII_DIM;
    ctx->ny = simulation_options_global->HII_DIM;
    ctx->nz = HII_D_PARA;
    ctx->total = (size_t)ctx->nx * (size_t)ctx->ny * (size_t)ctx->nz;
    ctx->k_total = (size_t)ctx->nx * (size_t)ctx->ny * (size_t)(ctx->nz / 2 + 1);
    ctx->n_radii = n_radii;
    ctx->use_ts_fluct = astro_options_global->USE_TS_FLUCT ? 1 : 0;
    ctx->use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    ctx->use_reion_fb = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK ? 1 : 0;
    ctx->vcb_flucts = (matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) ? 1 : 0;
    ctx->recomb_model = astro_options_global->RECOMB_MODEL;
    ctx->cell_recomb = astro_options_global->CELL_RECOMB ? 1 : 0;
    ctx->minimize_memory = matter_options_global->MINIMIZE_MEMORY ? 1 : 0;
    ctx->ionise_entire_sphere = astro_options_global->IONISE_ENTIRE_SPHERE ? 1 : 0;
    ctx->use_exp_filter = astro_options_global->USE_EXP_FILTER ? 1 : 0;
    ctx->prev_first_z = (params->prev_redshift < 1.0) ? 1 : 0;
    ctx->stream = p21_cuda_stream();

    cudaStream_t stream = ctx->stream;
    const size_t n = ctx->total;
    const size_t fbytes = n * sizeof(float);
    const size_t cbytes = ctx->k_total * sizeof(cufftComplex);
    const int grid = (int)((n + kBlockThreads - 1) / kBlockThreads);
    bool recomb_used = (ctx->recomb_model != RECOMB_MODEL_NO);

    if (fbytes >= (1ULL << 30)) {
        size_t total_mem = p21_cuda_total_mem_bytes();
        size_t target_free = fbytes * 10;
        if (total_mem > 0) {
            target_free = std::min<size_t>(target_free, (total_mem * 3) / 4);
        }
        p21_cuda_registry_evict_to_free(target_free, stream);
    }

    ctx->d_dens =
        (const float *)p21_cuda_registry_get_or_upload(perturbed_field->density, fbytes, stream);
    bool ok = (ctx->d_dens != nullptr);

    if (ok && ctx->use_ts_fluct) {
        ctx->d_ts_xe = (const float *)p21_cuda_registry_get_or_upload(
            spin_temp->xray_ionised_fraction, fbytes, stream);
        ctx->d_ts_tk = (const float *)p21_cuda_registry_get_or_upload(
            spin_temp->kinetic_temp_neutral, fbytes, stream);
        if (!ctx->d_ts_xe || !ctx->d_ts_tk) ok = false;
        if (ok && ctx->use_mcgs) {
            ctx->d_ts_j21 =
                (const float *)p21_cuda_registry_get_or_upload(spin_temp->J_21_LW, fbytes, stream);
            if (!ctx->d_ts_j21) ok = false;
        }
    }

    if (ok && !params->lagrangian_source_grids && ctx->vcb_flucts && ctx->use_mcgs) {
        ctx->d_vcb =
            (const float *)p21_cuda_registry_get_or_upload(ini_boxes->lowres_vcb, fbytes, stream);
        if (!ctx->d_vcb) ok = false;
    }

    if (ok && !ctx->prev_first_z) {
        ctx->d_prev_zre = (const float *)p21_cuda_registry_get_or_upload(
            previous_ionize_box->z_reion, fbytes, stream);
        if (!ctx->d_prev_zre) ok = false;
        if (ok && ctx->use_reion_fb) {
            ctx->d_prev_gamma12 = (const float *)p21_cuda_registry_get_or_upload(
                previous_ionize_box->ionisation_rate_G12, fbytes, stream);
            if (!ctx->d_prev_gamma12) ok = false;
        }
        if (ok && ctx->use_mcgs && !params->lagrangian_source_grids) {
            ctx->d_prev_dens = (const float *)p21_cuda_registry_get_or_upload(
                previous_perturbed_field->density, fbytes, stream);
            if (!ctx->d_prev_dens) ok = false;
        }
        if (ok && ctx->recomb_model == RECOMB_MODEL_INHOMOGENEOUS) {
            ctx->d_prev_nrec = (const float *)p21_cuda_registry_get_or_upload(
                previous_ionize_box->cumulative_recombinations, fbytes, stream);
            if (!ctx->d_prev_nrec) ok = false;
        }
        if (ok && params->need_mcgs_nion) {
            size_t pbytes = (size_t)n_radii * fbytes;
            ctx->d_prev_nion_acg = (const float *)p21_cuda_registry_get_or_upload(
                previous_ionize_box->nion_conditional_filtered_acg, pbytes, stream);
            ctx->d_prev_nion_mcg = (const float *)p21_cuda_registry_get_or_upload(
                previous_ionize_box->nion_conditional_filtered_mcg, pbytes, stream);
            if (!ctx->d_prev_nion_acg || !ctx->d_prev_nion_mcg) ok = false;
        }
    }

    if (ok && params->lagrangian_source_grids) {
        ctx->d_stars_in = (const float *)p21_cuda_registry_get_or_upload(emissivity_fields->n_ion,
                                                                         fbytes, stream);
        if (!ctx->d_stars_in) ok = false;
        if (ok && recomb_used) {
            ctx->d_sfr_in = (const float *)p21_cuda_registry_get_or_upload(
                emissivity_fields->fesc_weighted_sfrd, fbytes, stream);
            if (!ctx->d_sfr_in) ok = false;
        }
    }

    /* Output buffers */
    if (ok) {
        ctx->d_z_reion = (float *)p21_cuda_pool_alloc(fbytes, stream);
        ctx->d_neutral_fraction = (float *)p21_cuda_pool_alloc(fbytes, stream);
        if (!ctx->d_z_reion || !ctx->d_neutral_fraction) {
            ok = false;
        } else {
            fill_float_kernel<<<grid, kBlockThreads, 0, stream>>>(ctx->d_z_reion, n, -1.0f);
            fill_float_kernel<<<grid, kBlockThreads, 0, stream>>>(ctx->d_neutral_fraction, n, 1.0f);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        if (ok && recomb_used) {
            ctx->d_ionisation_rate_G12 = (float *)p21_cuda_pool_alloc(fbytes, stream);
            if (!ctx->d_ionisation_rate_G12 ||
                !P21_CUDA_OK(cudaMemsetAsync(ctx->d_ionisation_rate_G12, 0, fbytes, stream))) {
                ok = false;
            }
        }
    }
    if (ok && !ctx->minimize_memory) {
        ctx->d_kinetic_temperature = (float *)p21_cuda_pool_alloc(fbytes, stream);
        if (!ctx->d_kinetic_temperature ||
            !P21_CUDA_OK(cudaMemsetAsync(ctx->d_kinetic_temperature, 0, fbytes, stream))) {
            ok = false;
        }
        if (ok && recomb_used) {
            ctx->d_mean_free_path = (float *)p21_cuda_pool_alloc(fbytes, stream);
            if (!ctx->d_mean_free_path ||
                !P21_CUDA_OK(cudaMemsetAsync(ctx->d_mean_free_path, 0, fbytes, stream))) {
                ok = false;
            }
        }
    }
    if (ok && ctx->recomb_model == RECOMB_MODEL_INHOMOGENEOUS) {
        ctx->d_cumulative_recombinations = (float *)p21_cuda_pool_alloc(fbytes, stream);
        if (!ctx->d_cumulative_recombinations ||
            !P21_CUDA_OK(cudaMemsetAsync(ctx->d_cumulative_recombinations, 0, fbytes, stream))) {
            ok = false;
        }
    }
    if (ok && !params->lagrangian_source_grids) {
        size_t alloc_radii = params->need_mcgs_nion ? (size_t)n_radii : 1ULL;
        ctx->nion_cond_bytes = alloc_radii * fbytes;
        ctx->d_nion_cond_acg = (float *)p21_cuda_pool_alloc(ctx->nion_cond_bytes, stream);
        if (!ctx->d_nion_cond_acg ||
            !P21_CUDA_OK(cudaMemsetAsync(ctx->d_nion_cond_acg, 0, ctx->nion_cond_bytes, stream))) {
            ok = false;
        }
        if (ok && params->need_mcgs_nion) {
            ctx->d_nion_cond_mcg = (float *)p21_cuda_pool_alloc(ctx->nion_cond_bytes, stream);
            if (!ctx->d_nion_cond_mcg ||
                !P21_CUDA_OK(
                    cudaMemsetAsync(ctx->d_nion_cond_mcg, 0, ctx->nion_cond_bytes, stream))) {
                ok = false;
            }
        }
    }

    /* Scratch & reduction buffers (k-space/filtered grids allocated in prepare_r2c so early-exit is fast) */
    int blocks = (int)((n + kBlockThreads - 1) / kBlockThreads);
    if (blocks > 1024) blocks = 1024;
    ctx->reduce_blocks = blocks;
    if (ok) {
        ctx->d_blk0 = (double *)p21_cuda_pool_alloc((size_t)blocks * sizeof(double), stream);
        ctx->d_blk1 = (double *)p21_cuda_pool_alloc((size_t)blocks * sizeof(double), stream);
        ctx->d_scalar2 = (double *)p21_cuda_pool_alloc(2 * sizeof(double), stream);
        ctx->d_nonfinite = (int *)p21_cuda_pool_alloc(sizeof(int), stream);
        if (!ctx->d_blk0 || !ctx->d_blk1 || !ctx->d_scalar2 || !ctx->d_nonfinite) ok = false;
    }

    /* Allocate mturn real grids early for calc_mcrit */
    if (ok && !params->lagrangian_source_grids) {
        if (ctx->use_reion_fb) {
            ctx->d_mturn_acg_real = (float *)p21_cuda_pool_alloc(fbytes, stream);
            if (!ctx->d_mturn_acg_real) ok = false;
        }
        if (ok && ctx->use_mcgs) {
            ctx->d_mturn_mcg_real = (float *)p21_cuda_pool_alloc(fbytes, stream);
            if (!ctx->d_mturn_mcg_real) ok = false;
        }
    }
    (void)cbytes;

    if (!ok) {
        p21_cuda_ion_end(ctx);
        return nullptr;
    }
    return ctx;
}

extern "C" int p21_cuda_ion_calc_mcrit(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                       double *log10_mturn_acg_avg, double *log10_mturn_mcg_avg) {
    if (!ctx || !params || !log10_mturn_acg_avg || !log10_mturn_mcg_avg) {
        return -1;
    }
    if (!ctx->use_mcgs && !ctx->use_reion_fb) {
        *log10_mturn_acg_avg = log10(params->mturn_acg_homogeneous);
        *log10_mturn_mcg_avg = 0.0;
        return 0;
    }

    cudaStream_t stream = ctx->stream;
    P21MturnParams mp{};
    mp.z = params->redshift;
    mp.mturn_acg_homogeneous = params->mturn_acg_homogeneous;
    mp.a_lw = astro_params_global->A_LW;
    mp.beta_lw = astro_params_global->BETA_LW;
    mp.a_vcb = astro_params_global->A_VCB;
    mp.beta_vcb = astro_params_global->BETA_VCB;
    mp.sigma_vcb = cosmo_tables_global->V_CB_AVG * sqrt(3.0 * M_PI / 8.0);
    mp.mturn_stellar_feedback = astro_params_global->M_TURN_STELLAR_FEEDBACK;
    mp.use_reion_feedback = ctx->use_reion_fb;
    mp.use_mcgs = ctx->use_mcgs;

    int below_zheatmax = (params->redshift < simulation_options_global->Z_HEAT_MAX) ? 1 : 0;
    ion_mcrit_kernel<<<ctx->reduce_blocks, kBlockThreads, 0, stream>>>(
        ctx->total, mp, (float)params->vcb_const, below_zheatmax, ctx->vcb_flucts, ctx->d_vcb,
        ctx->d_ts_j21, ctx->d_prev_gamma12, ctx->d_prev_zre, ctx->d_mturn_acg_real,
        ctx->d_mturn_mcg_real, ctx->d_blk0, ctx->d_blk1);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }
    finish_pair_sum_kernel<<<1, kBlockThreads, 0, stream>>>(ctx->d_blk0, ctx->d_blk1,
                                                            ctx->reduce_blocks, ctx->d_scalar2);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }
    double sums[2] = {0.0, 0.0};
    if (!P21_CUDA_OK(cudaMemcpyAsync(sums, ctx->d_scalar2, 2 * sizeof(double),
                                     cudaMemcpyDeviceToHost, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return -1;
    }

    *log10_mturn_acg_avg =
        ctx->use_reion_fb ? (sums[0] / (double)ctx->total) : log10(params->mturn_acg_homogeneous);
    *log10_mturn_mcg_avg = ctx->use_mcgs ? (sums[1] / (double)ctx->total) : 0.0;
    return 0;
}

extern "C" int p21_cuda_ion_set_fully_neutral(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                              double xion_recfast_z, IonizedBox *box,
                                              double *global_xH_out) {
    if (!ctx || !params || !box || !global_xH_out) {
        return -1;
    }
    cudaStream_t stream = ctx->stream;
    float global_xh_nofluct = (float)(1.0 - xion_recfast_z);
    set_fully_neutral_kernel<<<ctx->reduce_blocks, kBlockThreads, 0, stream>>>(
        ctx->total, ctx->use_ts_fluct, ctx->minimize_memory, global_xh_nofluct, params->TK_nofluct,
        params->adia_TK_term, ctx->d_dens, ctx->d_ts_xe, ctx->d_ts_tk, ctx->d_neutral_fraction,
        ctx->d_kinetic_temperature, ctx->d_blk0);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }

    double h_xh = global_xh_nofluct;
    if (ctx->use_ts_fluct) {
        finish_pair_sum_kernel<<<1, kBlockThreads, 0, stream>>>(ctx->d_blk0, ctx->d_blk0,
                                                                ctx->reduce_blocks, ctx->d_scalar2);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            return -1;
        }
        double pair[2] = {0.0, 0.0};
        if (!P21_CUDA_OK(cudaMemcpyAsync(pair, ctx->d_scalar2, 2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream))) {
            return -1;
        }
        /* Wait to read pair after sync below */
        size_t fbytes = ctx->total * sizeof(float);
        bool ok = P21_CUDA_OK(cudaMemcpyAsync(box->z_reion, ctx->d_z_reion, fbytes,
                                              cudaMemcpyDeviceToHost, stream)) &&
                  P21_CUDA_OK(cudaMemcpyAsync(box->neutral_fraction, ctx->d_neutral_fraction,
                                              fbytes, cudaMemcpyDeviceToHost, stream));
        if (ok) p21_cuda_stats_add_d2h(2 * fbytes);
        if (ok && !ctx->minimize_memory) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(box->kinetic_temperature, ctx->d_kinetic_temperature,
                                             fbytes, cudaMemcpyDeviceToHost, stream));
            if (ok) p21_cuda_stats_add_d2h(fbytes);
        }
        if (!ok || !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
            return -1;
        }
        h_xh = pair[0] / (double)ctx->total;
    } else {
        size_t fbytes = ctx->total * sizeof(float);
        bool ok = P21_CUDA_OK(cudaMemcpyAsync(box->z_reion, ctx->d_z_reion, fbytes,
                                              cudaMemcpyDeviceToHost, stream)) &&
                  P21_CUDA_OK(cudaMemcpyAsync(box->neutral_fraction, ctx->d_neutral_fraction,
                                              fbytes, cudaMemcpyDeviceToHost, stream));
        if (ok) p21_cuda_stats_add_d2h(2 * fbytes);
        if (ok && !ctx->minimize_memory) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(box->kinetic_temperature, ctx->d_kinetic_temperature,
                                             fbytes, cudaMemcpyDeviceToHost, stream));
            if (ok) p21_cuda_stats_add_d2h(fbytes);
        }
        if (!ok || !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
            return -1;
        }
    }

    size_t fbytes = ctx->total * sizeof(float);
    if (p21_cuda_registry_put_owned(box->z_reion, ctx->d_z_reion, fbytes, stream)) {
        ctx->d_z_reion = nullptr;
    }
    if (p21_cuda_registry_put_owned(box->neutral_fraction, ctx->d_neutral_fraction, fbytes,
                                    stream)) {
        ctx->d_neutral_fraction = nullptr;
    }

    *global_xH_out = h_xh;
    p21_cuda_stats_mut()->ionization_runs++;
    return 0;
}

extern "C" int p21_cuda_ion_prepare_r2c(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    if (!ctx || !params) {
        return -1;
    }
    cudaStream_t stream = ctx->stream;
    const size_t n = ctx->total;
    const size_t fbytes = n * sizeof(float);
    const size_t cbytes = ctx->k_total * sizeof(cufftComplex);
    const int grid = (int)((n + kBlockThreads - 1) / kBlockThreads);
    const int kgrid = (int)((ctx->k_total + kBlockThreads - 1) / kBlockThreads);
    const float inv_n = 1.0f / (float)n;
    const bool need_filter_recomb = (params->filter_recombinations && !ctx->prev_first_z);
    const bool need_prev_deltax =
        (!params->lagrangian_source_grids && ctx->use_mcgs && !ctx->prev_first_z);

    size_t n_unfilt = 1ULL;
    size_t n_filt = 1ULL;
    if (ctx->use_ts_fluct) {
        n_unfilt++;
        n_filt++;
    }
    if (need_filter_recomb) {
        n_unfilt++;
        n_filt++;
    }
    if (params->lagrangian_source_grids) {
        n_unfilt++;
        n_filt++;
        if (ctx->recomb_model != RECOMB_MODEL_NO) {
            n_unfilt++;
            n_filt++;
        }
    } else {
        if (ctx->use_reion_fb) n_unfilt++;
        if (ctx->use_mcgs) n_unfilt++;
        if (need_prev_deltax) {
            n_unfilt++;
            n_filt++;
        }
    }

    ctx->mgpu_mode = p21_mgpu_eor_enabled(n) && ctx->mgpu_grid.init(ctx->nx, ctx->ny, ctx->nz);

    int spill_env = read_ion_spill_kspace_env();
    if (spill_env >= 0) {
        ctx->spill_kspace = (spill_env > 0);
    } else {
        p21_cuda_pool_trim();
        size_t need_bytes = ctx->mgpu_mode
                                ? (n_filt * fbytes + fbytes +
                                   (n_unfilt + 1) * cbytes /
                                       (size_t)std::max(1, ctx->mgpu_grid.n_gpus))
                                : (n_unfilt * cbytes + n_filt * fbytes + cbytes + fbytes);
        size_t free_bytes =
            ctx->mgpu_mode ? p21_mgpu_min_free_mem(ctx->mgpu_grid) : p21_cuda_free_mem_bytes();
        ctx->spill_kspace = (need_bytes + (2ULL << 30) > free_bytes);
    }

    ctx->d_real_work = (float *)p21_cuda_pool_alloc(fbytes, stream);
    bool ok = (ctx->d_real_work != nullptr);
    auto alloc_yslab = [&](cufftComplex *(&ys)[kMgpuMaxDevices]) -> bool {
        for (int g = 0; g < ctx->mgpu_grid.n_gpus; ++g) {
            ys[g] = (cufftComplex *)p21_mgpu_malloc(
                ctx->mgpu_grid.devs[g], ctx->mgpu_grid.cplx_elems_y(g) * sizeof(cufftComplex),
                ctx->mgpu_grid.streams[g]);
            if (!ys[g]) {
                return false;
            }
        }
        return true;
    };
    if (ctx->mgpu_mode) {
        if (ok) ok = alloc_yslab(ctx->y_work);
    } else {
        ctx->d_k_work = (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream);
        ok = ok && (ctx->d_k_work != nullptr);
    }

    auto alloc_unfilt = [&](cufftComplex *&d_ptr, cufftComplex *(&y_ptr)[kMgpuMaxDevices],
                            cufftComplex *&h_ptr) {
        if (!ok) return;
        if (ctx->spill_kspace) {
            h_ptr = (cufftComplex *)malloc(cbytes);
            if (!h_ptr) ok = false;
        } else if (ctx->mgpu_mode) {
            if (!alloc_yslab(y_ptr)) ok = false;
        } else {
            d_ptr = (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream);
            if (!d_ptr) ok = false;
        }
    };

    alloc_unfilt(ctx->d_k_deltax_unfilt, ctx->y_deltax_unfilt, ctx->h_k_deltax_unfilt);

    if (ok && ctx->use_ts_fluct) {
        alloc_unfilt(ctx->d_k_xe_unfilt, ctx->y_xe_unfilt, ctx->h_k_xe_unfilt);
    }
    if (ok && need_filter_recomb) {
        alloc_unfilt(ctx->d_k_nrec_unfilt, ctx->y_nrec_unfilt, ctx->h_k_nrec_unfilt);
    }
    if (ok && params->lagrangian_source_grids) {
        alloc_unfilt(ctx->d_k_stars_unfilt, ctx->y_stars_unfilt, ctx->h_k_stars_unfilt);
        if (ok && ctx->recomb_model != RECOMB_MODEL_NO) {
            alloc_unfilt(ctx->d_k_sfr_unfilt, ctx->y_sfr_unfilt, ctx->h_k_sfr_unfilt);
        }
    } else if (ok) {
        if (ctx->use_reion_fb) {
            alloc_unfilt(ctx->d_k_mturn_acg_unfilt, ctx->y_mturn_acg_unfilt,
                         ctx->h_k_mturn_acg_unfilt);
        }
        if (ok && ctx->use_mcgs) {
            alloc_unfilt(ctx->d_k_mturn_mcg_unfilt, ctx->y_mturn_mcg_unfilt,
                         ctx->h_k_mturn_mcg_unfilt);
            if (ok && need_prev_deltax) {
                alloc_unfilt(ctx->d_k_prev_deltax_unfilt, ctx->y_prev_deltax_unfilt,
                             ctx->h_k_prev_deltax_unfilt);
            }
        }
    }

    if (!ok) {
        free_ion_rloop_scratch(ctx);
        return -1;
    }

    auto store_k_result = [&](cufftComplex *d_k_buf, cufftComplex *const *y_buf,
                              cufftComplex *h_k_dst) -> bool {
        if (!ctx->spill_kspace) {
            return true;
        }
        if (ctx->mgpu_mode) {
            size_t off = 0;
            for (int g = 0; g < ctx->mgpu_grid.n_gpus; ++g) {
                size_t elems = ctx->mgpu_grid.cplx_elems_y(g);
                if (!P21_CUDA_OK(cudaSetDevice(ctx->mgpu_grid.devs[g])) ||
                    !P21_CUDA_OK(cudaMemcpyAsync(h_k_dst + off, y_buf[g],
                                                 elems * sizeof(cufftComplex),
                                                 cudaMemcpyDeviceToHost,
                                                 ctx->mgpu_grid.streams[g])) ||
                    !P21_CUDA_OK(cudaStreamSynchronize(ctx->mgpu_grid.streams[g]))) {
                    cudaSetDevice(ctx->mgpu_grid.primary_dev);
                    return false;
                }
                off += elems;
            }
            cudaSetDevice(ctx->mgpu_grid.primary_dev);
            return true;
        }
        return P21_CUDA_OK(cudaMemcpyAsync(h_k_dst, d_k_buf, cbytes, cudaMemcpyDeviceToHost,
                                           stream)) &&
               P21_CUDA_OK(cudaStreamSynchronize(stream));
    };

    auto prepare_grid = [&](const float *d_src, cufftComplex *d_k_out,
                            cufftComplex *const *y_k_out, cufftComplex *h_k_out, double scale,
                            double lo, double hi) -> bool {
        clip_and_scale_kernel<<<grid, kBlockThreads, 0, stream>>>(d_src, ctx->d_real_work, n,
                                                                  scale, lo, hi);
        if (!P21_CUDA_OK(cudaGetLastError())) return false;
        if (ctx->mgpu_mode) {
            cufftComplex *const *y_target = ctx->spill_kspace ? ctx->y_work : y_k_out;
            if (!p21_mgpu_r2c_from_device(ctx->mgpu_grid, ctx->d_real_work, stream, (double)inv_n,
                                          const_cast<cufftComplex **>(y_target), nullptr)) {
                return false;
            }
            return store_k_result(nullptr, y_target, h_k_out);
        }
        cufftComplex *k_target = ctx->spill_kspace ? ctx->d_k_work : d_k_out;
        if (!p21_cuda_fft_r2c(ctx->d_real_work, k_target, ctx->nx, ctx->ny, ctx->nz, false,
                              stream)) {
            return false;
        }
        scale_complex_kernel<<<kgrid, kBlockThreads, 0, stream>>>(k_target, ctx->k_total, inv_n);
        if (!P21_CUDA_OK(cudaGetLastError())) return false;
        return store_k_result(k_target, nullptr, h_k_out);
    };

    auto prepare_precomputed = [&](const float *d_src_real, cufftComplex *d_k_out,
                                   cufftComplex *const *y_k_out,
                                   cufftComplex *h_k_out) -> bool {
        if (ctx->mgpu_mode) {
            cufftComplex *const *y_target = ctx->spill_kspace ? ctx->y_work : y_k_out;
            if (!p21_mgpu_r2c_from_device(ctx->mgpu_grid, d_src_real, stream, (double)inv_n,
                                          const_cast<cufftComplex **>(y_target), nullptr)) {
                return false;
            }
            return store_k_result(nullptr, y_target, h_k_out);
        }
        cufftComplex *k_target = ctx->spill_kspace ? ctx->d_k_work : d_k_out;
        if (!p21_cuda_fft_r2c(d_src_real, k_target, ctx->nx, ctx->ny, ctx->nz, false, stream)) {
            return false;
        }
        scale_complex_kernel<<<kgrid, kBlockThreads, 0, stream>>>(k_target, ctx->k_total, inv_n);
        if (!P21_CUDA_OK(cudaGetLastError())) return false;
        return store_k_result(k_target, nullptr, h_k_out);
    };

    ok = prepare_grid(ctx->d_dens, ctx->d_k_deltax_unfilt, ctx->y_deltax_unfilt,
                      ctx->h_k_deltax_unfilt, params->photoncons_adjustment_factor, -1.0, 1e6);
    if (ok && params->lagrangian_source_grids) {
        ok = prepare_grid(ctx->d_stars_in, ctx->d_k_stars_unfilt, ctx->y_stars_unfilt,
                          ctx->h_k_stars_unfilt, 1.0, 0.0, 1e20);
        if (ok && ctx->recomb_model != RECOMB_MODEL_NO) {
            ok = prepare_grid(ctx->d_sfr_in, ctx->d_k_sfr_unfilt, ctx->y_sfr_unfilt,
                              ctx->h_k_sfr_unfilt, 1.0, 0.0, 1e20);
        }
    } else if (ok) {
        if (ctx->use_reion_fb) {
            ok = prepare_precomputed(ctx->d_mturn_acg_real, ctx->d_k_mturn_acg_unfilt,
                                     ctx->y_mturn_acg_unfilt, ctx->h_k_mturn_acg_unfilt);
        }
        if (ok && ctx->use_mcgs) {
            if (need_prev_deltax) {
                ok = prepare_grid(ctx->d_prev_dens, ctx->d_k_prev_deltax_unfilt,
                                  ctx->y_prev_deltax_unfilt, ctx->h_k_prev_deltax_unfilt, 1.0,
                                  -1.0, 1e6);
            }
            if (ok) {
                ok = prepare_precomputed(ctx->d_mturn_mcg_real, ctx->d_k_mturn_mcg_unfilt,
                                         ctx->y_mturn_mcg_unfilt, ctx->h_k_mturn_mcg_unfilt);
            }
        }
    }
    if (ok && ctx->use_ts_fluct) {
        ok = prepare_grid(ctx->d_ts_xe, ctx->d_k_xe_unfilt, ctx->y_xe_unfilt, ctx->h_k_xe_unfilt,
                          1.0, 0.0, 1.0);
    }
    if (ok && need_filter_recomb) {
        ok = prepare_grid(ctx->d_prev_nrec, ctx->d_k_nrec_unfilt, ctx->y_nrec_unfilt,
                          ctx->h_k_nrec_unfilt, 1.0, 0.0, 1e20);
    }

    p21_cuda_pool_free(ctx->d_real_work, stream);
    ctx->d_real_work = nullptr;

    /* Allocate filtered real-space and EDT scratch only after R2C frees d_real_work. */
    if (ok) {
        ctx->d_deltax_filt = (float *)p21_cuda_pool_alloc(fbytes, stream);
        if (!ctx->d_deltax_filt) ok = false;
    }
    if (ok && ctx->use_ts_fluct) {
        ctx->d_xe_filt = (float *)p21_cuda_pool_alloc(fbytes, stream);
        if (!ctx->d_xe_filt) ok = false;
    }
    if (ok && need_filter_recomb) {
        ctx->d_nrec_filt = (float *)p21_cuda_pool_alloc(fbytes, stream);
        if (!ctx->d_nrec_filt) ok = false;
    }
    if (ok && params->lagrangian_source_grids) {
        ctx->d_stars_filt = (float *)p21_cuda_pool_alloc(fbytes, stream);
        if (!ctx->d_stars_filt) ok = false;
        if (ok && ctx->recomb_model != RECOMB_MODEL_NO) {
            ctx->d_sfr_filt = (float *)p21_cuda_pool_alloc(fbytes, stream);
            if (!ctx->d_sfr_filt) ok = false;
        }
    } else if (ok && ctx->use_mcgs && need_prev_deltax) {
        ctx->d_prev_deltax_filt = (float *)p21_cuda_pool_alloc(fbytes, stream);
        if (!ctx->d_prev_deltax_filt) ok = false;
    }
    if (ok && ctx->ionise_entire_sphere) {
        size_t ibytes = n * sizeof(int);
        ctx->d_sphere_mask = (uint8_t *)p21_cuda_pool_alloc(n * sizeof(uint8_t), stream);
        ctx->d_edt_a = (int *)p21_cuda_pool_alloc(ibytes, stream);
        ctx->d_edt_b = (int *)p21_cuda_pool_alloc(ibytes, stream);
        ctx->d_n_centers = (int *)p21_cuda_pool_alloc(sizeof(int), stream);
        if (!ctx->d_sphere_mask || !ctx->d_edt_a || !ctx->d_edt_b || !ctx->d_n_centers) ok = false;
    }
    return ok ? 0 : -1;
}

extern "C" int p21_cuda_ion_filter_and_extrema(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                               const P21IonRadiusSpec *rspec, double *min_dens,
                                               double *max_dens, double *prev_min_dens,
                                               double *prev_max_dens) {
    if (!ctx || !params || !rspec) {
        return -1;
    }
    cudaStream_t stream = ctx->stream;
    const size_t n = ctx->total;
    const size_t cbytes = ctx->k_total * sizeof(cufftComplex);
    const double box_len = simulation_options_global->BOX_LEN;
    const double ncf = simulation_options_global->NON_CUBIC_FACTOR;
    const bool need_filter_recomb = (params->filter_recombinations && !ctx->prev_first_z);
    const bool need_prev_deltax =
        (!params->lagrangian_source_grids && ctx->use_mcgs && !ctx->prev_first_z);

    auto filter_and_c2r = [&](const cufftComplex *d_k_unfilt, cufftComplex *const *y_k_unfilt,
                              const cufftComplex *h_k_unfilt, float *d_real_out, int ftype,
                              float R_param) -> bool {
        if (ctx->mgpu_mode) {
            cufftComplex *const *src_y = y_k_unfilt;
            if (ctx->spill_kspace) {
                if (!h_k_unfilt) {
                    return false;
                }
                size_t off = 0;
                for (int g = 0; g < ctx->mgpu_grid.n_gpus; ++g) {
                    size_t elems = ctx->mgpu_grid.cplx_elems_y(g);
                    if (!P21_CUDA_OK(cudaSetDevice(ctx->mgpu_grid.devs[g])) ||
                        !P21_CUDA_OK(cudaMemcpyAsync(ctx->y_work[g], h_k_unfilt + off,
                                                     elems * sizeof(cufftComplex),
                                                     cudaMemcpyHostToDevice,
                                                     ctx->mgpu_grid.streams[g])) ||
                        !P21_CUDA_OK(cudaStreamSynchronize(ctx->mgpu_grid.streams[g]))) {
                        cudaSetDevice(ctx->mgpu_grid.primary_dev);
                        return false;
                    }
                    off += elems;
                }
                cudaSetDevice(ctx->mgpu_grid.primary_dev);
                src_y = ctx->y_work;
            }
            if (rspec->R_index > 0) {
                if (!p21_mgpu_filter_yslabs(ctx->mgpu_grid, src_y, ctx->y_work, box_len, ncf,
                                            ftype, (float)rspec->R, R_param, 0.0f, 1.0f)) {
                    return false;
                }
            } else if (!ctx->spill_kspace) {
                for (int g = 0; g < ctx->mgpu_grid.n_gpus; ++g) {
                    size_t elems = ctx->mgpu_grid.cplx_elems_y(g);
                    if (!P21_CUDA_OK(cudaSetDevice(ctx->mgpu_grid.devs[g])) ||
                        !P21_CUDA_OK(cudaMemcpyAsync(ctx->y_work[g], y_k_unfilt[g],
                                                     elems * sizeof(cufftComplex),
                                                     cudaMemcpyDeviceToDevice,
                                                     ctx->mgpu_grid.streams[g])) ||
                        !P21_CUDA_OK(cudaStreamSynchronize(ctx->mgpu_grid.streams[g]))) {
                        cudaSetDevice(ctx->mgpu_grid.primary_dev);
                        return false;
                    }
                }
                cudaSetDevice(ctx->mgpu_grid.primary_dev);
            }
            return p21_mgpu_c2r_to_device(ctx->mgpu_grid, ctx->y_work, d_real_out, stream, nullptr);
        }

        const cufftComplex *src_for_filter = d_k_unfilt;
        if (ctx->spill_kspace) {
            if (!h_k_unfilt ||
                !P21_CUDA_OK(cudaMemcpyAsync(ctx->d_k_work, h_k_unfilt, cbytes,
                                             cudaMemcpyHostToDevice, stream))) {
                return false;
            }
            src_for_filter = ctx->d_k_work;
        }
        if (rspec->R_index > 0) {
            if (!p21_cuda_copy_and_filter_box(src_for_filter, ctx->d_k_work, ctx->nx, ctx->ny,
                                              ctx->nz, box_len, ncf, ftype, (float)rspec->R,
                                              R_param, 0.0f, 1.0f, stream)) {
                return false;
            }
        } else if (!ctx->spill_kspace) {
            if (!P21_CUDA_OK(cudaMemcpyAsync(ctx->d_k_work, d_k_unfilt, cbytes,
                                             cudaMemcpyDeviceToDevice, stream))) {
                return false;
            }
        }
        return p21_cuda_fft_c2r(ctx->d_k_work, d_real_out, ctx->nx, ctx->ny, ctx->nz, false,
                                stream);
    };

    bool ok = filter_and_c2r(ctx->d_k_deltax_unfilt, ctx->y_deltax_unfilt, ctx->h_k_deltax_unfilt,
                             ctx->d_deltax_filt, params->hii_filter, 0.0f);
    if (ok && ctx->use_ts_fluct) {
        ok = filter_and_c2r(ctx->d_k_xe_unfilt, ctx->y_xe_unfilt, ctx->h_k_xe_unfilt,
                            ctx->d_xe_filt, params->hii_filter, 0.0f);
    }
    if (ok && need_filter_recomb) {
        ok = filter_and_c2r(ctx->d_k_nrec_unfilt, ctx->y_nrec_unfilt, ctx->h_k_nrec_unfilt,
                            ctx->d_nrec_filt, params->hii_filter, 0.0f);
    }
    if (ok && params->lagrangian_source_grids) {
        int hf_type = ctx->use_exp_filter ? FILTER_EXP_MFP : params->hii_filter;
        float mfp_p = (float)params->mfp_meandens;
        ok = filter_and_c2r(ctx->d_k_stars_unfilt, ctx->y_stars_unfilt, ctx->h_k_stars_unfilt,
                            ctx->d_stars_filt, hf_type, mfp_p);
        if (ok && ctx->recomb_model != RECOMB_MODEL_NO) {
            ok = filter_and_c2r(ctx->d_k_sfr_unfilt, ctx->y_sfr_unfilt, ctx->h_k_sfr_unfilt,
                                ctx->d_sfr_filt, hf_type, mfp_p);
        }
    } else if (ok) {
        if (ctx->use_reion_fb) {
            ok = filter_and_c2r(ctx->d_k_mturn_acg_unfilt, ctx->y_mturn_acg_unfilt,
                                ctx->h_k_mturn_acg_unfilt, ctx->d_mturn_acg_real,
                                params->hii_filter, 0.0f);
        }
        if (ok && ctx->use_mcgs) {
            if (need_prev_deltax) {
                ok = filter_and_c2r(ctx->d_k_prev_deltax_unfilt, ctx->y_prev_deltax_unfilt,
                                    ctx->h_k_prev_deltax_unfilt, ctx->d_prev_deltax_filt,
                                    params->hii_filter, 0.0f);
            }
            if (ok) {
                ok = filter_and_c2r(ctx->d_k_mturn_mcg_unfilt, ctx->y_mturn_mcg_unfilt,
                                    ctx->h_k_mturn_mcg_unfilt, ctx->d_mturn_mcg_real,
                                    params->hii_filter, 0.0f);
            }
        }
    }
    if (!ok) {
        return -1;
    }

    if (!params->lagrangian_source_grids) {
        const int blocks = ctx->reduce_blocks;
        const int grid = (int)((n + kBlockThreads - 1) / kBlockThreads);
        float init_val = 0.0f;

        if (!P21_CUDA_OK(cudaMemcpyAsync(&init_val, ctx->d_deltax_filt, sizeof(float),
                                         cudaMemcpyDeviceToHost, stream)) ||
            !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
            return -1;
        }
        clip_and_extrema_kernel<<<blocks, kBlockThreads, 0, stream>>>(ctx->d_deltax_filt, n, -1.0,
                                                                      1e6, ctx->d_blk0, ctx->d_blk1);
        ok = P21_CUDA_OK(cudaGetLastError());
        if (ok) {
            finish_minmax_kernel<<<1, kBlockThreads, 0, stream>>>(ctx->d_blk0, ctx->d_blk1, blocks,
                                                                  init_val, ctx->d_scalar2);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        double mm[2] = {0.0, 0.0};
        if (ok) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(mm, ctx->d_scalar2, 2 * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream)) &&
                 P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
        if (!ok) return -1;
        *min_dens = mm[0] - 0.001;
        *max_dens = mm[1] + 0.001;

        if (ctx->use_reion_fb) {
            clip_simple_kernel<<<grid, kBlockThreads, 0, stream>>>(ctx->d_mturn_acg_real, n, 0.0f,
                                                                   10.0f);
            if (!P21_CUDA_OK(cudaGetLastError())) return -1;
        }
        if (ctx->use_mcgs) {
            if (need_prev_deltax) {
                if (!P21_CUDA_OK(cudaMemcpyAsync(&init_val, ctx->d_prev_deltax_filt, sizeof(float),
                                                 cudaMemcpyDeviceToHost, stream)) ||
                    !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                    return -1;
                }
                clip_and_extrema_kernel<<<blocks, kBlockThreads, 0, stream>>>(
                    ctx->d_prev_deltax_filt, n, -1.0, 1e6, ctx->d_blk0, ctx->d_blk1);
                ok = P21_CUDA_OK(cudaGetLastError());
                if (ok) {
                    finish_minmax_kernel<<<1, kBlockThreads, 0, stream>>>(
                        ctx->d_blk0, ctx->d_blk1, blocks, init_val, ctx->d_scalar2);
                    ok = P21_CUDA_OK(cudaGetLastError());
                }
                if (ok) {
                    ok = P21_CUDA_OK(cudaMemcpyAsync(mm, ctx->d_scalar2, 2 * sizeof(double),
                                                     cudaMemcpyDeviceToHost, stream)) &&
                         P21_CUDA_OK(cudaStreamSynchronize(stream));
                }
                if (!ok) return -1;
                *prev_min_dens = mm[0] - 0.001;
                *prev_max_dens = mm[1] + 0.001;
            } else {
                *prev_min_dens = -1.001;
                *prev_max_dens = -0.999;
            }

            clip_simple_kernel<<<grid, kBlockThreads, 0, stream>>>(ctx->d_mturn_mcg_real, n, 0.0f,
                                                                   10.0f);
            if (!P21_CUDA_OK(cudaGetLastError())) return -1;
        }
    }

    return 0;
}

extern "C" int p21_cuda_ion_calc_cond_nion(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                           P21IonRadiusSpec *rspec, int need_prev_ion) {
    if (!ctx || !params || !rspec) {
        return -1;
    }
    cudaStream_t stream = ctx->stream;
    int fc_r_idx = params->need_mcgs_nion ? rspec->R_index : 0;

    P21DevTable1Df t_nion_1d{}, t_nion_1d_prev{};
    P21DevTable2Df t_nion_2d_acg{}, t_nion_2d_mcg{}, t_nion_2d_acg_prev{}, t_nion_2d_mcg_prev{};
    bool ok = true;
    if (!params->lagrangian_source_grids) {
        P21ConditionalTables htbl{};
        p21_get_conditional_tables(&htbl);
        if (ctx->use_reion_fb && params->mass_dep_zeta) {
            ok = ok && p21_upload_rg2d_f(htbl.nion_2d_acg, &t_nion_2d_acg, stream);
        } else {
            ok = ok && p21_upload_rg1d_f(htbl.nion_1d_acg, &t_nion_1d, stream);
        }
        if (ctx->use_mcgs) {
            ok = ok && p21_upload_rg2d_f(htbl.nion_2d_mcg, &t_nion_2d_mcg, stream);
            if (need_prev_ion) {
                if (ctx->use_reion_fb && params->mass_dep_zeta) {
                    ok = ok && p21_upload_rg2d_f(htbl.nion_2d_acg_prev, &t_nion_2d_acg_prev, stream);
                } else {
                    ok = ok && p21_upload_rg1d_f(htbl.nion_1d_acg_prev, &t_nion_1d_prev, stream);
                }
                ok = ok && p21_upload_rg2d_f(htbl.nion_2d_mcg_prev, &t_nion_2d_mcg_prev, stream);
            }
        }
    }

    if (ok) {
        ok = P21_CUDA_OK(cudaMemsetAsync(ctx->d_nonfinite, 0, sizeof(int), stream));
    }
    if (ok) {
        calc_cond_nion_kernel<<<ctx->reduce_blocks, kBlockThreads, 0, stream>>>(
            ctx->total, fc_r_idx, params->lagrangian_source_grids, params->need_mcgs_nion,
            need_prev_ion, ctx->use_reion_fb, ctx->use_mcgs, params->mass_dep_zeta,
            params->filter_recombinations, ctx->use_ts_fluct,
            (ctx->recomb_model != RECOMB_MODEL_NO) ? 1 : 0, params->mturn_acg_homogeneous,
            params->atomic_cooling_threshold, params->nion_prefactor_acg,
            params->nion_prefactor_mcg, (float)(-1.0 + FRACT_FLOAT_ERR), ctx->prev_first_z,
            t_nion_1d, t_nion_1d_prev, t_nion_2d_acg, t_nion_2d_mcg, t_nion_2d_acg_prev,
            t_nion_2d_mcg_prev, ctx->d_deltax_filt, ctx->d_nrec_filt, ctx->d_xe_filt,
            ctx->d_stars_filt, ctx->d_sfr_filt, ctx->d_mturn_acg_real, ctx->d_mturn_mcg_real,
            ctx->d_prev_deltax_filt, ctx->d_prev_nion_acg, ctx->d_prev_nion_mcg,
            ctx->d_nion_cond_acg, ctx->d_nion_cond_mcg, ctx->d_blk0, ctx->d_blk1,
            ctx->d_nonfinite);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        finish_pair_sum_kernel<<<1, kBlockThreads, 0, stream>>>(ctx->d_blk0, ctx->d_blk1,
                                                                ctx->reduce_blocks, ctx->d_scalar2);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    double sums[2] = {0.0, 0.0};
    int h_nonfinite = 0;
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(sums, ctx->d_scalar2, 2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(&h_nonfinite, ctx->d_nonfinite, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_free_dev_table(&t_nion_1d, stream);
    p21_free_dev_table(&t_nion_1d_prev, stream);
    p21_free_dev_table(&t_nion_2d_acg, stream);
    p21_free_dev_table(&t_nion_2d_mcg, stream);
    p21_free_dev_table(&t_nion_2d_acg_prev, stream);
    p21_free_dev_table(&t_nion_2d_mcg_prev, stream);

    if (!ok) {
        return -1;
    }
    if (h_nonfinite != 0 || !isfinite(sums[0]) || !isfinite(sums[1])) {
        return InfinityorNaNError;
    }

    rspec->nion_conditional_ave_acg = sums[0] / (double)ctx->total;
    rspec->nion_conditional_ave_mcg = sums[1] / (double)ctx->total;
    return 0;
}

extern "C" int p21_cuda_ion_find_regions(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                         const P21IonRadiusSpec *rspec, double nion_uncond_acg,
                                         double nion_uncond_mcg, double nion_limit_acg,
                                         double nion_limit_mcg, float prev_homo_nrec) {
    if (!ctx || !params || !rspec) {
        return -1;
    }
    cudaStream_t stream = ctx->stream;
    const size_t n = ctx->total;
    const int grid = (int)((n + kBlockThreads - 1) / kBlockThreads);
    const int fc_r_idx = params->need_mcgs_nion ? rspec->R_index : 0;

    double mean_fix_acg = 1.0, mean_fix_mcg = 1.0;
    if (params->fix_mean) {
        mean_fix_acg = nion_uncond_acg / rspec->nion_conditional_ave_acg;
        if (ctx->use_mcgs) {
            mean_fix_mcg = nion_uncond_mcg / rspec->nion_conditional_ave_mcg;
        }
    }

    if (!ctx->ionise_entire_sphere) {
        find_ionised_regions_center_kernel<<<grid, kBlockThreads, 0, stream>>>(
            n, rspec->R_index, fc_r_idx, params->lagrangian_source_grids, params->need_mcgs_nion,
            params->mass_dep_zeta, ctx->use_mcgs, ctx->recomb_model, ctx->cell_recomb,
            ctx->use_ts_fluct, ctx->minimize_memory, rspec->R, params->redshift,
            params->photoncons_adjustment_factor, params->gamma_prefactor, params->T_re,
            params->TK_nofluct, params->adia_TK_term, mean_fix_acg, mean_fix_mcg, nion_limit_acg,
            nion_limit_mcg, prev_homo_nrec, FRACT_FLOAT_ERR, TINY, ctx->d_dens, ctx->d_deltax_filt,
            ctx->d_stars_filt, ctx->d_sfr_filt, ctx->d_nion_cond_acg, ctx->d_nion_cond_mcg,
            ctx->d_prev_nrec, ctx->d_nrec_filt, ctx->d_xe_filt, ctx->d_ts_tk, ctx->d_prev_zre,
            ctx->d_neutral_fraction, ctx->d_ionisation_rate_G12, ctx->d_mean_free_path,
            ctx->d_z_reion, ctx->d_kinetic_temperature);
        return P21_CUDA_OK(cudaGetLastError()) ? 0 : -1;
    }

    /* IONISE_ENTIRE_SPHERE mode: exact 3D periodic Euclidean Distance Transform */
    if (!P21_CUDA_OK(cudaMemsetAsync(ctx->d_n_centers, 0, sizeof(int), stream))) {
        return -1;
    }
    find_ionised_centers_sphere_kernel<<<grid, kBlockThreads, 0, stream>>>(
        n, rspec->R_index, fc_r_idx, params->lagrangian_source_grids, params->need_mcgs_nion,
        params->mass_dep_zeta, ctx->use_mcgs, ctx->recomb_model, ctx->cell_recomb,
        ctx->use_ts_fluct, ctx->minimize_memory, rspec->R, params->redshift,
        params->photoncons_adjustment_factor, params->gamma_prefactor, mean_fix_acg, mean_fix_mcg,
        nion_limit_acg, nion_limit_mcg, prev_homo_nrec, FRACT_FLOAT_ERR, ctx->d_dens,
        ctx->d_deltax_filt, ctx->d_stars_filt, ctx->d_sfr_filt, ctx->d_nion_cond_acg,
        ctx->d_nion_cond_mcg, ctx->d_prev_nrec, ctx->d_nrec_filt, ctx->d_xe_filt, ctx->d_prev_zre,
        ctx->d_neutral_fraction, ctx->d_ionisation_rate_G12, ctx->d_mean_free_path, ctx->d_z_reion,
        ctx->d_edt_a, ctx->d_sphere_mask, ctx->d_n_centers);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }

    int h_n_centers = 0;
    if (!P21_CUDA_OK(cudaMemcpyAsync(&h_n_centers, ctx->d_n_centers, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return -1;
    }

    if (h_n_centers > 0) {
        double box_len = simulation_options_global->BOX_LEN;
        float r_pix_f = (float)(rspec->R / box_len) * (float)ctx->nx;
        float rsq_curr_index = r_pix_f * r_pix_f;
        int r_ceil = (int)ceil((double)r_pix_f);

        auto run_axis = [&](int axis, const int *in_buf, int *out_buf) -> bool {
            int L = (axis == 0) ? ctx->nx : ((axis == 1) ? ctx->ny : ctx->nz);
            int r_lim = (r_ceil < L / 2) ? r_ceil : (L / 2);
            if (L <= 1024) {
                int num_lines =
                    (axis == 0) ? (ctx->ny * ctx->nz)
                                : ((axis == 1) ? (ctx->nx * ctx->nz) : (ctx->nx * ctx->ny));
                int tpb = (L < kBlockThreads) ? L : kBlockThreads;
                if (tpb < 1) tpb = 1;
                size_t smem = (size_t)L * sizeof(int);
                edt_1d_periodic_shared_kernel<<<num_lines, tpb, smem, stream>>>(
                    in_buf, out_buf, ctx->nx, ctx->ny, ctx->nz, axis, r_lim);
            } else {
                edt_1d_periodic_global_kernel<<<grid, kBlockThreads, 0, stream>>>(
                    in_buf, out_buf, ctx->nx, ctx->ny, ctx->nz, axis, r_lim);
            }
            return P21_CUDA_OK(cudaGetLastError());
        };

        if (!run_axis(0, ctx->d_edt_a, ctx->d_edt_b) ||
            !run_axis(1, ctx->d_edt_b, ctx->d_edt_a) ||
            !run_axis(2, ctx->d_edt_a, ctx->d_edt_b)) {
            return -1;
        }
        apply_sphere_edt_kernel<<<grid, kBlockThreads, 0, stream>>>(n, ctx->d_edt_b, rsq_curr_index,
                                                                    ctx->d_neutral_fraction);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            return -1;
        }
    }

    if (rspec->R_index == 0) {
        assign_partial_ionization_kernel<<<grid, kBlockThreads, 0, stream>>>(
            n, params->lagrangian_source_grids, params->need_mcgs_nion, params->mass_dep_zeta,
            ctx->use_mcgs, ctx->use_ts_fluct, ctx->minimize_memory,
            params->photoncons_adjustment_factor, params->T_re, params->TK_nofluct,
            params->adia_TK_term, mean_fix_acg, mean_fix_mcg, nion_limit_acg, nion_limit_mcg, TINY,
            ctx->d_dens, ctx->d_deltax_filt, ctx->d_stars_filt, ctx->d_nion_cond_acg,
            ctx->d_nion_cond_mcg, ctx->d_xe_filt, ctx->d_ts_tk, ctx->d_neutral_fraction,
            ctx->d_kinetic_temperature);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            return -1;
        }
    }

    return 0;
}

extern "C" int p21_cuda_ion_set_ionized_temps(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    if (!ctx || !params) {
        return -1;
    }
    free_ion_rloop_scratch(ctx);
    if (ctx->minimize_memory) {
        return 0;
    }
    cudaStream_t stream = ctx->stream;
    const int grid = (int)((ctx->total + kBlockThreads - 1) / kBlockThreads);
    if (!P21_CUDA_OK(cudaMemsetAsync(ctx->d_nonfinite, 0, sizeof(int), stream))) {
        return -1;
    }
    set_ionized_temps_kernel<<<grid, kBlockThreads, 0, stream>>>(
        ctx->total, ctx->use_ts_fluct, (float)params->stored_redshift, (float)params->T_re,
        params->TK_nofluct, params->adia_TK_term, TINY, ctx->d_z_reion, ctx->d_neutral_fraction,
        ctx->d_dens, ctx->d_ts_tk, ctx->d_kinetic_temperature, ctx->d_nonfinite);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }
    int h_nonfinite = 0;
    if (!P21_CUDA_OK(cudaMemcpyAsync(&h_nonfinite, ctx->d_nonfinite, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return -1;
    }
    if (h_nonfinite != 0) {
        return InfinityorNaNError;
    }
    return 0;
}

extern "C" int p21_cuda_ion_global_means(P21CudaIonCtx *ctx, double *global_xH_out,
                                         float *global_gamma12_out) {
    if (!ctx || !global_xH_out || !global_gamma12_out) {
        return -1;
    }
    free_ion_rloop_scratch(ctx);
    cudaStream_t stream = ctx->stream;
    int homo = (ctx->recomb_model == RECOMB_MODEL_HOMOGENEOUS) ? 1 : 0;
    ion_global_means_kernel<<<ctx->reduce_blocks, kBlockThreads, 0, stream>>>(
        ctx->total, homo, ctx->d_neutral_fraction, ctx->d_ionisation_rate_G12, ctx->d_blk0,
        ctx->d_blk1);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }
    finish_pair_sum_kernel<<<1, kBlockThreads, 0, stream>>>(ctx->d_blk0, ctx->d_blk1,
                                                            ctx->reduce_blocks, ctx->d_scalar2);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }
    double sums[2] = {0.0, 0.0};
    if (!P21_CUDA_OK(cudaMemcpyAsync(sums, ctx->d_scalar2, 2 * sizeof(double),
                                     cudaMemcpyDeviceToHost, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return -1;
    }
    double gxh = sums[0] / (float)ctx->total;
    if (!isfinite(gxh)) {
        return InfinityorNaNError;
    }
    *global_xH_out = gxh;
    *global_gamma12_out = (float)(sums[1] / (float)ctx->total);
    return 0;
}

extern "C" int p21_cuda_ion_recomb_inhomo(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    if (!ctx || !params) {
        return -1;
    }
    if (ctx->recomb_model != RECOMB_MODEL_INHOMOGENEOUS) {
        return 0;
    }

    cudaStream_t stream = ctx->stream;
    const double *d_mhr_y = nullptr, *d_mhr_c = nullptr;
    int nz = 0, ng = 0;
    double del_z = 0.2, lng_min = -10.0, del_lng = 0.1;
    if (!build_and_cache_mhr_spline(stream, &d_mhr_y, &d_mhr_c, &nz, &ng, &del_z, &lng_min,
                                    &del_lng)) {
        return -1;
    }

    if (!P21_CUDA_OK(cudaMemsetAsync(ctx->d_nonfinite, 0, sizeof(int), stream))) {
        return -1;
    }
    const int grid = (int)((ctx->total + kBlockThreads - 1) / kBlockThreads);
    inhomo_recomb_kernel<<<grid, kBlockThreads, 0, stream>>>(
        ctx->total, params->stored_redshift, params->fabs_dtdz, params->dz, nz, ng, del_z, lng_min,
        del_lng, FRACT_FLOAT_ERR, d_mhr_y, d_mhr_c, ctx->d_dens, ctx->d_ionisation_rate_G12,
        ctx->d_neutral_fraction, ctx->d_prev_nrec, ctx->d_cumulative_recombinations,
        ctx->d_nonfinite);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }
    int h_nonfinite = 0;
    if (!P21_CUDA_OK(cudaMemcpyAsync(&h_nonfinite, ctx->d_nonfinite, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return -1;
    }
    if (h_nonfinite != 0) {
        return InfinityorNaNError;
    }
    return 0;
}

extern "C" int p21_cuda_ion_finish(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                   IonizedBox *box) {
    if (!ctx || !params || !box) {
        return -1;
    }
    cudaStream_t stream = ctx->stream;
    const size_t fbytes = ctx->total * sizeof(float);
    bool recomb_used = (ctx->recomb_model != RECOMB_MODEL_NO);

    bool ok = true;
    auto copy_d2h = [&](float *hptr, const float *dptr, size_t bytes) {
        if (!ok || !hptr || !dptr || bytes == 0) return;
        ok = P21_CUDA_OK(cudaMemcpyAsync(hptr, dptr, bytes, cudaMemcpyDeviceToHost, stream));
        if (ok) p21_cuda_stats_add_d2h(bytes);
    };

    copy_d2h(box->neutral_fraction, ctx->d_neutral_fraction, fbytes);
    copy_d2h(box->z_reion, ctx->d_z_reion, fbytes);
    if (recomb_used) {
        copy_d2h(box->ionisation_rate_G12, ctx->d_ionisation_rate_G12, fbytes);
    } else if (ok && box->ionisation_rate_G12) {
        memset(box->ionisation_rate_G12, 0, fbytes);
        p21_cuda_invalidate_host_ptr(box->ionisation_rate_G12);
    }
    if (!ctx->minimize_memory) {
        copy_d2h(box->kinetic_temperature, ctx->d_kinetic_temperature, fbytes);
        if (recomb_used) {
            copy_d2h(box->mean_free_path, ctx->d_mean_free_path, fbytes);
        }
    }
    if (ctx->recomb_model == RECOMB_MODEL_INHOMOGENEOUS) {
        copy_d2h(box->cumulative_recombinations, ctx->d_cumulative_recombinations, fbytes);
    }
    if (!params->lagrangian_source_grids) {
        copy_d2h(box->nion_conditional_filtered_acg, ctx->d_nion_cond_acg, ctx->nion_cond_bytes);
        if (params->need_mcgs_nion) {
            copy_d2h(box->nion_conditional_filtered_mcg, ctx->d_nion_cond_mcg,
                     ctx->nion_cond_bytes);
        }
    }

    if (ok) {
        ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
    }
    if (!ok) {
        return -1;
    }

    auto own_or_free = [&](const void *hptr, float *&dbuf, size_t bytes) {
        if (!dbuf) return;
        if (hptr && bytes > 0 && p21_cuda_registry_put_owned(hptr, dbuf, bytes, stream)) {
            dbuf = nullptr;
        }
    };
    own_or_free(box->neutral_fraction, ctx->d_neutral_fraction, fbytes);
    own_or_free(box->z_reion, ctx->d_z_reion, fbytes);
    own_or_free(box->ionisation_rate_G12, ctx->d_ionisation_rate_G12, fbytes);
    if (ctx->recomb_model == RECOMB_MODEL_INHOMOGENEOUS) {
        own_or_free(box->cumulative_recombinations, ctx->d_cumulative_recombinations, fbytes);
    }
    if (!params->lagrangian_source_grids) {
        own_or_free(box->nion_conditional_filtered_acg, ctx->d_nion_cond_acg, ctx->nion_cond_bytes);
        if (params->need_mcgs_nion) {
            own_or_free(box->nion_conditional_filtered_mcg, ctx->d_nion_cond_mcg,
                        ctx->nion_cond_bytes);
        }
    }

    p21_cuda_stats_mut()->ionization_runs++;
    return 0;
}

extern "C" void p21_cuda_ion_end(P21CudaIonCtx *ctx) {
    if (!ctx) {
        return;
    }
    cudaStream_t stream = ctx->stream;
    free_ion_rloop_scratch(ctx);
    p21_cuda_pool_free(ctx->d_prev_dens_owned, stream);
    p21_cuda_pool_free(ctx->d_prev_zre_owned, stream);
    p21_cuda_pool_free(ctx->d_prev_gamma12_owned, stream);
    p21_cuda_pool_free(ctx->d_prev_nrec_owned, stream);

    p21_cuda_pool_free(ctx->d_blk0, stream);
    p21_cuda_pool_free(ctx->d_blk1, stream);
    p21_cuda_pool_free(ctx->d_scalar2, stream);
    p21_cuda_pool_free(ctx->d_nonfinite, stream);

    p21_cuda_pool_free(ctx->d_z_reion, stream);
    p21_cuda_pool_free(ctx->d_neutral_fraction, stream);
    p21_cuda_pool_free(ctx->d_ionisation_rate_G12, stream);
    p21_cuda_pool_free(ctx->d_mean_free_path, stream);
    p21_cuda_pool_free(ctx->d_kinetic_temperature, stream);
    p21_cuda_pool_free(ctx->d_cumulative_recombinations, stream);
    p21_cuda_pool_free(ctx->d_nion_cond_acg, stream);
    p21_cuda_pool_free(ctx->d_nion_cond_mcg, stream);

    delete ctx;
}

static constexpr int kMhrPanels = 8;
static constexpr int kMhrThreads = kMhrPanels * 32;

__global__ static void mhr_fill_table_kernel(int nz, float del_z, int ng, double lng_min,
                                             float del_lng, double n_h0, double corr_he,
                                             double alpha_b_1e4, double lnd_min,
                                             double d_lnd_panel,
                                             const double * __restrict__ d_a,
                                             const double * __restrict__ d_c,
                                             const double * __restrict__ d_beta,
                                             double * __restrict__ d_out) {
    const int gamma_ct = (int)blockIdx.x;
    const int z_ct = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (z_ct >= nz || gamma_ct >= ng) {
        return;
    }

    const int panel = tid >> 5;
    const int node = tid & 31;
    const int idx16 = (node < 16) ? (15 - node) : (node - 16);
    const double sign = (node < 16) ? -1.0 : 1.0;
    const double x_gl = sign * kGL32PosX[idx16];
    const double w_gl = kGL32PosW[idx16];

    const double p_a = lnd_min + (double)panel * d_lnd_panel;
    const double half = 0.5 * d_lnd_panel;
    const double lnD = p_a + half * (1.0 + x_gl);

    const double z = (double)((float)z_ct * del_z);
    const double ln_gamma = lng_min + (double)((float)gamma_ct * del_lng);
    const double gamma12_bg = (double)((float)exp(ln_gamma));

    const double a0 = d_a[z_ct];
    const double c0 = d_c[z_ct];
    const double beta0 = d_beta[z_ct];
    const double one_p_z = 1.0 + z;
    const double avenH = n_h0 * one_p_z * one_p_z * one_p_z;
    const double d_ss = 26.7 * pow(one_p_z / 10.0, -3.0) * pow(gamma12_bg, 2.0 / 3.0);

    const double del = exp(lnD);
    const double ratio_dss = del / d_ss;
    const double gamma = gamma12_bg * (0.98 * pow(1.0 + pow(ratio_dss, 1.64), -2.28) +
                                       0.02 * pow(1.0 + ratio_dss, -0.84));
    const double n_H = avenH * del;
    const double gamma_cgs = gamma * 1e-12;

    double x_HI;
    if (gamma_cgs <= 0.0) {
        x_HI = 1.0;
    } else {
        const double chi = (1.0 + corr_he) * n_H * alpha_b_1e4 / gamma_cgs;
        if (chi < TINY) {
            x_HI = 0.0;
        } else if (chi < 1e-5) {
            x_HI = chi;
        } else {
            const double b = -2.0 - gamma_cgs / (n_H * (1.0 + corr_he) * alpha_b_1e4);
            const double disc = fmax(0.0, b * b - 4.0);
            x_HI = (-b - sqrt(disc)) * 0.5;
        }
    }

    const double x_e = 1.0 - x_HI;
    const double inv_sigma = (3.0 * one_p_z) / (2.0 * 7.61);
    const double arg = (pow(del, -2.0 / 3.0) - c0) * inv_sigma;
    const double p_delta = a0 * exp(-0.5 * arg * arg) * pow(del, beta0);
    const double integrand = 1e15 * n_H * p_delta * alpha_b_1e4 * x_e * x_e * del * del;

    __shared__ double sdata[kMhrThreads];
    sdata[tid] = half * w_gl * integrand;
    __syncthreads();

    for (int s = kMhrThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[(size_t)z_ct * (size_t)ng + (size_t)gamma_ct] = sdata[0];
    }
}

extern "C" int p21_cuda_fill_mhr_table(int nz, float del_z, int ng, double lng_min, float del_lng,
                                       double n_h0, double y_he, double alpha_b_1e4,
                                       const double *a_mhr, const double *c_mhr,
                                       const double *beta_mhr, double *rr_table_out) {
    if (nz <= 0 || nz > 65535 || ng <= 0 || !a_mhr || !c_mhr || !beta_mhr || !rr_table_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }
    cudaStream_t stream = p21_cuda_stream();

    const size_t z_bytes = (size_t)nz * sizeof(double);
    const size_t out_bytes = (size_t)nz * (size_t)ng * sizeof(double);
    const size_t total_bytes = 3 * z_bytes + out_bytes;

    double *d_buf = (double *)p21_cuda_pool_alloc(total_bytes, stream);
    if (!d_buf) {
        return -1;
    }

    double *d_a = d_buf;
    double *d_c = d_buf + (size_t)nz;
    double *d_beta = d_buf + 2 * (size_t)nz;
    double *d_out = d_buf + 3 * (size_t)nz;

    std::vector<double> h_params(3 * (size_t)nz);
    memcpy(h_params.data(), a_mhr, z_bytes);
    memcpy(h_params.data() + (size_t)nz, c_mhr, z_bytes);
    memcpy(h_params.data() + 2 * (size_t)nz, beta_mhr, z_bytes);

    bool ok = P21_CUDA_OK(cudaMemcpyAsync(d_buf, h_params.data(), 3 * z_bytes,
                                          cudaMemcpyHostToDevice, stream));
    if (ok) {
        const double denom = 4.0 / y_he - 3.0;
        if (denom == 0.0) {
            p21_cuda_pool_free(d_buf, stream);
            return -1;
        }
        const double corr_he = 1.0 / denom;
        const double lnd_min = log(0.01);
        const double lnd_max = log(200.0);
        const double d_lnd_panel = (lnd_max - lnd_min) / (double)kMhrPanels;
        const dim3 grid((unsigned int)ng, (unsigned int)nz, 1);
        mhr_fill_table_kernel<<<grid, kMhrThreads, 0, stream>>>(
            nz, del_z, ng, lng_min, del_lng, n_h0, corr_he, alpha_b_1e4, lnd_min, d_lnd_panel, d_a,
            d_c, d_beta, d_out);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(rr_table_out, d_out, out_bytes, cudaMemcpyDeviceToHost,
                                         stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_cuda_pool_free(d_buf, stream);
    return ok ? 0 : -1;
}
