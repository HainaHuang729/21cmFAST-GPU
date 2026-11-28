#include <math.h>
#include <mutex>
#include <string.h>
#include <unordered_map>
#include <vector>

#include "cuda_common.cuh"
#include "cuda_mgpu.cuh"
#include "cuda_tables.cuh"

namespace {

constexpr int kBlockThreads = 256;

struct DeviceRadSetupCache {
    const int *key_m_xhii = nullptr;
    const float *inverse_val_host = nullptr;
    const double *freq_heat_host = nullptr;
    const double *freq_ion_host = nullptr;
    const double *freq_lya_host = nullptr;
    size_t num_pixels = 0;
    int num_R = 0;

    int *d_m_xhii_low = nullptr;
    float *d_inverse_val = nullptr;
    double *d_freq_heat = nullptr;
    double *d_freq_heat_diff = nullptr;
    double *d_freq_ion = nullptr;
    double *d_freq_ion_diff = nullptr;
    double *d_freq_lya = nullptr;
    double *d_freq_lya_diff = nullptr;
};

struct DeviceRadAccumCache {
    const double *key_xray_ion = nullptr;
    size_t num_pixels = 0;
    int last_R_ct = -1;
    int use_xray_heat = 0;
    int use_mcgs = 0;
    int use_lya_heat = 0;

    double *d_xray_heating = nullptr;
    double *d_xray_ion = nullptr;
    double *d_xray_lya = nullptr;
    double *d_lya_cont_inj = nullptr;
    double *d_lya_cont = nullptr;
    double *d_lya_inj = nullptr;
    double *d_lyw = nullptr;
};

struct XhiiInterpConst {
    float xhii[x_int_NXHII];
    float inv_diff[x_int_NXHII - 1];
};

std::mutex g_rad_mu;
std::unordered_map<const int *, DeviceRadSetupCache> g_setup_cache;
std::unordered_map<const double *, DeviceRadAccumCache> g_accum_cache;

struct DeviceElecInterpCache {
    float *d_buf = nullptr;
    const float *d_xhii = nullptr;
    const float *d_energy = nullptr;
    const float *d_fheat = nullptr;
    const float *d_nlya = nullptr;
    const float *d_nion_hi = nullptr;
    const float *d_nion_hei = nullptr;
    const float *d_nion_heii = nullptr;
};
DeviceElecInterpCache g_elec_cache{};

struct FreqIntKernelParams {
    double h_frac;
    double he_frac;
    double ev_to_hz;
    double h_p;
    double nu_ion_hi;
    double nu_ion_hei;
    double nu_ion_heii;
    double nu_x_thresh_hz;
    double nu_x_max_hz;
    double x_ray_spec_index;
    double tiny_val;
    double lya_prefactor;
};

void free_elec_interp_cache(cudaStream_t stream) {
    if (g_elec_cache.d_buf) {
        p21_cuda_pool_free(g_elec_cache.d_buf, stream);
    }
    g_elec_cache = DeviceElecInterpCache{};
}

bool ensure_elec_interp_uploaded(cudaStream_t stream, DeviceElecInterpCache *out) {
    {
        std::lock_guard<std::mutex> lock(g_rad_mu);
        if (g_elec_cache.d_buf) {
            *out = g_elec_cache;
            return true;
        }
    }

    const float *h_xhii = nullptr;
    const float *h_energy = nullptr;
    const float *h_fheat = nullptr;
    const float *h_nlya = nullptr;
    const float *h_nhi = nullptr;
    const float *h_nhei = nullptr;
    const float *h_nheii = nullptr;
    if (!p21_get_elec_interp_tables(&h_xhii, &h_energy, &h_fheat, &h_nlya, &h_nhi, &h_nhei,
                                    &h_nheii)) {
        return false;
    }

    const size_t n_xhii = x_int_NXHII;
    const size_t n_energy = x_int_NENERGY;
    const size_t n_grid = (size_t)x_int_NXHII * (size_t)x_int_NENERGY;
    const size_t total_floats = n_xhii + n_energy + 5 * n_grid;
    std::vector<float> h_pack(total_floats);
    size_t off = 0;
    memcpy(h_pack.data() + off, h_xhii, n_xhii * sizeof(float));
    off += n_xhii;
    memcpy(h_pack.data() + off, h_energy, n_energy * sizeof(float));
    off += n_energy;
    memcpy(h_pack.data() + off, h_fheat, n_grid * sizeof(float));
    off += n_grid;
    memcpy(h_pack.data() + off, h_nlya, n_grid * sizeof(float));
    off += n_grid;
    memcpy(h_pack.data() + off, h_nhi, n_grid * sizeof(float));
    off += n_grid;
    memcpy(h_pack.data() + off, h_nhei, n_grid * sizeof(float));
    off += n_grid;
    memcpy(h_pack.data() + off, h_nheii, n_grid * sizeof(float));

    std::lock_guard<std::mutex> lock(g_rad_mu);
    if (g_elec_cache.d_buf) {
        *out = g_elec_cache;
        return true;
    }

    const size_t bytes = total_floats * sizeof(float);
    float *d_buf = (float *)p21_cuda_pool_alloc(bytes, stream);
    if (!d_buf) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_buf, h_pack.data(), bytes, cudaMemcpyHostToDevice, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        p21_cuda_pool_free(d_buf, stream);
        return false;
    }

    DeviceElecInterpCache entry{};
    entry.d_buf = d_buf;
    size_t dev_off = 0;
    entry.d_xhii = d_buf + dev_off;
    dev_off += n_xhii;
    entry.d_energy = d_buf + dev_off;
    dev_off += n_energy;
    entry.d_fheat = d_buf + dev_off;
    dev_off += n_grid;
    entry.d_nlya = d_buf + dev_off;
    dev_off += n_grid;
    entry.d_nion_hi = d_buf + dev_off;
    dev_off += n_grid;
    entry.d_nion_hei = d_buf + dev_off;
    dev_off += n_grid;
    entry.d_nion_heii = d_buf + dev_off;
    g_elec_cache = entry;
    *out = entry;
    return true;
}

__device__ inline double p21_hei_ion_crosssec(double nu, const FreqIntKernelParams &p) {
    if (nu < p.nu_ion_hei) return 0.0;
    const double x = nu / p.ev_to_hz / 13.61 - 0.4434;
    const double y = sqrt(x * x + 2.136 * 2.136);
    return 9.492e-16 * ((x - 1.0) * (x - 1.0) + 2.039 * 2.039) * pow(y, (0.5 * 3.188 - 5.5)) *
           pow(1.0 + sqrt(y / 1.469), -3.188);
}

__device__ inline double p21_heii_ion_crosssec(double nu, const FreqIntKernelParams &p) {
    if (nu < p.nu_ion_heii) return 0.0;
    if (nu == p.nu_ion_heii) nu += p.tiny_val;
    const double epsilon = sqrt(nu / p.nu_ion_heii - 1.0);
    return (6.3e-18 / 4.0) * pow(p.nu_ion_heii / nu, 4.0) *
           exp(4.0 - (4.0 * atan(epsilon) / epsilon)) / (1.0 - exp(-2.0 * M_PI / epsilon));
}

__device__ inline double p21_hi_ion_crosssec(double nu, const FreqIntKernelParams &p) {
    if (nu < p.nu_ion_hi) return 0.0;
    if (nu == p.nu_ion_hi) nu += p.tiny_val;
    const double epsilon = sqrt(nu / p.nu_ion_hi - 1.0);
    return 6.3e-18 * pow(p.nu_ion_hi / nu, 4.0) * exp(4.0 - (4.0 * atan(epsilon) / epsilon)) /
           (1.0 - exp(-2.0 * M_PI / epsilon));
}

__device__ inline float p21_interp_elec_1(const float * __restrict__ tbl, int off0, int off1,
                                          int n_low, int n_high, float de, float de_span,
                                          float xhii_frac) {
    const float v00 = tbl[off0 + n_low];
    const float v01 = tbl[off0 + n_high];
    const float v10 = tbl[off1 + n_low];
    const float v11 = tbl[off1 + n_high];
    const float elow = ((v01 - v00) / de_span) * de + v00;
    const float ehigh = ((v11 - v10) / de_span) * de + v10;
    return ((ehigh - elow) * xhii_frac) + elow;
}

__device__ inline void p21_interp_elec_5(float en, int m_low, int m_high, float xhii_frac,
                                         const float * __restrict__ d_energy,
                                         const float * __restrict__ d_fheat,
                                         const float * __restrict__ d_nlya,
                                         const float * __restrict__ d_nhi,
                                         const float * __restrict__ d_nhei,
                                         const float * __restrict__ d_nheii, float *out_fheat,
                                         float *out_nlya, float *out_nion_sum) {
    if ((double)en > 0.999 * (double)d_energy[x_int_NENERGY - 1]) {
        en = (float)((double)d_energy[x_int_NENERGY - 1] * 0.999);
    } else if (en < d_energy[0]) {
        *out_fheat = 1.0f;
        *out_nlya = 0.0f;
        *out_nion_sum = 0.0f;
        return;
    }

    int n_low;
    if (en < 1008.88f) {
        n_low = (int)(log((double)en / 10.0) / 1.98026273e-2);
    } else {
        n_low = 232 + (int)(log((double)en / 1008.88) / 9.53101798e-2);
    }
    if (n_low < 0) n_low = 0;
    if (n_low >= x_int_NENERGY - 1) n_low = x_int_NENERGY - 2;
    const int n_high = n_low + 1;

    const float de_span = d_energy[n_high] - d_energy[n_low];
    const float de = en - d_energy[n_low];
    const int off0 = m_low * x_int_NENERGY;
    const int off1 = m_high * x_int_NENERGY;

    *out_fheat = p21_interp_elec_1(d_fheat, off0, off1, n_low, n_high, de, de_span, xhii_frac);
    *out_nlya = p21_interp_elec_1(d_nlya, off0, off1, n_low, n_high, de, de_span, xhii_frac);
    *out_nion_sum =
        p21_interp_elec_1(d_nhi, off0, off1, n_low, n_high, de, de_span, xhii_frac) +
        p21_interp_elec_1(d_nhei, off0, off1, n_low, n_high, de, de_span, xhii_frac) +
        p21_interp_elec_1(d_nheii, off0, off1, n_low, n_high, de, de_span, xhii_frac);
}

__global__ void fill_freqint_kernel(int num_R, FreqIntKernelParams params,
                                    const double * __restrict__ d_lower_limits,
                                    const float * __restrict__ d_xhii,
                                    const float * __restrict__ d_energy,
                                    const float * __restrict__ d_fheat,
                                    const float * __restrict__ d_nlya,
                                    const float * __restrict__ d_nhi,
                                    const float * __restrict__ d_nhei,
                                    const float * __restrict__ d_nheii,
                                    double * __restrict__ d_heat_out,
                                    double * __restrict__ d_ion_out,
                                    double * __restrict__ d_lya_out) {
    const int R_ct = (int)blockIdx.x;
    const int x_e_ct = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (R_ct >= num_R || x_e_ct >= x_int_NXHII) {
        return;
    }

    const size_t out_idx = (size_t)x_e_ct * (size_t)num_R + (size_t)R_ct;
    const double nu_min = d_lower_limits[R_ct];
    const double nu_max = params.nu_x_max_hz;
    if (nu_min <= 0.0 || nu_min >= nu_max) {
        if (tid == 0) {
            d_heat_out[out_idx] = 0.0;
            d_ion_out[out_idx] = 0.0;
            d_lya_out[out_idx] = 0.0;
        }
        return;
    }

    const float x_e = d_xhii[x_e_ct];
    float xhii_call = x_e;
    if ((double)xhii_call > (double)d_xhii[x_int_NXHII - 1] * 0.999) {
        xhii_call = (float)((double)d_xhii[x_int_NXHII - 1] * 0.999);
    } else if (xhii_call < d_xhii[0]) {
        xhii_call = (float)(1.001 * (double)d_xhii[0]);
    }
    int m_low = x_int_NXHII - 1;
    while (m_low > 0 && xhii_call < d_xhii[m_low]) {
        m_low--;
    }
    if (m_low >= x_int_NXHII - 1) m_low = x_int_NXHII - 2;
    const int m_high = m_low + 1;
    const float xhii_frac = (xhii_call - d_xhii[m_low]) / (d_xhii[m_high] - d_xhii[m_low]);

    const int node = tid & 31;
    const int base_panel = tid >> 5;
    const int idx16 = (node < 16) ? (15 - node) : (node - 16);
    const double sign = (node < 16) ? -1.0 : 1.0;
    const double x_gl = sign * kGL32PosX[idx16];
    const double w_gl = kGL32PosW[idx16];

    const double u_min = log(nu_min);
    const double u_max = log(nu_max);
    constexpr int kFreqSubPasses = 4;
    constexpr int kFreqPanels = kFreqSubPasses * 8;
    const double d_u_panel = (u_max - u_min) / (double)kFreqPanels;
    const double half = 0.5 * d_u_panel;

    const double one_minus_xe = (double)(1.0f - x_e);
    const double xe_dbl = (double)x_e;

    double sum_heat = 0.0;
    double sum_ion = 0.0;
    double sum_lya = 0.0;

    for (int pass = 0; pass < kFreqSubPasses; pass++) {
        const int panel = pass * 8 + base_panel;
        const double u = u_min + (double)panel * d_u_panel + half * (1.0 + x_gl);
        const double nu = exp(u);

        const double sig_hi = p21_hi_ion_crosssec(nu, params);
        const double sig_hei = p21_hei_ion_crosssec(nu, params);
        const double sig_heii = p21_heii_ion_crosssec(nu, params);
        const double spec_sed = pow(nu / params.nu_x_thresh_hz, -params.x_ray_spec_index - 1.0);

        double heat_sp = 0.0;
        double ion_sp = 0.0;
        double lya_sp = 0.0;

        if (sig_hi > 0.0) {
            float fh = 0.0f, nl = 0.0f, ns = 0.0f;
            p21_interp_elec_5((float)((nu - params.nu_ion_hi) / params.ev_to_hz), m_low, m_high,
                              xhii_frac, d_energy, d_fheat, d_nlya, d_nhi, d_nhei, d_nheii, &fh, &nl,
                              &ns);
            const double w_hi = params.h_frac * one_minus_xe * sig_hi;
            heat_sp += (double)fh * params.h_p * (nu - params.nu_ion_hi) * w_hi;
            ion_sp += ((double)ns + 1.0) * w_hi;
            lya_sp += (double)nl * w_hi;
        }
        if (sig_hei > 0.0) {
            float fh = 0.0f, nl = 0.0f, ns = 0.0f;
            p21_interp_elec_5((float)((nu - params.nu_ion_hei) / params.ev_to_hz), m_low, m_high,
                              xhii_frac, d_energy, d_fheat, d_nlya, d_nhi, d_nhei, d_nheii, &fh, &nl,
                              &ns);
            const double w_hei = params.he_frac * one_minus_xe * sig_hei;
            heat_sp += (double)fh * params.h_p * (nu - params.nu_ion_hei) * w_hei;
            ion_sp += ((double)ns + 1.0) * w_hei;
            lya_sp += (double)nl * w_hei;
        }
        if (sig_heii > 0.0) {
            float fh = 0.0f, nl = 0.0f, ns = 0.0f;
            p21_interp_elec_5((float)((nu - params.nu_ion_heii) / params.ev_to_hz), m_low, m_high,
                              xhii_frac, d_energy, d_fheat, d_nlya, d_nhi, d_nhei, d_nheii, &fh, &nl,
                              &ns);
            const double w_heii = params.he_frac * xe_dbl * sig_heii;
            heat_sp += (double)fh * params.h_p * (nu - params.nu_ion_heii) * w_heii;
            ion_sp += ((double)ns + 1.0) * w_heii;
            lya_sp += (double)nl * w_heii;
        }

        const double weight = half * w_gl * nu * spec_sed;
        sum_heat += weight * heat_sp;
        sum_ion += weight * ion_sp;
        sum_lya += weight * lya_sp;
    }

    __shared__ double s_heat[kBlockThreads];
    __shared__ double s_ion[kBlockThreads];
    __shared__ double s_lya[kBlockThreads];
    s_heat[tid] = sum_heat;
    s_ion[tid] = sum_ion;
    s_lya[tid] = sum_lya;
    __syncthreads();

    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_heat[tid] += s_heat[tid + s];
            s_ion[tid] += s_ion[tid + s];
            s_lya[tid] += s_lya[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_heat_out[out_idx] = s_heat[0];
        d_ion_out[out_idx] = s_ion[0];
        d_lya_out[out_idx] = s_lya[0] * params.lya_prefactor;
    }
}

void free_setup_entry(DeviceRadSetupCache *e, cudaStream_t stream) {
    p21_cuda_pool_free(e->d_m_xhii_low, stream);
    p21_cuda_pool_free(e->d_inverse_val, stream);
    p21_cuda_pool_free(e->d_freq_heat, stream);
    p21_cuda_pool_free(e->d_freq_heat_diff, stream);
    p21_cuda_pool_free(e->d_freq_ion, stream);
    p21_cuda_pool_free(e->d_freq_ion_diff, stream);
    p21_cuda_pool_free(e->d_freq_lya, stream);
    p21_cuda_pool_free(e->d_freq_lya_diff, stream);
    *e = DeviceRadSetupCache{};
}

void free_accum_entry(DeviceRadAccumCache *e, cudaStream_t stream) {
    p21_cuda_pool_free(e->d_xray_heating, stream);
    p21_cuda_pool_free(e->d_xray_ion, stream);
    p21_cuda_pool_free(e->d_xray_lya, stream);
    p21_cuda_pool_free(e->d_lya_cont_inj, stream);
    p21_cuda_pool_free(e->d_lya_cont, stream);
    p21_cuda_pool_free(e->d_lya_inj, stream);
    p21_cuda_pool_free(e->d_lyw, stream);
    *e = DeviceRadAccumCache{};
}

__global__ void sum_float_block_kernel(const float *arr, size_t n, double *block_sums) {
    __shared__ double sdata[kBlockThreads];
    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double acc = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        acc += (double)arr[i];
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_double_final_kernel(const double *block_sums, int nblocks, double *out) {
    __shared__ double sdata[kBlockThreads];
    size_t tid = threadIdx.x;
    double acc = 0.0;
    for (int i = (int)tid; i < nblocks; i += kBlockThreads) {
        acc += block_sums[i];
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out = sdata[0];
    }
}

__global__ void setup_xhii_interp_kernel(const float *prev_xe, size_t n, XhiiInterpConst c,
                                         int *m_xhii_low_out, float *inverse_val_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float xHII_call = prev_xe[idx];
    float upper = (float)((double)c.xhii[x_int_NXHII - 1] * 0.999);
    float lower = (float)(1.001 * (double)c.xhii[0]);
    if (xHII_call > upper) {
        xHII_call = upper;
    } else if (xHII_call < c.xhii[0]) {
        xHII_call = lower;
    }

    int low = x_int_NXHII - 1;
    while (low > 0 && xHII_call < c.xhii[low]) {
        low--;
    }
    m_xhii_low_out[idx] = low;
    inverse_val_out[idx] = (xHII_call - c.xhii[low]) * c.inv_diff[low];
}

__global__ void clamp_nonneg_kernel(float *arr, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && arr[idx] < 0.0f) {
        arr[idx] = 0.0f;
    }
}

__global__ void lerp_field_kernel(size_t n, const float *__restrict__ a,
                                  const float *__restrict__ b, double w, float *out,
                                  int clamp_nonneg) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    double wa = 1.0 - w;
    float v = (float)(wa * (double)a[idx] + w * (double)b[idx]);
    if (clamp_nonneg && v < 0.0f) {
        v = 0.0f;
    }
    out[idx] = v;
}

__global__ void accumulate_radiation_shell_kernel(
    size_t n, int R_ct, int num_R, int use_xray_heat, int use_mcgs, int use_lya_heat,
    int lya_ms, double z_edge_factor, double xray_R_factor, double lyw_pref_acg,
    double lyw_pref_mcg, double lya_cont_pref_acg, double lya_inj_pref_acg,
    double lya_cont_inj_pref_acg, double lya_cont_pref_mcg, double lya_inj_pref_mcg,
    double lya_cont_inj_pref_mcg, const int *m_xhii_low, const float *inverse_val,
    const double *freq_heat, const double *freq_heat_diff, const double *freq_ion,
    const double *freq_ion_diff, const double *freq_lya, const double *freq_lya_diff,
    float *filt_sfrd_acg_lya, float *filt_xray, float *filt_sfrd_mcg_lya,
    float *filt_sfrd_acg_lw, float *filt_sfrd_mcg_lw, double *xray_heating,
    double *xray_ion, double *xray_lya, double *lya_cont_inj, double *lya_cont,
    double *lya_inj, double *lyw) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float f_sfr_acg = filt_sfrd_acg_lya[idx];
    if (f_sfr_acg < 0.0f) {
        f_sfr_acg = 0.0f;
        filt_sfrd_acg_lya[idx] = 0.0f;
    }
    float f_xr = filt_xray[idx];
    if (f_xr < 0.0f) {
        f_xr = 0.0f;
        filt_xray[idx] = 0.0f;
    }

    double sfrd_term_lya_acg = (double)f_sfr_acg * z_edge_factor;
    double xray_emiss_term = (double)f_xr * z_edge_factor * xray_R_factor * 1e38;
    double sfrd_term_lya_mcg = 0.0;
    double sfrd_term_lw_acg = 0.0;
    double sfrd_term_lw_mcg = 0.0;

    if (use_mcgs) {
        if (lya_ms) {
            float f_lw_acg = filt_sfrd_acg_lw[idx];
            if (f_lw_acg < 0.0f) {
                f_lw_acg = 0.0f;
                filt_sfrd_acg_lw[idx] = 0.0f;
            }
            sfrd_term_lw_acg = (double)f_lw_acg * z_edge_factor;
        } else {
            sfrd_term_lw_acg = sfrd_term_lya_acg;
        }

        float f_sfr_mcg = filt_sfrd_mcg_lya[idx];
        if (f_sfr_mcg < 0.0f) {
            f_sfr_mcg = 0.0f;
            filt_sfrd_mcg_lya[idx] = 0.0f;
        }
        sfrd_term_lya_mcg = (double)f_sfr_mcg * z_edge_factor;

        if (lya_ms) {
            float f_lw_mcg = filt_sfrd_mcg_lw[idx];
            if (f_lw_mcg < 0.0f) {
                f_lw_mcg = 0.0f;
                filt_sfrd_mcg_lw[idx] = 0.0f;
            }
            sfrd_term_lw_mcg = (double)f_lw_mcg * z_edge_factor;
        } else {
            sfrd_term_lw_mcg = sfrd_term_lya_mcg;
        }
    }

    int xidx = m_xhii_low[idx];
    double ival = (double)inverse_val[idx];
    int ft_idx = xidx * num_R + R_ct;

    double f_heat = freq_heat_diff[ft_idx] * ival + freq_heat[ft_idx];
    double f_ion = freq_ion_diff[ft_idx] * ival + freq_ion[ft_idx];
    double f_lya = freq_lya_diff[ft_idx] * ival + freq_lya[ft_idx];

    if (use_xray_heat) {
        xray_heating[idx] += xray_emiss_term * f_heat;
    }
    xray_ion[idx] += xray_emiss_term * f_ion;
    xray_lya[idx] += xray_emiss_term * f_lya;

    if (use_mcgs) {
        lyw[idx] += sfrd_term_lw_acg * lyw_pref_acg + sfrd_term_lw_mcg * lyw_pref_mcg;
    }
    if (use_lya_heat) {
        lya_cont[idx] +=
            sfrd_term_lya_acg * lya_cont_pref_acg + sfrd_term_lya_mcg * lya_cont_pref_mcg;
        lya_inj[idx] +=
            sfrd_term_lya_acg * lya_inj_pref_acg + sfrd_term_lya_mcg * lya_inj_pref_mcg;
    } else {
        lya_cont_inj[idx] +=
            sfrd_term_lya_acg * lya_cont_inj_pref_acg + sfrd_term_lya_mcg * lya_cont_inj_pref_mcg;
    }
}

__global__ void multiply_radiation_constants_kernel(
    size_t n, int use_xray_heat, int use_mcgs, int use_lya_heat, const float *density,
    const float *prev_xe, double xray_vol, double xray_heat_factor, double xray_lya_factor,
    double lyw_factor, double lya_factor, double *xray_heating, double *xray_ion, double *xray_lya,
    double *lya_cont_inj, double *lya_cont, double *lya_inj, double *lyw) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    double curr_delta = (double)density[idx];
    double p_xe = (double)prev_xe[idx];

    if (use_xray_heat) {
        xray_heating[idx] *= xray_heat_factor / (1.0 + p_xe);
    }
    xray_ion[idx] *= xray_vol;
    xray_lya[idx] *= xray_lya_factor * (1.0 + curr_delta);

    if (use_mcgs) {
        lyw[idx] *= lyw_factor;
    }
    if (use_lya_heat) {
        lya_cont[idx] *= lya_factor;
        lya_inj[idx] *= lya_factor;
    } else {
        lya_cont_inj[idx] *= lya_factor;
    }
}

bool mgpu_alloc_rad_slabs(const P21MgpuGrid &grid, std::vector<cufftComplex *> &y) {
    y.assign((size_t)grid.n_gpus, nullptr);
    for (int g = 0; g < grid.n_gpus; ++g) {
        y[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
            grid.devs[g], grid.cplx_elems_y(g) * sizeof(cufftComplex), grid.streams[g]);
        if (!y[(size_t)g]) {
            return false;
        }
    }
    return true;
}

void mgpu_free_rad_slabs(const P21MgpuGrid &grid, std::vector<cufftComplex *> &y) {
    for (int g = 0; g < (int)y.size() && g < grid.n_gpus; ++g) {
        if (y[(size_t)g]) {
            p21_mgpu_free(grid.devs[g], y[(size_t)g], grid.streams[g]);
            y[(size_t)g] = nullptr;
        }
    }
    y.clear();
}

bool mgpu_filter_device_real_pair(float *d_in_and_out1, float *d_opt_out2, int nx, int ny, int nz,
                                  double box_len, double ncf, int filter_type1, int filter_type2,
                                  double R_inner, double R_outer, double R_star,
                                  cudaStream_t stream) {
    P21MgpuGrid grid;
    if (!grid.init(nx, ny, nz)) {
        return false;
    }
    std::vector<cufftComplex *> d_y_work, d_y_unfilt;
    bool dual = (d_opt_out2 != nullptr);
    bool ok = mgpu_alloc_rad_slabs(grid, d_y_work) && (!dual || mgpu_alloc_rad_slabs(grid, d_y_unfilt));
    float pre_scale = 1.0f / (float)((size_t)nx * (size_t)ny * (size_t)nz);
    if (ok) {
        cufftComplex **r2c_dst = dual ? d_y_unfilt.data() : d_y_work.data();
        ok = p21_mgpu_r2c_from_device(grid, d_in_and_out1, stream, 1.0, r2c_dst, nullptr) &&
             p21_mgpu_filter_yslabs(grid, r2c_dst, d_y_work.data(), box_len, ncf, filter_type1,
                                    (float)R_inner, (float)R_outer, (float)R_star, pre_scale) &&
             p21_mgpu_c2r_to_device(grid, d_y_work.data(), d_in_and_out1, stream, nullptr);
        if (ok && dual) {
            ok = p21_mgpu_filter_yslabs(grid, d_y_unfilt.data(), d_y_work.data(), box_len, ncf,
                                        filter_type2, (float)R_inner, (float)R_outer, (float)R_star,
                                        pre_scale) &&
                 p21_mgpu_c2r_to_device(grid, d_y_work.data(), d_opt_out2, stream, nullptr);
        }
    }
    mgpu_free_rad_slabs(grid, d_y_unfilt);
    mgpu_free_rad_slabs(grid, d_y_work);
    grid.destroy();
    return ok;
}

/* Filter one real grid on GPU without polluting DeviceFieldRegistry with ephemeral interp boxes. */
bool filter_one_ephemeral_shell(const float *host_src, int nx, int ny, int nz, double box_len,
                                double ncf, int R_ct, double R_inner, double R_outer, double R_star,
                                int filter_type, bool keep_kspace, float **d_real_out,
                                cufftComplex **d_kspace_out, cudaStream_t stream) {
    size_t n = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t fbytes = n * sizeof(float);
    size_t cbytes = (size_t)nx * (size_t)ny * (size_t)(nz / 2 + 1) * sizeof(cufftComplex);

    float *d_out = (float *)p21_cuda_pool_alloc(fbytes, stream);
    if (!d_out) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_out, host_src, fbytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_out, stream);
        return false;
    }

    if (R_ct == 0 || R_inner <= 0.0) {
        int grid = (int)((n + kBlockThreads - 1) / kBlockThreads);
        clamp_nonneg_kernel<<<grid, kBlockThreads, 0, stream>>>(d_out, n);
        bool ok = P21_CUDA_OK(cudaGetLastError());
        if (!ok) {
            p21_cuda_pool_free(d_out, stream);
            return false;
        }
        *d_real_out = d_out;
        *d_kspace_out = nullptr;
        return true;
    }

    if (!keep_kspace && p21_mgpu_eor_enabled(n)) {
        if (!mgpu_filter_device_real_pair(d_out, nullptr, nx, ny, nz, box_len, ncf, filter_type,
                                          filter_type, R_inner, R_outer, R_star, stream)) {
            p21_cuda_pool_free(d_out, stream);
            return false;
        }
        *d_real_out = d_out;
        *d_kspace_out = nullptr;
        return true;
    }

    cufftComplex *d_k_work = (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream);
    cufftComplex *d_k_unfilt =
        keep_kspace ? (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream) : nullptr;
    if (!d_k_work || (keep_kspace && !d_k_unfilt)) {
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_k_unfilt, stream);
        p21_cuda_pool_free(d_k_work, stream);
        return false;
    }

    float pre_scale = 1.0f / (float)n;
    bool ok;
    if (keep_kspace) {
        ok = p21_cuda_fft_r2c(d_out, d_k_unfilt, nx, ny, nz, false, stream) &&
             p21_cuda_copy_and_filter_box(d_k_unfilt, d_k_work, nx, ny, nz, box_len, ncf,
                                          filter_type, (float)R_inner, (float)R_outer,
                                          (float)R_star, pre_scale, stream) &&
             p21_cuda_fft_c2r(d_k_work, d_out, nx, ny, nz, false, stream);
    } else {
        ok = p21_cuda_fft_r2c(d_out, d_k_work, nx, ny, nz, false, stream) &&
             p21_cuda_copy_and_filter_box(d_k_work, d_k_work, nx, ny, nz, box_len, ncf, filter_type,
                                          (float)R_inner, (float)R_outer, (float)R_star, pre_scale,
                                          stream) &&
             p21_cuda_fft_c2r(d_k_work, d_out, nx, ny, nz, false, stream);
    }

    p21_cuda_pool_free(d_k_work, stream);
    if (!ok) {
        p21_cuda_pool_free(d_k_unfilt, stream);
        p21_cuda_pool_free(d_out, stream);
        return false;
    }
    *d_kspace_out = keep_kspace ? d_k_unfilt : nullptr;
    *d_real_out = d_out;
    return true;
}

bool filter_from_existing_kspace(const cufftComplex *d_k_unfilt, int nx, int ny, int nz,
                                 double box_len, double ncf, double R_inner, double R_outer,
                                 double R_star, float **d_real_out, cudaStream_t stream) {
    size_t n = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t fbytes = n * sizeof(float);
    size_t cbytes = (size_t)nx * (size_t)ny * (size_t)(nz / 2 + 1) * sizeof(cufftComplex);

    float *d_out = (float *)p21_cuda_pool_alloc(fbytes, stream);
    cufftComplex *d_k_work = (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream);
    if (!d_out || !d_k_work) {
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_k_work, stream);
        return false;
    }

    float pre_scale = 1.0f / (float)n;
    bool ok = p21_cuda_copy_and_filter_box(
                  d_k_unfilt, d_k_work, nx, ny, nz, box_len, ncf,
                  FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, (float)R_inner, (float)R_outer,
                  (float)R_star, pre_scale, stream) &&
              p21_cuda_fft_c2r(d_k_work, d_out, nx, ny, nz, false, stream);
    p21_cuda_pool_free(d_k_work, stream);
    if (!ok) {
        p21_cuda_pool_free(d_out, stream);
        return false;
    }
    *d_real_out = d_out;
    return true;
}

bool upload_setup_entry(RadiationFieldsSetup *rad_setup, size_t n, int num_R,
                        const float *d_prev_xe, bool compute_xhii_on_gpu, cudaStream_t stream,
                        DeviceRadSetupCache *out) {
    size_t ibytes = n * sizeof(int);
    size_t fbytes = n * sizeof(float);
    size_t tbl_elems = (size_t)x_int_NXHII * (size_t)num_R;
    size_t tbytes = tbl_elems * sizeof(double);

    DeviceRadSetupCache e{};
    e.key_m_xhii = rad_setup->m_xHII_low_box;
    e.inverse_val_host = rad_setup->inverse_val_box;
    e.freq_heat_host = rad_setup->freq_int_heat_tbl;
    e.freq_ion_host = rad_setup->freq_int_ion_tbl;
    e.freq_lya_host = rad_setup->freq_int_lya_tbl;
    e.num_pixels = n;
    e.num_R = num_R;

    e.d_m_xhii_low = (int *)p21_cuda_pool_alloc(ibytes, stream);
    e.d_inverse_val = (float *)p21_cuda_pool_alloc(fbytes, stream);
    e.d_freq_heat = (double *)p21_cuda_pool_alloc(tbytes, stream);
    e.d_freq_heat_diff = (double *)p21_cuda_pool_alloc(tbytes, stream);
    e.d_freq_ion = (double *)p21_cuda_pool_alloc(tbytes, stream);
    e.d_freq_ion_diff = (double *)p21_cuda_pool_alloc(tbytes, stream);
    e.d_freq_lya = (double *)p21_cuda_pool_alloc(tbytes, stream);
    e.d_freq_lya_diff = (double *)p21_cuda_pool_alloc(tbytes, stream);
    if (!e.d_m_xhii_low || !e.d_inverse_val || !e.d_freq_heat || !e.d_freq_heat_diff ||
        !e.d_freq_ion || !e.d_freq_ion_diff || !e.d_freq_lya || !e.d_freq_lya_diff) {
        free_setup_entry(&e, stream);
        return false;
    }

    bool ok = true;
    if (compute_xhii_on_gpu) {
        XhiiInterpConst hconst{};
        for (int i = 0; i < x_int_NXHII; ++i) {
            hconst.xhii[i] = x_int_XHII[i];
        }
        for (int i = 0; i < x_int_NXHII - 1; ++i) {
            hconst.inv_diff[i] = rad_setup->inverse_diff[i];
        }
        int grid = (int)((n + kBlockThreads - 1) / kBlockThreads);
        setup_xhii_interp_kernel<<<grid, kBlockThreads, 0, stream>>>(
            d_prev_xe, n, hconst, e.d_m_xhii_low, e.d_inverse_val);
        ok = P21_CUDA_OK(cudaGetLastError());
    } else {
        ok = P21_CUDA_OK(cudaMemcpyAsync(e.d_m_xhii_low, rad_setup->m_xHII_low_box, ibytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(e.d_inverse_val, rad_setup->inverse_val_box, fbytes,
                                         cudaMemcpyHostToDevice, stream));
    }

    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(e.d_freq_heat, rad_setup->freq_int_heat_tbl, tbytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(e.d_freq_heat_diff, rad_setup->freq_int_heat_tbl_diff,
                                         tbytes, cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(e.d_freq_ion, rad_setup->freq_int_ion_tbl, tbytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(e.d_freq_ion_diff, rad_setup->freq_int_ion_tbl_diff,
                                         tbytes, cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(e.d_freq_lya, rad_setup->freq_int_lya_tbl, tbytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(e.d_freq_lya_diff, rad_setup->freq_int_lya_tbl_diff,
                                         tbytes, cudaMemcpyHostToDevice, stream));
    }
    if (ok && compute_xhii_on_gpu) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(rad_setup->m_xHII_low_box, e.d_m_xhii_low, ibytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(rad_setup->inverse_val_box, e.d_inverse_val, fbytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
        if (ok) {
            p21_cuda_stats_add_d2h(ibytes + fbytes);
        }
    }

    if (!ok) {
        free_setup_entry(&e, stream);
        return false;
    }
    *out = e;
    return true;
}

bool acquire_rad_setup_cache(RadiationFieldsSetup *rad_setup, size_t total, int num_R,
                             cudaStream_t stream, DeviceRadSetupCache *setup_out) {
    bool need_setup_upload = false;
    {
        std::lock_guard<std::mutex> lock(g_rad_mu);
        auto it = g_setup_cache.find(rad_setup->m_xHII_low_box);
        if (it == g_setup_cache.end() || it->second.num_pixels != total ||
            it->second.num_R != num_R) {
            /* Keep at most 1 setup cache entry active across redshifts to save VRAM on 512^3+. */
            for (auto &kv : g_setup_cache) {
                free_setup_entry(&kv.second, stream);
            }
            g_setup_cache.clear();
            need_setup_upload = true;
        } else {
            *setup_out = it->second;
        }
    }
    if (need_setup_upload) {
        if (!upload_setup_entry(rad_setup, total, num_R, nullptr, false, stream, setup_out)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(g_rad_mu);
        for (auto &kv : g_setup_cache) {
            free_setup_entry(&kv.second, stream);
        }
        g_setup_cache.clear();
        g_setup_cache.emplace(rad_setup->m_xHII_low_box, *setup_out);
    }
    return true;
}

bool acquire_rad_accum_cache(RadiationFields *radiation_fields, int R_ct, size_t total,
                             int use_xray_heat, int use_mcgs, int use_lya_heat, bool zero_init,
                             cudaStream_t stream, DeviceRadAccumCache *accum_out) {
    size_t dbytes = total * sizeof(double);
    DeviceRadAccumCache accum{};
    bool need_accum_init = zero_init;
    if (!zero_init) {
        std::lock_guard<std::mutex> lock(g_rad_mu);
        auto it = g_accum_cache.find(radiation_fields->xray_ionization_rate);
        if (it == g_accum_cache.end() || R_ct >= it->second.last_R_ct ||
            it->second.num_pixels != total || it->second.use_xray_heat != use_xray_heat ||
            it->second.use_mcgs != use_mcgs || it->second.use_lya_heat != use_lya_heat) {
            if (it != g_accum_cache.end()) {
                free_accum_entry(&it->second, stream);
                g_accum_cache.erase(it);
            }
            need_accum_init = true;
        } else {
            accum = it->second;
            it->second.last_R_ct = R_ct;
        }
    } else {
        std::lock_guard<std::mutex> lock(g_rad_mu);
        auto it = g_accum_cache.find(radiation_fields->xray_ionization_rate);
        if (it != g_accum_cache.end()) {
            free_accum_entry(&it->second, stream);
            g_accum_cache.erase(it);
        }
    }

    if (need_accum_init) {
        accum.key_xray_ion = radiation_fields->xray_ionization_rate;
        accum.num_pixels = total;
        accum.last_R_ct = R_ct;
        accum.use_xray_heat = use_xray_heat;
        accum.use_mcgs = use_mcgs;
        accum.use_lya_heat = use_lya_heat;

        bool ok_alloc = true;
        accum.d_xray_ion = (double *)p21_cuda_pool_alloc(dbytes, stream);
        accum.d_xray_lya = (double *)p21_cuda_pool_alloc(dbytes, stream);
        if (!accum.d_xray_ion || !accum.d_xray_lya) ok_alloc = false;
        if (use_xray_heat) {
            accum.d_xray_heating = (double *)p21_cuda_pool_alloc(dbytes, stream);
            if (!accum.d_xray_heating) ok_alloc = false;
        }
        if (use_lya_heat) {
            accum.d_lya_cont = (double *)p21_cuda_pool_alloc(dbytes, stream);
            accum.d_lya_inj = (double *)p21_cuda_pool_alloc(dbytes, stream);
            if (!accum.d_lya_cont || !accum.d_lya_inj) ok_alloc = false;
        } else {
            accum.d_lya_cont_inj = (double *)p21_cuda_pool_alloc(dbytes, stream);
            if (!accum.d_lya_cont_inj) ok_alloc = false;
        }
        if (use_mcgs) {
            accum.d_lyw = (double *)p21_cuda_pool_alloc(dbytes, stream);
            if (!accum.d_lyw) ok_alloc = false;
        }
        if (!ok_alloc) {
            free_accum_entry(&accum, stream);
            return false;
        }

        auto init_buf = [&](double *dst, const double *host_src) -> bool {
            if (zero_init) {
                return P21_CUDA_OK(cudaMemsetAsync(dst, 0, dbytes, stream));
            }
            return P21_CUDA_OK(cudaMemcpyAsync(dst, host_src, dbytes, cudaMemcpyHostToDevice,
                                               stream));
        };

        bool ok_init = init_buf(accum.d_xray_ion, radiation_fields->xray_ionization_rate) &&
                       init_buf(accum.d_xray_lya, radiation_fields->xray_lya_flux);
        if (ok_init && use_xray_heat) {
            ok_init = init_buf(accum.d_xray_heating, radiation_fields->xray_heating_rate);
        }
        if (ok_init && use_lya_heat) {
            ok_init = init_buf(accum.d_lya_cont, radiation_fields->lya_flux_continuum) &&
                      init_buf(accum.d_lya_inj, radiation_fields->lya_flux_injected);
        } else if (ok_init) {
            ok_init = init_buf(accum.d_lya_cont_inj, radiation_fields->lya_flux_continuum_injected);
        }
        if (ok_init && use_mcgs) {
            ok_init = init_buf(accum.d_lyw, radiation_fields->lyw_flux);
        }
        if (!ok_init) {
            free_accum_entry(&accum, stream);
            return false;
        }

        if (!zero_init) {
            std::lock_guard<std::mutex> lock(g_rad_mu);
            g_accum_cache.emplace(radiation_fields->xray_ionization_rate, accum);
        }
    }

    *accum_out = accum;
    return true;
}

void erase_rad_accum_cache(const double *key, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_rad_mu);
    auto it = g_accum_cache.find(key);
    if (it != g_accum_cache.end()) {
        free_accum_entry(&it->second, stream);
        g_accum_cache.erase(it);
    }
}

bool launch_rad_shell_accumulate(size_t total, int R_ct, int num_R, float redshift,
                                 int use_xray_heat, int use_mcgs, int use_lya_heat, int lya_ms,
                                 const RadiationFieldsSetup *rad_setup,
                                 const DeviceRadSetupCache &setup_entry,
                                 const DeviceRadAccumCache &accum, float *d_sfr_acg_lya,
                                 float *d_xray_filt, float *d_sfr_mcg_lya, float *d_sfr_acg_lw,
                                 float *d_sfr_mcg_lw, cudaStream_t stream) {
    double zpp = rad_setup->zpp_avg[R_ct];
    double dzpp = (R_ct == 0) ? (redshift - rad_setup->zpp_edges[0])
                              : (rad_setup->zpp_edges[R_ct - 1] - rad_setup->zpp_edges[R_ct]);
    double z_edge_factor = fabs(dzpp * dtdz(zpp));
    double xray_R_factor = pow(1.0 + zpp, -(astro_params_global->X_RAY_SPEC_INDEX));

    double lyw_pref_acg = use_mcgs ? rad_setup->lyw_flux_prefactor_acg[R_ct] : 0.0;
    double lyw_pref_mcg = use_mcgs ? rad_setup->lyw_flux_prefactor_mcg[R_ct] : 0.0;
    double lya_cont_pref_acg =
        use_lya_heat ? rad_setup->lya_flux_continuum_prefactor_acg[R_ct] : 0.0;
    double lya_inj_pref_acg =
        use_lya_heat ? rad_setup->lya_flux_injected_prefactor_acg[R_ct] : 0.0;
    double lya_cont_inj_pref_acg =
        use_lya_heat ? 0.0 : rad_setup->lya_flux_continuum_injected_prefactor_acg[R_ct];
    double lya_cont_pref_mcg =
        (use_mcgs && use_lya_heat) ? rad_setup->lya_flux_continuum_prefactor_mcg[R_ct] : 0.0;
    double lya_inj_pref_mcg =
        (use_mcgs && use_lya_heat) ? rad_setup->lya_flux_injected_prefactor_mcg[R_ct] : 0.0;
    double lya_cont_inj_pref_mcg =
        (use_mcgs && !use_lya_heat) ? rad_setup->lya_flux_continuum_injected_prefactor_mcg[R_ct]
                                    : 0.0;

    int grid = (int)((total + kBlockThreads - 1) / kBlockThreads);
    accumulate_radiation_shell_kernel<<<grid, kBlockThreads, 0, stream>>>(
        total, R_ct, num_R, use_xray_heat, use_mcgs, use_lya_heat, lya_ms, z_edge_factor,
        xray_R_factor, lyw_pref_acg, lyw_pref_mcg, lya_cont_pref_acg, lya_inj_pref_acg,
        lya_cont_inj_pref_acg, lya_cont_pref_mcg, lya_inj_pref_mcg, lya_cont_inj_pref_mcg,
        setup_entry.d_m_xhii_low, setup_entry.d_inverse_val, setup_entry.d_freq_heat,
        setup_entry.d_freq_heat_diff, setup_entry.d_freq_ion, setup_entry.d_freq_ion_diff,
        setup_entry.d_freq_lya, setup_entry.d_freq_lya_diff, d_sfr_acg_lya, d_xray_filt,
        d_sfr_mcg_lya, d_sfr_acg_lw, d_sfr_mcg_lw, accum.d_xray_heating, accum.d_xray_ion,
        accum.d_xray_lya, accum.d_lya_cont_inj, accum.d_lya_cont, accum.d_lya_inj, accum.d_lyw);
    return P21_CUDA_OK(cudaGetLastError());
}

bool copy_filtered_shell_r0_d2h(size_t fbytes, int use_mcgs, int lya_ms,
                                RadiationFieldsSetup *rad_setup, const float *d_sfr_acg_lya,
                                const float *d_xray_filt, const float *d_sfr_mcg_lya,
                                const float *d_sfr_acg_lw, const float *d_sfr_mcg_lw,
                                cudaStream_t stream) {
    bool ok = true;
    if (rad_setup->filtered_sfrd_acg_for_lya && d_sfr_acg_lya) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(rad_setup->filtered_sfrd_acg_for_lya, d_sfr_acg_lya,
                                         fbytes, cudaMemcpyDeviceToHost, stream));
        if (ok) p21_cuda_stats_add_d2h(fbytes);
    }
    if (ok && rad_setup->filtered_xray_emissivity && d_xray_filt) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(rad_setup->filtered_xray_emissivity, d_xray_filt, fbytes,
                                         cudaMemcpyDeviceToHost, stream));
        if (ok) p21_cuda_stats_add_d2h(fbytes);
    }
    if (ok && use_mcgs && rad_setup->filtered_sfrd_mcg_for_lya && d_sfr_mcg_lya) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(rad_setup->filtered_sfrd_mcg_for_lya, d_sfr_mcg_lya,
                                         fbytes, cudaMemcpyDeviceToHost, stream));
        if (ok) p21_cuda_stats_add_d2h(fbytes);
    }
    if (ok && use_mcgs && lya_ms) {
        if (rad_setup->filtered_sfrd_acg_for_lw && d_sfr_acg_lw) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(rad_setup->filtered_sfrd_acg_for_lw, d_sfr_acg_lw,
                                             fbytes, cudaMemcpyDeviceToHost, stream));
            if (ok) p21_cuda_stats_add_d2h(fbytes);
        }
        if (ok && rad_setup->filtered_sfrd_mcg_for_lw && d_sfr_mcg_lw) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(rad_setup->filtered_sfrd_mcg_for_lw, d_sfr_mcg_lw,
                                             fbytes, cudaMemcpyDeviceToHost, stream));
            if (ok) p21_cuda_stats_add_d2h(fbytes);
        }
    }
    return ok;
}

bool finalize_rad_fields_r0(float redshift, size_t total, int use_xray_heat, int use_mcgs,
                            int use_lya_heat, const float *d_density, const float *d_prev_xe,
                            DeviceRadAccumCache *accum, RadiationFields *radiation_fields,
                            cudaStream_t stream) {
    size_t dbytes = total * sizeof(double);
    double luminosity_conversion_factor;
    if (fabs(astro_params_global->X_RAY_SPEC_INDEX - 1.0) < 1e-6) {
        luminosity_conversion_factor =
            (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz *
            log(astro_params_global->NU_X_BAND_MAX / (astro_params_global->NU_X_THRESH));
        luminosity_conversion_factor = 1.0 / luminosity_conversion_factor;
    } else {
        luminosity_conversion_factor =
            pow((astro_params_global->NU_X_BAND_MAX) * physconst.eV_to_Hz,
                1.0 - (astro_params_global->X_RAY_SPEC_INDEX)) -
            pow((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz,
                1.0 - (astro_params_global->X_RAY_SPEC_INDEX));
        luminosity_conversion_factor = 1.0 / luminosity_conversion_factor;
        luminosity_conversion_factor *=
            pow((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz,
                -(astro_params_global->X_RAY_SPEC_INDEX)) *
            (1.0 - (astro_params_global->X_RAY_SPEC_INDEX));
    }
    luminosity_conversion_factor /= (physconst.h_p);

    double xray_prefactor =
        luminosity_conversion_factor /
        ((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz) * physconst.c_cms *
        pow(1.0 + redshift, astro_params_global->X_RAY_SPEC_INDEX + 3.0);
    double Nb_zp = N_b0 * (1.0 + redshift) * (1.0 + redshift) * (1.0 + redshift);
    double lya_star_prefactor = physconst.c_cms / (4.0 * M_PI) * physconst.Msun /
                                physconst.m_p * (1.0 - 0.75 * cosmo_params_global->Y_He);
    double volunit_inv = pow(physconst.cm_per_Mpc, -3);

    double xray_vol = xray_prefactor * volunit_inv;
    double xray_heat_factor = xray_vol * 2.0 / 3.0 / physconst.k_B;
    double xray_lya_factor = xray_vol * Nb_zp;
    double lya_factor = lya_star_prefactor * volunit_inv;
    double lyw_factor = lya_factor * physconst.h_p * physconst.nu_Ly_alpha /
                        (physconst.nu_ion_HI - physconst.nu_LW_thresh) * 1e21;

    int grid = (int)((total + kBlockThreads - 1) / kBlockThreads);
    multiply_radiation_constants_kernel<<<grid, kBlockThreads, 0, stream>>>(
        total, use_xray_heat, use_mcgs, use_lya_heat, d_density, d_prev_xe, xray_vol,
        xray_heat_factor, xray_lya_factor, lyw_factor, lya_factor, accum->d_xray_heating,
        accum->d_xray_ion, accum->d_xray_lya, accum->d_lya_cont_inj, accum->d_lya_cont,
        accum->d_lya_inj, accum->d_lyw);
    bool ok = P21_CUDA_OK(cudaGetLastError());

    const bool lazy_d2h = p21_cuda_get_lazy_d2h() != 0;
    if (ok && lazy_d2h) {
        auto try_lazy = [&](const void *hptr, double *&dbuf) {
            if (dbuf && hptr && p21_cuda_registry_put_owned_lazy(hptr, dbuf, dbytes, stream)) {
                dbuf = nullptr;
            }
        };
        if (use_xray_heat) try_lazy(radiation_fields->xray_heating_rate, accum->d_xray_heating);
        try_lazy(radiation_fields->xray_ionization_rate, accum->d_xray_ion);
        try_lazy(radiation_fields->xray_lya_flux, accum->d_xray_lya);
        if (use_mcgs) try_lazy(radiation_fields->lyw_flux, accum->d_lyw);
        if (use_lya_heat) {
            try_lazy(radiation_fields->lya_flux_continuum, accum->d_lya_cont);
            try_lazy(radiation_fields->lya_flux_injected, accum->d_lya_inj);
        } else {
            try_lazy(radiation_fields->lya_flux_continuum_injected, accum->d_lya_cont_inj);
        }
    }

    bool need_sync = false;
    auto copy_d2h = [&](double *hptr, const double *dptr) {
        if (!ok || !hptr || !dptr) return;
        ok = P21_CUDA_OK(cudaMemcpyAsync(hptr, dptr, dbytes, cudaMemcpyDeviceToHost, stream));
        if (ok) {
            p21_cuda_stats_add_d2h(dbytes);
            need_sync = true;
        }
    };
    if (ok && use_xray_heat) {
        copy_d2h(radiation_fields->xray_heating_rate, accum->d_xray_heating);
    }
    if (ok) {
        copy_d2h(radiation_fields->xray_ionization_rate, accum->d_xray_ion);
        copy_d2h(radiation_fields->xray_lya_flux, accum->d_xray_lya);
    }
    if (ok && use_mcgs) {
        copy_d2h(radiation_fields->lyw_flux, accum->d_lyw);
    }
    if (ok && use_lya_heat) {
        copy_d2h(radiation_fields->lya_flux_continuum, accum->d_lya_cont);
        copy_d2h(radiation_fields->lya_flux_injected, accum->d_lya_inj);
    } else if (ok) {
        copy_d2h(radiation_fields->lya_flux_continuum_injected, accum->d_lya_cont_inj);
    }
    if (ok && need_sync) {
        ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    if (ok) {
        auto own_or_free = [&](const void *hptr, double *&dbuf) {
            if (!dbuf) return;
            if (p21_cuda_registry_put_owned(hptr, dbuf, dbytes, stream)) {
                dbuf = nullptr;
            } else {
                p21_cuda_pool_free(dbuf, stream);
                dbuf = nullptr;
            }
        };
        own_or_free(radiation_fields->xray_heating_rate, accum->d_xray_heating);
        own_or_free(radiation_fields->xray_ionization_rate, accum->d_xray_ion);
        own_or_free(radiation_fields->xray_lya_flux, accum->d_xray_lya);
        own_or_free(radiation_fields->lyw_flux, accum->d_lyw);
        own_or_free(radiation_fields->lya_flux_continuum, accum->d_lya_cont);
        own_or_free(radiation_fields->lya_flux_injected, accum->d_lya_inj);
        own_or_free(radiation_fields->lya_flux_continuum_injected, accum->d_lya_cont_inj);
        return true;
    }

    free_accum_entry(accum, stream);
    return false;
}

}  // namespace

void p21_cuda_radiation_invalidate_host_ptr(const void *ptr) {
    if (!ptr) {
        return;
    }
    cudaStream_t stream = p21_cuda_stream();
    std::lock_guard<std::mutex> lock(g_rad_mu);
    for (auto it = g_setup_cache.begin(); it != g_setup_cache.end();) {
        const DeviceRadSetupCache &e = it->second;
        if ((const void *)e.key_m_xhii == ptr || (const void *)e.inverse_val_host == ptr ||
            (const void *)e.freq_heat_host == ptr || (const void *)e.freq_ion_host == ptr ||
            (const void *)e.freq_lya_host == ptr) {
            DeviceRadSetupCache copy = it->second;
            it = g_setup_cache.erase(it);
            free_setup_entry(&copy, stream);
        } else {
            ++it;
        }
    }
    auto ait = g_accum_cache.find((const double *)ptr);
    if (ait != g_accum_cache.end()) {
        DeviceRadAccumCache copy = ait->second;
        g_accum_cache.erase(ait);
        free_accum_entry(&copy, stream);
    }
}

void p21_cuda_radiation_clear_setup_cache(void) {
    cudaStream_t stream = p21_cuda_stream();
    std::lock_guard<std::mutex> lock(g_rad_mu);
    for (auto &kv : g_setup_cache) {
        free_setup_entry(&kv.second, stream);
    }
    g_setup_cache.clear();
}

void p21_cuda_radiation_clear(void) {
    cudaStream_t stream = p21_cuda_stream();
    std::lock_guard<std::mutex> lock(g_rad_mu);
    for (auto &kv : g_setup_cache) {
        free_setup_entry(&kv.second, stream);
    }
    g_setup_cache.clear();
    for (auto &kv : g_accum_cache) {
        free_accum_entry(&kv.second, stream);
    }
    g_accum_cache.clear();
    free_elec_interp_cache(stream);
}

extern "C" int p21_cuda_setup_radiation_mean_xe(TsBox *previous_spin_temp, double *x_e_ave_out) {
    if (!previous_spin_temp || !previous_spin_temp->xray_ionised_fraction || !x_e_ave_out ||
        !p21_cuda_get_enabled()) {
        return -1;
    }
    if (simulation_options_global->HII_DIM <= 1 || !p21_cuda_ensure_init()) {
        return -1;
    }

    size_t total = (size_t)simulation_options_global->HII_DIM *
                   (size_t)simulation_options_global->HII_DIM * (size_t)HII_D_PARA;
    size_t bytes = total * sizeof(float);
    cudaStream_t stream = p21_cuda_stream();

    const float *d_prev_xe = (const float *)p21_cuda_registry_get_or_upload(
        previous_spin_temp->xray_ionised_fraction, bytes, stream);
    if (!d_prev_xe) {
        return -1;
    }

    int blocks = (int)((total + kBlockThreads - 1) / kBlockThreads);
    if (blocks > 1024) {
        blocks = 1024;
    }
    double *d_blocks = (double *)p21_cuda_pool_alloc((size_t)blocks * sizeof(double), stream);
    double *d_sum = (double *)p21_cuda_pool_alloc(sizeof(double), stream);
    if (!d_blocks || !d_sum) {
        p21_cuda_pool_free(d_blocks, stream);
        p21_cuda_pool_free(d_sum, stream);
        return -1;
    }

    sum_float_block_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_prev_xe, total, d_blocks);
    bool ok = P21_CUDA_OK(cudaGetLastError());
    if (ok) {
        reduce_double_final_kernel<<<1, kBlockThreads, 0, stream>>>(d_blocks, blocks, d_sum);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    double h_sum = 0.0;
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost,
                                         stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_cuda_pool_free(d_blocks, stream);
    p21_cuda_pool_free(d_sum, stream);
    if (!ok) {
        return -1;
    }

    *x_e_ave_out = h_sum / (float)total;
    return 0;
}

extern "C" int p21_cuda_setup_radiation_tables(TsBox *previous_spin_temp,
                                               RadiationFieldsSetup *rad_setup) {
    if (!previous_spin_temp || !rad_setup || !previous_spin_temp->xray_ionised_fraction ||
        !rad_setup->m_xHII_low_box || !rad_setup->inverse_val_box || !p21_cuda_get_enabled()) {
        return -1;
    }
    if (simulation_options_global->HII_DIM <= 1 || !p21_cuda_ensure_init()) {
        return -1;
    }

    size_t total = (size_t)simulation_options_global->HII_DIM *
                   (size_t)simulation_options_global->HII_DIM * (size_t)HII_D_PARA;
    size_t bytes = total * sizeof(float);
    int num_R = astro_params_global->N_STEP_TS;
    cudaStream_t stream = p21_cuda_stream();

    const float *d_prev_xe = (const float *)p21_cuda_registry_get_or_upload(
        previous_spin_temp->xray_ionised_fraction, bytes, stream);
    if (!d_prev_xe) {
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_rad_mu);
        for (auto &kv : g_setup_cache) {
            free_setup_entry(&kv.second, stream);
        }
        g_setup_cache.clear();
    }

    DeviceRadSetupCache entry{};
    if (!upload_setup_entry(rad_setup, total, num_R, d_prev_xe, true, stream, &entry)) {
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_rad_mu);
        for (auto &kv : g_setup_cache) {
            free_setup_entry(&kv.second, stream);
        }
        g_setup_cache.clear();
        g_setup_cache.emplace(rad_setup->m_xHII_low_box, entry);
    }

    p21_cuda_stats_mut()->radiation_setup_runs++;
    return 0;
}

extern "C" int p21_cuda_update_radiation_fields(float redshift, EmissivityFields *emissivity_fields,
                                                int R_ct, double R_star,
                                                PerturbedField *perturbed_field,
                                                TsBox *previous_spin_temp,
                                                RadiationFieldsSetup *rad_setup,
                                                RadiationFields *radiation_fields) {
    if (!emissivity_fields || !perturbed_field || !previous_spin_temp || !rad_setup ||
        !radiation_fields || !p21_cuda_get_enabled()) {
        return -1;
    }
    if (simulation_options_global->HII_DIM <= 1 || !p21_cuda_ensure_init()) {
        return -1;
    }

    const int nx = simulation_options_global->HII_DIM;
    const int ny = simulation_options_global->HII_DIM;
    const int nz = HII_D_PARA;
    const size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    const size_t fbytes = total * sizeof(float);
    const double box_len = simulation_options_global->BOX_LEN;
    const double ncf = simulation_options_global->NON_CUBIC_FACTOR;
    const int num_R = astro_params_global->N_STEP_TS;
    const int use_xray_heat = astro_options_global->USE_X_RAY_HEATING ? 1 : 0;
    const int use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    const int use_lya_heat = astro_options_global->USE_LYA_HEATING ? 1 : 0;
    const int lya_ms = astro_options_global->LYA_MULTIPLE_SCATTERING ? 1 : 0;
    cudaStream_t stream = p21_cuda_stream();

    DeviceRadSetupCache setup_entry{};
    if (!acquire_rad_setup_cache(rad_setup, total, num_R, stream, &setup_entry)) {
        return -1;
    }

    DeviceRadAccumCache accum{};
    if (!acquire_rad_accum_cache(radiation_fields, R_ct, total, use_xray_heat, use_mcgs,
                                 use_lya_heat, false, stream, &accum)) {
        return -1;
    }

    int lya_filter_type = lya_ms ? FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING
                                 : FILTER_SPHERICAL_SHELL_STRAIGHT_LINE;
    double R_inner = (R_ct == 0) ? 0.0 : rad_setup->R_values[R_ct - 1];
    double R_outer = rad_setup->R_values[R_ct];

    float *d_sfr_acg_lya = nullptr;
    float *d_xray_filt = nullptr;
    float *d_sfr_mcg_lya = nullptr;
    float *d_sfr_acg_lw = nullptr;
    float *d_sfr_mcg_lw = nullptr;
    cufftComplex *d_k_acg = nullptr;
    cufftComplex *d_k_mcg = nullptr;
    cufftComplex *d_k_unused = nullptr;

    bool keep_k_lw = (use_mcgs && lya_ms && R_ct > 0 && R_inner > 0.0);
    bool mgpu_dual = (keep_k_lw && p21_mgpu_eor_enabled(total));
    bool ok = true;
    if (mgpu_dual) {
        size_t fbytes = total * sizeof(float);
        auto run_mgpu_pair = [&](const float *h_src, float **d_lya, float **d_lw) -> bool {
            *d_lya = (float *)p21_cuda_pool_alloc(fbytes, stream);
            *d_lw = (float *)p21_cuda_pool_alloc(fbytes, stream);
            if (!*d_lya || !*d_lw ||
                !P21_CUDA_OK(cudaMemcpyAsync(*d_lya, h_src, fbytes, cudaMemcpyHostToDevice, stream))) {
                return false;
            }
            return mgpu_filter_device_real_pair(
                *d_lya, *d_lw, nx, ny, nz, box_len, ncf, lya_filter_type,
                FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, R_inner, R_outer, R_star, stream);
        };
        ok = run_mgpu_pair(emissivity_fields->sfrd_acg, &d_sfr_acg_lya, &d_sfr_acg_lw) &&
             filter_one_ephemeral_shell(emissivity_fields->xray_emissivity, nx, ny, nz, box_len, ncf,
                                        R_ct, R_inner, R_outer, R_star,
                                        FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, false, &d_xray_filt,
                                        &d_k_unused, stream) &&
             run_mgpu_pair(emissivity_fields->sfrd_mcg, &d_sfr_mcg_lya, &d_sfr_mcg_lw);
    } else {
        ok = filter_one_ephemeral_shell(emissivity_fields->sfrd_acg, nx, ny, nz, box_len, ncf, R_ct,
                                        R_inner, R_outer, R_star, lya_filter_type, keep_k_lw,
                                        &d_sfr_acg_lya, &d_k_acg, stream) &&
             filter_one_ephemeral_shell(emissivity_fields->xray_emissivity, nx, ny, nz, box_len,
                                        ncf, R_ct, R_inner, R_outer, R_star,
                                        FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, false, &d_xray_filt,
                                        &d_k_unused, stream);

        if (ok && use_mcgs) {
            ok = filter_one_ephemeral_shell(emissivity_fields->sfrd_mcg, nx, ny, nz, box_len, ncf,
                                            R_ct, R_inner, R_outer, R_star, lya_filter_type,
                                            keep_k_lw, &d_sfr_mcg_lya, &d_k_mcg, stream);
            if (ok && lya_ms) {
                if (R_ct > 0 && R_inner > 0.0) {
                    ok = filter_from_existing_kspace(d_k_acg, nx, ny, nz, box_len, ncf, R_inner,
                                                     R_outer, R_star, &d_sfr_acg_lw, stream) &&
                         filter_from_existing_kspace(d_k_mcg, nx, ny, nz, box_len, ncf, R_inner,
                                                     R_outer, R_star, &d_sfr_mcg_lw, stream);
                } else {
                    ok = filter_one_ephemeral_shell(
                             emissivity_fields->sfrd_acg, nx, ny, nz, box_len, ncf, R_ct, R_inner,
                             R_outer, R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, false,
                             &d_sfr_acg_lw, &d_k_unused, stream) &&
                         filter_one_ephemeral_shell(
                             emissivity_fields->sfrd_mcg, nx, ny, nz, box_len, ncf, R_ct, R_inner,
                             R_outer, R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, false,
                             &d_sfr_mcg_lw, &d_k_unused, stream);
                }
            }
        }
    }

    p21_cuda_pool_free(d_k_acg, stream);
    p21_cuda_pool_free(d_k_mcg, stream);

    if (ok) {
        ok = launch_rad_shell_accumulate(total, R_ct, num_R, redshift, use_xray_heat, use_mcgs,
                                         use_lya_heat, lya_ms, rad_setup, setup_entry, accum,
                                         d_sfr_acg_lya, d_xray_filt, d_sfr_mcg_lya, d_sfr_acg_lw,
                                         d_sfr_mcg_lw, stream);
    }

    if (ok && R_ct == 0) {
        ok = copy_filtered_shell_r0_d2h(fbytes, use_mcgs, lya_ms, rad_setup, d_sfr_acg_lya,
                                        d_xray_filt, d_sfr_mcg_lya, d_sfr_acg_lw, d_sfr_mcg_lw,
                                        stream);
    }

    p21_cuda_pool_free(d_sfr_acg_lya, stream);
    p21_cuda_pool_free(d_xray_filt, stream);
    p21_cuda_pool_free(d_sfr_mcg_lya, stream);
    p21_cuda_pool_free(d_sfr_acg_lw, stream);
    p21_cuda_pool_free(d_sfr_mcg_lw, stream);

    if (!ok) {
        erase_rad_accum_cache(radiation_fields->xray_ionization_rate, stream);
        return -1;
    }

    if (R_ct == 0) {
        const float *d_density = (const float *)p21_cuda_registry_get_or_upload(
            perturbed_field->density, fbytes, stream);
        const float *d_prev_xe = (const float *)p21_cuda_registry_get_or_upload(
            previous_spin_temp->xray_ionised_fraction, fbytes, stream);
        if (!d_density || !d_prev_xe) {
            erase_rad_accum_cache(radiation_fields->xray_ionization_rate, stream);
            return -1;
        }
        {
            std::lock_guard<std::mutex> lock(g_rad_mu);
            g_accum_cache.erase(radiation_fields->xray_ionization_rate);
        }
        if (!finalize_rad_fields_r0(redshift, total, use_xray_heat, use_mcgs, use_lya_heat,
                                    d_density, d_prev_xe, &accum, radiation_fields, stream)) {
            return -1;
        }
    }

    p21_cuda_stats_mut()->radiation_update_runs++;
    return 0;
}

extern "C" int p21_cuda_compute_radiation_shells(float redshift, int n_shells, const int *R_ct_arr,
                                                 double R_star, const double *interp_params,
                                                 EmissivityFields **ef_desc_arr,
                                                 EmissivityFields **ef_prog_arr,
                                                 RadiationFieldsSetup *rad_setup,
                                                 PerturbedField *perturbed_field,
                                                 TsBox *previous_spin_temp,
                                                 RadiationFields *radiation_fields) {
    if (n_shells <= 0 || !R_ct_arr || !interp_params || !ef_desc_arr || !ef_prog_arr ||
        !rad_setup || !perturbed_field || !previous_spin_temp || !radiation_fields ||
        !p21_cuda_get_enabled()) {
        return -1;
    }
    if (R_ct_arr[n_shells - 1] != 0 || simulation_options_global->HII_DIM <= 1 ||
        !p21_cuda_ensure_init()) {
        return -1;
    }

    const int nx = simulation_options_global->HII_DIM;
    const int ny = simulation_options_global->HII_DIM;
    const int nz = HII_D_PARA;
    const size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    const size_t fbytes = total * sizeof(float);
    const size_t cbytes = (size_t)nx * (size_t)ny * (size_t)(nz / 2 + 1) * sizeof(cufftComplex);
    const double box_len = simulation_options_global->BOX_LEN;
    const double ncf = simulation_options_global->NON_CUBIC_FACTOR;
    const int num_R = astro_params_global->N_STEP_TS;
    const int use_xray_heat = astro_options_global->USE_X_RAY_HEATING ? 1 : 0;
    const int use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    const int use_lya_heat = astro_options_global->USE_LYA_HEATING ? 1 : 0;
    const int lya_ms = astro_options_global->LYA_MULTIPLE_SCATTERING ? 1 : 0;
    const int grid = (int)((total + kBlockThreads - 1) / kBlockThreads);
    const float pre_scale = 1.0f / (float)total;
    const int lya_filter_type = lya_ms ? FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING
                                       : FILTER_SPHERICAL_SHELL_STRAIGHT_LINE;
    cudaStream_t stream = p21_cuda_stream();

    DeviceRadSetupCache setup_entry{};
    if (!acquire_rad_setup_cache(rad_setup, total, num_R, stream, &setup_entry)) {
        return -1;
    }

    const float *d_density =
        (const float *)p21_cuda_registry_get_or_upload(perturbed_field->density, fbytes, stream);
    const float *d_prev_xe = (const float *)p21_cuda_registry_get_or_upload(
        previous_spin_temp->xray_ionised_fraction, fbytes, stream);
    if (!d_density || !d_prev_xe) {
        return -1;
    }

    DeviceRadAccumCache accum{};
    if (!acquire_rad_accum_cache(radiation_fields, R_ct_arr[0], total, use_xray_heat, use_mcgs,
                                 use_lya_heat, true, stream, &accum)) {
        return -1;
    }

    const bool need_k_unfilt = (use_mcgs && lya_ms);
    const bool mgpu_eor = p21_mgpu_eor_enabled(total);
    float *d_sfr_acg_lya = (float *)p21_cuda_pool_alloc(fbytes, stream);
    float *d_xray_filt = (float *)p21_cuda_pool_alloc(fbytes, stream);
    float *d_sfr_mcg_lya = use_mcgs ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_sfr_acg_lw =
        need_k_unfilt ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_sfr_mcg_lw =
        need_k_unfilt ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    P21MgpuGrid mgpu_grid{};
    std::vector<cufftComplex *> y_unfilt, y_work;
    cufftComplex *d_k_unfilt = nullptr;
    cufftComplex *d_k_work = nullptr;
    bool k_alloc_ok = true;
    if (mgpu_eor) {
        k_alloc_ok = mgpu_grid.init(nx, ny, nz) && mgpu_alloc_rad_slabs(mgpu_grid, y_work) &&
                     (!need_k_unfilt || mgpu_alloc_rad_slabs(mgpu_grid, y_unfilt));
    } else {
        d_k_unfilt =
            need_k_unfilt ? (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream) : nullptr;
        d_k_work = (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream);
        k_alloc_ok = (d_k_work != nullptr) && (!need_k_unfilt || d_k_unfilt != nullptr);
    }

    bool ok = d_sfr_acg_lya && d_xray_filt && k_alloc_ok && (!use_mcgs || d_sfr_mcg_lya) &&
              (!need_k_unfilt || (d_sfr_acg_lw && d_sfr_mcg_lw));

    for (int s = 0; ok && s < n_shells; ++s) {
        int R_ct = R_ct_arr[s];
        double w = interp_params[s];
        EmissivityFields *ef_d = ef_desc_arr[s];
        EmissivityFields *ef_p = ef_prog_arr[s];
        if (!ef_d || !ef_p) {
            ok = false;
            break;
        }
        double R_inner = (R_ct == 0) ? 0.0 : rad_setup->R_values[R_ct - 1];
        double R_outer = rad_setup->R_values[R_ct];
        bool no_fft = (R_ct == 0 || R_inner <= 0.0);

        auto run_field = [&](const float *h_desc, const float *h_prog, int ftype, bool also_lw,
                             float *d_out, float *d_out_lw) -> bool {
            if (!h_desc || !h_prog) return false;
            const float *d_desc =
                (const float *)p21_cuda_registry_get_or_upload(h_desc, fbytes, stream);
            const float *d_prog =
                (const float *)p21_cuda_registry_get_or_upload(h_prog, fbytes, stream);
            if (!d_desc || !d_prog) return false;
            if (no_fft) {
                lerp_field_kernel<<<grid, kBlockThreads, 0, stream>>>(total, d_desc, d_prog, w,
                                                                      d_out, 1);
                return P21_CUDA_OK(cudaGetLastError());
            }
            lerp_field_kernel<<<grid, kBlockThreads, 0, stream>>>(total, d_desc, d_prog, w, d_out,
                                                                  0);
            if (!P21_CUDA_OK(cudaGetLastError())) return false;
            if (mgpu_eor) {
                if (also_lw && d_out_lw) {
                    return p21_mgpu_r2c_from_device(mgpu_grid, d_out, stream, 1.0,
                                                    y_unfilt.data(), nullptr) &&
                           p21_mgpu_filter_yslabs(mgpu_grid, y_unfilt.data(), y_work.data(),
                                                  box_len, ncf, ftype, (float)R_inner,
                                                  (float)R_outer, (float)R_star, pre_scale) &&
                           p21_mgpu_c2r_to_device(mgpu_grid, y_work.data(), d_out, stream,
                                                  nullptr) &&
                           p21_mgpu_filter_yslabs(mgpu_grid, y_unfilt.data(), y_work.data(),
                                                  box_len, ncf, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE,
                                                  (float)R_inner, (float)R_outer, (float)R_star,
                                                  pre_scale) &&
                           p21_mgpu_c2r_to_device(mgpu_grid, y_work.data(), d_out_lw, stream,
                                                  nullptr);
                }
                return p21_mgpu_r2c_from_device(mgpu_grid, d_out, stream, 1.0, y_work.data(),
                                                nullptr) &&
                       p21_mgpu_filter_yslabs(mgpu_grid, y_work.data(), y_work.data(), box_len, ncf,
                                              ftype, (float)R_inner, (float)R_outer, (float)R_star,
                                              pre_scale) &&
                       p21_mgpu_c2r_to_device(mgpu_grid, y_work.data(), d_out, stream, nullptr);
            }
            if (also_lw && d_out_lw && d_k_unfilt) {
                return p21_cuda_fft_r2c(d_out, d_k_unfilt, nx, ny, nz, false, stream) &&
                       p21_cuda_copy_and_filter_box(d_k_unfilt, d_k_work, nx, ny, nz, box_len, ncf,
                                                    ftype, (float)R_inner, (float)R_outer,
                                                    (float)R_star, pre_scale, stream) &&
                       p21_cuda_fft_c2r(d_k_work, d_out, nx, ny, nz, false, stream) &&
                       p21_cuda_copy_and_filter_box(
                           d_k_unfilt, d_k_work, nx, ny, nz, box_len, ncf,
                           FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, (float)R_inner, (float)R_outer,
                           (float)R_star, pre_scale, stream) &&
                       p21_cuda_fft_c2r(d_k_work, d_out_lw, nx, ny, nz, false, stream);
            }
            return p21_cuda_fft_r2c(d_out, d_k_work, nx, ny, nz, false, stream) &&
                   p21_cuda_copy_and_filter_box(d_k_work, d_k_work, nx, ny, nz, box_len, ncf, ftype,
                                                (float)R_inner, (float)R_outer, (float)R_star,
                                                pre_scale, stream) &&
                   p21_cuda_fft_c2r(d_k_work, d_out, nx, ny, nz, false, stream);
        };

        float *lw_acg_ptr = (use_mcgs && lya_ms) ? (no_fft ? d_sfr_acg_lya : d_sfr_acg_lw) : nullptr;
        float *lw_mcg_ptr = (use_mcgs && lya_ms) ? (no_fft ? d_sfr_mcg_lya : d_sfr_mcg_lw) : nullptr;

        ok = run_field(ef_d->sfrd_acg, ef_p->sfrd_acg, lya_filter_type, (bool)(use_mcgs && lya_ms),
                       d_sfr_acg_lya, lw_acg_ptr) &&
             run_field(ef_d->xray_emissivity, ef_p->xray_emissivity,
                       FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, false, d_xray_filt, nullptr);
        if (ok && use_mcgs) {
            ok = run_field(ef_d->sfrd_mcg, ef_p->sfrd_mcg, lya_filter_type, (bool)lya_ms,
                           d_sfr_mcg_lya, lw_mcg_ptr);
        }
        if (ok) {
            ok = launch_rad_shell_accumulate(total, R_ct, num_R, redshift, use_xray_heat, use_mcgs,
                                             use_lya_heat, lya_ms, rad_setup, setup_entry, accum,
                                             d_sfr_acg_lya, d_xray_filt, d_sfr_mcg_lya, lw_acg_ptr,
                                             lw_mcg_ptr, stream);
        }
        if (ok && R_ct == 0) {
            ok = copy_filtered_shell_r0_d2h(fbytes, use_mcgs, lya_ms, rad_setup, d_sfr_acg_lya,
                                            d_xray_filt, d_sfr_mcg_lya, lw_acg_ptr, lw_mcg_ptr,
                                            stream);
        }
    }

    p21_cuda_pool_free(d_sfr_acg_lya, stream);
    p21_cuda_pool_free(d_xray_filt, stream);
    p21_cuda_pool_free(d_sfr_mcg_lya, stream);
    p21_cuda_pool_free(d_sfr_acg_lw, stream);
    p21_cuda_pool_free(d_sfr_mcg_lw, stream);
    p21_cuda_pool_free(d_k_unfilt, stream);
    p21_cuda_pool_free(d_k_work, stream);
    if (mgpu_eor) {
        mgpu_free_rad_slabs(mgpu_grid, y_unfilt);
        mgpu_free_rad_slabs(mgpu_grid, y_work);
        mgpu_grid.destroy();
    }

    if (!ok) {
        free_accum_entry(&accum, stream);
        return -1;
    }

    if (!finalize_rad_fields_r0(redshift, total, use_xray_heat, use_mcgs, use_lya_heat, d_density,
                                d_prev_xe, &accum, radiation_fields, stream)) {
        return -1;
    }

    p21_cuda_stats_mut()->radiation_update_runs += (unsigned long long)n_shells;
    return 0;
}

extern "C" int p21_cuda_fill_freqint_tables(double zp, int num_R, const double *lower_int_limits,
                                            double hubble_zp, RadiationFieldsSetup *rad_setup) {
    (void)zp;
    if (num_R <= 0 || num_R > 65535 || hubble_zp <= 0.0 || !lower_int_limits || !rad_setup ||
        !p21_cuda_get_enabled()) {
        return -1;
    }
    if (!rad_setup->freq_int_heat_tbl || !rad_setup->freq_int_ion_tbl ||
        !rad_setup->freq_int_lya_tbl || !rad_setup->freq_int_heat_tbl_diff ||
        !rad_setup->freq_int_ion_tbl_diff || !rad_setup->freq_int_lya_tbl_diff ||
        !rad_setup->inverse_diff) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    DeviceElecInterpCache elec{};
    if (!ensure_elec_interp_uploaded(stream, &elec)) {
        return -1;
    }

    const float *h_xhii = nullptr;
    if (!p21_get_elec_interp_tables(&h_xhii, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) ||
        !h_xhii) {
        return -1;
    }

    const size_t n_entries = (size_t)x_int_NXHII * (size_t)num_R;
    const size_t limit_bytes = (size_t)num_R * sizeof(double);
    const size_t tbl_bytes = n_entries * sizeof(double);
    const size_t total_bytes = limit_bytes + 3 * tbl_bytes;

    double *d_buf = (double *)p21_cuda_pool_alloc(total_bytes, stream);
    if (!d_buf) {
        return -1;
    }

    double *d_lower = d_buf;
    double *d_heat = d_buf + (size_t)num_R;
    double *d_ion = d_heat + n_entries;
    double *d_lya = d_ion + n_entries;

    FreqIntKernelParams params{};
    params.h_frac = H_FRAC;
    params.he_frac = HE_FRAC;
    params.ev_to_hz = physconst.eV_to_Hz;
    params.h_p = physconst.h_p;
    params.nu_ion_hi = physconst.nu_ion_HI;
    params.nu_ion_hei = physconst.nu_ion_HeI;
    params.nu_ion_heii = physconst.nu_ion_HeII;
    params.nu_x_thresh_hz = (double)astro_params_global->NU_X_THRESH * physconst.eV_to_Hz;
    params.nu_x_max_hz = astro_params_global->NU_X_MAX * physconst.eV_to_Hz;
    params.x_ray_spec_index = astro_params_global->X_RAY_SPEC_INDEX;
    params.tiny_val = TINY;
    params.lya_prefactor = physconst.c_cms / (4.0 * M_PI) / physconst.nu_Ly_alpha / hubble_zp;

    if (params.nu_x_thresh_hz <= 0.0) {
        p21_cuda_pool_free(d_buf, stream);
        return -1;
    }

    bool ok = P21_CUDA_OK(
        cudaMemcpyAsync(d_lower, lower_int_limits, limit_bytes, cudaMemcpyHostToDevice, stream));
    if (ok) {
        const dim3 grid((unsigned int)num_R, (unsigned int)x_int_NXHII, 1);
        fill_freqint_kernel<<<grid, kBlockThreads, 0, stream>>>(
            num_R, params, d_lower, elec.d_xhii, elec.d_energy, elec.d_fheat, elec.d_nlya,
            elec.d_nion_hi, elec.d_nion_hei, elec.d_nion_heii, d_heat, d_ion, d_lya);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(rad_setup->freq_int_heat_tbl, d_heat, tbl_bytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(rad_setup->freq_int_ion_tbl, d_ion, tbl_bytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(rad_setup->freq_int_lya_tbl, d_lya, tbl_bytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_cuda_pool_free(d_buf, stream);
    if (!ok) {
        return -1;
    }

    for (int x_e_ct = 1; x_e_ct < x_int_NXHII; x_e_ct++) {
        for (int R_ct = 0; R_ct < num_R; R_ct++) {
            const size_t cur = (size_t)x_e_ct * (size_t)num_R + (size_t)R_ct;
            const size_t prev = (size_t)(x_e_ct - 1) * (size_t)num_R + (size_t)R_ct;
            rad_setup->freq_int_heat_tbl_diff[prev] =
                rad_setup->freq_int_heat_tbl[cur] - rad_setup->freq_int_heat_tbl[prev];
            rad_setup->freq_int_ion_tbl_diff[prev] =
                rad_setup->freq_int_ion_tbl[cur] - rad_setup->freq_int_ion_tbl[prev];
            rad_setup->freq_int_lya_tbl_diff[prev] =
                rad_setup->freq_int_lya_tbl[cur] - rad_setup->freq_int_lya_tbl[prev];
        }
    }
    for (int x_e_ct = 0; x_e_ct < x_int_NXHII - 1; x_e_ct++) {
        rad_setup->inverse_diff[x_e_ct] = 1.0 / (h_xhii[x_e_ct + 1] - h_xhii[x_e_ct]);
    }
    return 0;
}
