#include <algorithm>
#include <atomic>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

#include "cuda_common.cuh"
#include "cuda_mgpu.cuh"
#include "cuda_philox.cuh"
#include "cuda_rng.cuh"

namespace {

struct CosmoKernelParams {
    int power_spectrum;
    int v_cb_model;
    double sound_horizon;
    double alpha_nu;
    double beta_c;
    double omhh;
    double f_nu;
    double theta_cmb;
    double sigma_norm;
    double eh_ratio_at_kmax;
    double ps_norm;
    double power_index;
    double omm;
    double omb;
    double hlittle;
    double c_kms;
    const double *class_dens_table;
    int class_dens_rows;
    const double *class_vcb_table;
    int class_vcb_rows;
};

struct HostSpline {
    std::vector<double> packed; /* [5 * n]: x, y, b, c, d */
    int n = 0;
};

/* Natural cubic spline coefficients matching gsl_interp_cspline. */
bool build_natural_cubic_spline(const double *x, const double *y, int n, HostSpline *out) {
    if (!x || !y || n < 3) {
        return false;
    }
    std::vector<double> h(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
        if (h[i] <= 0.0) {
            return false;
        }
    }

    /* Solve tridiagonal system for c[1..n-2] (c[0] = c[n-1] = 0). */
    std::vector<double> c(n, 0.0);
    std::vector<double> cp(n, 0.0);
    std::vector<double> dp(n, 0.0);
    for (int i = 1; i < n - 1; ++i) {
        double a_i = h[i - 1];
        double b_i = 2.0 * (h[i - 1] + h[i]);
        double c_i = (i + 1 < n - 1) ? h[i] : 0.0;
        double r_i = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        double denom = b_i - a_i * cp[i - 1];
        cp[i] = c_i / denom;
        dp[i] = (r_i - a_i * dp[i - 1]) / denom;
    }
    for (int i = n - 2; i >= 1; --i) {
        c[i] = dp[i] - cp[i] * c[i + 1];
    }

    out->n = n;
    out->packed.assign((size_t)5 * n, 0.0);
    for (int i = 0; i < n; ++i) {
        out->packed[i] = x[i];
        out->packed[n + i] = y[i];
        out->packed[3 * n + i] = c[i];
    }
    for (int i = 0; i < n - 1; ++i) {
        out->packed[2 * n + i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
        out->packed[4 * n + i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
    }
    return true;
}

__device__ static inline double dev_eval_spline(const double *table, int n, double k) {
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (k < table[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    double dx = k - table[lo];
    return table[n + lo] +
           dx * (table[2 * n + lo] + dx * (table[3 * n + lo] + dx * table[4 * n + lo]));
}

__device__ static inline double dev_transfer_eh(double k, const CosmoKernelParams &p) {
    const double n_nu = 1.0;
    double q = k * (p.theta_cmb * p.theta_cmb) / p.omhh;
    double sqrt_a = sqrt(p.alpha_nu);
    double gamma_eff =
        sqrt_a + (1.0 - sqrt_a) / (1.0 + pow(0.43 * k * p.sound_horizon, 4.0));
    double q_eff = q / gamma_eff;
    double tf = log(M_E + 1.84 * p.beta_c * sqrt_a * q_eff);
    tf /= tf + q_eff * q_eff * (14.4 + 325.0 / (1.0 + 60.5 * pow(q_eff, 1.11)));
    double q_nu = 3.92 * q / sqrt(p.f_nu / n_nu);
    tf *= 1.0 + (1.2 * pow(p.f_nu, 0.64) * pow(n_nu, 0.3 + 0.6 * p.f_nu)) /
                    (pow(q_nu, -1.6) + pow(q_nu, 0.8));
    return tf;
}

__device__ static inline double dev_transfer_function(double k, const CosmoKernelParams &p) {
    switch (p.power_spectrum) {
        case POWER_SPECTRUM_EH:
            return dev_transfer_eh(k, p);
        case POWER_SPECTRUM_BBKS: {
            double gamma = p.omm * p.hlittle * exp(-p.omb - (p.omb / p.omm));
            double q = k / (p.hlittle * gamma);
            return (log(1.0 + 2.34 * q) / (2.34 * q)) *
                   pow(1.0 + 3.89 * q + pow(16.1 * q, 2.0) + pow(5.46 * q, 3.0) +
                           pow(6.71 * q, 4.0),
                       -0.25);
        }
        case POWER_SPECTRUM_EFSTATHIOU: {
            double gamma = p.omm * p.hlittle * p.hlittle;
            double aa = 6.4 / gamma;
            double bb = 3.0 / gamma;
            double cc = 1.7 / gamma;
            double nu = 1.13;
            return pow(1.0 + pow(aa * k + pow(bb * k, 1.5) + cc * cc * k * k, nu), -1.0 / nu);
        }
        case POWER_SPECTRUM_PEEBLES: {
            double gamma = p.omm * p.hlittle * exp(-p.omb - (p.omb / p.omm));
            double aa = 8.0 / (p.hlittle * gamma);
            double bb = 4.7 / pow(p.hlittle * gamma, 2.0);
            return 1.0 + aa * k + bb * k * k;
        }
        case POWER_SPECTRUM_WHITE: {
            double gamma = p.omm * p.hlittle * p.hlittle * exp(-p.omb - (p.omb / p.omm));
            double aa = 1.7 / gamma;
            double bb = 9.0 / pow(gamma, 1.5);
            double cc = 1.0 / (gamma * gamma);
            return 139.284 / (1.0 + aa * k + bb * pow(k, 1.5) + cc * k * k);
        }
        case POWER_SPECTRUM_CLASS: {
            int n = p.class_dens_rows;
            const double *tbl = p.class_dens_table;
            if (k > tbl[n - 1]) {
                return p.eh_ratio_at_kmax * dev_transfer_eh(k, p) * k * k;
            }
            return dev_eval_spline(tbl, n, k);
        }
        default:
            return 0.0;
    }
}

__device__ static inline double dev_power_in_k(double k, const CosmoKernelParams &p) {
    if (k == 0.0) {
        return 0.0;
    }
    double tf = dev_transfer_function(k, p);
    if (p.power_spectrum < POWER_SPECTRUM_CLASS) {
        tf *= k * k;
    }
    double primordial = p.ps_norm * pow(k / 0.05, p.power_index - 1.0);
    double pk = p.sigma_norm * primordial * tf * tf / (k * k * k);
    if (p.power_spectrum == POWER_SPECTRUM_CLASS && p.v_cb_model != V_CB_MODEL_NO) {
        double lr = log(k / 300.0);
        pk *= 1.0 - 0.24 * exp(-(lr * lr) / (2.0 * 0.9 * 0.9));
    }
    return pk;
}

__device__ static inline double dev_power_in_vcb(double k, const CosmoKernelParams &p) {
    if (k == 0.0) {
        return 0.0;
    }
    int n = p.class_vcb_rows;
    const double *tbl = p.class_vcb_table;
    double tv = 0.0;
    if (k > tbl[n - 1]) {
        tv = exp(log(tbl[2 * n - 1]) +
                 (log(tbl[2 * n - 1]) - log(tbl[2 * n - 2])) /
                     (log(tbl[n - 1]) - log(tbl[n - 2])) * (log(k) - log(tbl[n - 1])));
    } else {
        tv = dev_eval_spline(tbl, n, k);
    }
    double primordial = p.ps_norm * pow(k / 0.05, p.power_index - 1.0);
    return p.sigma_norm * primordial * tv * tv / (k * k * k);
}

__global__ void scale_real_kernel(const float *src, float *dst, size_t n, double factor) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)((double)src[idx] * factor);
    }
}

__global__ void fill_ic_modes_slice_kernel(cufftComplex *box, const double *gauss, int first_x,
                                           int count_x, int ny, int nz, double box_len_x,
                                           double box_len_y, double box_len_z, double volume,
                                           CosmoKernelParams params) {
    int hz = nz / 2 + 1;
    size_t local_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slice_modes = (size_t)count_x * ny * hz;
    if (local_idx >= slice_modes) {
        return;
    }

    int n_z = (int)(local_idx % hz);
    size_t q = local_idx / hz;
    int n_y = (int)(q % ny);
    int n_x = first_x + (int)(q / ny);

    double kx = p21_index_to_k(n_x, box_len_x, ny); /* hi_dim[0] == hi_dim[1] == DIM */
    double ky = p21_index_to_k(n_y, box_len_y, ny);
    double kz = p21_index_to_k(n_z, box_len_z, nz);
    double k_mag = sqrt(kx * kx + ky * ky + kz * kz);
    double p = dev_power_in_k(k_mag, params);
    double amp = sqrt(volume * p / 2.0);
    double a = gauss[2 * local_idx];
    double b = gauss[2 * local_idx + 1];

    size_t out_idx = ((size_t)n_x * ny + (size_t)n_y) * hz + (size_t)n_z;
    box[out_idx] = make_float2((float)(amp * a), (float)(amp * b));
}

__global__ void fill_ic_modes_fast_kernel(cufftComplex *box, uint64_t seed, int nx, int ny, int nz,
                                          double box_len_x, double box_len_y, double box_len_z,
                                          double volume, CosmoKernelParams params) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_modes = (size_t)nx * ny * hz;
    if (idx >= total_modes) {
        return;
    }

    int n_z = (int)(idx % hz);
    size_t q = idx / hz;
    int n_y = (int)(q % ny);
    int n_x = (int)(q / ny);

    double kx = p21_index_to_k(n_x, box_len_x, nx);
    double ky = p21_index_to_k(n_y, box_len_y, ny);
    double kz = p21_index_to_k(n_z, box_len_z, nz);
    double k_mag = sqrt(kx * kx + ky * ky + kz * kz);
    double p = dev_power_in_k(k_mag, params);
    double amp = sqrt(volume * p / 2.0);

    P21PhiloxRng rng;
    rng.init(seed, (uint64_t)idx, 0x49U);
    double a = 0.0;
    double b = 0.0;
    rng.ugaussian_pair(&a, &b);

    box[idx] = make_float2((float)(amp * a), (float)(amp * b));
}

__global__ void enforce_real_planes_general_kernel(cufftComplex *modes, int nx, int ny, int nz) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane_sz = (size_t)nx * ny;
    if (index >= 2 * plane_sz) {
        return;
    }
    int mx = nx / 2;
    int my = ny / 2;
    int mz = nz / 2;
    int hz = nz / 2 + 1;
    int plane_k = (int)(index % 2);
    if (plane_k && mz == 0) {
        return;
    }
    int k = plane_k ? mz : 0;
    size_t q = index / 2;
    int i = (int)(q / ny);
    int j = (int)(q % ny);
    size_t pos = ((size_t)i * ny + (size_t)j) * hz + (size_t)k;

    if (i == 0 && j == 0 && k == 0) {
        modes[0] = make_float2(0.0f, 0.0f);
    } else if ((i == 0 || i == mx) && (j == 0 || j == my)) {
        modes[pos].y = 0.0f;
    } else if (i > 0 && i < mx) {
        if (j == 0 || j == my) {
            cufftComplex v = modes[((size_t)(nx - i) * ny + (size_t)j) * hz + (size_t)k];
            modes[pos] = make_float2(v.x, -v.y);
        } else if (j < my || j > ny - my) {
            int sy = ny - j;
            cufftComplex v = modes[((size_t)(nx - i) * ny + (size_t)sy) * hz + (size_t)k];
            modes[pos] = make_float2(v.x, -v.y);
        }
    } else if ((i == 0 || i == mx) && j > 0 && j < my) {
        cufftComplex v = modes[((size_t)i * ny + (size_t)(ny - j)) * hz + (size_t)k];
        modes[pos] = make_float2(v.x, -v.y);
    }
}

__global__ void resample_hi_to_out_kernel(const float *hi_real, int nx, int ny, int nz,
                                          int hi_z_stride, float *out_real, int ox, int oy, int oz,
                                          double dim_ratio, double volume, bool divide_vol,
                                          bool accum_sq) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)ox * oy * oz;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % oz);
    size_t q = idx / oz;
    int j = (int)(q % oy);
    int i = (int)(q / oy);

    int ri = p21_resample_coord(i, dim_ratio);
    int rj = p21_resample_coord(j, dim_ratio);
    int rk = p21_resample_coord(k, dim_ratio);
    float val = hi_real[p21_grid_idx(ri, rj, rk, ny, hi_z_stride)];

    if (accum_sq) {
        double dv = (double)val;
        out_real[idx] += (float)(dv * dv);
    } else if (divide_vol) {
        out_real[idx] = (float)((double)val / volume);
    } else {
        out_real[idx] = val;
    }
}

__global__ void scale_padded_real_kernel(float *buf, int nx, int ny, int nz, int z_stride,
                                         double factor) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * nz;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % nz);
    size_t q = idx / nz;
    int j = (int)(q % ny);
    int i = (int)(q / ny);
    size_t pidx = p21_grid_idx(i, j, k, ny, z_stride);
    buf[pidx] = (float)((double)buf[pidx] * factor);
}

__global__ void pack_padded_to_dense_kernel(const float *src, float *dst, int nx, int ny, int nz,
                                            int z_stride, double scale) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * nz;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % nz);
    size_t q = idx / nz;
    int j = (int)(q % ny);
    int i = (int)(q / ny);
    double val = (double)src[p21_grid_idx(i, j, k, ny, z_stride)];
    dst[idx] = (scale == 1.0) ? (float)val : (float)(val * scale);
}

__device__ static inline double mgpu_tophat_w(double kR) {
    if (kR < 1e-4) {
        return 1.0 - kR * kR / 10.0;
    }
    return 3.0 * pow(kR, -3) * (sin(kR) - cos(kR) * kR);
}

__global__ void fill_ic_modes_fast_yslab_kernel(cufftComplex *y_slab, uint64_t seed, int y0,
                                                 int sy, int nx, int ny, int nz, double box_len_x,
                                                 double box_len_y, double box_len_z, double volume,
                                                 CosmoKernelParams params) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sy * (size_t)hz * (size_t)nx;
    if (idx >= total) {
        return;
    }
    int n_x = (int)(idx % (size_t)nx);
    size_t q = idx / (size_t)nx;
    int n_z = (int)(q % (size_t)hz);
    int n_y = y0 + (int)(q / (size_t)hz);

    double kx = p21_index_to_k(n_x, box_len_x, nx);
    double ky = p21_index_to_k(n_y, box_len_y, ny);
    double kz = p21_index_to_k(n_z, box_len_z, nz);
    double k_mag = sqrt(kx * kx + ky * ky + kz * kz);
    double p = dev_power_in_k(k_mag, params);
    double amp = sqrt(volume * p / 2.0);

    uint64_t canon_idx =
        ((uint64_t)n_x * (uint64_t)ny + (uint64_t)n_y) * (uint64_t)hz + (uint64_t)n_z;
    P21PhiloxRng rng;
    rng.init(seed, canon_idx, 0x49U);
    double a = 0.0;
    double b = 0.0;
    rng.ugaussian_pair(&a, &b);

    y_slab[idx] = make_float2((float)(amp * a), (float)(amp * b));
}

__global__ void filter_tophat_yslab_kernel(cufftComplex *y_slab, int y0, int sy, int nx, int ny,
                                           int nz, double dk_x, double dk_y, double dk_z,
                                           float tophat_R) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sy * (size_t)hz * (size_t)nx;
    if (idx >= total) {
        return;
    }
    int n_x = (int)(idx % (size_t)nx);
    size_t q = idx / (size_t)nx;
    int n_z = (int)(q % (size_t)hz);
    int n_y = y0 + (int)(q / (size_t)hz);

    float kx = (n_x > nx / 2) ? (float)((n_x - nx) * dk_x) : (float)(n_x * dk_x);
    float ky = (n_y > ny / 2) ? (float)((n_y - ny) * dk_y) : (float)(n_y * dk_y);
    float kz = (float)(n_z * dk_z);
    float kR = sqrtf(kx * kx + ky * ky + kz * kz) * tophat_R;
    double w = mgpu_tophat_w((double)kR);

    cufftComplex v = y_slab[idx];
    y_slab[idx] = make_float2((float)((double)v.x * w), (float)((double)v.y * w));
}

__global__ void vcb_kspace_axis_yslab_kernel(const cufftComplex *saved, cufftComplex *out, int y0,
                                             int sy, int nx, int ny, int nz, double len_x,
                                             double len_y, double len_z, int axis,
                                             CosmoKernelParams params) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sy * (size_t)hz * (size_t)nx;
    if (idx >= total) {
        return;
    }
    int n_x = (int)(idx % (size_t)nx);
    size_t q = idx / (size_t)nx;
    int n_z = (int)(q % (size_t)hz);
    int n_y = y0 + (int)(q / (size_t)hz);
    if (n_x == 0 && n_y == 0 && n_z == 0) {
        out[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_mag = sqrt(kx * kx + ky * ky + kz * kz);
    double p = dev_power_in_k(k_mag, params);
    double p_vcb = dev_power_in_vcb(k_mag, params);
    double k_comp = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
    double factor = (k_comp / k_mag) * sqrt(p_vcb / p) * params.c_kms;

    cufftComplex v = saved[idx];
    out[idx] = make_float2((float)(-(double)v.y * factor), (float)((double)v.x * factor));
}

__global__ void f_gradient_yslab_kernel(const cufftComplex *in, cufftComplex *out, int y0, int sy,
                                        int nx, int ny, int nz, double len_x, double len_y,
                                        double len_z, int axis, int do_tophat, float tophat_R) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sy * (size_t)hz * (size_t)nx;
    if (idx >= total) {
        return;
    }
    int n_x = (int)(idx % (size_t)nx);
    size_t q = idx / (size_t)nx;
    int n_z = (int)(q % (size_t)hz);
    int n_y = y0 + (int)(q / (size_t)hz);
    if (n_x == 0 && n_y == 0 && n_z == 0) {
        out[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_sq = kx * kx + ky * ky + kz * kz;
    double k_comp = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
    double factor = k_comp / k_sq;
    if (do_tophat) {
        float kR = (float)sqrt(k_sq) * tophat_R;
        factor *= mgpu_tophat_w((double)kR);
    }

    cufftComplex v = in[idx];
    out[idx] = make_float2((float)(-(double)v.y * factor), (float)((double)v.x * factor));
}

__global__ void f_laplacian_yslab_kernel(const cufftComplex *in, cufftComplex *out, int y0, int sy,
                                         int nx, int ny, int nz, double len_x, double len_y,
                                         double len_z, int ax0, int ax1) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sy * (size_t)hz * (size_t)nx;
    if (idx >= total) {
        return;
    }
    int n_x = (int)(idx % (size_t)nx);
    size_t q = idx / (size_t)nx;
    int n_z = (int)(q % (size_t)hz);
    int n_y = y0 + (int)(q / (size_t)hz);
    if (n_x == 0 && n_y == 0 && n_z == 0) {
        out[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_sq = kx * kx + ky * ky + kz * kz;
    double k0 = (ax0 == 0) ? kx : ((ax0 == 1) ? ky : kz);
    double k1 = (ax1 == 0) ? kx : ((ax1 == 1) ? ky : kz);
    double factor = -(k0 * k1) / k_sq;

    cufftComplex v = in[idx];
    out[idx] = make_float2((float)((double)v.x * factor), (float)((double)v.y * factor));
}

__global__ void resample_strip_to_out_kernel(const float *d_strip, int x_global_start, int ny,
                                             int nz, int ox_g_start, int ox_begin, int ox_count,
                                             int oy, int oz, double dim_ratio, double volume,
                                             bool divide_vol, bool accum_sq, float *d_out_lo) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)ox_count * (size_t)oy * (size_t)oz;
    if (idx >= total) {
        return;
    }
    int ok = (int)(idx % (size_t)oz);
    size_t q = idx / (size_t)oz;
    int oj = (int)(q % (size_t)oy);
    int oi = ox_begin + (int)(q / (size_t)oy);

    int ri = p21_resample_coord(oi, dim_ratio) - x_global_start;
    int rj = p21_resample_coord(oj, dim_ratio);
    int rk = p21_resample_coord(ok, dim_ratio);
    float val = d_strip[p21_grid_idx(ri, rj, rk, ny, nz)];

    size_t out_idx = (((size_t)(oi - ox_g_start) * (size_t)oy + (size_t)oj) * (size_t)oz) +
                     (size_t)ok;
    if (accum_sq) {
        double dv = (double)val;
        d_out_lo[out_idx] += (float)(dv * dv);
    } else if (divide_vol) {
        d_out_lo[out_idx] = (float)((double)val / volume);
    } else {
        d_out_lo[out_idx] = val;
    }
}

__global__ void sub_sq_2lpt_dense_kernel(float *rhs, const float *phi_ab, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double c = (double)phi_ab[idx];
        rhs[idx] = (float)((double)rhs[idx] - c * c);
    }
}

__global__ void accum_2lpt_d01_dense_kernel(float *rhs, float *diag_acc, const float *d1,
                                          size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double d0 = (double)diag_acc[idx];
        double d1v = (double)d1[idx];
        rhs[idx] = (float)((double)rhs[idx] + d0 * d1v);
        diag_acc[idx] = (float)(d0 + d1v);
    }
}

__global__ void accum_2lpt_d01_d2_dense_kernel(float *rhs, const float *diag_acc, const float *d2,
                                             size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double d01 = (double)diag_acc[idx];
        double d2v = (double)d2[idx];
        rhs[idx] = (float)((double)rhs[idx] + d01 * d2v);
    }
}

bool mgpu_ic_force_enabled(void) {
    const char *env = getenv("P21_CUDA_FORCE_MGPU");
    return env && strcmp(env, "1") == 0;
}

bool mgpu_ic_force_single_slab(void) {
    const char *env = getenv("P21_CUDA_MGPU_SINGLE_SLAB");
    return env && strcmp(env, "1") == 0;
}

bool mgpu_ic_force_2lpt_host_stream(void) {
    const char *env = getenv("P21_CUDA_MGPU_2LPT_HOST_STREAM");
    return env && strcmp(env, "1") == 0;
}

template <typename Fn>
bool mgpu_ic_parallel(int n_workers, Fn &&fn) {
    if (n_workers <= 1) {
        return n_workers == 1 ? fn(0) : true;
    }
    std::atomic<bool> ok{true};
    std::vector<std::thread> threads;
    threads.reserve((size_t)n_workers);
    for (int w = 0; w < n_workers; ++w) {
        threads.emplace_back([&, w]() {
            if (!fn(w)) {
                ok.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    return ok.load(std::memory_order_relaxed);
}

struct IcHostBounce {
    void *ptr = nullptr;
    bool pinned = false;

    bool alloc(size_t bytes) {
        size_t n = bytes > 0 ? bytes : 256;
        if (cudaHostAlloc(&ptr, n, cudaHostAllocPortable) == cudaSuccess && ptr != nullptr) {
            pinned = true;
            return true;
        }
        cudaGetLastError();
        ptr = malloc(n);
        pinned = false;
        return ptr != nullptr;
    }

    void free_mem(void) {
        if (!ptr) {
            return;
        }
        if (pinned) {
            cudaFreeHost(ptr);
            cudaGetLastError();
        } else {
            ::free(ptr);
        }
        ptr = nullptr;
        pinned = false;
    }
};

void compute_ox_range(int ox_dim, double dim_ratio, int x_begin, int x_end, int *ox_start_out,
                      int *ox_count_out) {
    int first = ox_dim;
    int last = -1;
    for (int oi = 0; oi < ox_dim; ++oi) {
        int ri = p21_resample_coord(oi, dim_ratio);
        if (ri >= x_begin && ri < x_end) {
            if (oi < first) first = oi;
            if (oi > last) last = oi;
        }
    }
    if (last < first) {
        *ox_start_out = 0;
        *ox_count_out = 0;
    } else {
        *ox_start_out = first;
        *ox_count_out = last - first + 1;
    }
}

__global__ void vcb_kspace_axis_kernel(const cufftComplex *saved, cufftComplex *out, int nx, int ny,
                                       int nz, double len_x, double len_y, double len_z, int axis,
                                       CosmoKernelParams params) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * hz;
    if (idx >= total) {
        return;
    }
    if (idx == 0) {
        out[0] = make_float2(0.0f, 0.0f);
        return;
    }
    int n_z = (int)(idx % hz);
    size_t q = idx / hz;
    int n_y = (int)(q % ny);
    int n_x = (int)(q / ny);

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_mag = sqrt(kx * kx + ky * ky + kz * kz);
    double p = dev_power_in_k(k_mag, params);
    double p_vcb = dev_power_in_vcb(k_mag, params);
    double k_comp = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
    double factor = (k_comp / k_mag) * sqrt(p_vcb / p) * params.c_kms;

    cufftComplex v = saved[idx];
    out[idx] = make_float2((float)(-(double)v.y * factor), (float)((double)v.x * factor));
}

__global__ void vcb_finalize_kernel(float *vcb, size_t n, double volume) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vcb[idx] = (float)(sqrt((double)vcb[idx]) / volume);
    }
}

__global__ void f_gradient_kernel(const cufftComplex *in, cufftComplex *out, int nx, int ny, int nz,
                                  double len_x, double len_y, double len_z, int axis) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * hz;
    if (idx >= total) {
        return;
    }
    if (idx == 0) {
        out[0] = make_float2(0.0f, 0.0f);
        return;
    }
    int n_z = (int)(idx % hz);
    size_t q = idx / hz;
    int n_y = (int)(q % ny);
    int n_x = (int)(q / ny);

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_sq = kx * kx + ky * ky + kz * kz;
    double k_comp = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
    double factor = k_comp / k_sq;

    cufftComplex v = in[idx];
    out[idx] = make_float2((float)(-(double)v.y * factor), (float)((double)v.x * factor));
}

__global__ void f_laplacian_kernel(const cufftComplex *in, cufftComplex *out, int nx, int ny, int nz,
                                   double len_x, double len_y, double len_z, int ax0, int ax1) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * hz;
    if (idx >= total) {
        return;
    }
    if (idx == 0) {
        out[0] = make_float2(0.0f, 0.0f);
        return;
    }
    int n_z = (int)(idx % hz);
    size_t q = idx / hz;
    int n_y = (int)(q % ny);
    int n_x = (int)(q / ny);

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_sq = kx * kx + ky * ky + kz * kz;
    double k0 = (ax0 == 0) ? kx : ((ax0 == 1) ? ky : kz);
    double k1 = (ax1 == 0) ? kx : ((ax1 == 1) ? ky : kz);
    double factor = -(k0 * k1) / k_sq;

    cufftComplex v = in[idx];
    out[idx] = make_float2((float)((double)v.x * factor), (float)((double)v.y * factor));
}

__global__ void sub_sq_padded_2lpt_kernel(float *rhs, const float *phi_ab_padded, int nx, int ny,
                                          int nz, int z_stride) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * nz;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % nz);
    size_t q = idx / nz;
    int j = (int)(q % ny);
    int i = (int)(q / ny);
    double c_ij = (double)phi_ab_padded[p21_grid_idx(i, j, k, ny, z_stride)];
    float acc = rhs[idx];
    acc -= (float)(c_ij * c_ij);
    rhs[idx] = acc;
}

__global__ void accum_2lpt_d01_padded_kernel(float *rhs, float *diag_acc, const float *d1_padded,
                                             int nx, int ny, int nz, int z_stride) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * nz;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % nz);
    size_t q = idx / nz;
    int j = (int)(q % ny);
    int i = (int)(q / ny);
    double d0 = (double)diag_acc[idx];
    double d1 = (double)d1_padded[p21_grid_idx(i, j, k, ny, z_stride)];
    float acc = rhs[idx];
    acc += (float)(d0 * d1);
    rhs[idx] = acc;
    diag_acc[idx] = (float)(d0 + d1);
}

__global__ void accum_2lpt_d01_d2_padded_kernel(float *rhs, const float *diag_acc,
                                                const float *d2_padded, int nx, int ny, int nz,
                                                int z_stride) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * nz;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % nz);
    size_t q = idx / nz;
    int j = (int)(q % ny);
    int i = (int)(q / ny);
    double d01 = (double)diag_acc[idx];
    double d2 = (double)d2_padded[p21_grid_idx(i, j, k, ny, z_stride)];
    float acc = rhs[idx];
    acc += (float)(d01 * d2);
    rhs[idx] = acc;
}

bool upload_and_register_d2h(float *host_ptr, const float *d_src, size_t nbytes,
                             cudaStream_t stream, bool register_in_cache = true) {
    if (!host_ptr || !d_src || nbytes == 0) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(host_ptr, d_src, nbytes, cudaMemcpyDeviceToHost, stream))) {
        return false;
    }
    p21_cuda_stats_add_d2h(nbytes);
    if (!P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return false;
    }
    if (register_in_cache && p21_cuda_registry_put_copy(host_ptr, d_src, nbytes, stream)) {
        return true;
    }
    p21_cuda_invalidate_host_ptr(host_ptr);
    return true;
}

bool d2h_padded_and_register(float *host_ptr, float *d_padded, int nx, int ny, int nz, int z_stride,
                             double scale, cudaStream_t stream, bool register_in_cache = true) {
    if (!host_ptr || !d_padded || nx <= 0 || ny <= 0 || nz <= 0 || z_stride < nz) {
        return false;
    }
    const size_t hi_pixels = (size_t)nx * ny * nz;
    const size_t hi_bytes = hi_pixels * sizeof(float);
    const int tpb = 256;
    const int g = (int)((hi_pixels + tpb - 1) / tpb);

    size_t max_entry = p21_cuda_registry_max_entry_bytes();
    if (register_in_cache && hi_bytes <= max_entry &&
        p21_cuda_free_mem_bytes() >= hi_bytes + 512ULL * 1024 * 1024) {
        float *d_dense = (float *)p21_cuda_pool_alloc(hi_bytes, stream);
        if (d_dense) {
            pack_padded_to_dense_kernel<<<g, tpb, 0, stream>>>(d_padded, d_dense, nx, ny, nz,
                                                               z_stride, scale);
            bool ok = P21_CUDA_OK(cudaGetLastError()) &&
                      P21_CUDA_OK(cudaMemcpyAsync(host_ptr, d_dense, hi_bytes,
                                                  cudaMemcpyDeviceToHost, stream));
            if (ok) {
                p21_cuda_stats_add_d2h(hi_bytes);
                ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
            }
            if (!ok) {
                p21_cuda_pool_free(d_dense, stream);
                return false;
            }
            if (!p21_cuda_registry_put_owned(host_ptr, d_dense, hi_bytes, stream)) {
                p21_cuda_pool_free(d_dense, stream);
                p21_cuda_invalidate_host_ptr(host_ptr);
            }
            return true;
        }
    }

    if (scale != 1.0) {
        scale_padded_real_kernel<<<g, tpb, 0, stream>>>(d_padded, nx, ny, nz, z_stride, scale);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            return false;
        }
    }
    if (!P21_CUDA_OK(cudaMemcpy2DAsync(host_ptr, (size_t)nz * sizeof(float), d_padded,
                                       (size_t)z_stride * sizeof(float),
                                       (size_t)nz * sizeof(float), (size_t)nx * ny,
                                       cudaMemcpyDeviceToHost, stream))) {
        return false;
    }
    p21_cuda_stats_add_d2h(hi_bytes);
    if (!P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return false;
    }
    p21_cuda_invalidate_host_ptr(host_ptr);
    return true;
}

int run_mgpu_compute_initial_conditions(unsigned long long int random_seed, InitialConditions *boxes,
                                        bool non_zero_input, int nx, int ny, int nz, int lx, int ly,
                                        int lz, int pt_x, int pt_y, int pt_z, double box_len_x,
                                        double box_len_y, double box_len_z, double volume,
                                        double dim_ratio_hi_lo, double dim_ratio_hi_pt,
                                        float tophat_R, bool perturb_hires, float *vel_pointers[3],
                                        float *vel_pointers_2lpt[3],
                                        const CosmoKernelParams &base_cosmo,
                                        const HostSpline &dens_spline, const HostSpline &vcb_spline) {
    P21MgpuGrid grid;
    if (!grid.init(nx, ny, nz)) {
        return -1;
    }
    const int G = grid.n_gpus;
    const size_t hi_pixels = (size_t)nx * (size_t)ny * (size_t)nz;
    const size_t plane_real = (size_t)ny * (size_t)nz;
    const int tpb = 256;
    const double dk_x = 2.0 * M_PI / box_len_x;
    const double dk_y = 2.0 * M_PI / box_len_y;
    const double dk_z = 2.0 * M_PI / box_len_z;

    size_t max_slab_bytes = 0;
    size_t max_real_x_bytes = 0;
    for (int g = 0; g < G; ++g) {
        max_slab_bytes = std::max(max_slab_bytes, grid.cplx_elems_y(g) * sizeof(cufftComplex));
        max_real_x_bytes = std::max(max_real_x_bytes, grid.real_elems_x(g) * sizeof(float));
    }
    const size_t min_free = p21_mgpu_min_free_mem(grid);
    const bool single_slab_mode =
        mgpu_ic_force_single_slab() || (min_free < 2ULL * max_slab_bytes + (2048ULL << 20));
    void *h_stage = nullptr;
    if (min_free < 4ULL * max_slab_bytes + (768ULL << 20)) {
        h_stage = malloc(grid.total_cplx_elems() * sizeof(cufftComplex));
        if (!h_stage) {
            grid.destroy();
            return -1;
        }
    }

    std::vector<cufftComplex *> d_saved((size_t)G, nullptr), d_work((size_t)G, nullptr);
    std::vector<double *> d_cdens((size_t)G, nullptr), d_cvcb((size_t)G, nullptr);
    std::vector<CosmoKernelParams> dev_cosmo((size_t)G, base_cosmo);
    std::vector<IcHostBounce> h_strip_bounce((size_t)G);
    cufftComplex *h_saved_input = nullptr;
    bool ok = true;

    const size_t strip_cap_bytes = 64ULL << 20;
    const size_t bounce_bytes = std::max(strip_cap_bytes, plane_real * sizeof(float));
    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        size_t bytes_y = grid.cplx_elems_y(g) * sizeof(cufftComplex);
        d_work[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(grid.devs[g], bytes_y, grid.streams[g]);
        if (!single_slab_mode) {
            d_saved[(size_t)g] =
                (cufftComplex *)p21_mgpu_malloc(grid.devs[g], bytes_y, grid.streams[g]);
        }
        if (!d_work[(size_t)g] || (!single_slab_mode && !d_saved[(size_t)g]) ||
            !h_strip_bounce[(size_t)g].alloc(bounce_bytes)) {
            ok = false;
            break;
        }
        if (matter_options_global->POWER_SPECTRUM == POWER_SPECTRUM_CLASS) {
            size_t db = dens_spline.packed.size() * sizeof(double);
            d_cdens[(size_t)g] = (double *)p21_mgpu_malloc(grid.devs[g], db, grid.streams[g]);
            if (!d_cdens[(size_t)g] ||
                !P21_CUDA_OK(cudaMemcpyAsync(d_cdens[(size_t)g], dens_spline.packed.data(), db,
                                             cudaMemcpyHostToDevice, grid.streams[g]))) {
                ok = false;
                break;
            }
            dev_cosmo[(size_t)g].class_dens_table = d_cdens[(size_t)g];
            if (matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) {
                size_t vb = vcb_spline.packed.size() * sizeof(double);
                d_cvcb[(size_t)g] = (double *)p21_mgpu_malloc(grid.devs[g], vb, grid.streams[g]);
                if (!d_cvcb[(size_t)g] ||
                    !P21_CUDA_OK(cudaMemcpyAsync(d_cvcb[(size_t)g], vcb_spline.packed.data(), vb,
                                                 cudaMemcpyHostToDevice, grid.streams[g]))) {
                    ok = false;
                    break;
                }
                dev_cosmo[(size_t)g].class_vcb_table = d_cvcb[(size_t)g];
            }
        }
    }
    if (ok && single_slab_mode && non_zero_input) {
        h_saved_input = (cufftComplex *)malloc(grid.total_cplx_elems() * sizeof(cufftComplex));
        if (!h_saved_input) {
            ok = false;
        }
    }

    auto cleanup_all = [&]() {
        for (int g = 0; g < G; ++g) {
            p21_mgpu_free(grid.devs[g], d_cvcb[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_cdens[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_work[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_saved[(size_t)g], grid.streams[g]);
            h_strip_bounce[(size_t)g].free_mem();
        }
        if (h_saved_input) {
            free(h_saved_input);
            h_saved_input = nullptr;
        }
        if (h_stage) {
            free(h_stage);
            h_stage = nullptr;
        }
        grid.destroy();
    };

    if (!ok) {
        cleanup_all();
        return -1;
    }

    auto spill_yslabs_to_host = [&](cufftComplex *const *d_src, cufftComplex *h_buf) -> bool {
        return mgpu_ic_parallel(G, [&](int g) -> bool {
            cudaSetDevice(grid.devs[g]);
            size_t elems_y = grid.cplx_elems_y(g);
            size_t bytes_y = elems_y * sizeof(cufftComplex);
            cufftComplex *dst =
                h_buf + (size_t)grid.y0[g] * (size_t)grid.hz * (size_t)grid.nx;
            return P21_CUDA_OK(cudaMemcpyAsync(dst, d_src[(size_t)g], bytes_y,
                                               cudaMemcpyDeviceToHost, grid.streams[g])) &&
                   P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
        });
    };

    auto upload_yslabs_from_host = [&](const cufftComplex *h_buf, cufftComplex **d_dst) -> bool {
        return mgpu_ic_parallel(G, [&](int g) -> bool {
            cudaSetDevice(grid.devs[g]);
            size_t elems_y = grid.cplx_elems_y(g);
            size_t bytes_y = elems_y * sizeof(cufftComplex);
            const cufftComplex *src =
                h_buf + (size_t)grid.y0[g] * (size_t)grid.hz * (size_t)grid.nx;
            return P21_CUDA_OK(cudaMemcpyAsync(d_dst[(size_t)g], src, bytes_y,
                                               cudaMemcpyHostToDevice, grid.streams[g])) &&
                   P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
        });
    };

    auto generate_base_modes_fast = [&](cufftComplex **d_dst) -> bool {
        bool lok = mgpu_ic_parallel(G, [&](int g) -> bool {
            cudaSetDevice(grid.devs[g]);
            size_t elems_y = grid.cplx_elems_y(g);
            int kg = (int)((elems_y + tpb - 1) / tpb);
            fill_ic_modes_fast_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                d_dst[(size_t)g], (uint64_t)random_seed, grid.y0[g], grid.sy[g], nx, ny, nz,
                box_len_x, box_len_y, box_len_z, volume, dev_cosmo[(size_t)g]);
            return P21_CUDA_OK(cudaGetLastError()) &&
                   P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
        });
        if (lok) {
            lok = p21_mgpu_enforce_real_planes_yslabs(grid, d_dst);
        }
        return lok;
    };

    auto prepare_base_kspace_in_work = [&](void) -> bool {
        if (!single_slab_mode) {
            return mgpu_ic_parallel(G, [&](int g) -> bool {
                cudaSetDevice(grid.devs[g]);
                size_t bytes_y = grid.cplx_elems_y(g) * sizeof(cufftComplex);
                return P21_CUDA_OK(cudaMemcpyAsync(d_work[(size_t)g], d_saved[(size_t)g], bytes_y,
                                                   cudaMemcpyDeviceToDevice, grid.streams[g])) &&
                       P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
            });
        }
        if (non_zero_input) {
            return upload_yslabs_from_host(h_saved_input, d_work.data());
        }
        return generate_base_modes_fast(d_work.data());
    };

    auto c2r_to_host = [&](float *h_dst, double post_scale) -> bool {
        return p21_mgpu_fft_c2r_strips(
            grid, d_work.data(), h_stage,
            [&](int g, int x_global_start, int cur_nx, float *d_strip, cudaStream_t stream) -> bool {
                size_t elems = (size_t)cur_nx * plane_real;
                size_t bytes = elems * sizeof(float);
                if (post_scale != 1.0) {
                    int sg = (int)((elems + tpb - 1) / tpb);
                    scale_real_kernel<<<sg, tpb, 0, stream>>>(d_strip, d_strip, elems, post_scale);
                    if (!P21_CUDA_OK(cudaGetLastError())) {
                        return false;
                    }
                }
                void *bounce = h_strip_bounce[(size_t)g].ptr;
                if (!P21_CUDA_OK(cudaMemcpyAsync(bounce, d_strip, bytes, cudaMemcpyDeviceToHost,
                                                 stream)) ||
                    !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                    return false;
                }
                memcpy(h_dst + (size_t)x_global_start * plane_real, bounce, bytes);
                return true;
            });
    };

    auto c2r_resample_to_host = [&](int ox, int oy, int oz, double dim_ratio, bool divide_vol,
                                    float *h_dst) -> bool {
        std::vector<int> ox_start((size_t)G, 0), ox_cnt((size_t)G, 0);
        std::vector<float *> d_out((size_t)G, nullptr);
        for (int g = 0; g < G; ++g) {
            compute_ox_range(ox, dim_ratio, grid.x0[g], grid.x0[g] + grid.sx[g],
                             &ox_start[(size_t)g], &ox_cnt[(size_t)g]);
            if (ox_cnt[(size_t)g] > 0) {
                cudaSetDevice(grid.devs[g]);
                d_out[(size_t)g] = (float *)p21_mgpu_malloc(
                    grid.devs[g], (size_t)ox_cnt[(size_t)g] * (size_t)oy * (size_t)oz * sizeof(float),
                    grid.streams[g]);
                if (!d_out[(size_t)g]) {
                    for (int k = 0; k <= g; ++k) p21_mgpu_free(grid.devs[k], d_out[(size_t)k], grid.streams[k]);
                    return false;
                }
            }
        }
        bool cok = p21_mgpu_fft_c2r_strips(
            grid, d_work.data(), h_stage,
            [&](int g, int x_global_start, int cur_nx, float *d_strip, cudaStream_t stream) -> bool {
                int bs = 0, bc = 0;
                compute_ox_range(ox, dim_ratio, x_global_start, x_global_start + cur_nx, &bs, &bc);
                if (bc <= 0) {
                    return true;
                }
                size_t elems = (size_t)bc * (size_t)oy * (size_t)oz;
                int rg = (int)((elems + tpb - 1) / tpb);
                resample_strip_to_out_kernel<<<rg, tpb, 0, stream>>>(
                    d_strip, x_global_start, ny, nz, ox_start[(size_t)g], bs, bc, oy, oz, dim_ratio,
                    volume, divide_vol, false, d_out[(size_t)g]);
                return P21_CUDA_OK(cudaGetLastError());
            });
        if (cok) {
            cok = mgpu_ic_parallel(G, [&](int g) -> bool {
                if (ox_cnt[(size_t)g] <= 0) {
                    return true;
                }
                cudaSetDevice(grid.devs[g]);
                size_t bytes = (size_t)ox_cnt[(size_t)g] * (size_t)oy * (size_t)oz * sizeof(float);
                float *dst = h_dst + (size_t)ox_start[(size_t)g] * (size_t)oy * (size_t)oz;
                return P21_CUDA_OK(cudaMemcpyAsync(dst, d_out[(size_t)g], bytes,
                                                   cudaMemcpyDeviceToHost, grid.streams[g])) &&
                       P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
            });
        }
        for (int g = 0; g < G; ++g) {
            p21_mgpu_free(grid.devs[g], d_out[(size_t)g], grid.streams[g]);
        }
        return cok;
    };

    /* 1. Initial density modes */
    if (non_zero_input) {
        cufftComplex **base_dst = single_slab_mode ? d_work.data() : d_saved.data();
        ok = p21_mgpu_fft_r2c_from_host(grid, boxes->hires_density, volume / (double)hi_pixels,
                                        base_dst, h_stage);
        if (ok && single_slab_mode) {
            ok = spill_yslabs_to_host(d_work.data(), h_saved_input);
        }
    } else {
        cufftComplex **base_dst = single_slab_mode ? d_work.data() : d_saved.data();
        ok = generate_base_modes_fast(base_dst);
        if (ok && !single_slab_mode) {
            ok = prepare_base_kspace_in_work();
        }
        if (ok) ok = c2r_to_host(boxes->hires_density, 1.0 / volume);
    }

    /* 2. Filtered lowres density */
    if (ok) {
        ok = prepare_base_kspace_in_work();
        if (ok && nx != lx) {
            ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                cudaSetDevice(grid.devs[g]);
                size_t elems_y = grid.cplx_elems_y(g);
                int kg = (int)((elems_y + tpb - 1) / tpb);
                filter_tophat_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                    d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny, nz, dk_x, dk_y, dk_z,
                    tophat_R);
                return P21_CUDA_OK(cudaGetLastError()) &&
                       P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
            });
        }
        if (ok) ok = c2r_resample_to_host(lx, ly, lz, dim_ratio_hi_lo, true, boxes->lowres_density);
    }

    /* 3. Relative velocity (CLASS FLUCTS) */
    if (ok && matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) {
        std::vector<int> ox_start((size_t)G, 0), ox_cnt((size_t)G, 0);
        std::vector<float *> d_vcb((size_t)G, nullptr);
        for (int g = 0; g < G; ++g) {
            compute_ox_range(lx, dim_ratio_hi_lo, grid.x0[g], grid.x0[g] + grid.sx[g],
                             &ox_start[(size_t)g], &ox_cnt[(size_t)g]);
            if (ox_cnt[(size_t)g] > 0) {
                cudaSetDevice(grid.devs[g]);
                size_t bytes = (size_t)ox_cnt[(size_t)g] * (size_t)ly * (size_t)lz * sizeof(float);
                d_vcb[(size_t)g] = (float *)p21_mgpu_malloc(grid.devs[g], bytes, grid.streams[g]);
                if (!d_vcb[(size_t)g] || !P21_CUDA_OK(cudaMemsetAsync(d_vcb[(size_t)g], 0, bytes, grid.streams[g]))) {
                    ok = false;
                    break;
                }
            }
        }
        for (int ii = 0; ok && ii < 3; ++ii) {
            ok = prepare_base_kspace_in_work();
            if (ok) {
                ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                    cudaSetDevice(grid.devs[g]);
                    size_t elems_y = grid.cplx_elems_y(g);
                    int kg = (int)((elems_y + tpb - 1) / tpb);
                    vcb_kspace_axis_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                        d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny, nz,
                        box_len_x, box_len_y, box_len_z, ii, dev_cosmo[(size_t)g]);
                    if (!P21_CUDA_OK(cudaGetLastError())) {
                        return false;
                    }
                    if (nx != lx) {
                        filter_tophat_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                            d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny, nz, dk_x, dk_y,
                            dk_z, tophat_R);
                        if (!P21_CUDA_OK(cudaGetLastError())) {
                            return false;
                        }
                    }
                    return P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                });
            }
            if (ok) {
                ok = p21_mgpu_fft_c2r_strips(
                    grid, d_work.data(), h_stage,
                    [&](int g, int x_global_start, int cur_nx, float *d_strip,
                        cudaStream_t stream) -> bool {
                        int bs = 0, bc = 0;
                        compute_ox_range(lx, dim_ratio_hi_lo, x_global_start,
                                         x_global_start + cur_nx, &bs, &bc);
                        if (bc <= 0) return true;
                        size_t elems = (size_t)bc * (size_t)ly * (size_t)lz;
                        int rg = (int)((elems + tpb - 1) / tpb);
                        resample_strip_to_out_kernel<<<rg, tpb, 0, stream>>>(
                            d_strip, x_global_start, ny, nz, ox_start[(size_t)g], bs, bc, ly, lz,
                            dim_ratio_hi_lo, volume, false, true, d_vcb[(size_t)g]);
                        return P21_CUDA_OK(cudaGetLastError());
                    });
            }
        }
        if (ok) {
            ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                if (ox_cnt[(size_t)g] <= 0) {
                    return true;
                }
                cudaSetDevice(grid.devs[g]);
                size_t elems = (size_t)ox_cnt[(size_t)g] * (size_t)ly * (size_t)lz;
                int lg = (int)((elems + tpb - 1) / tpb);
                vcb_finalize_kernel<<<lg, tpb, 0, grid.streams[g]>>>(d_vcb[(size_t)g], elems,
                                                                      volume);
                float *dst = boxes->lowres_vcb + (size_t)ox_start[(size_t)g] * (size_t)ly * (size_t)lz;
                return P21_CUDA_OK(cudaGetLastError()) &&
                       P21_CUDA_OK(cudaMemcpyAsync(dst, d_vcb[(size_t)g], elems * sizeof(float),
                                                   cudaMemcpyDeviceToHost, grid.streams[g])) &&
                       P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
            });
        }
        for (int g = 0; g < G; ++g) {
            p21_mgpu_free(grid.devs[g], d_vcb[(size_t)g], grid.streams[g]);
        }
    }

    /* 4. 1LPT velocities */
    const int do_tophat_vel = (!perturb_hires && nx != lx) ? 1 : 0;
    for (int ii = 0; ok && ii < 3; ++ii) {
        ok = prepare_base_kspace_in_work();
        if (ok) {
            ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                cudaSetDevice(grid.devs[g]);
                size_t elems_y = grid.cplx_elems_y(g);
                int kg = (int)((elems_y + tpb - 1) / tpb);
                f_gradient_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                    d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny, nz,
                    box_len_x, box_len_y, box_len_z, ii, do_tophat_vel, tophat_R);
                return P21_CUDA_OK(cudaGetLastError()) &&
                       P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
            });
        }
        if (ok) {
            if (perturb_hires) {
                ok = c2r_to_host(vel_pointers[ii], 1.0 / volume);
            } else {
                ok = c2r_resample_to_host(pt_x, pt_y, pt_z, dim_ratio_hi_pt, true, vel_pointers[ii]);
            }
        }
    }

    /* 5. 2LPT velocities */
    if (ok && matter_options_global->PERTURB_ALGORITHM == PERTURB_ALGORITHM_2LPT) {
        const size_t min_free_2lpt = p21_mgpu_min_free_mem(grid);
        const bool fit_2lpt_gpu = !mgpu_ic_force_2lpt_host_stream() &&
                                  (min_free_2lpt >= 2ULL * max_real_x_bytes + (768ULL << 20));
        const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        cufftComplex *h_phi2_k = nullptr;

        if (fit_2lpt_gpu) {
            std::vector<float *> d_phi2((size_t)G, nullptr), d_diag((size_t)G, nullptr);
            for (int g = 0; g < G && ok; ++g) {
                cudaSetDevice(grid.devs[g]);
                size_t rb = grid.real_elems_x(g) * sizeof(float);
                d_phi2[(size_t)g] = (float *)p21_mgpu_malloc(grid.devs[g], rb, grid.streams[g]);
                d_diag[(size_t)g] = (float *)p21_mgpu_malloc(grid.devs[g], rb, grid.streams[g]);
                if (!d_phi2[(size_t)g] || !d_diag[(size_t)g] ||
                    !P21_CUDA_OK(cudaMemsetAsync(d_phi2[(size_t)g], 0, rb, grid.streams[g]))) {
                    ok = false;
                }
            }
            for (int p_idx = 0; ok && p_idx < 3; ++p_idx) {
                ok = prepare_base_kspace_in_work();
                if (ok) {
                    ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                        cudaSetDevice(grid.devs[g]);
                        int kg = (int)((grid.cplx_elems_y(g) + tpb - 1) / tpb);
                        f_laplacian_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                            d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny,
                            nz, box_len_x, box_len_y, box_len_z, pairs[p_idx][0], pairs[p_idx][1]);
                        return P21_CUDA_OK(cudaGetLastError()) &&
                               P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                    });
                }
                if (ok) {
                    ok = p21_mgpu_fft_c2r_strips(
                        grid, d_work.data(), h_stage,
                        [&](int g, int x_global_start, int cur_nx, float *d_strip,
                            cudaStream_t stream) -> bool {
                            size_t elems = (size_t)cur_nx * plane_real;
                            int x_loc = x_global_start - grid.x0[g];
                            float *rhs = d_phi2[(size_t)g] + (size_t)x_loc * plane_real;
                            int sg = (int)((elems + tpb - 1) / tpb);
                            sub_sq_2lpt_dense_kernel<<<sg, tpb, 0, stream>>>(rhs, d_strip, elems);
                            return P21_CUDA_OK(cudaGetLastError());
                        });
                }
            }
            /* Diagonal a=0 -> d_diag (+ optional host hires_vx_2LPT) */
            if (ok) ok = prepare_base_kspace_in_work();
            if (ok) {
                ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                    cudaSetDevice(grid.devs[g]);
                    int kg = (int)((grid.cplx_elems_y(g) + tpb - 1) / tpb);
                    f_laplacian_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                        d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny, nz,
                        box_len_x, box_len_y, box_len_z, 0, 0);
                    return P21_CUDA_OK(cudaGetLastError()) &&
                           P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                });
            }
            if (ok) {
                ok = p21_mgpu_fft_c2r_strips(
                    grid, d_work.data(), h_stage,
                    [&](int g, int x_global_start, int cur_nx, float *d_strip,
                        cudaStream_t stream) -> bool {
                        size_t elems = (size_t)cur_nx * plane_real;
                        int x_loc = x_global_start - grid.x0[g];
                        if (!P21_CUDA_OK(cudaMemcpyAsync(
                                d_diag[(size_t)g] + (size_t)x_loc * plane_real, d_strip,
                                elems * sizeof(float), cudaMemcpyDeviceToDevice, stream))) {
                            return false;
                        }
                        if (!perturb_hires && boxes->hires_vx_2LPT) {
                            size_t bytes = elems * sizeof(float);
                            void *bounce = h_strip_bounce[(size_t)g].ptr;
                            if (!P21_CUDA_OK(cudaMemcpyAsync(bounce, d_strip, bytes,
                                                             cudaMemcpyDeviceToHost, stream)) ||
                                !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                                return false;
                            }
                            memcpy(boxes->hires_vx_2LPT + (size_t)x_global_start * plane_real,
                                   bounce, bytes);
                        }
                        return true;
                    });
            }
            /* Diagonal a=1 -> accum d0*d1, d_diag=d0+d1 (+ optional host hires_vy_2LPT) */
            if (ok) ok = prepare_base_kspace_in_work();
            if (ok) {
                ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                    cudaSetDevice(grid.devs[g]);
                    int kg = (int)((grid.cplx_elems_y(g) + tpb - 1) / tpb);
                    f_laplacian_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                        d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny, nz,
                        box_len_x, box_len_y, box_len_z, 1, 1);
                    return P21_CUDA_OK(cudaGetLastError()) &&
                           P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                });
            }
            if (ok) {
                ok = p21_mgpu_fft_c2r_strips(
                    grid, d_work.data(), h_stage,
                    [&](int g, int x_global_start, int cur_nx, float *d_strip,
                        cudaStream_t stream) -> bool {
                        size_t elems = (size_t)cur_nx * plane_real;
                        int x_loc = x_global_start - grid.x0[g];
                        size_t off = (size_t)x_loc * plane_real;
                        if (!perturb_hires && boxes->hires_vy_2LPT) {
                            size_t bytes = elems * sizeof(float);
                            void *bounce = h_strip_bounce[(size_t)g].ptr;
                            if (!P21_CUDA_OK(cudaMemcpyAsync(bounce, d_strip, bytes,
                                                             cudaMemcpyDeviceToHost, stream)) ||
                                !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                                return false;
                            }
                            memcpy(boxes->hires_vy_2LPT + (size_t)x_global_start * plane_real,
                                   bounce, bytes);
                        }
                        int sg = (int)((elems + tpb - 1) / tpb);
                        accum_2lpt_d01_dense_kernel<<<sg, tpb, 0, stream>>>(
                            d_phi2[(size_t)g] + off, d_diag[(size_t)g] + off, d_strip, elems);
                        return P21_CUDA_OK(cudaGetLastError());
                    });
            }
            /* Diagonal a=2 -> accum (d0+d1)*d2 (+ optional host hires_vz_2LPT) */
            if (ok) ok = prepare_base_kspace_in_work();
            if (ok) {
                ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                    cudaSetDevice(grid.devs[g]);
                    int kg = (int)((grid.cplx_elems_y(g) + tpb - 1) / tpb);
                    f_laplacian_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                        d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny, nz,
                        box_len_x, box_len_y, box_len_z, 2, 2);
                    return P21_CUDA_OK(cudaGetLastError()) &&
                           P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                });
            }
            if (ok) {
                ok = p21_mgpu_fft_c2r_strips(
                    grid, d_work.data(), h_stage,
                    [&](int g, int x_global_start, int cur_nx, float *d_strip,
                        cudaStream_t stream) -> bool {
                        size_t elems = (size_t)cur_nx * plane_real;
                        int x_loc = x_global_start - grid.x0[g];
                        size_t off = (size_t)x_loc * plane_real;
                        if (!perturb_hires && boxes->hires_vz_2LPT) {
                            size_t bytes = elems * sizeof(float);
                            void *bounce = h_strip_bounce[(size_t)g].ptr;
                            if (!P21_CUDA_OK(cudaMemcpyAsync(bounce, d_strip, bytes,
                                                             cudaMemcpyDeviceToHost, stream)) ||
                                !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                                return false;
                            }
                            memcpy(boxes->hires_vz_2LPT + (size_t)x_global_start * plane_real,
                                   bounce, bytes);
                        }
                        int sg = (int)((elems + tpb - 1) / tpb);
                        accum_2lpt_d01_d2_dense_kernel<<<sg, tpb, 0, stream>>>(
                            d_phi2[(size_t)g] + off, d_diag[(size_t)g] + off, d_strip, elems);
                        return P21_CUDA_OK(cudaGetLastError());
                    });
            }
            for (int g = 0; g < G; ++g) {
                p21_mgpu_free(grid.devs[g], d_diag[(size_t)g], grid.streams[g]);
            }
            if (ok) {
                double norm_2lpt = 1.0 / (volume * volume * (double)hi_pixels);
                ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                    cudaSetDevice(grid.devs[g]);
                    size_t elems = grid.real_elems_x(g);
                    int hg = (int)((elems + tpb - 1) / tpb);
                    scale_real_kernel<<<hg, tpb, 0, grid.streams[g]>>>(
                        d_phi2[(size_t)g], d_phi2[(size_t)g], elems, norm_2lpt);
                    return P21_CUDA_OK(cudaGetLastError()) &&
                           P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                });
                if (ok) {
                    cufftComplex **phi2_dst = single_slab_mode ? d_work.data() : d_saved.data();
                    ok = p21_mgpu_fft_r2c_slabs(grid, d_phi2.data(), phi2_dst, h_stage);
                }
            }
            for (int g = 0; g < G; ++g) {
                p21_mgpu_free(grid.devs[g], d_phi2[(size_t)g], grid.streams[g]);
            }
        } else {
            /* Streaming host-accumulated 2LPT (Direction P: stream diagonals a=0,1,2 through one
             * host buffer instead of three full diag grids, keeping 2048^3 peak host RAM < 160 GiB). */
            float *h_phi2 = (float *)calloc(hi_pixels, sizeof(float));
            float *h_diag_own = nullptr;
            float *h_diag = nullptr;
            if (h_phi2) {
                if (perturb_hires && boxes->hires_vx_2LPT) {
                    h_diag = boxes->hires_vx_2LPT;
                } else {
                    h_diag_own = (float *)malloc(hi_pixels * sizeof(float));
                    h_diag = h_diag_own;
                }
            }
            if (!h_phi2 || !h_diag) {
                free(h_diag_own);
                free(h_phi2);
                ok = false;
            } else {
                for (int p_idx = 0; ok && p_idx < 3; ++p_idx) {
                    ok = prepare_base_kspace_in_work();
                    if (ok) {
                        ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                            cudaSetDevice(grid.devs[g]);
                            int kg = (int)((grid.cplx_elems_y(g) + tpb - 1) / tpb);
                            f_laplacian_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                                d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx,
                                ny, nz, box_len_x, box_len_y, box_len_z, pairs[p_idx][0],
                                pairs[p_idx][1]);
                            return P21_CUDA_OK(cudaGetLastError()) &&
                                   P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                        });
                    }
                    if (ok) {
                        ok = p21_mgpu_fft_c2r_strips(
                            grid, d_work.data(), h_stage,
                            [&](int g, int x_global_start, int cur_nx, float *d_strip,
                                cudaStream_t stream) -> bool {
                                size_t elems = (size_t)cur_nx * plane_real;
                                size_t bytes = elems * sizeof(float);
                                float *bounce = (float *)h_strip_bounce[(size_t)g].ptr;
                                if (!P21_CUDA_OK(cudaMemcpyAsync(bounce, d_strip, bytes,
                                                                 cudaMemcpyDeviceToHost, stream)) ||
                                    !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                                    return false;
                                }
                                float *rhs = h_phi2 + (size_t)x_global_start * plane_real;
                                for (size_t i = 0; i < elems; ++i) {
                                    double c = (double)bounce[i];
                                    rhs[i] = (float)((double)rhs[i] - c * c);
                                }
                                return true;
                            });
                    }
                }

                float *opt_diag_out[3] = {
                    (!perturb_hires ? boxes->hires_vx_2LPT : nullptr),
                    (!perturb_hires ? boxes->hires_vy_2LPT : nullptr),
                    (!perturb_hires ? boxes->hires_vz_2LPT : nullptr),
                };
                for (int a = 0; ok && a < 3; ++a) {
                    ok = prepare_base_kspace_in_work();
                    if (ok) {
                        ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                            cudaSetDevice(grid.devs[g]);
                            int kg = (int)((grid.cplx_elems_y(g) + tpb - 1) / tpb);
                            f_laplacian_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                                d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx,
                                ny, nz, box_len_x, box_len_y, box_len_z, a, a);
                            return P21_CUDA_OK(cudaGetLastError()) &&
                                   P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                        });
                    }
                    if (ok) {
                        ok = p21_mgpu_fft_c2r_strips(
                            grid, d_work.data(), h_stage,
                            [&](int g, int x_global_start, int cur_nx, float *d_strip,
                                cudaStream_t stream) -> bool {
                                size_t elems = (size_t)cur_nx * plane_real;
                                size_t bytes = elems * sizeof(float);
                                float *bounce = (float *)h_strip_bounce[(size_t)g].ptr;
                                if (!P21_CUDA_OK(cudaMemcpyAsync(bounce, d_strip, bytes,
                                                                 cudaMemcpyDeviceToHost, stream)) ||
                                    !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                                    return false;
                                }
                                size_t offset = (size_t)x_global_start * plane_real;
                                if (opt_diag_out[a]) {
                                    memcpy(opt_diag_out[a] + offset, bounce, bytes);
                                }
                                float *diag = h_diag + offset;
                                float *rhs = h_phi2 + offset;
                                if (a == 0) {
                                    memcpy(diag, bounce, bytes);
                                } else if (a == 1) {
                                    for (size_t i = 0; i < elems; ++i) {
                                        double d0 = (double)diag[i];
                                        double d1 = (double)bounce[i];
                                        rhs[i] = (float)((double)rhs[i] + d0 * d1);
                                        diag[i] = (float)(d0 + d1);
                                    }
                                } else {
                                    for (size_t i = 0; i < elems; ++i) {
                                        double d01 = (double)diag[i];
                                        double d2 = (double)bounce[i];
                                        rhs[i] = (float)((double)rhs[i] + d01 * d2);
                                    }
                                }
                                return true;
                            });
                    }
                }
                free(h_diag_own);
                h_diag_own = nullptr;
                h_diag = nullptr;
                if (ok) {
                    double norm_2lpt = 1.0 / (volume * volume * (double)hi_pixels);
                    cufftComplex **phi2_dst = single_slab_mode ? d_work.data() : d_saved.data();
                    ok = p21_mgpu_fft_r2c_from_host(grid, h_phi2, norm_2lpt, phi2_dst, h_stage);
                }
                free(h_phi2);
                h_phi2 = nullptr;
            }
        }

        if (ok && single_slab_mode) {
            if (h_saved_input) {
                h_phi2_k = h_saved_input;
                h_saved_input = nullptr;
            } else {
                h_phi2_k = (cufftComplex *)malloc(grid.total_cplx_elems() * sizeof(cufftComplex));
            }
            ok = (h_phi2_k != nullptr) && spill_yslabs_to_host(d_work.data(), h_phi2_k);
        }

        for (int ii = 0; ok && ii < 3; ++ii) {
            if (single_slab_mode) {
                ok = upload_yslabs_from_host(h_phi2_k, d_work.data());
            } else {
                ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                    cudaSetDevice(grid.devs[g]);
                    size_t bytes_y = grid.cplx_elems_y(g) * sizeof(cufftComplex);
                    return P21_CUDA_OK(cudaMemcpyAsync(d_work[(size_t)g], d_saved[(size_t)g],
                                                       bytes_y, cudaMemcpyDeviceToDevice,
                                                       grid.streams[g])) &&
                           P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                });
            }
            if (ok) {
                ok = mgpu_ic_parallel(G, [&](int g) -> bool {
                    cudaSetDevice(grid.devs[g]);
                    int kg = (int)((grid.cplx_elems_y(g) + tpb - 1) / tpb);
                    f_gradient_yslab_kernel<<<kg, tpb, 0, grid.streams[g]>>>(
                        d_work[(size_t)g], d_work[(size_t)g], grid.y0[g], grid.sy[g], nx, ny, nz,
                        box_len_x, box_len_y, box_len_z, ii, do_tophat_vel, tophat_R);
                    return P21_CUDA_OK(cudaGetLastError()) &&
                           P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
                });
            }
            if (ok) {
                if (perturb_hires) {
                    ok = c2r_to_host(vel_pointers_2lpt[ii], 1.0);
                } else {
                    ok = c2r_resample_to_host(pt_x, pt_y, pt_z, dim_ratio_hi_pt, false,
                                              vel_pointers_2lpt[ii]);
                }
            }
        }
        if (h_phi2_k) {
            free(h_phi2_k);
            h_phi2_k = nullptr;
        }
    }

    cleanup_all();
    if (!ok) {
        return -1;
    }

    p21_cuda_invalidate_host_ptr(boxes->hires_density);
    p21_cuda_invalidate_host_ptr(boxes->lowres_density);
    if (boxes->lowres_vcb) p21_cuda_invalidate_host_ptr(boxes->lowres_vcb);
    for (int ii = 0; ii < 3; ++ii) {
        p21_cuda_invalidate_host_ptr(vel_pointers[ii]);
        if (vel_pointers_2lpt[ii]) p21_cuda_invalidate_host_ptr(vel_pointers_2lpt[ii]);
    }
    if (!perturb_hires) {
        if (boxes->hires_vx_2LPT) p21_cuda_invalidate_host_ptr(boxes->hires_vx_2LPT);
        if (boxes->hires_vy_2LPT) p21_cuda_invalidate_host_ptr(boxes->hires_vy_2LPT);
        if (boxes->hires_vz_2LPT) p21_cuda_invalidate_host_ptr(boxes->hires_vz_2LPT);
    }

    P21CudaStats *st = p21_cuda_stats_mut();
    st->mgpu_ic_runs++;
    st->ic_runs++;
    return 0;
}

}  // namespace

extern "C" int p21_cuda_compute_initial_conditions(unsigned long long int random_seed,
                                                   InitialConditions *boxes) {
    if (!boxes || !p21_cuda_get_enabled() || !p21_cuda_ensure_init()) {
        return -1;
    }

    const int nx = simulation_options_global->DIM;
    const int ny = simulation_options_global->DIM;
    const int nz = D_PARA;
    const int lx = simulation_options_global->HII_DIM;
    const int ly = simulation_options_global->HII_DIM;
    const int lz = HII_D_PARA;
    const int n_threads = simulation_options_global->N_THREADS;

    if (nx < 2 || ny < 2 || nz < 2 || lx < 2 || ly < 2 || lz < 2 || n_threads < 1) {
        return -1;
    }
    if (matter_options_global->POWER_SPECTRUM < 0 ||
        matter_options_global->POWER_SPECTRUM > POWER_SPECTRUM_CLASS) {
        return -1;
    }
    if (matter_options_global->POWER_SPECTRUM == POWER_SPECTRUM_CLASS) {
        if (!cosmo_tables_global || !cosmo_tables_global->transfer_density ||
            cosmo_tables_global->transfer_density->size < 3) {
            return -1;
        }
        if (matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS &&
            (!cosmo_tables_global->transfer_vcb || cosmo_tables_global->transfer_vcb->size < 3)) {
            return -1;
        }
    }

    const double box_len_x = simulation_options_global->BOX_LEN;
    const double box_len_y = simulation_options_global->BOX_LEN;
    const double ncf = simulation_options_global->NON_CUBIC_FACTOR;
    const double box_len_z = ncf * box_len_x;
    const double volume = box_len_x * box_len_y * box_len_z;
    const double dim_ratio_hi_lo = nx / (double)lx;
    const float tophat_R = (float)(physconst.l_factor * box_len_x / (lx + 0.0));

    const bool perturb_hires = matter_options_global->PERTURB_ON_HIGH_RES;
    const int pt_x = perturb_hires ? nx : lx;
    const int pt_y = perturb_hires ? ny : ly;
    const int pt_z = perturb_hires ? nz : lz;
    const double dim_ratio_hi_pt = nx / (double)pt_x;

    float *vel_pointers[3] = {
        perturb_hires ? boxes->hires_vx : boxes->lowres_vx,
        perturb_hires ? boxes->hires_vy : boxes->lowres_vy,
        perturb_hires ? boxes->hires_vz : boxes->lowres_vz,
    };
    float *vel_pointers_2lpt[3] = {
        perturb_hires ? boxes->hires_vx_2LPT : boxes->lowres_vx_2LPT,
        perturb_hires ? boxes->hires_vy_2LPT : boxes->lowres_vy_2LPT,
        perturb_hires ? boxes->hires_vz_2LPT : boxes->lowres_vz_2LPT,
    };

    const size_t hi_pixels = (size_t)nx * ny * nz;
    const size_t lo_pixels = (size_t)lx * ly * lz;
    const size_t pt_pixels = (size_t)pt_x * pt_y * pt_z;
    const size_t hi_k_pixels = (size_t)nx * ny * (nz / 2 + 1);
    const int hi_r_stride = 2 * (nz / 2 + 1);
    const size_t hi_bytes = hi_pixels * sizeof(float);
    const size_t lo_bytes = lo_pixels * sizeof(float);
    const size_t pt_bytes = pt_pixels * sizeof(float);
    const size_t hi_k_bytes = hi_k_pixels * sizeof(cufftComplex);

    cudaStream_t stream = p21_cuda_stream();

    bool non_zero_input = false;
    for (size_t idx = 0; idx < hi_pixels; ++idx) {
        if (boxes->hires_density[idx] != 0.0f) {
            non_zero_input = true;
            break;
        }
    }

    bool ok = true;
    double cosmo[9] = {};
    p21_get_cosmo_constants(cosmo);

    HostSpline dens_spline, vcb_spline;
    if (matter_options_global->POWER_SPECTRUM == POWER_SPECTRUM_CLASS) {
        if (!build_natural_cubic_spline(cosmo_tables_global->transfer_density->x_values,
                                        cosmo_tables_global->transfer_density->y_values,
                                        cosmo_tables_global->transfer_density->size,
                                        &dens_spline)) {
            ok = false;
        }
        if (ok && matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) {
            if (!build_natural_cubic_spline(cosmo_tables_global->transfer_vcb->x_values,
                                            cosmo_tables_global->transfer_vcb->y_values,
                                            cosmo_tables_global->transfer_vcb->size,
                                            &vcb_spline)) {
                ok = false;
            }
        }
    }

    double *d_class_dens = nullptr;
    double *d_class_vcb = nullptr;
    CosmoKernelParams cosmo_params = {
        matter_options_global->POWER_SPECTRUM,
        matter_options_global->V_CB_MODEL,
        cosmo[0],
        cosmo[1],
        cosmo[2],
        cosmo[3],
        cosmo[4],
        cosmo[6],
        cosmo[7],
        cosmo[8],
        cosmo_tables_global->ps_norm,
        cosmo_params_global->POWER_INDEX,
        cosmo_params_global->OMm,
        cosmo_params_global->OMb,
        cosmo_params_global->hlittle,
        physconst.c_kms,
        nullptr,
        dens_spline.n,
        nullptr,
        vcb_spline.n,
    };

    if (ok && p21_mgpu_effective_ranks() >= 2 &&
        (p21_cuda_get_fast_ic() != 0 || non_zero_input) &&
        (hi_bytes >= 512ULL * 1024 * 1024 || mgpu_ic_force_enabled())) {
        if (run_mgpu_compute_initial_conditions(
                random_seed, boxes, non_zero_input, nx, ny, nz, lx, ly, lz, pt_x, pt_y, pt_z,
                box_len_x, box_len_y, box_len_z, volume, dim_ratio_hi_lo, dim_ratio_hi_pt,
                tophat_R, perturb_hires, vel_pointers, vel_pointers_2lpt, cosmo_params, dens_spline,
                vcb_spline) == 0) {
            return 0;
        }
    }

    if (hi_bytes >= 256ULL * 1024 * 1024) {
        size_t total_mem = p21_cuda_total_mem_bytes();
        size_t target_free = hi_bytes * 5;
        if (total_mem > 0) {
            target_free = std::min<size_t>(target_free, (total_mem * 3) / 4);
        }
        p21_cuda_registry_evict_to_free(target_free, stream);
    }

    cufftComplex *d_kspace_saved = (cufftComplex *)p21_cuda_pool_alloc(hi_k_bytes, stream);
    cufftComplex *d_kspace_work = nullptr;
    float *d_real_lo = (float *)p21_cuda_pool_alloc(lo_bytes, stream);
    float *d_real_pt = perturb_hires ? nullptr : d_real_lo;
    double *d_gauss = nullptr;
    float *d_diag_acc = nullptr;
    float *d_phi2_rhs = nullptr;

    ok = ok && d_kspace_saved && d_real_lo;
    const int tpb = 256;

    if (ok && matter_options_global->POWER_SPECTRUM == POWER_SPECTRUM_CLASS) {
        d_class_dens =
            (double *)p21_cuda_pool_alloc(dens_spline.packed.size() * sizeof(double), stream);
        if (!d_class_dens ||
            !P21_CUDA_OK(cudaMemcpyAsync(d_class_dens, dens_spline.packed.data(),
                                         dens_spline.packed.size() * sizeof(double),
                                         cudaMemcpyHostToDevice, stream))) {
            ok = false;
        } else {
            cosmo_params.class_dens_table = d_class_dens;
        }
        if (ok && matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) {
            d_class_vcb =
                (double *)p21_cuda_pool_alloc(vcb_spline.packed.size() * sizeof(double), stream);
            if (!d_class_vcb ||
                !P21_CUDA_OK(cudaMemcpyAsync(d_class_vcb, vcb_spline.packed.data(),
                                             vcb_spline.packed.size() * sizeof(double),
                                             cudaMemcpyHostToDevice, stream))) {
                ok = false;
            } else {
                cosmo_params.class_vcb_table = d_class_vcb;
            }
        }
    }

    if (ok && non_zero_input) {
        d_kspace_work = (cufftComplex *)p21_cuda_pool_alloc(hi_k_bytes, stream);
        float *d_tmp_real = (float *)d_kspace_work;
        if (!d_kspace_work ||
            !P21_CUDA_OK(cudaMemcpyAsync(d_tmp_real, boxes->hires_density, hi_bytes,
                                         cudaMemcpyHostToDevice, stream))) {
            ok = false;
        } else {
            if (!p21_cuda_registry_put_copy(boxes->hires_density, d_tmp_real, hi_bytes, stream)) {
                p21_cuda_invalidate_host_ptr(boxes->hires_density);
            }
            int g = (int)((hi_pixels + tpb - 1) / tpb);
            scale_real_kernel<<<g, tpb, 0, stream>>>(d_tmp_real, d_tmp_real, hi_pixels,
                                                     volume / (double)hi_pixels);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_r2c(d_tmp_real, d_kspace_saved, nx, ny, nz, false, stream);
        }
    } else if (ok) {
        if (p21_cuda_get_fast_ic() != 0) {
            int fg = (int)((hi_k_pixels + tpb - 1) / tpb);
            fill_ic_modes_fast_kernel<<<fg, tpb, 0, stream>>>(
                d_kspace_saved, (uint64_t)random_seed, nx, ny, nz, box_len_x, box_len_y,
                box_len_z, volume, cosmo_params);
            ok = P21_CUDA_OK(cudaGetLastError());
        } else {
            std::vector<uint32_t> seeds((size_t)n_threads);
            if (!p21_cuda_seed_rng_threads(random_seed, n_threads, seeds.data(), stream)) {
                ok = false;
            }
            int max_slice_x = (nx + n_threads - 1) / n_threads;
            size_t max_slice_modes = (size_t)max_slice_x * ny * (nz / 2 + 1);
            d_gauss = (double *)p21_cuda_pool_alloc(max_slice_modes * 2 * sizeof(double), stream);
            if (!d_gauss) {
                ok = false;
            }

            for (int t = 0; ok && t < n_threads; ++t) {
                int q = nx / n_threads;
                int rem = nx % n_threads;
                int first = t * q + (t < rem ? t : rem);
                int count_x = q + (t < rem ? 1 : 0);
                if (count_x <= 0) {
                    continue;
                }
                size_t slice_modes = (size_t)count_x * ny * (nz / 2 + 1);
                if (!p21_cuda_fill_ugaussian_thread(t % 5, seeds[(size_t)t], d_gauss,
                                                    slice_modes * 2, stream)) {
                    ok = false;
                    break;
                }
                int g = (int)((slice_modes + tpb - 1) / tpb);
                fill_ic_modes_slice_kernel<<<g, tpb, 0, stream>>>(
                    d_kspace_saved, d_gauss, first, count_x, ny, nz, box_len_x, box_len_y,
                    box_len_z, volume, cosmo_params);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
            p21_cuda_pool_free(d_gauss, stream);
            d_gauss = nullptr;
        }

        if (ok) {
            d_kspace_work = (cufftComplex *)p21_cuda_pool_alloc(hi_k_bytes, stream);
            if (!d_kspace_work) {
                ok = false;
            }
        }
        if (ok) {
            size_t plane_items = (size_t)2 * nx * ny;
            int pg = (int)((plane_items + tpb - 1) / tpb);
            enforce_real_planes_general_kernel<<<pg, tpb, 0, stream>>>(d_kspace_saved, nx, ny, nz);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 P21_CUDA_OK(cudaMemcpyAsync(d_kspace_work, d_kspace_saved, hi_k_bytes,
                                             cudaMemcpyDeviceToDevice, stream)) &&
                 p21_cuda_fft_c2r(d_kspace_work, (float *)d_kspace_work, nx, ny, nz, true, stream);
        }
        if (ok) {
            ok = d2h_padded_and_register(boxes->hires_density, (float *)d_kspace_work, nx, ny, nz,
                                         hi_r_stride, 1.0 / volume, stream);
        }
    }

    /* Filtered lowres density */
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_kspace_work, d_kspace_saved, hi_k_bytes,
                                         cudaMemcpyDeviceToDevice, stream));
        if (ok && nx != lx) {
            ok = p21_cuda_filter_box(d_kspace_work, nx, ny, nz, box_len_x, ncf, FILTER_TOPHAT,
                                     tophat_R, 0.0f, 0.0f, stream);
        }
        if (ok) {
            ok = p21_cuda_fft_c2r(d_kspace_work, (float *)d_kspace_work, nx, ny, nz, true, stream);
        }
        if (ok) {
            int lg = (int)((lo_pixels + tpb - 1) / tpb);
            resample_hi_to_out_kernel<<<lg, tpb, 0, stream>>>(
                (const float *)d_kspace_work, nx, ny, nz, hi_r_stride, d_real_lo, lx, ly, lz,
                dim_ratio_hi_lo, volume, true, false);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 upload_and_register_d2h(boxes->lowres_density, d_real_lo, lo_bytes, stream);
        }
    }

    /* Relative velocity (CLASS FLUCTS) */
    if (ok && matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) {
        ok = P21_CUDA_OK(cudaMemsetAsync(d_real_lo, 0, lo_bytes, stream));
        int kg = (int)((hi_k_pixels + tpb - 1) / tpb);
        int lg = (int)((lo_pixels + tpb - 1) / tpb);
        for (int ii = 0; ok && ii < 3; ++ii) {
            vcb_kspace_axis_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny,
                                                           nz, box_len_x, box_len_y, box_len_z, ii,
                                                           cosmo_params);
            ok = P21_CUDA_OK(cudaGetLastError());
            if (ok && nx != lx) {
                ok = p21_cuda_filter_box(d_kspace_work, nx, ny, nz, box_len_x, ncf, FILTER_TOPHAT,
                                         tophat_R, 0.0f, 0.0f, stream);
            }
            if (ok) {
                ok = p21_cuda_fft_c2r(d_kspace_work, (float *)d_kspace_work, nx, ny, nz, true,
                                      stream);
            }
            if (ok) {
                resample_hi_to_out_kernel<<<lg, tpb, 0, stream>>>(
                    (const float *)d_kspace_work, nx, ny, nz, hi_r_stride, d_real_lo, lx, ly, lz,
                    dim_ratio_hi_lo, volume, false, true);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
        }
        if (ok) {
            vcb_finalize_kernel<<<lg, tpb, 0, stream>>>(d_real_lo, lo_pixels, volume);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 upload_and_register_d2h(boxes->lowres_vcb, d_real_lo, lo_bytes, stream);
        }
    }

    if (perturb_hires && d_real_lo) {
        p21_cuda_pool_free(d_real_lo, stream);
        d_real_lo = nullptr;
    }

    /* 1LPT velocities */
    if (ok) {
        int kg = (int)((hi_k_pixels + tpb - 1) / tpb);
        int ptg = (int)((pt_pixels + tpb - 1) / tpb);
        for (int ii = 0; ok && ii < 3; ++ii) {
            f_gradient_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                      box_len_x, box_len_y, box_len_z, ii);
            ok = P21_CUDA_OK(cudaGetLastError());
            if (ok && !perturb_hires && nx != lx) {
                ok = p21_cuda_filter_box(d_kspace_work, nx, ny, nz, box_len_x, ncf, FILTER_TOPHAT,
                                         tophat_R, 0.0f, 0.0f, stream);
            }
            if (ok) {
                ok = p21_cuda_fft_c2r(d_kspace_work, (float *)d_kspace_work, nx, ny, nz, true,
                                      stream);
            }
            if (ok) {
                if (perturb_hires) {
                    ok = d2h_padded_and_register(vel_pointers[ii], (float *)d_kspace_work, nx, ny,
                                                 nz, hi_r_stride, 1.0 / volume, stream);
                } else {
                    resample_hi_to_out_kernel<<<ptg, tpb, 0, stream>>>(
                        (const float *)d_kspace_work, nx, ny, nz, hi_r_stride, d_real_pt, pt_x,
                        pt_y, pt_z, dim_ratio_hi_pt, volume, true, false);
                    ok = P21_CUDA_OK(cudaGetLastError()) &&
                         upload_and_register_d2h(vel_pointers[ii], d_real_pt, pt_bytes, stream);
                }
            }
        }
    }

    /* 2LPT velocities (2-buffer exact algebraic decomposition: d0*d1 + (d0+d1)*d2) */
    if (ok && matter_options_global->PERTURB_ALGORITHM == PERTURB_ALGORITHM_2LPT) {
        d_phi2_rhs = (float *)p21_cuda_pool_alloc(hi_bytes, stream);
        if (!d_phi2_rhs || !P21_CUDA_OK(cudaMemsetAsync(d_phi2_rhs, 0, hi_bytes, stream))) {
            ok = false;
        }

        int kg = (int)((hi_k_pixels + tpb - 1) / tpb);
        int hg = (int)((hi_pixels + tpb - 1) / tpb);

        const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        for (int p_idx = 0; ok && p_idx < 3; ++p_idx) {
            int a = pairs[p_idx][0];
            int b = pairs[p_idx][1];
            f_laplacian_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                       box_len_x, box_len_y, box_len_z, a, b);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_c2r(d_kspace_work, (float *)d_kspace_work, nx, ny, nz, true, stream);
            if (ok) {
                sub_sq_padded_2lpt_kernel<<<hg, tpb, 0, stream>>>(
                    d_phi2_rhs, (const float *)d_kspace_work, nx, ny, nz, hi_r_stride);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
        }

        if (ok) {
            d_diag_acc = (float *)p21_cuda_pool_alloc(hi_bytes, stream);
            if (!d_diag_acc) {
                ok = false;
            }
        }

        /* Diagonal a=0 into dense d_diag_acc */
        if (ok) {
            f_laplacian_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                       box_len_x, box_len_y, box_len_z, 0, 0);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_c2r(d_kspace_work, d_diag_acc, nx, ny, nz, false, stream);
            if (ok && !perturb_hires && boxes->hires_vx_2LPT) {
                ok = upload_and_register_d2h(boxes->hires_vx_2LPT, d_diag_acc, hi_bytes, stream,
                                             false);
            }
        }

        /* Diagonal a=1 into in-place padded d_kspace_work */
        if (ok) {
            f_laplacian_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                       box_len_x, box_len_y, box_len_z, 1, 1);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_c2r(d_kspace_work, (float *)d_kspace_work, nx, ny, nz, true, stream);
            if (ok && !perturb_hires && boxes->hires_vy_2LPT) {
                ok = d2h_padded_and_register(boxes->hires_vy_2LPT, (float *)d_kspace_work, nx, ny,
                                             nz, hi_r_stride, 1.0, stream, false);
            }
            if (ok) {
                accum_2lpt_d01_padded_kernel<<<hg, tpb, 0, stream>>>(
                    d_phi2_rhs, d_diag_acc, (const float *)d_kspace_work, nx, ny, nz, hi_r_stride);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
        }

        /* Diagonal a=2 into in-place padded d_kspace_work */
        if (ok) {
            f_laplacian_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                       box_len_x, box_len_y, box_len_z, 2, 2);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_c2r(d_kspace_work, (float *)d_kspace_work, nx, ny, nz, true, stream);
            if (ok && !perturb_hires && boxes->hires_vz_2LPT) {
                ok = d2h_padded_and_register(boxes->hires_vz_2LPT, (float *)d_kspace_work, nx, ny,
                                             nz, hi_r_stride, 1.0, stream, false);
            }
            if (ok) {
                accum_2lpt_d01_d2_padded_kernel<<<hg, tpb, 0, stream>>>(
                    d_phi2_rhs, d_diag_acc, (const float *)d_kspace_work, nx, ny, nz, hi_r_stride);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
        }

        p21_cuda_pool_free(d_diag_acc, stream);
        d_diag_acc = nullptr;

        if (ok) {
            double norm_2lpt = 1.0 / (volume * volume * (double)hi_pixels);
            scale_real_kernel<<<hg, tpb, 0, stream>>>(d_phi2_rhs, d_phi2_rhs, hi_pixels, norm_2lpt);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_r2c(d_phi2_rhs, d_kspace_saved, nx, ny, nz, false, stream);
        }

        p21_cuda_pool_free(d_phi2_rhs, stream);
        d_phi2_rhs = nullptr;

        int ptg = (int)((pt_pixels + tpb - 1) / tpb);
        for (int ii = 0; ok && ii < 3; ++ii) {
            f_gradient_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                      box_len_x, box_len_y, box_len_z, ii);
            ok = P21_CUDA_OK(cudaGetLastError());
            if (ok && !perturb_hires && nx != lx) {
                ok = p21_cuda_filter_box(d_kspace_work, nx, ny, nz, box_len_x, ncf, FILTER_TOPHAT,
                                         tophat_R, 0.0f, 0.0f, stream);
            }
            if (ok) {
                ok = p21_cuda_fft_c2r(d_kspace_work, (float *)d_kspace_work, nx, ny, nz, true,
                                      stream);
            }
            if (ok) {
                if (perturb_hires) {
                    ok = d2h_padded_and_register(vel_pointers_2lpt[ii], (float *)d_kspace_work, nx,
                                                 ny, nz, hi_r_stride, 1.0, stream);
                } else {
                    resample_hi_to_out_kernel<<<ptg, tpb, 0, stream>>>(
                        (const float *)d_kspace_work, nx, ny, nz, hi_r_stride, d_real_pt, pt_x,
                        pt_y, pt_z, dim_ratio_hi_pt, volume, false, false);
                    ok = P21_CUDA_OK(cudaGetLastError()) &&
                         upload_and_register_d2h(vel_pointers_2lpt[ii], d_real_pt, pt_bytes,
                                                 stream);
                }
            }
        }
    }

    p21_cuda_pool_free(d_phi2_rhs, stream);
    p21_cuda_pool_free(d_diag_acc, stream);
    p21_cuda_pool_free(d_class_vcb, stream);
    p21_cuda_pool_free(d_class_dens, stream);
    p21_cuda_pool_free(d_real_lo, stream);
    p21_cuda_pool_free(d_kspace_work, stream);
    p21_cuda_pool_free(d_kspace_saved, stream);

    if (ok && P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        p21_cuda_stats_mut()->ic_runs++;
        return 0;
    }
    return -1;
}
