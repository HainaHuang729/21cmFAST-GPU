#ifndef P21_CUDA_FILTERS_CUH
#define P21_CUDA_FILTERS_CUH

#include "cuda_common.cuh"

namespace {

__device__ static inline double p21_real_tophat_w(double kR) {
    if (kR < 1e-4) {
        return 1.0 - kR * kR / 10.0;
    }
    return 3.0 * pow(kR, -3) * (sin(kR) - cos(kR) * kR);
}

__device__ static inline double p21_sharp_k_w(double kR) {
    return (kR * 0.413566994 > 1.0) ? 0.0 : 1.0;
}

__device__ static inline double p21_gaussian_w(double kR_sq) {
    return exp(-0.643 * 0.643 * kR_sq / 2.0);
}

__device__ static inline double p21_exp_mfp_w(double k, double R, double mfp, double exp_term) {
    double kR = k * R;
    double ratio = mfp / R;
    if (kR < 1e-4) {
        double r2 = ratio * ratio;
        double r3 = r2 * ratio;
        double ts_0 = 6.0 * r3 - exp_term * (6.0 * r3 + 6.0 * r2 + 3.0 * ratio);
        return ts_0 + (exp_term * (2.0 * r2 + 0.5 * ratio) - 2.0 * ts_0 * r2) * kR * kR;
    }
    double r2 = ratio * ratio;
    double r3 = r2 * ratio;
    double f = (kR * kR * r2 + 2.0 * ratio + 1.0) * ratio * cos(kR);
    f += (kR * kR * (r2 - r3) + ratio + 1.0) * sin(kR) / kR;
    f *= exp_term;
    f -= 2.0 * r2;
    double d = (kR * ratio) * (kR * ratio) + 1.0;
    f *= -3.0 * ratio / (d * d);
    return f;
}

__device__ static inline double p21_spherical_shell_w(double k, double R_inner, double R_outer) {
    double kR_inner = k * R_inner;
    double kR_outer = k * R_outer;
    if (kR_outer < 1e-4) {
        double q = R_inner / R_outer;
        return 1.0 - kR_outer * kR_outer / 10.0 * (pow(q, 5) - 1.0) / (pow(q, 3) - 1.0);
    }
    return 3.0 / (pow(kR_outer, 3) - pow(kR_inner, 3)) *
           (sin(kR_outer) - cos(kR_outer) * kR_outer - sin(kR_inner) + cos(kR_inner) * kR_inner);
}

__device__ static inline double p21_gammainv(double x) {
    if (x > 0.0) {
        return (x > 171.6) ? 0.0 : (1.0 / tgamma(x));
    }
    return tgamma(1.0 - x) * sinpi(x) / M_PI;
}

__device__ static inline double p21_asymptotic_2F3(double kR, double alpha, double beta) {
    double a1 = (2.0 + alpha) / 2.0;
    double a2 = (3.0 + alpha) / 2.0;
    double b1 = 5.0 / 2.0;
    double b2 = (2.0 + alpha + beta) / 2.0;
    double b3 = (3.0 + alpha + beta) / 2.0;

    double gamma_b1 = 3.0 / 4.0;
    double gamma_b2_over_a1, gamma_b3_over_a2;
    double gamma_a1 = 0.0, gamma_a2 = 0.0;

    if (a1 < 20.0) {
        gamma_a1 = tgamma(a1);
        gamma_a2 = tgamma(a2);
        gamma_b2_over_a1 = tgamma(b2) / gamma_a1;
        gamma_b3_over_a2 = tgamma(b3) / gamma_a2;
    } else {
        double y = beta / 2.0;
        gamma_b2_over_a1 =
            pow(a1, y) *
            exp((a1 + y - 0.5) * (y / a1 - y * y / (2.0 * a1 * a1) + y * y * y / (3.0 * a1 * a1 * a1)) -
                y);
        gamma_b3_over_a2 =
            pow(a2, y) *
            exp((a2 + y - 0.5) * (y / a2 - y * y / (2.0 * a2 * a1) + y * y * y / (3.0 * a2 * a2 * a2)) -
                y);
    }

    double decay_term1 = 0.0;
    double decay_term2 = 0.0;
    if (alpha < 10.0) {
        double gamma_b1_minus_a1_inv = p21_gammainv(b1 - a1);
        double gamma_b1_minus_a2_inv = p21_gammainv(b1 - a2);
        double gamma_b2_minus_a2_inv = p21_gammainv(b2 - a2);
        decay_term1 = M_PI * gamma_a1 * gamma_b1_minus_a1_inv / tgamma(b2 - a1) / tgamma(b3 - a1) /
                      pow(kR / 2.0, alpha + 2.0);
        decay_term2 = -2.0 * M_PI * gamma_a2 * gamma_b1_minus_a2_inv * gamma_b2_minus_a2_inv /
                      tgamma(b3 - a2) / pow(kR / 2.0, alpha + 3.0);
    }

    double phase = kR - M_PI * (2.0 + beta) / 2.0;
    double F_asymp = (cos(phase) - (1.0 + (alpha - 1.0) * beta) / kR * sin(phase)) /
                     pow(kR / 2.0, beta + 2.0);
    F_asymp += decay_term1 + decay_term2;
    F_asymp *= gamma_b1 * gamma_b2_over_a1 * gamma_b3_over_a2;
    return F_asymp;
}

__device__ static inline double p21_hyper_2F3(double kR, double alpha, double beta) {
    if (kR == 0.0) {
        return 1.0;
    }
    if (beta == 0.0) {
        return 3.0 / pow(kR, 3) * (sin(kR) - cos(kR) * kR);
    }
    if (kR < 30.0) {
        constexpr int kMaxTerms = 1000;
        double sum = 0.0;
        double term = 1.0;
        for (int n = 1; n < kMaxTerms; ++n) {
            sum += term;
            term *= -1.0 / (1.0 + beta / (alpha + 2.0 * n)) /
                    (1.0 + beta / (alpha + 1.0 + 2.0 * n)) * kR * kR / (2.0 * n) /
                    (2.0 * n + 3.0);
            if (fabs(term) < fabs(sum) * 1e-4) {
                break;
            }
        }
        return sum;
    } else {
        double F_ms = p21_asymptotic_2F3(kR, alpha, beta);
        double F_sl = 3.0 / pow(kR, 3) * (sin(kR) - cos(kR) * kR);
        return (fabs(F_ms) < fabs(F_sl)) ? F_ms : F_sl;
    }
}

__device__ static inline double p21_multiple_scattering_w(double k, double R_inner, double R_outer,
                                                          double alpha_in, double beta_in,
                                                          double alpha_out, double beta_out) {
    double kR_in = k * R_inner;
    double kR_out = k * R_outer;
    double r_out3 = R_outer * R_outer * R_outer;
    double r_in3 = R_inner * R_inner * R_inner;
    double W = r_out3 * p21_hyper_2F3(kR_out, alpha_out, beta_out) -
               r_in3 * p21_hyper_2F3(kR_in, alpha_in, beta_in);
    return W / (r_out3 - r_in3);
}

__device__ static inline double p21_eval_filter_weight(int filter_type, float k_mag_sq, float R,
                                                       float R_param, double exp_term,
                                                       double alpha_in, double beta_in,
                                                       double alpha_out, double beta_out) {
    if (filter_type == FILTER_TOPHAT) {
        return p21_real_tophat_w((double)(sqrtf(k_mag_sq) * R));
    } else if (filter_type == FILTER_SHARP_K) {
        return p21_sharp_k_w((double)(sqrtf(k_mag_sq) * R));
    } else if (filter_type == FILTER_GAUSSIAN) {
        return p21_gaussian_w((double)(k_mag_sq * R * R));
    } else if (filter_type == FILTER_EXP_MFP) {
        return p21_exp_mfp_w(sqrt((double)k_mag_sq), (double)R, (double)R_param, exp_term);
    } else if (filter_type == FILTER_SPHERICAL_SHELL_STRAIGHT_LINE) {
        return p21_spherical_shell_w(sqrt((double)k_mag_sq), (double)R, (double)R_param);
    } else if (filter_type == FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING) {
        return p21_multiple_scattering_w(sqrt((double)k_mag_sq), (double)R, (double)R_param,
                                         alpha_in, beta_in, alpha_out, beta_out);
    }
    return 1.0;
}

}  // namespace

#endif /* P21_CUDA_FILTERS_CUH */
