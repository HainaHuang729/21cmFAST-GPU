#ifndef USE_CUDA

#include <string.h>

#include "21cmFAST.h"

static int g_p21_cuda_enabled = 0;
static int g_p21_cuda_mode = 0;
static int g_p21_cuda_fast_ic = 0;
static int g_p21_cuda_fast_halo_sampler = 0;

int p21_cuda_is_compiled(void) { return 0; }

int p21_cuda_is_available(void) { return 0; }

void p21_cuda_set_enabled(int enabled) { g_p21_cuda_enabled = enabled ? 1 : 0; }

int p21_cuda_get_enabled(void) { return g_p21_cuda_enabled; }

void p21_cuda_set_mode(int mode) { g_p21_cuda_mode = mode ? 1 : 0; }

int p21_cuda_get_mode(void) { return g_p21_cuda_mode; }

void p21_cuda_set_fast_ic(int enabled) { g_p21_cuda_fast_ic = enabled ? 1 : 0; }

int p21_cuda_get_fast_ic(void) { return g_p21_cuda_fast_ic; }

void p21_cuda_set_fast_halo_sampler(int enabled) {
    g_p21_cuda_fast_halo_sampler = enabled ? 1 : 0;
}

int p21_cuda_get_fast_halo_sampler(void) { return g_p21_cuda_fast_halo_sampler; }

void p21_cuda_reset_stats(void) {}

void p21_cuda_get_stats(P21CudaStats *out) {
    if (out) {
        memset(out, 0, sizeof(*out));
        out->device_id = -1;
        out->enabled = g_p21_cuda_enabled;
    }
}

void p21_cuda_cleanup(void) {}

void p21_cuda_invalidate_host_ptr(const void *ptr) { (void)ptr; }

void p21_cuda_registry_clear(void) {}

int p21_cuda_compute_initial_conditions(unsigned long long int random_seed,
                                        InitialConditions *boxes) {
    (void)random_seed;
    (void)boxes;
    return -1;
}

int p21_cuda_compute_perturbed_field(float redshift, InitialConditions *boxes,
                                     PerturbedField *perturbed_field) {
    (void)redshift;
    (void)boxes;
    (void)perturbed_field;
    return -1;
}

int p21_cuda_compute_brightness_temp(float redshift, TsBox *spin_temp, IonizedBox *ionized_box,
                                     PerturbedField *perturb_field, BrightnessTemp *box) {
    (void)redshift;
    (void)spin_temp;
    (void)ionized_box;
    (void)perturb_field;
    (void)box;
    return -1;
}

int p21_cuda_setup_radiation_mean_xe(TsBox *previous_spin_temp, double *x_e_ave_out) {
    (void)previous_spin_temp;
    (void)x_e_ave_out;
    return -1;
}

int p21_cuda_setup_radiation_tables(TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup) {
    (void)previous_spin_temp;
    (void)rad_setup;
    return -1;
}

int p21_cuda_fill_freqint_tables(double zp, int num_R, const double *lower_int_limits,
                                 double hubble_zp, RadiationFieldsSetup *rad_setup) {
    (void)zp;
    (void)num_R;
    (void)lower_int_limits;
    (void)hubble_zp;
    (void)rad_setup;
    return -1;
}

int p21_cuda_update_radiation_fields(float redshift, EmissivityFields *emissivity_fields, int R_ct,
                                     double R_star, PerturbedField *perturbed_field,
                                     TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup,
                                     RadiationFields *radiation_fields) {
    (void)redshift;
    (void)emissivity_fields;
    (void)R_ct;
    (void)R_star;
    (void)perturbed_field;
    (void)previous_spin_temp;
    (void)rad_setup;
    (void)radiation_fields;
    return -1;
}

int p21_cuda_compute_radiation_shells(float redshift, int n_shells, const int *R_ct_arr,
                                      double R_star, const double *interp_params,
                                      EmissivityFields **ef_desc_arr,
                                      EmissivityFields **ef_prog_arr,
                                      RadiationFieldsSetup *rad_setup,
                                      PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                                      RadiationFields *radiation_fields) {
    (void)redshift;
    (void)n_shells;
    (void)R_ct_arr;
    (void)R_star;
    (void)interp_params;
    (void)ef_desc_arr;
    (void)ef_prog_arr;
    (void)rad_setup;
    (void)perturbed_field;
    (void)previous_spin_temp;
    (void)radiation_fields;
    return -1;
}

int p21_cuda_init_first_ts(TsBox *box, float *dens, float z, double xe_recfast, double tk_recfast,
                           double cT_ad) {
    (void)box;
    (void)dens;
    (void)z;
    (void)xe_recfast;
    (void)tk_recfast;
    (void)cT_ad;
    return -1;
}

int p21_cuda_compute_ts_box(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                            RadiationFields *radiation_fields, TsBox *previous_spin_temp,
                            InitialConditions *ini_boxes, TsBox *this_spin_temp) {
    (void)redshift;
    (void)prev_redshift;
    (void)perturbed_field;
    (void)radiation_fields;
    (void)previous_spin_temp;
    (void)ini_boxes;
    (void)this_spin_temp;
    return -1;
}

P21CudaEmissivityCtx *p21_cuda_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, PerturbedField *perturbed_field,
    TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, const ScalingConstants *consts,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out) {
    (void)redshift;
    (void)ini_boxes;
    (void)perturbed_field;
    (void)previous_spin_temp;
    (void)previous_ionize_box;
    (void)consts;
    (void)log10_mturn_averages;
    (void)min_dens_out;
    (void)max_dens_out;
    return NULL;
}

int p21_cuda_emissivity_eval_and_finish(P21CudaEmissivityCtx *ctx, const ScalingConstants *ev_consts,
                                        const P21EmissivityAverages *uhmf_averages,
                                        EmissivityFields *emissivity_fields) {
    (void)ctx;
    (void)ev_consts;
    (void)uhmf_averages;
    (void)emissivity_fields;
    return -1;
}

void p21_cuda_emissivity_abort(P21CudaEmissivityCtx *ctx) { (void)ctx; }

P21CudaIonCtx *p21_cuda_ion_begin(const P21IonBoxParams *params, int n_radii,
                                  PerturbedField *perturbed_field,
                                  PerturbedField *previous_perturbed_field,
                                  IonizedBox *previous_ionize_box, TsBox *spin_temp,
                                  EmissivityFields *emissivity_fields,
                                  InitialConditions *ini_boxes) {
    (void)params;
    (void)n_radii;
    (void)perturbed_field;
    (void)previous_perturbed_field;
    (void)previous_ionize_box;
    (void)spin_temp;
    (void)emissivity_fields;
    (void)ini_boxes;
    return NULL;
}

int p21_cuda_ion_calc_mcrit(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                            double *log10_mturn_acg_avg, double *log10_mturn_mcg_avg) {
    (void)ctx;
    (void)params;
    (void)log10_mturn_acg_avg;
    (void)log10_mturn_mcg_avg;
    return -1;
}

int p21_cuda_ion_set_fully_neutral(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                   double xion_recfast_z, IonizedBox *box, double *global_xH_out) {
    (void)ctx;
    (void)params;
    (void)xion_recfast_z;
    (void)box;
    (void)global_xH_out;
    return -1;
}

int p21_cuda_ion_prepare_r2c(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    (void)ctx;
    (void)params;
    return -1;
}

int p21_cuda_ion_filter_and_extrema(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                    const P21IonRadiusSpec *rspec, double *min_dens,
                                    double *max_dens, double *prev_min_dens,
                                    double *prev_max_dens) {
    (void)ctx;
    (void)params;
    (void)rspec;
    (void)min_dens;
    (void)max_dens;
    (void)prev_min_dens;
    (void)prev_max_dens;
    return -1;
}

int p21_cuda_ion_calc_cond_nion(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                P21IonRadiusSpec *rspec, int need_prev_ion) {
    (void)ctx;
    (void)params;
    (void)rspec;
    (void)need_prev_ion;
    return -1;
}

int p21_cuda_ion_find_regions(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                              const P21IonRadiusSpec *rspec, double nion_uncond_acg,
                              double nion_uncond_mcg, double nion_limit_acg, double nion_limit_mcg,
                              float prev_homo_nrec) {
    (void)ctx;
    (void)params;
    (void)rspec;
    (void)nion_uncond_acg;
    (void)nion_uncond_mcg;
    (void)nion_limit_acg;
    (void)nion_limit_mcg;
    (void)prev_homo_nrec;
    return -1;
}

int p21_cuda_ion_set_ionized_temps(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    (void)ctx;
    (void)params;
    return -1;
}

int p21_cuda_ion_global_means(P21CudaIonCtx *ctx, double *global_xH_out, float *global_gamma12_out) {
    (void)ctx;
    (void)global_xH_out;
    (void)global_gamma12_out;
    return -1;
}

int p21_cuda_ion_recomb_inhomo(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    (void)ctx;
    (void)params;
    return -1;
}

int p21_cuda_ion_finish(P21CudaIonCtx *ctx, const P21IonBoxParams *params, IonizedBox *box) {
    (void)ctx;
    (void)params;
    (void)box;
    return -1;
}

void p21_cuda_ion_end(P21CudaIonCtx *ctx) { (void)ctx; }

int p21_cuda_fill_mhr_table(int nz, float del_z, int ng, double lng_min, float del_lng,
                            double n_h0, double y_he, double alpha_b_1e4, const double *a_mhr,
                            const double *c_mhr, const double *beta_mhr, double *rr_table_out) {
    (void)nz;
    (void)del_z;
    (void)ng;
    (void)lng_min;
    (void)del_lng;
    (void)n_h0;
    (void)y_he;
    (void)alpha_b_1e4;
    (void)a_mhr;
    (void)c_mhr;
    (void)beta_mhr;
    (void)rr_table_out;
    return -1;
}

P21CudaLagEmissCtx *p21_cuda_lag_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, TsBox *previous_spin_temp,
    IonizedBox *previous_ionize_box, const ScalingConstants *consts, int need_integral,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out) {
    (void)redshift;
    (void)ini_boxes;
    (void)previous_spin_temp;
    (void)previous_ionize_box;
    (void)consts;
    (void)need_integral;
    (void)log10_mturn_averages;
    (void)min_dens_out;
    (void)max_dens_out;
    return NULL;
}

int p21_cuda_lag_emissivity_add_halos(P21CudaLagEmissCtx *ctx, InitialConditions *ini_boxes,
                                      HaloCatalog *halos, const ScalingConstants *consts) {
    (void)ctx;
    (void)ini_boxes;
    (void)halos;
    (void)consts;
    return -1;
}

int p21_cuda_lag_emissivity_add_integral(P21CudaLagEmissCtx *ctx, InitialConditions *ini_boxes,
                                         const ScalingConstants *ev_consts,
                                         const P21EmissivityAverages *uhmf_averages) {
    (void)ctx;
    (void)ini_boxes;
    (void)ev_consts;
    (void)uhmf_averages;
    return -1;
}

int p21_cuda_lag_emissivity_finish(P21CudaLagEmissCtx *ctx, EmissivityFields *emissivity_fields) {
    (void)ctx;
    (void)emissivity_fields;
    return -1;
}

void p21_cuda_lag_emissivity_abort(P21CudaLagEmissCtx *ctx) { (void)ctx; }

int p21_cuda_compute_perturbed_halo_catalog(float redshift, InitialConditions *boxes,
                                            TsBox *prev_ts, IonizedBox *prev_ion,
                                            HaloCatalog *halos,
                                            PerturbedHaloCatalog *halos_perturbed) {
    (void)redshift;
    (void)boxes;
    (void)prev_ts;
    (void)prev_ion;
    (void)halos;
    (void)halos_perturbed;
    return -1;
}

P21CudaDexmCtx *p21_cuda_dexm_run(float redshift, InitialConditions *boxes, float M_MIN,
                                  float Delta_R, unsigned long long int *total_halo_num_out) {
    (void)redshift;
    (void)boxes;
    (void)M_MIN;
    (void)Delta_R;
    (void)total_halo_num_out;
    return NULL;
}

int p21_cuda_dexm_finish(P21CudaDexmCtx *ctx, HaloCatalog *halos_dexm,
                         float *halo_overlap_box_out) {
    (void)ctx;
    (void)halos_dexm;
    (void)halo_overlap_box_out;
    return -1;
}

void p21_cuda_dexm_abort(P21CudaDexmCtx *ctx) { (void)ctx; }

int p21_cuda_fill_dndm_tables(int nx, double xmin, double xmax, double ymin, double ymax,
                              double growth_out, double param, int from_catalog, double *nhalo_out,
                              double *fcoll_out) {
    (void)nx;
    (void)xmin;
    (void)xmax;
    (void)ymin;
    (void)ymax;
    (void)growth_out;
    (void)param;
    (void)from_catalog;
    (void)nhalo_out;
    (void)fcoll_out;
    return -1;
}

int p21_cuda_fill_dndm_inverse_table(int nx, int np, double xmin, double xmax, double lnM_min,
                                     double growth_out, double param, int from_catalog,
                                     double min_lp, double **z_arr_out) {
    (void)nx;
    (void)np;
    (void)xmin;
    (void)xmax;
    (void)lnM_min;
    (void)growth_out;
    (void)param;
    (void)from_catalog;
    (void)min_lp;
    (void)z_arr_out;
    return -1;
}

int p21_cuda_sample_halo_grids(unsigned long long int seed, double redshift, float *dens_field,
                               float *halo_overlap_box, HaloCatalog *halofield_large,
                               HaloCatalog *halofield_out, const P21HaloSamplingParams *hsp,
                               int *overflow_out, unsigned long long int *thread_counts_out) {
    (void)seed;
    (void)redshift;
    (void)dens_field;
    (void)halo_overlap_box;
    (void)halofield_large;
    (void)halofield_out;
    (void)hsp;
    (void)overflow_out;
    (void)thread_counts_out;
    return -1;
}

int p21_cuda_sample_halo_progenitors(unsigned long long int seed, double z_in, double z_out,
                                     HaloCatalog *halofield_in, HaloCatalog *halofield_out,
                                     const P21HaloSamplingParams *hsp, int *overflow_out,
                                     unsigned long long int *thread_counts_out) {
    (void)seed;
    (void)z_in;
    (void)z_out;
    (void)halofield_in;
    (void)halofield_out;
    (void)hsp;
    (void)overflow_out;
    (void)thread_counts_out;
    return -1;
}

int p21_cuda_fill_cond_emissivity_tables(int kind, double redshift, double growthf, double lnM_min,
                                         double lnM_max, double lnM_cond, double sigma_cond,
                                         int n_delta, double min_dens, double max_dens,
                                         int n_mturn, int do_acg_2d, double acg_l10_min,
                                         double acg_l10_max, int do_mcg_2d, double mcg_l10_min,
                                         double mcg_l10_max, const ScalingConstants *sc,
                                         float *acg_1d_out, float **acg_2d_out,
                                         float **mcg_2d_out) {
    (void)kind;
    (void)redshift;
    (void)growthf;
    (void)lnM_min;
    (void)lnM_max;
    (void)lnM_cond;
    (void)sigma_cond;
    (void)n_delta;
    (void)min_dens;
    (void)max_dens;
    (void)n_mturn;
    (void)do_acg_2d;
    (void)acg_l10_min;
    (void)acg_l10_max;
    (void)do_mcg_2d;
    (void)mcg_l10_min;
    (void)mcg_l10_max;
    (void)sc;
    (void)acg_1d_out;
    (void)acg_2d_out;
    (void)mcg_2d_out;
    return -1;
}

int p21_cuda_fill_uncond_emissivity_tables(
    int kind, int n_z, const double *z_vals, const double *growthf_vals, const double *lnMmin_vals,
    double lnMmax, const double *mturn_acg_hom_vals, const double *atomic_thresh_vals,
    const double *sfr_timescale_vals, int do_mcg, int n_mturn, double mcg_l10_min,
    double mcg_l10_max, const ScalingConstants *sc, double *acg_1d_out, double **mcg_2d_out) {
    (void)kind;
    (void)n_z;
    (void)z_vals;
    (void)growthf_vals;
    (void)lnMmin_vals;
    (void)lnMmax;
    (void)mturn_acg_hom_vals;
    (void)atomic_thresh_vals;
    (void)sfr_timescale_vals;
    (void)do_mcg;
    (void)n_mturn;
    (void)mcg_l10_min;
    (void)mcg_l10_max;
    (void)sc;
    (void)acg_1d_out;
    (void)mcg_2d_out;
    return -1;
}

int p21_cuda_include_dvdr_in_tau21(const float *h_tb, const float *h_los_vel,
                                   const float *h_tau_21, const double *h_H_s_inv, int nx, int ny,
                                   int nz, double dx_los_mpc, double max_dvdr, int use_ts_fluct,
                                   int periodic, float *h_out) {
    (void)h_tb;
    (void)h_los_vel;
    (void)h_tau_21;
    (void)h_H_s_inv;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)dx_los_mpc;
    (void)max_dvdr;
    (void)use_ts_fluct;
    (void)periodic;
    (void)h_out;
    return -1;
}

int p21_cuda_rsds_shift(const float *h_field, const float *h_disp, int n_los, int nz,
                        int n_rsd_subcells, int periodic, float *h_out) {
    (void)h_field;
    (void)h_disp;
    (void)n_los;
    (void)nz;
    (void)n_rsd_subcells;
    (void)periodic;
    (void)h_out;
    return -1;
}

int p21_cuda_rectilinear_slices(const float *h_box_a, const float *h_box_b, int nx, int ny, int nz,
                                int n_slices, const int *h_z_indices, const float *h_w_a,
                                const float *h_w_b, int kind, float *h_out) {
    (void)h_box_a;
    (void)h_box_b;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)n_slices;
    (void)h_z_indices;
    (void)h_w_a;
    (void)h_w_b;
    (void)kind;
    (void)h_out;
    return -1;
}

int p21_cuda_power_spectrum_1d(const float *h_field_a, const float *h_field_b, int nx, int ny,
                               int nz, double lx, double ly, double lz, int n_bins,
                               const double *h_k_edges, int skip_dc, double *h_k_mean_out,
                               double *h_pk_out, double *h_delta_sq_out,
                               unsigned long long *h_counts_out) {
    (void)h_field_a;
    (void)h_field_b;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)lx;
    (void)ly;
    (void)lz;
    (void)n_bins;
    (void)h_k_edges;
    (void)skip_dc;
    (void)h_k_mean_out;
    (void)h_pk_out;
    (void)h_delta_sq_out;
    (void)h_counts_out;
    return -1;
}

int p21_cuda_power_spectrum_2d(const float *h_field_a, const float *h_field_b, int nx, int ny,
                               int nz, double lx, double ly, double lz, int los_axis,
                               int n_kperp, const double *h_kperp_edges, int n_kpara,
                               const double *h_kpara_edges, int skip_dc,
                               double *h_kperp_mean_out, double *h_kpara_mean_out,
                               double *h_pk_out, unsigned long long *h_counts_out) {
    (void)h_field_a;
    (void)h_field_b;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)lx;
    (void)ly;
    (void)lz;
    (void)los_axis;
    (void)n_kperp;
    (void)h_kperp_edges;
    (void)n_kpara;
    (void)h_kpara_edges;
    (void)skip_dc;
    (void)h_kperp_mean_out;
    (void)h_kpara_mean_out;
    (void)h_pk_out;
    (void)h_counts_out;
    return -1;
}

int p21_cuda_angular_slice(const float *h_box_a, const float *h_box_b, int nx, int ny, int nz,
                           int n_coords, const double *h_coords_xyz, float w_a, float w_b,
                           int order, int kind, float *h_out) {
    (void)h_box_a;
    (void)h_box_b;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)n_coords;
    (void)h_coords_xyz;
    (void)w_a;
    (void)w_b;
    (void)order;
    (void)kind;
    (void)h_out;
    return -1;
}

int p21_cuda_angular_los_velocity(const float *h_vxa, const float *h_vya, const float *h_vza,
                                  const float *h_vxb, const float *h_vyb, const float *h_vzb,
                                  int nx, int ny, int nz, int n_coords,
                                  const double *h_coords_xyz, const double *h_unit_xyz, float w_a,
                                  float w_b, int order, float *h_out) {
    (void)h_vxa;
    (void)h_vya;
    (void)h_vza;
    (void)h_vxb;
    (void)h_vyb;
    (void)h_vzb;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)n_coords;
    (void)h_coords_xyz;
    (void)h_unit_xyz;
    (void)w_a;
    (void)w_b;
    (void)order;
    (void)h_out;
    return -1;
}

void p21_cuda_set_lazy_d2h(int enabled) { (void)enabled; }

int p21_cuda_get_lazy_d2h(void) { return 0; }

int p21_cuda_registry_sync_to_host(const void *host_ptr) {
    (void)host_ptr;
    return 0;
}

void p21_cuda_registry_sync_all_lazy(void) {}

int p21_cuda_lzf_compress_batch_f32(const float *host_chunks, int n_chunks, size_t chunk_elems,
                                    int shuffle, unsigned char *host_out, size_t max_out_bytes,
                                    size_t *out_sizes, int *has_nonfinite_out) {
    (void)host_chunks;
    (void)n_chunks;
    (void)chunk_elems;
    (void)shuffle;
    (void)host_out;
    (void)max_out_bytes;
    (void)out_sizes;
    (void)has_nonfinite_out;
    return -1;
}

int p21_cuda_device_count(void) { return 0; }

void p21_cuda_set_devices(const int *device_ids, int n_devices) {
    (void)device_ids;
    (void)n_devices;
}

int p21_cuda_get_num_devices(void) { return 0; }

int p21_cuda_get_devices(int *out_ids, int max_n) {
    (void)out_ids;
    (void)max_n;
    return 0;
}

int p21_cuda_mgpu_fft_r2c_host(const float *h_in, float *h_out_complex, int nx, int ny, int nz) {
    (void)h_in;
    (void)h_out_complex;
    (void)nx;
    (void)ny;
    (void)nz;
    return -1;
}

int p21_cuda_mgpu_fft_c2r_host(const float *h_in_complex, float *h_out_real, int nx, int ny,
                               int nz) {
    (void)h_in_complex;
    (void)h_out_real;
    (void)nx;
    (void)ny;
    (void)nz;
    return -1;
}

int p21_cuda_wedge_filter_3d(const float *h_in, float *h_out, int nx, int ny, int nz, double lx,
                             double ly, double lz, int los_axis, double slope, double buffer_mpc) {
    (void)h_in;
    (void)h_out;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)lx;
    (void)ly;
    (void)lz;
    (void)los_axis;
    (void)slope;
    (void)buffer_mpc;
    return -1;
}

int p21_cuda_add_thermal_noise_3d(const float *h_in, float *h_out, size_t n_elems, float sigma_mk,
                                  unsigned long long seed) {
    (void)h_in;
    (void)h_out;
    (void)n_elems;
    (void)sigma_mk;
    (void)seed;
    return -1;
}

int p21_cuda_field_moments_and_pdf(const float *h_in, size_t n_elems, int n_bins, double vmin,
                                   double vmax, double *mean_out, double *var_out, double *skew_out,
                                   double *kurt_out, unsigned long long *hist_out) {
    (void)h_in;
    (void)n_elems;
    (void)n_bins;
    (void)vmin;
    (void)vmax;
    (void)mean_out;
    (void)var_out;
    (void)skew_out;
    (void)kurt_out;
    (void)hist_out;
    return -1;
}

int p21_cuda_export_device_f32(const float *h_ptr, size_t n_elems, unsigned long long *d_ptr_out,
                               int *device_id_out) {
    (void)h_ptr;
    (void)n_elems;
    (void)d_ptr_out;
    (void)device_id_out;
    return -1;
}

void p21_cuda_free_exported_device_ptr(unsigned long long d_ptr) { (void)d_ptr; }

int p21_cuda_device_copy_to_host_f32(unsigned long long d_ptr, float *h_out, size_t n_elems) {
    (void)d_ptr;
    (void)h_out;
    (void)n_elems;
    return -1;
}

int p21_cuda_mock_observation_3d(const float *h_in, float *h_out, int nx, int ny, int nz,
                                 double lx, double ly, double lz, int los_axis,
                                 double sigma_perp_mpc, double channel_dr_mpc, int channel_mode,
                                 int apply_uv_mask, double k_perp_min, double k_perp_max,
                                 int apply_wedge, double wedge_slope, double wedge_buffer_mpc,
                                 int add_noise, int n_noise_table, double noise_kperp_max,
                                 const float *h_sigma_re_table, unsigned long long seed) {
    (void)h_in;
    (void)h_out;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)lx;
    (void)ly;
    (void)lz;
    (void)los_axis;
    (void)sigma_perp_mpc;
    (void)channel_dr_mpc;
    (void)channel_mode;
    (void)apply_uv_mask;
    (void)k_perp_min;
    (void)k_perp_max;
    (void)apply_wedge;
    (void)wedge_slope;
    (void)wedge_buffer_mpc;
    (void)add_noise;
    (void)n_noise_table;
    (void)noise_kperp_max;
    (void)h_sigma_re_table;
    (void)seed;
    return -1;
}

int p21_cuda_field_mean_f32(const float *h_in, size_t n_elems, double *mean_out) {
    (void)h_in;
    (void)n_elems;
    (void)mean_out;
    return -1;
}

int p21_cuda_bispectrum_3d(const float *h_field, int nx, int ny, int nz, double lx, double ly,
                           double lz, int remove_mean, int n_triangles, const double *h_k1_lo,
                           const double *h_k1_hi, const double *h_k2_lo, const double *h_k2_hi,
                           const double *h_k3_lo, const double *h_k3_hi, double *h_b_out,
                           double *h_q_out, double *h_pk1_out, double *h_pk2_out, double *h_pk3_out,
                           double *h_ntri_out) {
    (void)h_field;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)lx;
    (void)ly;
    (void)lz;
    (void)remove_mean;
    (void)n_triangles;
    (void)h_k1_lo;
    (void)h_k1_hi;
    (void)h_k2_lo;
    (void)h_k2_hi;
    (void)h_k3_lo;
    (void)h_k3_hi;
    (void)h_b_out;
    (void)h_q_out;
    (void)h_pk1_out;
    (void)h_pk2_out;
    (void)h_pk3_out;
    (void)h_ntri_out;
    return -1;
}

int p21_cuda_mfp_bsd(const float *h_field, int nx, int ny, int nz, double lx, double ly, double lz,
                     float threshold, int below_threshold, unsigned long long n_rays,
                     double step_cells, double max_dist, int n_bins, double r_min, double r_max,
                     int log_bins, unsigned long long seed, unsigned long long *h_hist_out,
                     double *mean_r_out, double *std_r_out, unsigned long long *n_ion_out) {
    (void)h_field;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)lx;
    (void)ly;
    (void)lz;
    (void)threshold;
    (void)below_threshold;
    (void)n_rays;
    (void)step_cells;
    (void)max_dist;
    (void)n_bins;
    (void)r_min;
    (void)r_max;
    (void)log_bins;
    (void)seed;
    (void)h_hist_out;
    (void)mean_r_out;
    (void)std_r_out;
    (void)n_ion_out;
    return -1;
}

int p21_cuda_minkowski_3d(const float *h_field, int nx, int ny, int nz, double lx, double ly,
                          double lz, int n_thresholds, const double *h_thresholds,
                          double *h_v0_out, double *h_v1_out, double *h_v2_out, double *h_v3_out,
                          long long *h_euler_out) {
    (void)h_field;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)lx;
    (void)ly;
    (void)lz;
    (void)n_thresholds;
    (void)h_thresholds;
    (void)h_v0_out;
    (void)h_v1_out;
    (void)h_v2_out;
    (void)h_v3_out;
    (void)h_euler_out;
    return -1;
}

int p21_cuda_cic_deposit_3d(const float *h_pos_xyz, const float *h_weights,
                            unsigned long long n_particles, int nx, int ny, int nz, double lx,
                            double ly, double lz, int overdensity, float *h_grid_out) {
    (void)h_pos_xyz;
    (void)h_weights;
    (void)n_particles;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)lx;
    (void)ly;
    (void)lz;
    (void)overdensity;
    (void)h_grid_out;
    return -1;
}

int p21_cuda_mgpu_filter_box_host(const float *h_in, float *h_out, int nx, int ny, int nz,
                                  double box_len, double non_cubic_factor, int filter_type, float R,
                                  float R_param, float R_star) {
    (void)h_in;
    (void)h_out;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)box_len;
    (void)non_cubic_factor;
    (void)filter_type;
    (void)R;
    (void)R_param;
    (void)R_star;
    return -1;
}

#else

typedef int p21_cuda_stubs_unused_translation_unit_t;

#endif /* USE_CUDA */
