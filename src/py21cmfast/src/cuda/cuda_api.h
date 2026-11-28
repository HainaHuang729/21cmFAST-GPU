#ifndef P21_CUDA_API_H
#define P21_CUDA_API_H

#include "../OutputStructs.h"
#include "../scaling_relations.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct P21EmissivityAverages {
    double halo_number;
    double halo_mass_density;
    double stellar_mass_density_acg;
    double stellar_mass_density_mcg;
    double sfrd_acg;
    double sfrd_mcg;
    double n_ion;
    double xray_emissivity;
    double fesc_weighted_sfrd;
} P21EmissivityAverages;

typedef struct P21IonBoxParams {
    double redshift;
    double stored_redshift;
    double prev_redshift;
    double growth_factor;
    double prev_growth_factor;
    double photoncons_adjustment_factor;
    double dz;
    double fabs_dtdz;
    int fix_mean;
    int mass_dep_zeta;
    int filter_recombinations;
    int lagrangian_source_grids;
    int need_mcgs_nion;
    int hii_filter;
    double T_re;
    double mfp_meandens;
    double gamma_prefactor;
    double nion_prefactor_acg;
    double nion_prefactor_mcg;
    double TK_nofluct;
    double adia_TK_term;
    double M_min;
    double vcb_const;
    double mturn_acg_homogeneous;
    double atomic_cooling_threshold;
} P21IonBoxParams;

typedef struct P21IonRadiusSpec {
    double R;
    double M_max_R;
    double ln_M_max_R;
    double sigma_maxmass;
    int R_index;
    double nion_conditional_ave_acg;
    double nion_conditional_ave_mcg;
} P21IonRadiusSpec;

typedef struct P21CudaEmissivityCtx P21CudaEmissivityCtx;
typedef struct P21CudaLagEmissCtx P21CudaLagEmissCtx;
typedef struct P21CudaIonCtx P21CudaIonCtx;
typedef struct P21CudaDexmCtx P21CudaDexmCtx;

typedef struct P21HaloSamplingParams {
    int from_catalog;
    double z_in;
    double z_out;
    double growth_in;
    double growth_out;
    double M_min;
    double lnM_min;
    double sigma_min;
    double M_max_tables;
    double lnM_max_tb;
    double M_cond;
    double lnM_cond;
    double sigma_cond;
    double corr_star;
    double corr_sfr;
    double corr_xray;
} P21HaloSamplingParams;

#define P21_CUDA_MODE_STRICT 0
#define P21_CUDA_MODE_FAST 1

int p21_cuda_is_compiled(void);
int p21_cuda_is_available(void);
int p21_cuda_device_count(void);
void p21_cuda_set_devices(const int *device_ids, int n_devices);
int p21_cuda_get_num_devices(void);
int p21_cuda_get_devices(int *out_ids, int max_n);
void p21_cuda_set_enabled(int enabled);
int p21_cuda_get_enabled(void);
void p21_cuda_set_mode(int mode);
int p21_cuda_get_mode(void);
void p21_cuda_set_fast_ic(int enabled);
int p21_cuda_get_fast_ic(void);
void p21_cuda_set_fast_halo_sampler(int enabled);
int p21_cuda_get_fast_halo_sampler(void);
void p21_cuda_set_lazy_d2h(int enabled);
int p21_cuda_get_lazy_d2h(void);
int p21_cuda_registry_sync_to_host(const void *host_ptr);
void p21_cuda_registry_sync_all_lazy(void);
void p21_cuda_reset_stats(void);
void p21_cuda_get_stats(P21CudaStats *out);
void p21_cuda_cleanup(void);
void p21_cuda_invalidate_host_ptr(const void *ptr);
void p21_cuda_registry_clear(void);

int p21_cuda_compute_initial_conditions(unsigned long long int random_seed,
                                        InitialConditions *boxes);

int p21_cuda_compute_perturbed_field(float redshift, InitialConditions *boxes,
                                     PerturbedField *perturbed_field);

int p21_cuda_compute_brightness_temp(float redshift, TsBox *spin_temp, IonizedBox *ionized_box,
                                     PerturbedField *perturb_field, BrightnessTemp *box);

int p21_cuda_setup_radiation_mean_xe(TsBox *previous_spin_temp, double *x_e_ave_out);
int p21_cuda_setup_radiation_tables(TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup);
int p21_cuda_fill_freqint_tables(double zp, int num_R, const double *lower_int_limits,
                                 double hubble_zp, RadiationFieldsSetup *rad_setup);
int p21_cuda_update_radiation_fields(float redshift, EmissivityFields *emissivity_fields, int R_ct,
                                     double R_star, PerturbedField *perturbed_field,
                                     TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup,
                                     RadiationFields *radiation_fields);
int p21_cuda_compute_radiation_shells(float redshift, int n_shells, const int *R_ct_arr,
                                      double R_star, const double *interp_params,
                                      EmissivityFields **ef_desc_arr,
                                      EmissivityFields **ef_prog_arr,
                                      RadiationFieldsSetup *rad_setup,
                                      PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                                      RadiationFields *radiation_fields);

int p21_cuda_init_first_ts(TsBox *box, float *dens, float z, double xe_recfast, double tk_recfast,
                           double cT_ad);
int p21_cuda_compute_ts_box(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                            RadiationFields *radiation_fields, TsBox *previous_spin_temp,
                            InitialConditions *ini_boxes, TsBox *this_spin_temp);

P21CudaEmissivityCtx *p21_cuda_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, PerturbedField *perturbed_field,
    TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, const ScalingConstants *consts,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out);
int p21_cuda_emissivity_eval_and_finish(P21CudaEmissivityCtx *ctx, const ScalingConstants *ev_consts,
                                        const P21EmissivityAverages *uhmf_averages,
                                        EmissivityFields *emissivity_fields);
void p21_cuda_emissivity_abort(P21CudaEmissivityCtx *ctx);

P21CudaIonCtx *p21_cuda_ion_begin(const P21IonBoxParams *params, int n_radii,
                                  PerturbedField *perturbed_field,
                                  PerturbedField *previous_perturbed_field,
                                  IonizedBox *previous_ionize_box, TsBox *spin_temp,
                                  EmissivityFields *emissivity_fields,
                                  InitialConditions *ini_boxes);
int p21_cuda_ion_calc_mcrit(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                            double *log10_mturn_acg_avg, double *log10_mturn_mcg_avg);
int p21_cuda_ion_set_fully_neutral(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                   double xion_recfast_z, IonizedBox *box, double *global_xH_out);
int p21_cuda_ion_prepare_r2c(P21CudaIonCtx *ctx, const P21IonBoxParams *params);
int p21_cuda_ion_filter_and_extrema(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                    const P21IonRadiusSpec *rspec, double *min_dens,
                                    double *max_dens, double *prev_min_dens,
                                    double *prev_max_dens);
int p21_cuda_ion_calc_cond_nion(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                P21IonRadiusSpec *rspec, int need_prev_ion);
int p21_cuda_ion_find_regions(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                              const P21IonRadiusSpec *rspec, double nion_uncond_acg,
                              double nion_uncond_mcg, double nion_limit_acg, double nion_limit_mcg,
                              float prev_homo_nrec);
int p21_cuda_ion_set_ionized_temps(P21CudaIonCtx *ctx, const P21IonBoxParams *params);
int p21_cuda_ion_global_means(P21CudaIonCtx *ctx, double *global_xH_out, float *global_gamma12_out);
int p21_cuda_ion_recomb_inhomo(P21CudaIonCtx *ctx, const P21IonBoxParams *params);
int p21_cuda_ion_finish(P21CudaIonCtx *ctx, const P21IonBoxParams *params, IonizedBox *box);
void p21_cuda_ion_end(P21CudaIonCtx *ctx);
int p21_cuda_fill_mhr_table(int nz, float del_z, int ng, double lng_min, float del_lng,
                            double n_h0, double y_he, double alpha_b_1e4, const double *a_mhr,
                            const double *c_mhr, const double *beta_mhr, double *rr_table_out);

/* Phase 3: Lagrangian emissivity, PerturbedHaloCatalog, DEXM-ESF, dNdM tables & fast sampler */
P21CudaLagEmissCtx *p21_cuda_lag_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, TsBox *previous_spin_temp,
    IonizedBox *previous_ionize_box, const ScalingConstants *consts, int need_integral,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out);
int p21_cuda_lag_emissivity_add_halos(P21CudaLagEmissCtx *ctx, InitialConditions *ini_boxes,
                                      HaloCatalog *halos, const ScalingConstants *consts);
int p21_cuda_lag_emissivity_add_integral(P21CudaLagEmissCtx *ctx, InitialConditions *ini_boxes,
                                         const ScalingConstants *ev_consts,
                                         const P21EmissivityAverages *uhmf_averages);
int p21_cuda_lag_emissivity_finish(P21CudaLagEmissCtx *ctx, EmissivityFields *emissivity_fields);
void p21_cuda_lag_emissivity_abort(P21CudaLagEmissCtx *ctx);

int p21_cuda_compute_perturbed_halo_catalog(float redshift, InitialConditions *boxes,
                                            TsBox *prev_ts, IonizedBox *prev_ion,
                                            HaloCatalog *halos,
                                            PerturbedHaloCatalog *halos_perturbed);

P21CudaDexmCtx *p21_cuda_dexm_run(float redshift, InitialConditions *boxes, float M_MIN,
                                  float Delta_R, unsigned long long int *total_halo_num_out);
int p21_cuda_dexm_finish(P21CudaDexmCtx *ctx, HaloCatalog *halos_dexm, float *halo_overlap_box_out);
void p21_cuda_dexm_abort(P21CudaDexmCtx *ctx);

int p21_cuda_fill_dndm_tables(int nx, double xmin, double xmax, double ymin, double ymax,
                              double growth_out, double param, int from_catalog, double *nhalo_out,
                              double *fcoll_out);
int p21_cuda_fill_dndm_inverse_table(int nx, int np, double xmin, double xmax, double lnM_min,
                                     double growth_out, double param, int from_catalog,
                                     double min_lp, double **z_arr_out);

int p21_cuda_sample_halo_grids(unsigned long long int seed, double redshift, float *dens_field,
                               float *halo_overlap_box, HaloCatalog *halofield_large,
                               HaloCatalog *halofield_out, const P21HaloSamplingParams *hsp,
                               int *overflow_out, unsigned long long int *thread_counts_out);
int p21_cuda_sample_halo_progenitors(unsigned long long int seed, double z_in, double z_out,
                                     HaloCatalog *halofield_in, HaloCatalog *halofield_out,
                                     const P21HaloSamplingParams *hsp, int *overflow_out,
                                     unsigned long long int *thread_counts_out);

int p21_cuda_fill_cond_emissivity_tables(int kind, double redshift, double growthf, double lnM_min,
                                         double lnM_max, double lnM_cond, double sigma_cond,
                                         int n_delta, double min_dens, double max_dens,
                                         int n_mturn, int do_acg_2d, double acg_l10_min,
                                         double acg_l10_max, int do_mcg_2d, double mcg_l10_min,
                                         double mcg_l10_max, const ScalingConstants *sc,
                                         float *acg_1d_out, float **acg_2d_out,
                                         float **mcg_2d_out);

int p21_cuda_fill_uncond_emissivity_tables(
    int kind, int n_z, const double *z_vals, const double *growthf_vals, const double *lnMmin_vals,
    double lnMmax, const double *mturn_acg_hom_vals, const double *atomic_thresh_vals,
    const double *sfr_timescale_vals, int do_mcg, int n_mturn, double mcg_l10_min,
    double mcg_l10_max, const ScalingConstants *sc, double *acg_1d_out, double **mcg_2d_out);

/* Direction D: RSDs, rectilinear lightcone slicing, and 1D/2D power spectra */
int p21_cuda_include_dvdr_in_tau21(const float *h_tb, const float *h_los_vel,
                                   const float *h_tau_21, const double *h_H_s_inv, int nx, int ny,
                                   int nz, double dx_los_mpc, double max_dvdr, int use_ts_fluct,
                                   int periodic, float *h_out);
int p21_cuda_rsds_shift(const float *h_field, const float *h_disp, int n_los, int nz,
                        int n_rsd_subcells, int periodic, float *h_out);
int p21_cuda_rectilinear_slices(const float *h_box_a, const float *h_box_b, int nx, int ny, int nz,
                                int n_slices, const int *h_z_indices, const float *h_w_a,
                                const float *h_w_b, int kind, float *h_out);
int p21_cuda_power_spectrum_1d(const float *h_field_a, const float *h_field_b, int nx, int ny,
                               int nz, double lx, double ly, double lz, int n_bins,
                               const double *h_k_edges, int skip_dc, double *h_k_mean_out,
                               double *h_pk_out, double *h_delta_sq_out,
                               unsigned long long *h_counts_out);
int p21_cuda_power_spectrum_2d(const float *h_field_a, const float *h_field_b, int nx, int ny,
                               int nz, double lx, double ly, double lz, int los_axis,
                               int n_kperp, const double *h_kperp_edges, int n_kpara,
                               const double *h_kpara_edges, int skip_dc,
                               double *h_kperp_mean_out, double *h_kpara_mean_out,
                               double *h_pk_out, unsigned long long *h_counts_out);

/* Direction H: AngularLightconer GPU sampling */
int p21_cuda_angular_slice(const float *h_box_a, const float *h_box_b, int nx, int ny, int nz,
                           int n_coords, const double *h_coords_xyz, float w_a, float w_b,
                           int order, int kind, float *h_out);
int p21_cuda_angular_los_velocity(const float *h_vxa, const float *h_vya, const float *h_vza,
                                  const float *h_vxb, const float *h_vyb, const float *h_vzb,
                                  int nx, int ny, int nz, int n_coords,
                                  const double *h_coords_xyz, const double *h_unit_xyz, float w_a,
                                  float w_b, int order, float *h_out);

/* Direction I: DeviceLZF HDF5 batch chunk compression */
int p21_cuda_lzf_compress_batch_f32(const float *host_chunks, int n_chunks, size_t chunk_elems,
                                    int shuffle, unsigned char *host_out, size_t max_out_bytes,
                                    size_t *out_sizes, int *has_nonfinite_out);

/* Direction G: Multi-GPU slab-decomposed 3D FFTs */
int p21_cuda_mgpu_fft_r2c_host(const float *h_in, float *h_out_complex, int nx, int ny, int nz);
int p21_cuda_mgpu_fft_c2r_host(const float *h_in_complex, float *h_out_real, int nx, int ny,
                               int nz);

/* Direction O: GPU-native observables & non-Gaussian statistics */
int p21_cuda_wedge_filter_3d(const float *h_in, float *h_out, int nx, int ny, int nz, double lx,
                             double ly, double lz, int los_axis, double slope, double buffer_mpc);
int p21_cuda_add_thermal_noise_3d(const float *h_in, float *h_out, size_t n_elems, float sigma_mk,
                                  unsigned long long seed);
int p21_cuda_field_moments_and_pdf(const float *h_in, size_t n_elems, int n_bins, double vmin,
                                   double vmax, double *mean_out, double *var_out, double *skew_out,
                                   double *kurt_out, unsigned long long *hist_out);

/* Direction Q: Zero-copy CUDA DLPack / device pointer interop */
int p21_cuda_export_device_f32(const float *h_ptr, size_t n_elems, unsigned long long *d_ptr_out,
                               int *device_id_out);
void p21_cuda_free_exported_device_ptr(unsigned long long d_ptr);
int p21_cuda_device_copy_to_host_f32(unsigned long long d_ptr, float *h_out, size_t n_elems);

/* Direction S: GPU-native interferometer mock observation */
int p21_cuda_mock_observation_3d(const float *h_in, float *h_out, int nx, int ny, int nz,
                                 double lx, double ly, double lz, int los_axis,
                                 double sigma_perp_mpc, double channel_dr_mpc, int channel_mode,
                                 int apply_uv_mask, double k_perp_min, double k_perp_max,
                                 int apply_wedge, double wedge_slope, double wedge_buffer_mpc,
                                 int add_noise, int n_noise_table, double noise_kperp_max,
                                 const float *h_sigma_re_table, unsigned long long seed);
int p21_cuda_field_mean_f32(const float *h_in, size_t n_elems, double *mean_out);

/* Phase 10 (v0.9.0-dev): Higher-order non-Gaussian & morphological topology */
int p21_cuda_bispectrum_3d(const float *h_field, int nx, int ny, int nz, double lx, double ly,
                           double lz, int remove_mean, int n_triangles, const double *h_k1_lo,
                           const double *h_k1_hi, const double *h_k2_lo, const double *h_k2_hi,
                           const double *h_k3_lo, const double *h_k3_hi, double *h_b_out,
                           double *h_q_out, double *h_pk1_out, double *h_pk2_out, double *h_pk3_out,
                           double *h_ntri_out);
int p21_cuda_mfp_bsd(const float *h_field, int nx, int ny, int nz, double lx, double ly, double lz,
                     float threshold, int below_threshold, unsigned long long n_rays,
                     double step_cells, double max_dist, int n_bins, double r_min, double r_max,
                     int log_bins, unsigned long long seed, unsigned long long *h_hist_out,
                     double *mean_r_out, double *std_r_out, unsigned long long *n_ion_out);
int p21_cuda_minkowski_3d(const float *h_field, int nx, int ny, int nz, double lx, double ly,
                          double lz, int n_thresholds, const double *h_thresholds,
                          double *h_v0_out, double *h_v1_out, double *h_v2_out, double *h_v3_out,
                          long long *h_euler_out);
int p21_cuda_cic_deposit_3d(const float *h_pos_xyz, const float *h_weights,
                            unsigned long long n_particles, int nx, int ny, int nz, double lx,
                            double ly, double lz, int overdensity, float *h_grid_out);

/* Direction P: Distributed multi-GPU Y-slab k-space filtering */
int p21_cuda_mgpu_filter_box_host(const float *h_in, float *h_out, int nx, int ny, int nz,
                                  double box_len, double non_cubic_factor, int filter_type, float R,
                                  float R_param, float R_star);

#ifdef __cplusplus
}
#endif

#endif /* P21_CUDA_API_H */
