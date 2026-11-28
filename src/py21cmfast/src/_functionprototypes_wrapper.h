/* This file contains the repeated function prototypes which are needed by CFFI
    to be included explicitly via ffi.cdef(), These are the only functions which
    are visible to the python wrapper */

/* OutputStruct COMPUTE FUNCTIONS */
int ComputeInitialConditions(unsigned long long int random_seed, InitialConditions *boxes);

int ComputePerturbedField(float redshift, InitialConditions *boxes,
                          PerturbedField *perturbed_field);

int ComputeHaloCatalog(float redshift_desc, float redshift, InitialConditions *boxes,
                       unsigned long long int random_seed, HaloCatalog *halos_desc,
                       HaloCatalog *halos);

int ComputePerturbedHaloCatalog(float redshift, InitialConditions *boxes, TsBox *prev_ts,
                                IonizedBox *prev_ion, HaloCatalog *halos,
                                PerturbedHaloCatalog *halos_perturbed);

int ComputeTsBox(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                 RadiationFields *radiation_fields, TsBox *previous_spin_temp,
                 InitialConditions *ini_boxes, TsBox *this_spin_temp);

int ComputeIonizedBox(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                      PerturbedField *previous_perturbed_field, IonizedBox *previous_ionize_box,
                      TsBox *spin_temp, EmissivityFields *emissivity_fields,
                      InitialConditions *ini_boxes, IonizedBox *box);

int ComputeBrightnessTemp(float redshift, TsBox *spin_temp, IonizedBox *ionized_box,
                          PerturbedField *perturb_field, BrightnessTemp *box);

int ComputeEmissivityFields(double redshift, InitialConditions *ini_boxes,
                            PerturbedField *perturbed_field, HaloCatalog *halos,
                            TsBox *previous_spin_temp, IonizedBox *previous_ionize_box,
                            EmissivityFields *emissivity_fields);

int SetupRadiationFields(float redshift, TsBox *previous_spin_temp,
                         RadiationFieldsSetup *rad_setup);

int UpdateRadiationFields(float redshift, EmissivityFields *emissivity_fields, int R_ct,
                          double R_star, PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                          RadiationFieldsSetup *rad_setup, RadiationFields *radiation_fields);
/*--------------------------*/

/* PHOTON CONSERVATION MODEL FUNCTIONS */
int InitialisePhotonCons();

int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline);
int ComputeZstart_PhotonCons(double *zstart);

void adjust_redshifts_for_photoncons(double z_step_factor, float *redshift, float *stored_redshift,
                                     float *absolute_delta_z);

void determine_deltaz_for_photoncons();

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic,
                         double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
                         double *PhotonCons_NFdata, double *PhotonCons_deltaz,
                         int *Ndata_PhotonCons);

void FreePhotonConsMemory();
extern bool photon_cons_allocated;

void set_alphacons_params(double norm, double slope);
/* ------------------------------- */

/* Non-OutputStruct data products */
int ComputeLF(int nbins, int component, int NUM_OF_REDSHIFT_FOR_LF, double *z_LF,
              double *M_TURNs_ACG, double *M_TURNs_MCG, double *M_uv_z, double *M_h_z,
              double *log10phi);

float ComputeTau(int NPoints, float *redshifts, float *global_xHI, float z_re_HeII);
/*-----------------------------*/

/* Initialisation functions needed in the wrapper*/
void init_ps();
int init_heat();
void init_MHR();
int CreateFFTWWisdoms();
void clean_fftw();
int test_dft_cube(int dim, int n_threads, int n_repeat);
void Broadcast_struct_global_noastro(SimulationOptions *simulation_options,
                                     MatterOptions *matter_options, CosmoParams *cosmo_params);
void Broadcast_struct_global_all(SimulationOptions *simulation_options,
                                 MatterOptions *matter_options, CosmoParams *cosmo_params,
                                 AstroParams *astro_params, AstroOptions *astro_options,
                                 CosmoTables *cosmo_tables);
void initialize_sigma_tables(float M_Min, float M_Max);
void initialise_GL(double lnM_Min, double lnM_Max);
/*---------------------------*/

/* Free memory routines*/
void destruct_heat();
void free_sigma_tables();
void free_ps();
void free_MHR();
void Free_cosmo_tables_global();

/*---------------------------*/

/* Intergration routines */
void get_sigma(int n_masses, double *mass_values, double *sigma_out, double *dsigmasqdm_out);
void get_condition_integrals(double redshift, double z_prev, int n_conditions, double *cond_values,
                             double *out_n_exp, double *out_m_exp);
void get_halo_chmf_interval(double redshift, double z_prev, int n_conditions, double *cond_values,
                            int n_masslim, double *lnM_lo, double *lnM_hi, double *out_n);
void get_halomass_at_probability(double redshift, double z_prev, int n_conditions,
                                 double *cond_values, double *probabilities, double *out_mass);
void get_unconditional_sfrd(int n_redshift, double *redshifts, double *log10_turnovers_acg,
                            double *log10_turnovers_mcg, double *out_sfrd_acg,
                            double *out_sfrd_mcg);
void get_unconditional_nion(int n_redshift, double *redshifts, double *log10_turnovers_acg,
                            double *log10_turnovers_mcg, double *out_nion_acg,
                            double *out_nion_mcg);
void get_conditional_fcoll_eps(double redshift, double R, int n_densities, double *densities,
                               double *out_fcoll, double *out_dfcoll);
void get_conditional_sfrd(double redshift, double R, int n_densities, double *densities,
                          double log10_mturn_acg, double log10_mturn_mcg, double *out_sfrd_acg,
                          double *out_sfrd_mcg);
void get_conditional_nion(double redshift, double R, int n_densities, double *densities,
                          double log10_mturn_acg, double log10_mturn_mcg, double *out_nion_acg,
                          double *out_nion_mcg);
void get_conditional_xray_emissivity(double redshift, double R, int n_densities, double *densities,
                                     double log10_mturn_acg, double log10_mturn_mcg,
                                     double *out_xray);
/*--------------------------------*/

/* Error framework testing */
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double *result);
void FunctionThatThrows();
/*------------------------*/

/* Test Outputs For Specific Models */
int single_test_sample(unsigned long long int seed, int n_condition, float *conditions,
                       float *cond_crd, double z_out, double z_in, int *out_n_tot, int *out_n_cell,
                       double *out_n_exp, double *out_m_cell, double *out_m_exp,
                       float *out_halo_masses, float *out_halo_coords);
// test function for getting halo properties from the wrapper, can use a lot of memory for large
// catalogs
int test_halo_props(double redshift, float *vcb_grid, float *J21_LW_grid, float *z_re_grid,
                    float *Gamma12_ion_grid, int n_halos, float *halo_masses, float *halo_coords,
                    float *star_rng, float *sfr_rnd, float *xray_rng, float *halo_props_out);
int test_filter(float *input_box, double R, double R_param, double R_star, int filter_flag,
                double *result);

// test functions for multiple scattering
double compute_mu_for_multiple_scattering(double x_em);
double compute_eta_for_multiple_scattering(double x_em);
double hyper_2F3(double kR, double alpha, double beta);
double test_alpha_for_multiple_scattering(double x_em);
double test_beta_for_multiple_scattering(double x_em);
double test_alpha_for_multiple_scattering_straight_line_limit(void);
double test_beta_for_multiple_scattering_straight_line_limit(void);

/* Functions required to access cosmology & mass functions directly */
double dicke(double z);
double sigma_z0(double M);
double dsigmasqdm_z0(double M);
double power_in_k(double k);
double power_in_vcb(double k);
double get_delta_crit(int HMF, double sigma, double growthf);
double atomic_cooling_threshold(float z);
double molecular_cooling_threshold_with_feedbacks(float z, float J_21_LW, float vcb);
double reionization_feedback(float z, float Gamma_halo_HII, float z_IN);
double unconditional_hmf(double growthf, double lnM, double z, int HMF);
double conditional_hmf(double growthf, double lnM, double delta, double sigma, int HMF);
double expected_nhalo(double redshift);
void compute_mturns(double z, float J_21_LW, float vcb, float Gamma12, float z_reion,
                    float *M_turn_acg, float *M_turn_mcg);

/* CUDA backend runtime and diagnostics */
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
int p21_cuda_compute_radiation_shells(float redshift, int n_shells, const int *R_ct_arr,
                                      double R_star, const double *interp_params,
                                      EmissivityFields **ef_desc_arr,
                                      EmissivityFields **ef_prog_arr,
                                      RadiationFieldsSetup *rad_setup,
                                      PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                                      RadiationFields *radiation_fields);
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
int p21_cuda_angular_slice(const float *h_box_a, const float *h_box_b, int nx, int ny, int nz,
                           int n_coords, const double *h_coords_xyz, float w_a, float w_b,
                           int order, int kind, float *h_out);
int p21_cuda_angular_los_velocity(const float *h_vxa, const float *h_vya, const float *h_vza,
                                  const float *h_vxb, const float *h_vyb, const float *h_vzb,
                                  int nx, int ny, int nz, int n_coords,
                                  const double *h_coords_xyz, const double *h_unit_xyz, float w_a,
                                  float w_b, int order, float *h_out);
int p21_cuda_lzf_compress_batch_f32(const float *host_chunks, int n_chunks, size_t chunk_elems,
                                    int shuffle, unsigned char *host_out, size_t max_out_bytes,
                                    size_t *out_sizes, int *has_nonfinite_out);
int p21_cuda_mgpu_fft_r2c_host(const float *h_in, float *h_out_complex, int nx, int ny, int nz);
int p21_cuda_mgpu_fft_c2r_host(const float *h_in_complex, float *h_out_real, int nx, int ny,
                               int nz);
int p21_cuda_wedge_filter_3d(const float *h_in, float *h_out, int nx, int ny, int nz, double lx,
                             double ly, double lz, int los_axis, double slope, double buffer_mpc);
int p21_cuda_add_thermal_noise_3d(const float *h_in, float *h_out, size_t n_elems, float sigma_mk,
                                  unsigned long long seed);
int p21_cuda_field_moments_and_pdf(const float *h_in, size_t n_elems, int n_bins, double vmin,
                                   double vmax, double *mean_out, double *var_out, double *skew_out,
                                   double *kurt_out, unsigned long long *hist_out);
int p21_cuda_export_device_f32(const float *h_ptr, size_t n_elems, unsigned long long *d_ptr_out,
                               int *device_id_out);
void p21_cuda_free_exported_device_ptr(unsigned long long d_ptr);
int p21_cuda_device_copy_to_host_f32(unsigned long long d_ptr, float *h_out, size_t n_elems);
int p21_cuda_mock_observation_3d(const float *h_in, float *h_out, int nx, int ny, int nz,
                                 double lx, double ly, double lz, int los_axis,
                                 double sigma_perp_mpc, double channel_dr_mpc, int channel_mode,
                                 int apply_uv_mask, double k_perp_min, double k_perp_max,
                                 int apply_wedge, double wedge_slope, double wedge_buffer_mpc,
                                 int add_noise, int n_noise_table, double noise_kperp_max,
                                 const float *h_sigma_re_table, unsigned long long seed);
int p21_cuda_field_mean_f32(const float *h_in, size_t n_elems, double *mean_out);
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
int p21_cuda_mgpu_filter_box_host(const float *h_in, float *h_out, int nx, int ny, int nz,
                                  double box_len, double non_cubic_factor, int filter_type, float R,
                                  float R_param, float R_star);
void p21_dft_clear_plan_cache(void);
void p21_dft_get_cache_stats(unsigned long long *hits_out, unsigned long long *misses_out,
                             int *entries_out);
void p21_dft_clear_scratch_pool(void);
void p21_dft_get_scratch_stats(unsigned long long *hits_out, unsigned long long *misses_out,
                               int *pooled_out);
int p21_cpu_rsds_shift_f32(const float *f_in, const double *d_in, int n_los, int nz,
                           int n_rsd_subcells, int periodic, int disp_prescaled, float *f_out);
int p21_cpu_rsds_shift_f64(const double *f_in, const double *d_in, int n_los, int nz,
                           int n_rsd_subcells, int periodic, int disp_prescaled, double *f_out);
/*-----------------------*/
