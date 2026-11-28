# Changelog

All notable milestones in **21cmCU** are documented in this file.

## [Unreleased]
### Direction P — Distributed Multi-GPU EoR & Sub-160 GiB `2048³` 2LPT Streaming
- Shared k-space filter weights in [`cuda_filters.cuh`](src/py21cmfast/src/cuda/cuda_filters.cuh) across `cuda_cufft.cu` and `cuda_mgpu.cu`; added `p21_mgpu_r2c_from_device`, `p21_mgpu_filter_yslabs`, `p21_mgpu_c2r_to_device`, `P21_CUDA_MGPU_VIRTUAL_RANKS`, and `p21_cuda_mgpu_filter_box_host` (`83` CUDA APIs / `47` CFFI exports).
- Upgraded `cuda_ic.cu` host-accumulated `2LPT` (`P21_CUDA_MGPU_2LPT_HOST_STREAM=1`) to stream diagonals `a=0, 1, 2` strip-by-strip through a single auxiliary buffer, reuse `h_saved_input` in `single_slab_mode`, and work cleanly with `PERTURB_ON_HIGH_RES=False` (`<160 GiB` peak host RAM at `2048³`).
- Wired distributed Y-slab filtering into `cuda_ionization.cu` and `cuda_radiation.cu` behind `p21_mgpu_eor_enabled`, updated [`src/py21cmcu/doctor.py`](src/py21cmcu/doctor.py), and added virtual-rank parity coverage in [`tests/native/test_21cmcu_phase15_direction_p.py`](tests/native/test_21cmcu_phase15_direction_p.py).

### Architecture Modernization
- Split `cuda_analysis.cu` (`3,056` lines) into four domain translation units (`cuda_lightcone_ps.cu`, `cuda_observables.cu`, `cuda_topology.cu`, `cuda_multiprobe.cu`) backed by shared RAII + grid-stride header [`cuda_analysis_common.cuh`](src/py21cmfast/src/cuda/cuda_analysis_common.cuh).
- Upgraded [`build_cffi.py`](build_cffi.py) with concurrent `nvcc` compilation (`P21_NVCC_JOBS`) and `build/compile_commands.json` generation, and added [`scripts/check_cuda_abi.py`](scripts/check_cuda_abi.py) (`--stub-skeleton`) wired into CI for 1:1 `cuda_api.h` / `cuda_stubs.c` / `cuda/*.cu` parity.
- Consolidated cross-module Python helpers into [`src/py21cmcu/_internal.py`](src/py21cmcu/_internal.py) and upgraded spectral/topological/multi-probe outputs to `Mapping`-compatible frozen dataclasses in [`src/py21cmcu/_results.py`](src/py21cmcu/_results.py) (`res["k"]` and `res.k`).

## [0.9.0] — 2026-11-28 (Higher-Order Topology, CRN Fisher, Multi-Probe CIC, Online SBI & Modern Architecture)
### Added & Improved
- **Architecture Modernization, Packaging & Automated Release**:
  - Adopted PEP 621 dynamic single-source versioning (`src/py21cmcu/_version.py`), PEP 561 `py.typed` marker, modular runtime facade (`src/py21cmcu/_runtime.py` + `cuda_api_count()`), and PEP 562 lazy top-level imports in [`src/py21cmcu/__init__.py`](src/py21cmcu/__init__.py).
  - Hardened [`src/py21cmcu/cli.py`](src/py21cmcu/cli.py) and [`src/py21cmcu/doctor.py`](src/py21cmcu/doctor.py) with `21cmcu version [--as-json]`, `api_count` reporting, and non-editable wheel guardrails in `21cmcu bench`.
  - Added Python `3.12`/`3.13` CI matrix + `packaging-smoke` wheel job ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), tag-triggered wheel & release automation ([`.github/workflows/release.yml`](.github/workflows/release.yml)), and [`scripts/build_wheels.sh`](scripts/build_wheels.sh).
  - Updated [`docs/paper/21cmcu_paper.tex`](docs/paper/21cmcu_paper.tex), [`docs/paper/21cmcu_paper.md`](docs/paper/21cmcu_paper.md), [`benchmarks/results_t4_v090.json`](benchmarks/results_t4_v090.json), [`benchmarks/reproduce_v090_tables.py`](benchmarks/reproduce_v090_tables.py), and [`notebooks/21cmCU_GPU_Verify.ipynb`](notebooks/21cmCU_GPU_Verify.ipynb) to `v0.9.0`.
- **Phase 11 (`v0.9.0`) — CRN Fisher Forecaster & Multi-Probe Cross-Correlations (`86` CUDA APIs)**:
  - Verified Colab Tesla T4 `256³` 1M-particle benchmarks: `cic_deposit_3d` `1.4336 s -> 0.0365 s` (**`39.29×`**, `max_abs_diff = 3.81e-6`), `cross_power_21cm_galaxy` `4.8076 s -> 0.2324 s` (**`20.68×`**, `max_abs_rk_diff = 2.91e-8`), `phase13` **`7 passed, 0 skipped`** on T4 & `97 passed, 64 skipped` in CPU CI.
  - Added native 86th CUDA API `p21_cuda_cic_deposit_3d` (`src/py21cmfast/src/cuda/cuda_analysis.cu`, `cuda_api.h`, `_functionprototypes_wrapper.h`, `cuda_stubs.c`) with periodic 3D Cloud-in-Cell deposition, optional overdensity normalization, $\mathrm{sinc}^2$ window deconvolution, and grid-stride loops for `256³+` volumes above `65535 × 256` threads.
  - Added [`src/py21cmcu/multiprobe.py`](src/py21cmcu/multiprobe.py): `cic_deposit_3d`, `load_gravitycu_halos` (GravityCU HDF5 + NumPy catalog support), `cross_power_21cm_galaxy`, `cross_power_21cm_galaxy_error`, and `ksz_momentum_map`.
  - Added [`src/py21cmcu/fisher.py`](src/py21cmcu/fisher.py) and [`src/py21cmcu/viz.py`](src/py21cmcu/viz.py): `FisherForecaster`, `FisherForecastResult`, `fisher_forecast_21cm` (Common-Random-Number low-variance derivatives + marginalized $1\sigma$ forecasts), and `plot_fisher_contours`.
  - Added runnable [`examples/09_fisher_and_multiprobe_forecast.py`](examples/09_fisher_and_multiprobe_forecast.py) and [`tests/native/test_21cmcu_phase13_fisher_multiprobe.py`](tests/native/test_21cmcu_phase13_fisher_multiprobe.py), wired into exports and GitHub `CPU CI`.
- **Phase 10 & Direction R GPU Verification + Early Unit-Test / Warning Fixes**:
  - Verified Phase 10 (`bispectrum_3d` `6.21×` at `128³` / `26.64×` at `256³`, `bubble_size_distribution_mfp` `72.87×` at `128³` / `62.53×` at `256³`, `minkowski_functionals_3d` `8.28×`–`8.48×`) and Direction R (`BatchPool` `2.368×` overall / `1.27 s` warm round 4; zero-disk `OnlineSBIDataset` CUDA tensor stream) on Colab Tesla T4 (`10 passed`, `0 skipped`).
  - Fixed `OnlineSBIDataset` infinite Sobol stream refill (`sobol.random(1 << chunk_m)`), `observables.py` Sobol power-of-two sampling, `interop.py` `PyCapsule_*` ctypes isolation (zero `PytestUnraisableExceptionWarning`), and `cli.py` `| None` type mapping for Click.
- **Phase 10 (`v0.9.0-dev`) — Higher-Order Non-Gaussian & Morphological Topology**:
  - Added native CUDA APIs `#83–#85` (`p21_cuda_bispectrum_3d`, `p21_cuda_mfp_bsd`, `p21_cuda_minkowski_3d`) in [`src/py21cmfast/src/cuda/cuda_analysis.cu`](src/py21cmfast/src/cuda/cuda_analysis.cu), `cuda_api.h`, `_functionprototypes_wrapper.h`, and `cuda_stubs.c`.
  - Added [`src/py21cmcu/topology.py`](src/py21cmcu/topology.py) exporting `bispectrum_3d` (`equilateral`/`squeezed`/`folded`/`isosceles`/custom triangles with 3-slot shell caching), `bubble_size_distribution_mfp` (warp-compacted Philox ray-traced MFP BSD + vectorized CPU fallback), and `minkowski_functionals_3d` (exact cubical-complex $V_0, V_1, V_2, V_3$ & Euler $\chi$).
  - Added `SBISimulator(..., summary="nongaussian")` in [`src/py21cmcu/inference.py`](src/py21cmcu/inference.py), [`examples/08_nongaussian_topology_sbi.py`](examples/08_nongaussian_topology_sbi.py), and [`tests/native/test_21cmcu_phase12_topology.py`](tests/native/test_21cmcu_phase12_topology.py) (**`85 passed, 54 skipped (GPU)`** across 11 native suites on cluster `b1`; `128³` CPU timings `0.523 s` Bispectrum / `0.037 s` 50k-ray MFP BSD / `0.173 s` 15-threshold Minkowski).
- **Direction R (`v0.8.0-dev`) — Persistent Multi-GPU cuFFT Plan Cache, Warm `BatchPool` & Zero-Disk Online SBI Pipeline**:
  - Added a 64-entry concurrency-safe per-device cuFFT LRU cache across Y-slab 2D R2C/C2R and 1D C2C transforms in [`src/py21cmfast/src/cuda/cuda_mgpu.cu`](src/py21cmfast/src/cuda/cuda_mgpu.cu) (`p21_mgpu_fft_clear_cache()` wired into `p21_cuda_cleanup()`).
  - Added warm persistent [`BatchPool`](src/py21cmcu/batch.py) (`map`, `imap(ordered=...)`, context manager, periodic registry clearing) and extended `BatchTask` / `BatchResult` (`instrument`, `lazy_d2h`, `compute_moments`, `oblique_lightcone`, `worker_pid`, HDF5 moments persistence). Measured **`2.643×`** speedup across 4 repeated 4-task `128/64` `SKA1-Low` rounds (`20.439 s` cold / 16 PIDs → **`7.733 s`** warm / 4 PIDs; warm rounds `1.227–1.343 s` vs `5.332–5.647 s`).
  - Added [`OnlineSBIDataset`](src/py21cmcu/inference.py) (`torch.utils.data.IterableDataset`-compatible when PyTorch is installed, zero hard dependency) and upgraded `SBISimulator` with `instrument` + `batch_pool` batching and zero-disk Sobol/Uniform Tensor/NumPy streaming.
  - Added [`examples/07_online_sbi_zero_disk.py`](examples/07_online_sbi_zero_disk.py) and [`tests/native/test_21cmcu_phase11_batchpool_sbi.py`](tests/native/test_21cmcu_phase11_batchpool_sbi.py) (`81 passed, 53 skipped (GPU)` across 10 native suites; wired into [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- **Bit-Exact CPU Backend Optimizations (`c010ce1` + Phase 10 Step 2)**:
  - Added a 32-entry thread-safe aligned FFTW plan LRU cache (`P21_FFTW_PLAN_CACHE=1`) and a 24-slot `fftwf_malloc` scratch pool (`P21_CPU_SCRATCH_POOL=1`, `P21_CPU_SCRATCH_POOL_MB=2048`) in [`src/py21cmfast/src/dft.c`](src/py21cmfast/src/dft.c) / `dft.h` wired into `IonisationBox.c`, `RadiationFields.c`, and `PerturbedField.c`, both cleaned in `clean_fftw()`.
  - Added hoisted-switch multi-box fused `k`-space filtering (`p21_filter_boxes_fused` in [`src/py21cmfast/src/filtering.c`](src/py21cmfast/src/filtering.c), `IonisationBox.c`, and `RadiationFields.c` `multi_shell_filter`, toggled via `P21_CPU_FILTER_FUSED=1`) and opt-in `-march=native` (`P21_CPU_NATIVE=1` in [`build_cffi.py`](build_cffi.py)).
  - Verified full 16-core CPU coeval speedups with **strict `max_abs = 0.0`** across every output array: **`1.227×`** on `256/128` `E-INTEGRAL + TS + RECOMB` (`9.0989 s → 7.4172 s`, `plan_cache=(494,6,6)`, `scratch_pool=(172,4,4)`), **`1.205×`** on `256/128` `L-INTEGRAL + TS + MCGS + EXP` (`11.6838 s → 9.6985 s`, `plan_cache=(814,8,8)`, `scratch_pool=(259,8,8)`), and **`1.114×`** on `512/256` `E-INTEGRAL + TS + RECOMB` (`63.2424 s → 56.7649 s`).
  - Added native C/OpenMP column-parallel `apply_rsds` / `rsds_shift` kernels (`p21_cpu_rsds_shift_f32` / `p21_cpu_rsds_shift_f64`) in [`src/py21cmfast/src/rsds_cpu.c`](src/py21cmfast/src/rsds_cpu.c) and [`src/py21cmfast/rsds.py`](src/py21cmfast/rsds.py) (`P21_CPU_RSDS_NATIVE=1` default) under strict non-FMA/non-reassociate FP guards (`#pragma STDC FP_CONTRACT OFF` and `no-fast-math`), accelerating `128 × 128 × 512` 16-core CPU `apply_rsds` from **`38.2230 s → 0.0967 s`** (**`395.26×`**) with **`max_abs = 0.0`**.
  - Added [`tests/native/test_21cmcu_phase10_cpu_opt.py`](tests/native/test_21cmcu_phase10_cpu_opt.py) (**`69 passed, 23 skipped`** across CPU native suites `phase0/3_d/9/10` on cluster; wired into [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- **Anti-Repetition Oblique Lightcones (`9b86822`)**:
  - Added `ObliqueRectilinearLightconer` and `quick_lightcone(..., oblique=True)` with GPU/CPU parity, using periodic oblique line-of-sight traversal (`direction=(3, 5, 11)`) to extend non-repeating path length by `12.45×`.
- **Upstream Comparison & 21cmFAST-Aligned Showcase (`3cf3700`, `b58d8cd`)**:
  - Added [`docs/UPSTREAM_AND_OPTIMIZATION.md`](docs/UPSTREAM_AND_OPTIMIZATION.md), corresponding Wiki documentation, and 21cmFAST `EoR`/`viridis`-aligned 2D/3D showcase figures in [`README.md`](README.md).

## [0.7.0] — 2026-11-27 (Directions O, Q, S & Phase 9)
### Added
- **Direction O (`8300144`) — GPU-Native Observables & Non-Gaussian Statistics (`cuda_analysis.cu`, `src/py21cmcu/observables.py`, `src/py21cmcu/analysis.py`, `src/py21cmcu/batch.py`)**:
  - Added GPU-native `p21_cuda_wedge_filter_3d` (cuFFT 3D R2C/C2R + `DeviceFieldRegistry` across `los_axis` ∈ {0, 1, 2}, **`23.86×`** speedup at `256³`) and `p21_cuda_add_thermal_noise_3d` (deterministic Philox Gaussian thermal noise, **`10.71×`** at `256³`).
  - Added `p21_cuda_field_moments_and_pdf` + `field_moments_and_pdf` (single-pass variance, skewness, excess kurtosis, and 1-point histogram PDF, **`26.43×`** at `256³`, `hist_diff=0`), plus 2D cylindrical lightcone chunk power spectra (`lightcone_chunk_power_spectra_2d`).
- **Direction Q (`8300144`) — Zero-Copy CUDA Interop & SBI/MCMC Inference (`cuda_mempool.cu`, `src/py21cmcu/interop.py`, `src/py21cmcu/inference.py`)**:
  - Added `CudaFieldView`, `to_cuda_view`, `lightcone_to_cuda_views`, and `to_torch` exposing pure-`ctypes` `__dlpack__`, `__dlpack_device__`, and `__cuda_array_interface__` (`0.527 ms`, `registry_h2d_bytes=0` under `lazy_d2h=True`; live zero-copy device pointer sharing verified on PyTorch `2.5.1+cu121` on RTX 4090).
  - Added `Likelihood21cm` and `SBISimulator`, verified with [`tests/native/test_21cmcu_phase7_oq.py`](tests/native/test_21cmcu_phase7_oq.py) (**`7/7` passing**) and [`examples/04_dlpack_and_sbi_inference.py`](examples/04_dlpack_and_sbi_inference.py).
- **Direction S (`4720a37`) — Radio Interferometer Mock Observation & Sensitivity Forecast Suite (`cuda_analysis.cu`, `src/py21cmcu/instrument.py`)**:
  - Added single-pass GPU `p21_cuda_mock_observation_3d` & `mock_observation_3d` combining UV coverage cuts, PSF Gaussian smoothing, LOS boxcar/Gaussian channel response, horizon/FoV foreground wedge exclusion, and $P_N(k_\perp, z)$-weighted Hermitian Philox thermal noise (**`71.65×`** on `SKA1-Low` `23.04 ms` vs `1650.48 ms` CPU and **`71.88×`** on `HERA-350` `22.48 ms` vs `1615.87 ms` CPU at `256³`).
  - Added `InterferometerConfig` & `INSTRUMENT_PRESETS` (`SKA1-Low`, `HERA-350`, `MWA-II`, `LOFAR`), analytic & explicit `antenna_xy_m` baseline density (`baseline_density_uv`), exact $\Omega_p \Omega_{\mathrm{eff}}$ thermal noise `thermal_noise_power_spectrum`, and 1D/2D SNR forecasters (`forecast_power_spectrum_error`, `forecast_power_spectrum_2d_error`), verified with [`tests/native/test_21cmcu_phase8_instrument.py`](tests/native/test_21cmcu_phase8_instrument.py) (**`7/7` passing**) and [`examples/05_interferometer_mock_and_forecast.py`](examples/05_interferometer_mock_and_forecast.py).
- **Phase 9 (`e35f58c` + Wiki `d825676`) — Lightcone Pipeline Optimization, One-Line Shortcuts & Interactive 3D WebGL Viewer**:
  - Added 82nd native CUDA API `p21_cuda_field_mean_f32` and zero-D2H `_get_coeval_array_no_sync` (`256/128` 12-node `L-INTEGRAL` lightcone **`1.496 s → 1.169 s`**, **`-21.9%`** wall time, **`-8.5%`** D2H bytes), plus $O(1)$-memory on-demand `stream_perturbed_fields=True`.
  - Added ergonomic `quick_coeval` and `quick_lightcone` (with causal-free `parallel_redshifts=True`, integrated telescope mocks, and HTML export) in [`src/py21cmcu/shortcuts.py`](src/py21cmcu/shortcuts.py).
  - Added offline WebGL2 3D raymarching HTML viewer & matplotlib 2D plotting helpers in [`src/py21cmcu/viz.py`](src/py21cmcu/viz.py) (`export_volume_3d_html`, `export_lightcone_3d_html`, `plot_lightcone_summary`, `plot_coeval_slices`, `plot_mock_comparison`), verified with [`tests/native/test_21cmcu_phase9_lightcone_viz.py`](tests/native/test_21cmcu_phase9_lightcone_viz.py) (**`7/7` passing**), [`examples/06_quick_lightcone_and_3d_viz.py`](examples/06_quick_lightcone_and_3d_viz.py), single-GPU fix in `test_mgpu_device_config_and_api`, and full GitHub Wiki documentation.

## [0.6.0] — 2026-11-27 (Directions K, L, M & N)
### Added
- **Direction K (`6e91af3`) — Multi-GPU Communication & Low-VRAM Streaming (`cuda_mgpu.cu`, `cuda_ic.cu`)**:
  - 64 MiB portable pinned bounce buffers (`MgpuHostBounce`), `std::thread` sub-tiled `X↔Y` host-staged transposes, cuFFT plan creation mutex safety, and parallel multi-GPU strip transfers (`1536³` Zel'dovich **`27.799 s`** / 2LPT **`75.231 s`** on `2× RTX 4090`, up to **`2.80×`** faster than `v0.5.0`; `1792³` 2LPT **`103.892 s`** on `3× RTX 4090`, **`3.52×`** faster).
  - Zero-`h_tmp_ab` 2LPT pinned streaming (saving `32 GiB` host RAM at `2048³`) and single-work-slab mode (`P21_CUDA_MGPU_SINGLE_SLAB`), unlocking **native in-VRAM `2048³` (`8.590 B` voxels) Zel'dovich IC on `3× RTX 4090` in `58.467 s`** (**`3.71×`** faster than `v0.5.0` host fallback).
  - Expanded `tests/native/test_21cmcu_phase5_g.py` (`14/14` passing with `test_mgpu_ic_single_slab_mode_parity`).
- **Direction L (`4c89439`) — SBI Observables & Resumable Pipeline**:
  - Added `src/py21cmcu/observables.py` (`horizon_slope`, `horizon_wedge_mask`, `apply_wedge_filter_3d`, `add_thermal_noise_3d`, `generate_sobol_parameter_grid`).
  - Extended `BatchTask` & `run_batch` (`wedge_filter`, `wedge_buffer`, `noise_sigma_mk`, `lightcone_out_path`, `lightcone_compression`, `checkpoint_h5`, `resume=True`) and added [`examples/sbi_lightcone_dataset.py`](examples/sbi_lightcone_dataset.py).
- **Direction M (`a8eff48`) — Diagnostics, CLI, User Guide & CI**:
  - Added `py21cmcu.doctor()` (`src/py21cmcu/doctor.py`) and extended `21cmcu` CLI (`src/py21cmcu/cli.py`) with `batch`, `doctor`, and `bench` subcommands (`tests/native/test_21cmcu_phase6_lm.py`, `4/4` passing).
  - Added [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) and expanded `.github/workflows/ci.yml` to lint `examples/` and execute `tests/native/test_21cmcu_phase6_lm.py`.
- **Direction N — `v0.6.0` Release Documentation, Figures & Methods Paper**:
  - Merged `benchmarks/results_rtx4090_v060.json`, updated `benchmarks/plot_benchmarks.py` & `docs/figures/*.png`, added standalone methods paper in [`docs/paper/`](docs/paper/21cmcu_paper.md), bumped version to `0.6.0`, and verified **`168/168` passing tests** (`-k "not speed"` across all 12 native GPU test suites on `3× RTX 4090`).

## [0.5.0] — 2026-11-27 (Direction G + Full Benchmarks & Figures)
### Added
- **Direction G (`9ec08a4`)**: Native multi-GPU 1D Y-slab domain decomposition (`cuda_mgpu.cu`/`.cuh`) for 3D R2C/C2R FFTs, `compute_initial_conditions` (`ZELDOVICH` + hybrid host-accumulated `2LPT`), and `power_spectrum_1d` / `power_spectrum_2d` with `CUDA_DEVICES` / `P21_CUDA_DEVICES` configuration, `cuda_device_count()`, `get_cuda_devices()`, `set_cuda_devices()`, `use_mode(..., devices=...)`, and `mgpu_fft_runs` / `mgpu_ic_runs` stats (`12/12` tests passing in `tests/native/test_21cmcu_phase5_g.py` across `2×` and `3× RTX 4090`).
- **Full Benchmark Sweep & Visualization (`docs/BENCHMARKS.md`, `benchmarks/plot_benchmarks.py`, `docs/figures/*.png`)**: measured on node `b1` (16 CPU threads vs `1×–3× RTX 4090`) into `benchmarks/results_rtx4090_v050.json`, demonstrating `492.74×` on `apply_rsds`, `82.05×` on `RectilinearLightconer`, `20.65×` on `AngularLightconer`, `12.36×` on 3z `L-INTEGRAL+TS`, `1.83×` HDF5 write speedup with `DeviceLZF`, `-21.7%` D2H with `CUDA_LAZY_D2H`, `1536³` (`46.131 s` Zel'dovich) & `1792³` (`98.095 s` Zel'dovich) native multi-GPU IC on `3× RTX 4090`, and `10,502.0 tasks/h` multi-GPU `run_batch`.
- **Full `v0.5.0` native GPU suite**: **`162/162`** passing tests (`-k "not speed"` across all 11 native test suites on `3× RTX 4090`).

## [0.4.0] — 2026-11-26 (Directions H, I & J)
### Added
- **Direction H (`ec27d60`)**: In-memory + atomic `.npz` disk caching for `CLASS` `CosmoTables` (`17.1857 s` cold → `0.002208 s` memory **`7,783×`** / `0.0028 s` disk **`6,214×`**, `py21cmcu.clear_class_cache()`), `AngularLightconer` `run_classy` caching, and native CUDA `p21_cuda_angular_slice` / `p21_cuda_angular_los_velocity` (`0.0934 s` GPU vs `0.1644 s` CPU on `64 × 128³` `order=1` slices — **`1.76×`**; `6/6` tests passing in `tests/native/test_21cmcu_phase4_h.py`).
- **Direction I (`0360f70`)**: Native CUDA `DeviceLZF` HDF5 compression (`cuda_io.cu`, `py21cmcu.save_lightcone`, **`2.91×`** faster than CPU LZF at `0.2366 s` vs `0.6893 s` on `129.11 MB` `256/128` lightcones) and opt-in `CUDA_LAZY_D2H` GPU-only intermediate residency (`use_mode(..., lazy_d2h=True)`, PEP 688 `_CudaBufferOwner` GC tracking, cutting `256/128` `L-INTEGRAL+TS` lightcone D2H transfers by **`24.3%`** (`1723.60 → 1304.17 MB`) and runtime `1.912 s → 1.322 s` (**`1.45×`**); `9/9` tests passing in `tests/native/test_21cmcu_phase4_i.py`).
- **Direction J**: Removed legacy v3 (`lightcone/`, `gpu/`), CuPy v4 adapter (`v4/`, `reports/`), and obsolete fixtures from `main` (preserved under git tag `archive-cupy-v4`), added GitHub Actions CPU CI (`.github/workflows/ci.yml`), bumped version to `0.4.0`, and verified **`150/150` passing tests** on RTX 4090 (`sm_89`).

## [0.3.0] — 2026-11-26 (Directions D, E & F)
### Added
- **Direction D (`fa22efc`)**: Native CUDA `include_dvdr_in_tau21`, `rsds_shift` / `apply_rsds` (`476.90×` speedup at `128×128×512`), `RectilinearLightconer` batch GPU slicing, `BrightnessTemp`/`tau_21` `DeviceFieldRegistry` retention, and cuFFT-backed `power_spectrum_1d` / `power_spectrum_2d` / `lightcone_chunk_power_spectra` (`19/19` tests passing in `tests/native/test_21cmcu_phase3_d.py`).
- **Direction E (`b8e7950`)**: Spawned multi-GPU / concurrent batch runner (`py21cmcu.run_batch`, `save_batch_h5`, `4,797.8 tasks/h` at `192/96` on RTX 4090 with `workers_per_gpu=2`) and 6-seed `strict` vs `fast` calibration across `E-INTEGRAL`, `L-INTEGRAL`, and `CHMF-SAMPLER+MCG+CLASS` (`docs/CALIBRATION.md`, `tests/native/test_21cmcu_phase3_e.py`).
- **Direction F**: `build_cffi.py` toolchain & `CUDA_ARCHS=auto` SM auto-detection (verified compiling `sm_89` on RTX 4090 without `CUDA_HOME`), unified `benchmarks/bench_21cmcu.py` CLI, and `v0.3.0` release verification (`135/135` passing tests across all native suites).

## [0.2.0] — 2026-11-26 (Phase 1 / C & Phase 2 / B)
### Added
- **Phase 1 / Direction C (`924f940`)**:
  - Parallel Philox 4×32-10 initial-conditions RNG (`cuda_philox.cuh`) and parallel 16 KB GPU chunk-tree hashing (`cuda_mempool.cu`).
  - Full GPU acceleration and table caching for `CHMF-SAMPLER`, `DEXM-ESF`, `L-INTEGRAL`, `USE_MCGS`, `V_CB_MODEL="FLUCTS"` (`CLASS`), `LYA_MULTIPLE_SCATTERING`, and conditional/unconditional emissivity tables.
  - Expanded test coverage in `tests/native/test_21cmcu_phase1.py`.
- **Phase 2 / Direction B (`f276a7a`)**:
  - 64-bit grid-index arithmetic across all CUDA kernels supporting `512³–1024³` boxes and beyond.
  - Configurable device registry budget (`P21_CUDA_REGISTRY_MB`) and stale-LRU reclamation on mempool OOM (`reclaim_stale_registry_for_oom`).
  - Low-VRAM 2LPT potential scheduling (`10.331 s` vs `85.784 s` CPU for `512/1024` lo-res 2LPT on RTX 4090 — **8.30×** speedup), `27.630 s` `512/1024` full coeval, and `29.673 s` `768³` EoR box within 24 GB VRAM.
  - Adaptive float32/float64 perturbation accumulation preserving deterministic CPU/GPU parity on smaller grids.
  - Regression & high-res test suite `tests/native/test_21cmcu_phase2.py` (`108/108` passing tests total on RTX 4090).

## [0.1.0] — 2026-11-26 (Phase 0)
### Added
- **`f3285f4`**: Created standalone **21cmCU** engine (`py21cmcu` / `21cmcu`) on top of native C/CUDA + cuFFT inside `c_21cmfast` (zero CuPy runtime dependency).
- Exposed dual execution modes `strict` (CPU-matched parity) and `fast` (statistically equivalent throughput mode).
- Fixed engine stability bugs in `photoncons.py`, `hmf.c` (`st_taylor_factor` null guard), and `interp_tables.c` / `cuda_ionization.cu` (`Nion_conditional_table1D_acg_prev`).
- Archived legacy CuPy v4 adapter and v3 CUDA paths at tag `archive-cupy-v4`.
