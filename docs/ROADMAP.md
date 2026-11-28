# 21cmCU Roadmap & Milestone Record

**21cmCU** is a standalone native C/CUDA + cuFFT 21cm cosmological simulation engine derived from the `21cmFAST-GPU` fork (21cmFAST v4 semantics) with full `py21cmcu` / `py21cmfast` API and CLI compatibility and zero runtime CuPy dependency.

## Core Architecture

- **Native C/CUDA + cuFFT inside `c_21cmfast`**: All GPU kernels (`cuda_*.cu`) compile directly into the CFFI shared library, eliminating Python/CuPy launch overhead and external CuPy requirements.
- **Dual precision modes (`CUDA_MODE`)**:
  - `strict`: CPU-matched parity mode using matched FFT conventions, deterministic RNG paths, and strict IEEE floating-point order.
  - `fast`: Statistically equivalent high-speed mode (parallel Philox 4×32-10 IC RNG, float32 accumulation where bounded, fast halo sampler / FFT paths) preserving unbiased global histories and power spectra.
- **End-to-end GPU residency (`DeviceFieldRegistry` + `cudaMemPool`)**: Host field pointers are tracked with parallel 16 KB chunk-tree cryptographic fingerprints (`F_hash`) so unchanged coeval/perturbation/emissivity fields remain GPU-resident across redshifts and radii with minimal or zero H2D transfers.
- **Cached cuFFT Plan LRU**: Up to 16 padded/unpadded R2C/C2R plans cached with external mempool work areas to handle `256³`–`1024³` transforms inside a single-GPU VRAM budget (e.g., 24 GB RTX 4090).

---

## Completed Milestones

### Phase 0 (`f3285f4`) — Engine Unification & Upstream Bug Fixes
- Branded standalone package as **`21cmCU`** (`py21cmcu` / `21cmcu` CLI) alongside full `py21cmfast` namespace compatibility.
- Preserved legacy CuPy v4 adapter and v3 CUDA sources under tag `archive-cupy-v4`.
- Engine stability fixes:
  - Photon-conservation calibration redshift/neutral-fraction length alignment (`photoncons.py`).
  - Safe null-output guard in Sheth–Tormen `st_taylor_factor` for `SAMPLE_METHOD="PARTITION"` (`hmf.c`).
  - Separate previous-redshift 1D ACG ionizing-efficiency table (`Nion_conditional_table1D_acg_prev`) for `E-INTEGRAL` + `USE_MCGS=True` (`interp_tables.c` / `cuda_ionization.cu`).

### Phase 1 / Direction C (`924f940`) — End-to-End GPU Halo, MCG & CLASS Residency
- Parallel **Philox** IC RNG and **GPU chunk-tree hashing** (`cuda_philox.cuh`, `cuda_mempool.cu`) eliminating host RNG and single-thread hashing bottlenecks at large `DIM`.
- Full GPU acceleration for `CHMF-SAMPLER`, `DEXM-ESF`, `L-INTEGRAL`, `USE_MCGS` molecular cooling, `V_CB_MODEL="FLUCTS"` (`CLASS`), `LYA_MULTIPLE_SCATTERING`, and conditional/unconditional emissivity integration tables (`cuda_tables_build.cu`, `cuda_sampler.cu`, `cuda_sources.cu`, `cuda_halos.cu`, `cuda_dexm.cu`).

### Phase 2 / Direction B (`f276a7a`) — Extreme-Scale (`512³–1024³`) Single-GPU Support
- **64-bit indexing audit**: Fixed 32-bit `int` element offsets across IC/2LPT, perturbation, spin temperature, radiation, and ionization kernels for `DIM >= 1290` (`N >= 2^31`).
- **Dynamic VRAM & registry budget (`P21_CUDA_REGISTRY_MB`)**: Capped registry footprint and added automatic stale-LRU eviction under mempool OOM pressure (`reclaim_stale_registry_for_oom`) so `768³` EoR and `512/1024` 2LPT coevals fit cleanly in 24 GB VRAM.
- **Large-box cuFFT / perturbation residency**: Reordered free-memory-aware 2LPT potential reuse, adaptive float/double perturbation accumulation (keeping deterministic double accumulation for `box_bytes < 2 GiB`), and resident `L-INTEGRAL` emissivity/ionization paths.

#### Verified RTX 4090 Benchmarks (`Phase 2 / f276a7a`, `N_THREADS=8`, `--mem=48G`)

| Benchmark | Config (`HII_DIM / DIM`, Box, Regime, Mode) | Wall Time | vs CPU (8T) | H2D Transfer | Peak / Post VRAM |
|---|---|---:|---:|---:|---|
| `B1` | `256 / 512`, `L-INTEGRAL + TS`, 2z (`fast`) | `5.070 s` | — | `0.00 GB` | `4.71 / 23.55 GB` |
| `B2` | `512 / 512`, `L-INTEGRAL + TS`, 2z (`fast`) | `32.087 s` | — | `10.50 GB` | `17.52 / 23.55 GB` |
| `B3` | `512 / 1024` `IC + PF` (2LPT `lo`, `fast`) | `10.331 s` | **`8.30×`** (`85.784 s`) | `1.50 GB` | `6.71 / 23.55 GB` |
| `B3` | `512 / 1024` `IC + PF` (2LPT `lo`, `strict`) | `11.574 s` | `7.41×` | `1.50 GB` | `6.74 / 23.55 GB` |
| `B4` | `512 / 1024` `IC + PF` (2LPT `hi`, `fast`) | `14.257 s` | **`6.47×`** (`92.196 s`) | `0.00 GB` | `18.52 / 23.55 GB` |
| `B4` | `512 / 1024` `IC + PF` (2LPT `hi`, `strict`) | `15.657 s` | `5.89×` | `0.00 GB` | `18.46 / 23.55 GB` |
| `B5` | `512 / 1024` Full Coeval (2LPT, `fast`, `xH=0.5671`) | `27.630 s` | — | `2.50 GB` | `15.58 / 23.55 GB` |
| `B6` | `768³` EoR (`E-INTEGRAL`, `fast`, `xH=0.2518`) | `29.673 s` | — | `13.50 GB` | `18.99 / 23.55 GB` |

*Test suite status at `f276a7a`: `108/108` passing across `test_cuda_phase1.py` (43), `test_cuda_phase2.py -k "not speed"` (27), `test_cuda_phase3.py` (19), `test_21cmcu_phase1.py` (7), and `test_21cmcu_phase2.py` (12).*

---

## Completed Directions (D → E → F, `v0.3.0`)

### Direction D (`fa22efc`) — Lightcone Pipeline, GPU RSDs & GPU Power Spectra
- **Native CUDA `include_dvdr_in_tau21` & `rsds_shift` (`cuda_analysis.cu`)**: Replaced CPU `scipy.fft`, `RegularGridInterpolator`, and `cosmotile.cic.cloud_in_cell_los` with fused column-parallel GPU kernels (`double` interpolation/CIC accumulation, zero `4×` host temporary).
  - Verified on RTX 4090 (`128 × 128 × 512`, `USE_TS_FLUCT=True`): `include_dvdr_in_tau21` **`0.0231 s`** vs `0.3223 s` CPU (**`13.94×`**); `apply_rsds` **`0.0794 s`** vs `37.8637 s` CPU (**`476.90×`**).
- **Native CUDA `RectilinearLightconer` Batch Slicing & `BrightnessTemp` Registry Retention (`cuda_brightness.cu`, `lightconers.py`)**: Batch slice interpolation in a single kernel directly from GPU-resident coeval arrays. End-to-end `256/128` `L-INTEGRAL` lightcone (`z=6.5–11`, `apply_rsds=True`, 777 slices) completes in **`4.066 s`** (`1,212` registry hits, `0.56 GB` H2D).
- **Native CUDA 1D & 2D Power Spectra (`power_spectrum_1d`, `power_spectrum_2d`, `lightcone_chunk_power_spectra`)**: cuFFT-backed spherical `P(k)` / `Δ²(k)` and cylindrical `P(k_⊥, k_∥)` matching `powerbox.get_power` (`19.68×` speedup at `128³`: `0.0067 s` vs `0.1311 s` CPU; `6` lightcone chunk PS in `0.1823 s`).
- Verified by `tests/native/test_21cmcu_phase3_d.py` (`19/19 passed`).

### Direction E (`b8e7950`) — Multi-GPU Batch / SBI Runner & `strict` vs `fast` Calibration
- **Spawned Multi-GPU / Multi-Worker Batch Runner (`py21cmcu.run_batch`, `save_batch_h5`)**: Concurrent in-memory sweeps with `CUDA_VISIBLE_DEVICES` isolation, GPU 1D/2D PS extraction, error isolation, and compact summary HDF5 output (`tests/native/test_21cmcu_phase3_e.py` `2/2 passed`).
  - RTX 4090 single-GPU throughput (`4 × 192/96` coevals, `fast`): `3.816 s` (`3,773.5 tasks/h`) with 1 worker; **`3.001 s` (`4,797.8 tasks/h`)** with `workers_per_gpu=2`.
- **Statistical Calibration (`docs/CALIBRATION.md`)**: 6-seed ensembles across `E-INTEGRAL`, `L-INTEGRAL`, and `CHMF-SAMPLER + USE_MCGS + CLASS` establish global agreement (`max |ΔxH| ≤ 0.00147`, `max |ΔTb| ≤ 0.3719 mK`, `max |ΔTs|/Ts ≤ 1.07%`) and unbiased 1D/2D power spectra (`median |Δ|/SEM = 0.68–1.11`).

### Direction F (`b04f812`, `v0.3.0`) — Build Auto-Detection & Release Polish
- **`build_cffi.py` Auto-Detection**: Version-aware CUDA toolkit discovery (preferring highest `/usr/local/cuda*`), `CUDA_ARCHS=auto` / `native` (`nvidia-smi` SM cap detection, verified compiling `sm_89` on RTX 4090), and low-priority `conda`/`venv`/`pkg-config` GSL & FFTW discovery.
- **Reproducible Benchmark CLI ([`benchmarks/bench_21cmcu.py`](../benchmarks/bench_21cmcu.py))**: Single-command `--suite core|highres|lightcone|batch|all` harness.
- **Full `v0.3.0` native test suite**: **`135/135` passing tests** on RTX 4090 (`phase0`, `phase1`, `phase2`, `phase3_d`, `phase3_e`, `cuda_phase1`, `cuda_phase2 -k "not speed"`, `cuda_phase3`).

---

## Completed Directions (H → I → J, `v0.4.0`)

### Direction H (`ec27d60`) — CLASS Startup Caching & GPU `AngularLightconer`
- **CLASS Transfer-Table & Lightconer Caching (`inputs.py`, `lightconers.py`, `py21cmcu.clear_class_cache`)**: Thread-safe bounded LRU memory cache + atomic `.npz` disk cache under `config["direc"]/class_tables_cache` (`P21_CLASS_CACHE`, `P21_CLASS_DISK_CACHE`) and `AngularLightconer` `run_classy` cache.
  - RTX 4090 (`192/64` `CHMF-SAMPLER + USE_MCGS + CLASS`): `CosmoTables` cold `17.1857 s` → **`0.002208 s`** memory (**`7,783×`**) / **`0.0028 s`** disk (**`6,214×`**); E2E `fast` coeval completes in **`7.713 s`** (`mem`) / **`7.795 s`** (`disk`).
- **Native CUDA `AngularLightconer` Kernels (`cuda_analysis.cu`)**: `p21_cuda_angular_slice` and `p21_cuda_angular_los_velocity` (`64 × 128³` `order=1` slice interpolation **`0.0934 s`** GPU vs `0.1644 s` CPU, **`1.76×`**).
- Verified by `tests/native/test_21cmcu_phase4_h.py` (`6/6 passed`).

### Direction I (`0360f70`) — Native CUDA `DeviceLZF` HDF5 & `CUDA_LAZY_D2H` Residency
- **Native CUDA `DeviceLZF` HDF5 (`cuda_io.cu`, `src/py21cmcu/io.py`)**: `py21cmcu.save_lightcone(..., compression="device-lzf")` compresses 3D `float32` lightcone chunks on GPU with HDF5 byte-shuffle + `liblzf`, non-finite flagging, incompressible fallback (`filter_mask`), and clean CPU fallback.
  - RTX 4090 `256/128` 3-field `LightCone` (`129.11 MB` raw): **`0.2366 s`** GPU vs `0.6893 s` CPU LZF (**`2.91×`**).
- **`CUDA_LAZY_D2H` Intermediate GPU Residency (`cuda_mempool.cu`, `cuda_radiation.cu`, `cuda_sources.cu`, `_cfg.py`, `arrays.py`, `outputs.py`)**: Opt-in GPU-only residency (`P21_CUDA_LAZY_D2H=1` or `use_mode(..., lazy_d2h=True)`) for `EmissivityFields` and `RadiationFields`, with PEP 688 GC-safe tracking (`_CudaBufferOwner`) and automatic materialization on host readback, eviction, or CPU fallback.
  - RTX 4090 `256/128` `L-INTEGRAL + TS` lightcone (`z=7–11`): reduces D2H traffic by **`24.3%`** (`1723.60 MB → 1304.17 MB`) and improves wall time from `1.912 s` to **`1.322 s`** (**`1.45×`**).
- Verified by `tests/native/test_21cmcu_phase4_i.py` (`9/9 passed`).

### Direction J (`v0.4.0`) — Legacy Removal, CPU CI & Suite Verification
- Removed legacy v3 (`lightcone/`, `gpu/`), CuPy v4 adapter (`v4/`, `reports/`), and obsolete top-level fixtures from `main` (preserved under git tag `archive-cupy-v4`).
- Added GitHub Actions CPU CI (`.github/workflows/ci.yml`: `USE_CUDA=0` build, `ruff`, CPU-safe native suite).
- Bumped package version to `0.4.0` and verified **`150/150` passing tests** on RTX 4090 (`cuda_arch=sm_89`).

---

## Completed Direction G (`v0.5.0`) — Native Multi-GPU 1D Slab Domain Decomposition

- **Multi-GPU Device Management (`cuda_runtime.cu`, `_cfg.py`, `py21cmcu`)**: added `CUDA_DEVICES` / `P21_CUDA_DEVICES`, `cuda_device_count()`, `get_cuda_devices()`, `set_cuda_devices()`, `use_mode(..., devices=...)`, and `mgpu_fft_runs` / `mgpu_ic_runs` runtime counters.
- **Y-Slab Distributed 3D FFT & IC (`cuda_mgpu.cu`, `cuda_ic.cu`, `cuda_analysis.cu`)**: implemented striped 2D R2C/C2R + adaptive P2P/host-staged X-to-Y transpose + 1D C2C transforms without external MPI dependencies, native multi-GPU `compute_initial_conditions` (`ZELDOVICH` + hybrid host-accumulated `2LPT`), and multi-GPU `power_spectrum_1d` / `power_spectrum_2d`.
  - Verified on `2×` & `3× RTX 4090` (`tests/native/test_21cmcu_phase5_g.py`, **`12/12 passed`**): native `1536³` (`46.131 s` Zel'dovich / `201.302 s` 2LPT) and `1792³` (`98.095 s` Zel'dovich / `365.663 s` 2LPT) beyond 24 GB single-card limits, graceful host fallback at `2048³` on 3 GPUs, and **`10,502.0 tasks/h`** multi-GPU `run_batch`.
- **Benchmark Suite & Publication Figures ([`docs/BENCHMARKS.md`](BENCHMARKS.md))**: complete CPU-16T vs `1×–3× RTX 4090` sweep (`benchmarks/results_rtx4090_v050.json`), visualization script (`benchmarks/plot_benchmarks.py`), and **`162/162` passing tests** (`-k "not speed"` across all 11 native suites).

---

## Completed Directions (K → L → M → N, `v0.6.0`)

### Direction K (`6e91af3`) — Multi-GPU Communication, Pinned-Bounce & Single-Slab `2048³`
- **Pinned host-bounce & parallel transposes (`cuda_mgpu.cu`)**: added 64 MiB portable `MgpuHostBounce`, `std::thread` sub-tiled `mgpu_transpose_x_to_y` / `mgpu_transpose_y_to_x`, cuFFT plan mutex safety (`g_mgpu_cufft_mtx`), and parallel multi-GPU strip transfers (`p21_mgpu_fft_r2c_from_host`, `p21_mgpu_fft_c2r_strips`).
- **Single-work-slab mode & zero-`h_tmp_ab` 2LPT streaming (`cuda_ic.cu`)**: eliminated the `32 GiB` `h_tmp_ab` host allocation at `2048³` and enabled single-work-slab Zel'dovich execution (`P21_CUDA_MGPU_SINGLE_SLAB=1` / auto), unlocking **native in-VRAM `2048³` (`8.590 B` voxels) Zel'dovich IC on `3× RTX 4090` in `58.467 s`** (**`3.71×`** vs `v0.5.0` host fallback) and **`2.80×` / `3.52×`** 2LPT speedups at `1536³` (`75.231 s`) and `1792³` (`103.892 s`).
- Verified in `tests/native/test_21cmcu_phase5_g.py` (**`14/14 passed`**, including `test_mgpu_ic_single_slab_mode_parity`).

### Direction L (`4c89439`) — SBI Observables & Resumable Pipeline
- **Observables module (`src/py21cmcu/observables.py`)**: `horizon_slope`, `horizon_wedge_mask`, `apply_wedge_filter_3d`, `add_thermal_noise_3d`, and `generate_sobol_parameter_grid`.
- **Resumable HDF5 checkpoints & lightcone archival (`src/py21cmcu/batch.py`, `examples/sbi_lightcone_dataset.py`)**: extended `BatchTask` (`wedge_filter`, `wedge_buffer`, `noise_sigma_mk`, `lightcone_out_path`, `lightcone_compression`) and `run_batch(..., checkpoint_h5=..., resume=True)` for fault-tolerant SBI dataset sweeps.

### Direction M (`a8eff48`) — Hardware Diagnostics, Extended CLI, User Guide & CI
- **Hardware & sizing doctor (`src/py21cmcu/doctor.py`)**: `py21cmcu.doctor(as_json=...)` checking CUDA runtime, `nvidia-smi topo -m`, compiler/GSL/FFTW bindings, and 24 GB multi-GPU sizing recommendations.
- **Extended CLI (`src/py21cmcu/cli.py`)**: added `21cmcu batch`, `21cmcu doctor`, and `21cmcu bench` subcommands alongside upstream `run` (`tests/native/test_21cmcu_phase6_lm.py`, `4/4 passed`).
- **Documentation & CI**: added [`docs/USER_GUIDE.md`](USER_GUIDE.md) and expanded `.github/workflows/ci.yml` to lint `examples` and test `test_21cmcu_phase6_lm.py`.

### Direction N (`v0.6.0`) — Benchmarks, Figures, Methods Draft & Release Verification
- Merged single-GPU and Direction K multi-GPU benchmarks into `benchmarks/results_rtx4090_v060.json`, updated `benchmarks/plot_benchmarks.py` (`mgpu_scaling.png` displaying native `2048³` Zel'dovich `58.5s`), authored standalone methods draft ([`docs/paper/21cmcu_paper.md`](paper/21cmcu_paper.md), `.tex`, `refs.bib`), bumped version to `0.6.0`, and verified **`168/168` passing tests** (`-k "not speed"` across all 12 native GPU test suites on `3× RTX 4090`).

---

## Completed Directions (O → Q → S & Phase 9, `v0.7.0`)

### Direction O (`8300144`) — GPU-Native Observables & Non-Gaussian Statistics
- **GPU Foreground Wedge & Thermal Noise (`cuda_analysis.cu`, `src/py21cmcu/observables.py`, `batch.py`)**: `p21_cuda_wedge_filter_3d` (cuFFT 3D R2C/C2R + `DeviceFieldRegistry` across `los_axis` ∈ {0, 1, 2}, **`23.86×`** at `256³`) and `p21_cuda_add_thermal_noise_3d` (Philox Gaussian noise, **`10.71×`** at `256³`).
- **One-Point Non-Gaussian Moments & PDF**: `p21_cuda_field_moments_and_pdf` + `field_moments_and_pdf` (**`26.43×`** at `256³`, histogram diff `0`), plus cross/2D lightcone chunk PS (`lightcone_chunk_power_spectra_2d`).

### Direction Q (`8300144`) — Zero-Copy CUDA Interop & `sbi` / `21CMMC` Inference
- **Zero-Copy Device Export (`cuda_mempool.cu`, `src/py21cmcu/interop.py`)**: `CudaFieldView` implementing pure-`ctypes` `__dlpack__`, `__dlpack_device__`, `__cuda_array_interface__`, and `to_torch` (`0.527 ms` with `registry_h2d_bytes = 0` under `lazy_d2h=True`; verified live zero-copy pointer sharing on PyTorch `2.5.1+cu121` RTX 4090).
- **Inference Primitives (`src/py21cmcu/inference.py`)**: `Likelihood21cm` and `SBISimulator` (with `tests/native/test_21cmcu_phase7_oq.py` `7/7 passed` & `examples/04_dlpack_and_sbi_inference.py`).

### Direction S (`4720a37`) — Radio Interferometer Mock Observation & Sensitivity Forecast Suite
- **Single-Pass GPU Mock Observation (`cuda_analysis.cu`, `src/py21cmcu/instrument.py`)**: `p21_cuda_mock_observation_3d` & `mock_observation_3d` combining UV coverage cuts, PSF Gaussian smoothing, LOS boxcar/Gaussian channel response, horizon/FoV foreground wedge exclusion, and $P_N(k_\perp, z)$-weighted Hermitian Philox thermal noise (**`71.65×`** on `SKA1-Low` `23.04 ms` vs `1650.48 ms` CPU and **`71.88×`** on `HERA-350` `22.48 ms` vs `1615.87 ms` CPU at `256³`).
- **Telescope Presets & Forecasts**: `InterferometerConfig`, `INSTRUMENT_PRESETS` (`SKA1-Low`, `HERA-350`, `MWA-II`, `LOFAR`), `baseline_density_uv` (analytic + explicit `antenna_xy_m`), exact $\Omega_p \Omega_{\mathrm{eff}}$ `thermal_noise_power_spectrum`, `forecast_power_spectrum_error`, and `forecast_power_spectrum_2d_error` (`tests/native/test_21cmcu_phase8_instrument.py` `7/7 passed`, `examples/05_interferometer_mock_and_forecast.py`).

### Phase 9 (`e35f58c`) — Lightcone Pipeline Optimization, `quick_lightcone`, & Interactive 3D Visualization
- **Zero-D2H Lightcone Means & Streaming (`cuda_analysis.cu`, `coeval.py`, `lightcone.py`, `lightconers.py`)**: 82nd CUDA API `p21_cuda_field_mean_f32` + `_get_coeval_array_no_sync` evaluate `global_quantities` and rectilinear slices directly on GPU-resident `device_only` arrays under `lazy_d2h=True` (`256/128` 12-node `L-INTEGRAL` lightcone **`1.496 s → 1.169 s`**, **`-21.9%`** wall time, **`-8.5%`** D2H); `stream_perturbed_fields=True` computes `PerturbedField` on demand inside `_redshift_loop_generator` to cut peak memory to $O(1)$.
- **One-Line API & Multi-GPU Redshift-Parallel Lightcones (`src/py21cmcu/shortcuts.py`)**: `quick_coeval` and `quick_lightcone` (with `parallel_redshifts=True` for causal-free EoR runs, integrated telescope mock, and HTML export).
- **Offline Interactive 3D WebGL2 & 2D Plotting (`src/py21cmcu/viz.py`)**: `export_volume_3d_html`, `export_lightcone_3d_html`, `plot_lightcone_summary`, `plot_coeval_slices`, and `plot_mock_comparison` (`tests/native/test_21cmcu_phase9_lightcone_viz.py` `7/7 passed`, `examples/06_quick_lightcone_and_3d_viz.py`, plus full GitHub Wiki sync).

---

## Completed Direction R (`v0.8.0-dev`) — Persistent Multi-GPU Plan Cache, Warm `BatchPool` & Zero-Disk Online SBI

- **Persistent Per-GPU cuFFT Plan Cache (`cuda_mgpu.cu`, `cuda_common.cuh`, `cuda_runtime.cu`)**: replaced per-call plan creation/destruction in `mgpu_local_2d_r2c`, `mgpu_local_2d_c2r`, and `mgpu_local_1d_c2c` with a mutex-guarded, concurrency-safe (`in_use`-tracked) 64-entry per-device LRU cache, cleaned safely by `p21_mgpu_fft_clear_cache()` inside `p21_cuda_cleanup()`.
- **Warm Persistent `BatchPool` & Integrated Post-Processing (`src/py21cmcu/batch.py`)**:
  - Extended `BatchTask` (`instrument`, `instrument_kwargs`, `lazy_d2h`, `compute_moments`, `moments_n_bins`, `oblique_lightcone`, `oblique_direction`) and `BatchResult` (`moments`, `worker_pid`, HDF5 roundtrip persistence), automatically enabling `KEEP_3D_VELOCITIES=True` for velocity-corrected oblique lightcones.
  - Added `BatchPool` (`map`, `imap(ordered=...)`, context manager, periodic `p21_cuda_registry_clear()` every `clear_registry_every` tasks) keeping worker processes, CUDA contexts, cuFFT plans, and CLASS tables resident across repeated SBI rounds; wired `run_batch(..., pool=...)`.
- **Zero-Disk `OnlineSBIDataset` (`src/py21cmcu/inference.py`, `src/py21cmcu/__init__.py`)**:
  - Upgraded `SBISimulator` with `instrument`, `batch_pool`, `lazy_d2h`, `oblique_lightcone`, and `BatchPool`-backed `"stats"`/`"ps1d"` batching.
  - Added `OnlineSBIDataset` (subclassing `torch.utils.data.IterableDataset` when PyTorch is present, zero hard PyTorch dependency) supporting finite Scrambled Sobol and infinite Uniform streaming into NumPy or GPU PyTorch tensors via DLPack.
- **Verification**:
  - `USE_CUDA=1 CUDA_ARCHS="86;89"` wheel build compiled all 18 CUDA units (`cuda_mgpu.cu`, `cuda_dlink.o`, `CUDA: True`).
  - **`81 passed, 53 skipped (GPU)`** across `tests/native/test_21cmcu_phase11_batchpool_sbi.py` and 9 existing native suites (`phase0`, `phase3_d`, `phase3_e`, `phase5_g`, `phase6_lm`, `phase7_oq`, `phase8_instrument`, `phase9_lightcone_viz`, `phase10_cpu_opt`) + [`examples/07_online_sbi_zero_disk.py`](../examples/07_online_sbi_zero_disk.py). *(Note: cluster GPUs on `phy-gpus` `debug` were 100% allocated during this verification run, so CUDA-only test cases were skipped.)*
  - 4-round × 4-task `128/64` `E-INTEGRAL` + `SKA1-Low` + moments CPU benchmark (`workers_per_gpu=4`, `threads_per_worker=4`): cold `run_batch` **`20.439 s`** (`16` worker PIDs, `[5.479, 5.332, 5.647, 3.982] s`) vs warm `BatchPool` **`7.733 s`** (`4` persistent worker PIDs, `[3.660, 1.228, 1.227, 1.343] s`) — **`2.643×` overall speedup** (**`~4.3×`** on warm rounds).

## Completed Phase 10 (`v0.9.0-dev`) — Higher-Order Non-Gaussian & Morphological Topology

- **Native CUDA APIs (`#83–#85`) in `cuda_analysis.cu` (`cuda_api.h`, `_functionprototypes_wrapper.h`, `cuda_stubs.c`) & [`src/py21cmcu/topology.py`](../src/py21cmcu/topology.py)**:
  - **`bispectrum_3d` (`p21_cuda_bispectrum_3d`)**: Fast 3D FFT shell-product bispectrum $B(k_1, k_2, k_3)$ & reduced bispectrum $Q$ supporting `equilateral`, `squeezed`, `folded`, `isosceles`, and custom triangle lists with 3-slot shell reuse across identical/adjacent $k$-shells.
  - **`bubble_size_distribution_mfp` (`p21_cuda_mfp_bsd`)**: Parallel Philox isotropic ray-traced Mean-Free-Path Bubble Size Distribution (MFP BSD) with warp-aggregated ionized-voxel compaction (`__ballot_sync`) and periodic boundary wrapping (plus vectorized active-ray NumPy CPU fallback).
  - **`minkowski_functionals_3d` (`p21_cuda_minkowski_3d`)**: Warp-reduced exact cubical-complex 3D Minkowski Functionals ($V_0, V_1, V_2, V_3$ and integer Euler characteristic $\chi$) with `kMkChunk=8` multi-threshold evaluation and Schmalzing & Buchert (1997) Crofton normalization.
- **Non-Gaussian SBI Integration (`src/py21cmcu/inference.py`, `src/py21cmcu/__init__.py`)**:
  - Added `SBISimulator(..., summary="nongaussian")` combining global/1-point moments (`5`), 1D $\Delta^2(k)$, equilateral $Q(k)$ (`6`), and standardized Minkowski functionals $V_1, V_2, V_3$ (`3 × 5`), exported top-level APIs (`bispectrum_3d`, `bubble_size_distribution_mfp`, `minkowski_functionals_3d`), and added [`examples/08_nongaussian_topology_sbi.py`](../examples/08_nongaussian_topology_sbi.py).
- **Verification (`b1` `USE_CUDA=1` CUDA 12.6 build + 16-core CPU)**:
  - `USE_CUDA=1 CUDA_ARCHS="86;89"` wheel compiled cleanly (`CUDA: True`, all 85 native CUDA symbols verified).
  - **`85 passed, 54 skipped (GPU)`** across [`tests/native/test_21cmcu_phase12_topology.py`](../tests/native/test_21cmcu_phase12_topology.py) and all 10 prior native suites: verified Gaussian $Q \approx 0$, local quadratic $Q \approx 2\lambda = 0.50$ (`equilateral`/`squeezed`/`folded` within `<15%`), isolated sphere MFP $\langle R \rangle \approx \frac{3}{4} R_0$ (`<5%`), exact cubical Euler numbers ($\chi = 1$ ball, $2$ shell, $0$ torus), `<2%` sphere $V_0,V_1,V_2$, resolved GRF parity, and EoR `SBISimulator(summary="nongaussian")` `(2, 34)` output.
  - `128³` 16-core CPU timings: `bispectrum_3d` (equilateral, 8 bins) `0.523 s`, `bubble_size_distribution_mfp` (50k rays) `0.037 s`, `minkowski_functionals_3d` (15 thresholds) `0.173 s`.
  - **Colab Tesla T4 Live GPU Parity & Speedup (`10 passed`, `0 skipped` across `phase11` + `phase12`)**:
    - `128³`: `bispectrum_3d` `0.8812 s -> 0.1420 s` (`6.21×`, `max_rel_diff_B = 1.31e-6`), `bubble_size_distribution_mfp` (200k rays) `0.6947 s -> 0.0095 s` (`72.87×`, `mean_r_rel_diff = 0.00366`), `minkowski_functionals_3d` `0.2960 s -> 0.0357 s` (`8.28×`, exact Euler $\chi$).
    - `256³`: `bispectrum_3d` `11.3274 s -> 0.4252 s` (`26.64×`, `max_rel_diff_B = 8.21e-7`), `bubble_size_distribution_mfp` (200k rays) `2.3150 s -> 0.0370 s` (`62.53×`, `mean_r_rel_diff = 0.00050`), `minkowski_functionals_3d` `3.0784 s -> 0.3632 s` (`8.48×`, exact Euler $\chi$).
    - `BatchPool` (`4×2`, `128/64` E-INTEGRAL + `SKA1-Low` + `moments` on T4): cold `26.7845 s` (`4` PIDs) -> warm `11.3101 s` (`1` PID, round 4 `1.2729 s`, `2.368×` overall); `OnlineSBIDataset` zero-disk PyTorch CUDA tensor streaming (`theta=(2,2)`, `field=(2,64,64,64)`, `is_cuda=True`, `h5_files_on_disk=0`).

### Completed: Phase 11 — CRN Fisher Forecaster & Multi-Probe Cross-Correlations (`v0.9.0-dev`)
- **86th Native CUDA API + Multi-Probe Gridding ([`src/py21cmcu/multiprobe.py`](../src/py21cmcu/multiprobe.py))**:
  - `p21_cuda_cic_deposit_3d` in [`cuda_analysis.cu`](../src/py21cmfast/src/cuda/cuda_analysis.cu) (exposed across `cuda_api.h`, `_functionprototypes_wrapper.h`, `cuda_stubs.c`) + vectorized CPU fallback `cic_deposit_3d` (periodic CIC deposition, optional overdensity normalization $\delta = \rho/\bar{\rho} - 1$, optional $\mathrm{sinc}^2$ window deconvolution, and grid-stride loops supporting `256³+` volumes above `65535 × 256` threads).
  - `load_gravitycu_halos`: normalized reader for GravityCU HDF5 (`halos/*`, `halo_*`), `.npz`/`.npy`, and structured catalog inputs.
  - `cross_power_21cm_galaxy`, `cross_power_21cm_galaxy_error`, `ksz_momentum_map`: 3D `21cm × galaxy` cross-power $P_{21,g}(k)$ & correlation coefficient $r(k)$ with per-bin cosmic+thermal+shot-noise uncertainties, and kinematic Sunyaev-Zel'dovich momentum proxy map $p_\mathrm{los} = (1+\delta)\,v_\mathrm{los}$ + transverse $\tau$-weighted integral.
- **CRN Fisher Forecaster & Contour Visualization ([`src/py21cmcu/fisher.py`](../src/py21cmcu/fisher.py), [`src/py21cmcu/viz.py`](../src/py21cmcu/viz.py))**:
  - `FisherForecaster`, `FisherForecastResult`, `fisher_forecast_21cm`: Common-Random-Number (`CRN`) finite-difference Fisher matrix with shared per-evaluation `seed_base` across $\pm h$ parameter perturbations, inverse/SVD covariance marginalization, $1\sigma$ uncertainties, and `to_dict()`/`from_dict()`.
  - `plot_fisher_contours`: $1\sigma$/$2\sigma$ Gaussian corner-plot contours for parameter subsets.
  - Runnable end-to-end showcase [`examples/09_fisher_and_multiprobe_forecast.py`](../examples/09_fisher_and_multiprobe_forecast.py) and test suite [`tests/native/test_21cmcu_phase13_fisher_multiprobe.py`](../tests/native/test_21cmcu_phase13_fisher_multiprobe.py) (`7 passed, 0 skipped` on Colab T4 GPU; `97 passed, 64 skipped` in GitHub `CPU CI`).
  - **Colab Tesla T4 `256³` 1M-Particle Benchmarks (`benchmarks/results_t4_v090.json`)**: `cic_deposit_3d` `1.4336 s -> 0.0365 s` (**`39.29×`**, `max_abs_diff = 3.81e-6`), `cross_power_21cm_galaxy` `4.8076 s -> 0.2324 s` (**`20.68×`**, `max_abs_rk_diff = 2.91e-8`), `Example 09` `total SNR = 20.13`, `kSZ rms = 0.1391 uK`.

---

### Completed Direction P (`v0.8.0` Carryover) — Distributed Multi-GPU EoR & Sub-160 GiB `2048³` 2LPT Streaming
- **Shared Filter Math & Y-Slab Filtering (`cuda_filters.cuh`, `cuda_cufft.cu`, `cuda_mgpu.cu`/`.cuh`)**:
  - Consolidated all 6 k-space filter weights (tophat, sharp-k, Gaussian, exponential MFP, spherical shell straight-line, and multiple-scattering `2F3` asymptotics) into [`cuda_filters.cuh`](../src/py21cmfast/src/cuda/cuda_filters.cuh).
  - Extended `cuda_mgpu.cu` with `P21_CUDA_MGPU_VIRTUAL_RANKS` (allowing 2+ virtual Y-slab ranks on single-GPU systems), `p21_mgpu_r2c_from_device`, `p21_mgpu_filter_yslabs`, `p21_mgpu_c2r_to_device`, and the C/CFFI `p21_cuda_mgpu_filter_box_host` entry point (`47` CFFI CUDA symbols / `83` CUDA APIs).
- **Sub-160 GiB `2048³` 2LPT Host Streaming (`cuda_ic.cu`)**:
  - Replaced the three simultaneous full-grid diagonal host buffers with strip-streamed accumulation across `a=0, 1, 2`, freed the single auxiliary diagonal buffer before `phi2` R2C, reused `h_saved_input` for `h_phi2_k` in `single_slab_mode`, removed the `boxes->hires_vx_2LPT` allocation dependency when `PERTURB_ON_HIGH_RES=False`, and added `P21_CUDA_MGPU_2LPT_HOST_STREAM=1`.
- **Distributed EoR Filtering (`cuda_ionization.cu`, `cuda_radiation.cu`)**:
  - Wired `p21_mgpu_eor_enabled` distributed Y-slab FFT and k-space filtering into both `IonizedBox` (including `P21_CUDA_ION_SPILL_KSPACE` host spilling) and `RadiationFields` / `SpinTemperature`, verified against single-GPU parity using virtual ranks in [`tests/native/test_21cmcu_phase15_direction_p.py`](../tests/native/test_21cmcu_phase15_direction_p.py).

---

## Future Roadmap

### Release `v1.0.0` — Multi-Node NCCL, Large-Volume Benchmarks & Paper Finalization
- **Multi-GPU Multi-Probe Scaling**: Joint EoR + galaxy + CMB secondary likelihood benchmarks across `256³–1024³` volumes on multi-GPU nodes.
- **Multi-Node / NVLink Scaling**: Optional NCCL all-to-all collective backend in `cuda_mgpu.cu` for `4×–8×` NVLink/NVSwitch nodes (`2048³–4096³` grids).
- **Methods Paper Archival**: Finalize [`docs/paper/21cmcu_paper.tex`](paper/21cmcu_paper.tex) with multi-GPU multi-probe benchmarks, Zenodo archival DOI, and frozen reproduction tables (`benchmarks/reproduce_v090_tables.py`). *(Packaging/wheel CI, `release.yml`, PEP 621/561/562 layout, and `21cmcu version` already shipped in `v0.9.0`.)*
