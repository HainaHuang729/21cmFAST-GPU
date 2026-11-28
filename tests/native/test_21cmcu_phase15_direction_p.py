"""Phase 15 tests (Direction P): Distributed multi-GPU EoR filtering & sub-160 GiB 2LPT streaming."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import py21cmcu
import pytest
from py21cmfast.c_21cmfast import ffi, lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _numpy_tophat_gaussian_sharpk_ref(
    field: np.ndarray, box_len: float, ncf: float, filter_type: int, radius: float
) -> np.ndarray:
    """Exact NumPy 3D R2C/C2R reference for Tophat (0), Sharp-k (1), and Gaussian (2)."""
    nx, ny, nz = field.shape
    dk_x = 2.0 * np.pi / box_len
    dk_y = 2.0 * np.pi / box_len
    dk_z = 2.0 * np.pi / (box_len * ncf)

    kx = np.fft.fftfreq(nx, d=1.0 / nx).astype(np.float64) * dk_x
    ky = np.fft.fftfreq(ny, d=1.0 / ny).astype(np.float64) * dk_y
    kz = np.arange(nz // 2 + 1, dtype=np.float64) * dk_z

    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    k_sq = kx3 * kx3 + ky3 * ky3 + kz3 * kz3
    k = np.sqrt(k_sq)

    if filter_type == 0:
        kr = k * radius
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(
                kr < 1e-4,
                1.0 - kr * kr / 10.0,
                3.0 * np.power(np.where(kr > 0, kr, 1.0), -3) * (np.sin(kr) - np.cos(kr) * kr),
            )
    elif filter_type == 1:
        w = np.where(k * radius * 0.413566994 > 1.0, 0.0, 1.0)
    elif filter_type == 2:
        w = np.exp(-0.643 * 0.643 * k_sq * (radius * radius) / 2.0)
    else:
        raise ValueError(f"Unsupported reference filter_type={filter_type}")

    kfield = np.fft.rfftn(field.astype(np.float64)) * w
    return np.fft.irfftn(kfield, s=(nx, ny, nz)).astype(np.float32)


def test_direction_p_doctor_and_abi_exports() -> None:
    """Verify doctor() reports <160 GiB 2048^3 2LPT RAM and filter ABI symbol is exported."""
    report = py21cmcu.doctor(as_json=False)
    sizing = {row["grid"]: row for row in report["grid_sizing"]}
    assert sizing["2048^3 (8.59B, Single-Slab Zel'dovich / 2LPT)"]["host_ram_gb"] == (
        "72 (Zel) / <160 (2LPT)"
    )

    assert hasattr(lib, "p21_cuda_mgpu_filter_box_host")
    assert py21cmcu.cuda_api_count() >= 47

    if lib.p21_cuda_is_compiled() == 0:
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        out = np.zeros((4, 4, 4), dtype=np.float32)
        rc = lib.p21_cuda_mgpu_filter_box_host(
            ffi.cast("const float *", ffi.from_buffer(arr)),
            ffi.cast("float *", ffi.from_buffer(out)),
            4,
            4,
            4,
            16.0,
            1.0,
            0,
            2.0,
            0.0,
            0.0,
        )
        assert rc == -1


@requires_cuda_runtime
@pytest.mark.parametrize(
    ("filter_type", "radius", "r_param", "r_star"),
    [
        (0, 6.0, 0.0, 0.0),
        (1, 6.0, 0.0, 0.0),
        (2, 5.0, 0.0, 0.0),
        (3, 6.0, 4.5, 0.0),
        (4, 4.0, 8.0, 0.0),
        (5, 4.0, 8.0, 2.0),
    ],
)
def test_mgpu_filter_box_host_parity_virtual_ranks(
    monkeypatch: pytest.MonkeyPatch,
    filter_type: int,
    radius: float,
    r_param: float,
    r_star: float,
) -> None:
    """Verify 2-virtual-rank Y-slab k-space filter matches 1-rank & NumPy references."""
    nx, ny, nz = 32, 32, 48
    box_len = 64.0
    ncf = 1.5
    rng = np.random.default_rng(1500 + filter_type)
    src = rng.standard_normal((nx, ny, nz)).astype(np.float32)

    py21cmcu.cleanup_cuda()
    monkeypatch.delenv("P21_CUDA_MGPU_VIRTUAL_RANKS", raising=False)
    ref_out = np.empty_like(src)
    rc_ref = lib.p21_cuda_mgpu_filter_box_host(
        ffi.cast("const float *", ffi.from_buffer(src)),
        ffi.cast("float *", ffi.from_buffer(ref_out)),
        nx,
        ny,
        nz,
        box_len,
        ncf,
        filter_type,
        radius,
        r_param,
        r_star,
    )
    assert rc_ref == 0

    if filter_type in (0, 1, 2):
        np_ref = _numpy_tophat_gaussian_sharpk_ref(src, box_len, ncf, filter_type, radius)
        np.testing.assert_allclose(ref_out, np_ref, rtol=1e-4, atol=1e-4)

    py21cmcu.reset_cuda_stats()
    monkeypatch.setenv("P21_CUDA_MGPU_VIRTUAL_RANKS", "2")
    monkeypatch.setenv("P21_CUDA_FORCE_MGPU", "1")
    mgpu_out = np.empty_like(src)
    rc_mgpu = lib.p21_cuda_mgpu_filter_box_host(
        ffi.cast("const float *", ffi.from_buffer(src)),
        ffi.cast("float *", ffi.from_buffer(mgpu_out)),
        nx,
        ny,
        nz,
        box_len,
        ncf,
        filter_type,
        radius,
        r_param,
        r_star,
    )
    assert rc_mgpu == 0
    assert py21cmcu.cuda_stats()["mgpu_fft_runs"] >= 2
    np.testing.assert_allclose(mgpu_out, ref_out, rtol=1e-4, atol=1e-4)
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
@pytest.mark.parametrize("perturb_hires", [False, True])
def test_mgpu_2lpt_host_stream_virtual_ranks_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, perturb_hires: bool
) -> None:
    """Verify streaming host-accumulated 2LPT IC across virtual ranks matches 1-GPU baseline."""
    inputs = py21cmcu.InputParameters(random_seed=9150 + int(perturb_hires)).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=48.0,
        PERTURB_ALGORITHM="2LPT",
        PERTURB_ON_HIGH_RES=perturb_hires,
        N_THREADS=2,
    )

    py21cmcu.cleanup_cuda()
    monkeypatch.delenv("P21_CUDA_MGPU_VIRTUAL_RANKS", raising=False)
    monkeypatch.delenv("P21_CUDA_MGPU_2LPT_HOST_STREAM", raising=False)
    monkeypatch.delenv("P21_CUDA_MGPU_SINGLE_SLAB", raising=False)
    monkeypatch.delenv("P21_CUDA_FORCE_MGPU", raising=False)

    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0], direc=str(tmp_path / "ref")):
        ic_ref = py21cmcu.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)

    py21cmcu.reset_cuda_stats()
    monkeypatch.setenv("P21_CUDA_MGPU_VIRTUAL_RANKS", "2")
    monkeypatch.setenv("P21_CUDA_FORCE_MGPU", "1")
    monkeypatch.setenv("P21_CUDA_MGPU_2LPT_HOST_STREAM", "1")
    monkeypatch.setenv("P21_CUDA_MGPU_SINGLE_SLAB", "1")

    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0], direc=str(tmp_path / "mgpu")):
        ic_mgpu = py21cmcu.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)

    st = py21cmcu.cuda_stats()
    assert st["mgpu_ic_runs"] >= 1
    assert st["mgpu_fft_runs"] >= 1

    fields = (
        ["hires_density", "hires_vx", "hires_vy", "hires_vz", "hires_vx_2LPT", "hires_vy_2LPT", "hires_vz_2LPT"]
        if perturb_hires
        else [
            "hires_density",
            "lowres_density",
            "lowres_vx",
            "lowres_vy",
            "lowres_vz",
            "lowres_vx_2LPT",
            "lowres_vy_2LPT",
            "lowres_vz_2LPT",
        ]
    )
    for name in fields:
        arr_m = np.asarray(getattr(ic_mgpu, name), dtype=np.float64)
        arr_r = np.asarray(getattr(ic_ref, name), dtype=np.float64)
        np.testing.assert_allclose(arr_m, arr_r, rtol=1e-4, atol=1e-4, err_msg=f"Mismatch in {name}")

    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
@pytest.mark.parametrize("spill_kspace", ["0", "1"])
def test_mgpu_eor_virtual_ranks_coeval_ts_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spill_kspace: str
) -> None:
    """Verify full TS+EoR coeval simulation with distributed Y-slab filtering matches 1-GPU."""
    inputs = py21cmcu.InputParameters(
        random_seed=9160 + int(spill_kspace),
        node_redshifts=(12.0, 10.0),
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=48.0,
        N_THREADS=2,
        Z_HEAT_MAX=12.0,
        SOURCE_MODEL="E-INTEGRAL",
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        LYA_MULTIPLE_SCATTERING=True,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
    )

    py21cmcu.cleanup_cuda()
    monkeypatch.delenv("P21_CUDA_MGPU_VIRTUAL_RANKS", raising=False)
    monkeypatch.delenv("P21_CUDA_FORCE_MGPU", raising=False)
    monkeypatch.delenv("P21_CUDA_ION_SPILL_KSPACE", raising=False)

    with py21cmcu.use_mode("strict", use_cuda=True, devices=[0], direc=str(tmp_path / "ref")):
        coev_ref = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=False)

    py21cmcu.reset_cuda_stats()
    monkeypatch.setenv("P21_CUDA_MGPU_VIRTUAL_RANKS", "2")
    monkeypatch.setenv("P21_CUDA_FORCE_MGPU", "1")
    monkeypatch.setenv("P21_CUDA_ION_SPILL_KSPACE", spill_kspace)

    with py21cmcu.use_mode("strict", use_cuda=True, devices=[0], direc=str(tmp_path / "mgpu")):
        coev_mgpu = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=False)

    st = py21cmcu.cuda_stats()
    assert st["mgpu_fft_runs"] >= 4
    assert st["ionization_runs"] >= 2

    for ref_c, mgpu_c in zip(coev_ref, coev_mgpu, strict=True):
        for name in ("neutral_fraction", "kinetic_temperature", "brightness_temp"):
            a_ref = np.asarray(getattr(ref_c, name), dtype=np.float64)
            a_mgpu = np.asarray(getattr(mgpu_c, name), dtype=np.float64)
            assert np.isfinite(a_mgpu).all()
            np.testing.assert_allclose(
                a_mgpu, a_ref, rtol=1e-3, atol=1e-3, err_msg=f"Mismatch in {name}"
            )

    py21cmcu.cleanup_cuda()
