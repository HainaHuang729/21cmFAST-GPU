"""Hardware, CUDA, toolchain, and memory-capacity diagnostics for 21cmCU."""

from __future__ import annotations

import ctypes.util
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

_GRID_RECOMMENDATIONS = [
    {
        "grid": "512^3 -> 256^3 (Coeval L-INTEGRAL+TS)",
        "min_gpus_24gb": 1,
        "vram_per_gpu_gb": "~5",
        "host_ram_gb": "8",
    },
    {
        "grid": "768^3 (EoR Coeval)",
        "min_gpus_24gb": 1,
        "vram_per_gpu_gb": "~19",
        "host_ram_gb": "16",
    },
    {
        "grid": "1024^3 (2LPT IC + PF)",
        "min_gpus_24gb": 1,
        "vram_per_gpu_gb": "~7–18",
        "host_ram_gb": "24",
    },
    {
        "grid": "1536^3 (3.62B, Multi-GPU 2LPT IC)",
        "min_gpus_24gb": 2,
        "vram_per_gpu_gb": "~15 (2 GPUs)",
        "host_ram_gb": "90",
    },
    {
        "grid": "1792^3 (5.75B, Multi-GPU 2LPT IC)",
        "min_gpus_24gb": 3,
        "vram_per_gpu_gb": "~16 (3 GPUs)",
        "host_ram_gb": "135",
    },
    {
        "grid": "2048^3 (8.59B, Single-Slab Zel'dovich / 2LPT)",
        "min_gpus_24gb": 3,
        "vram_per_gpu_gb": "~11.6 (3 GPUs)",
        "host_ram_gb": "72 (Zel) / <160 (2LPT)",
    },
]


def _run_cmd(cmd: list[str]) -> str | None:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def doctor(*, as_json: bool = False) -> dict[str, Any]:
    """Collect 21cmCU runtime, GPU topology, dependency, and sizing diagnostics."""
    import py21cmcu
    from py21cmfast._cfg import config

    gpus: list[dict[str, Any]] = []
    smi_out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi_out:
        for line in smi_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_mib": int(parts[2]),
                        "compute_cap": parts[3],
                    }
                )

    topo = _run_cmd(["nvidia-smi", "topo", "-m"]) if len(gpus) >= 2 else None
    nvcc_bin = shutil.which("nvcc") or (
        "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None
    )
    nvcc_out = _run_cmd([nvcc_bin, "--version"]) if nvcc_bin else None

    report: dict[str, Any] = {
        "versions": {
            "py21cmcu": py21cmcu.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": int(os.cpu_count() or 1),
        },
        "cuda": {
            "compiled": bool(py21cmcu.cuda_compiled()),
            "available": bool(py21cmcu.cuda_available()),
            "api_count": int(py21cmcu.cuda_api_count()),
            "device_count": int(py21cmcu.cuda_device_count()),
            "active_devices": list(py21cmcu.get_cuda_devices()),
            "use_cuda_config": bool(config["USE_CUDA"]),
            "cuda_mode": py21cmcu.get_cuda_mode(),
            "cuda_lazy_d2h": bool(config["CUDA_LAZY_D2H"]),
            "gpus": gpus,
            "nvlink_or_pcie_topo": topo,
        },
        "toolchain": {
            "nvcc": nvcc_bin,
            "nvcc_banner": nvcc_out.splitlines()[-1] if nvcc_out else None,
            "gcc": shutil.which("gcc"),
            "libgsl": ctypes.util.find_library("gsl"),
            "libfftw3f": ctypes.util.find_library("fftw3f"),
        },
        "grid_sizing": _GRID_RECOMMENDATIONS,
    }
    if as_json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


__all__ = ["doctor"]
