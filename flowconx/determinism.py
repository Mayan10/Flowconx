"""Seeding and run provenance.

The Phase 0 audit found that the previous training script seeded only
``torch.manual_seed`` and recorded nothing about the environment, so no number
could be attributed to a code state (AUDIT.md 4, R3-R4). This module is the
fix: one call seeds every generator the pipeline touches, and one call
captures everything a reviewer needs to tell two runs apart.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# Set before the first CUDA context is created, or cuBLAS ignores it and
# torch.use_deterministic_algorithms raises at the first matmul.
CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"


def seed_everything(seed: int, deterministic: bool = True) -> Dict[str, object]:
    """Seed Python, NumPy and Torch; optionally force deterministic kernels.

    Returns what it actually managed to set, because determinism is not fully
    available on every backend and a run must record which guarantees it had
    rather than claim all of them.
    """
    state: Dict[str, object] = {"seed": seed, "deterministic_requested": deterministic}
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        state["torch"] = "absent"
        return state

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    state["torch"] = torch.__version__

    if not deterministic:
        state["deterministic_algorithms"] = False
        return state

    os.environ.setdefault(CUBLAS_WORKSPACE_ENV, ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        state["deterministic_algorithms"] = True
    except (RuntimeError, AttributeError) as exc:
        # MPS in particular does not implement deterministic variants of every
        # kernel. Recording the failure is more useful than pretending.
        state["deterministic_algorithms"] = False
        state["deterministic_error"] = str(exc)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        state["cudnn_deterministic"] = True
    return state


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker from the torch seed, not from the clock."""
    import torch

    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def git_state() -> Dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    dirty = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # A run made on a dirty tree is not reproducible from the commit
        # alone, and the paper needs to know that before the number is used.
        "dirty": "yes" if dirty else "no",
        "dirty_files": dirty[:2000],
    }


def library_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for module in ("numpy", "pandas", "scipy", "sklearn", "torch", "xgboost", "yaml", "matplotlib"):
        try:
            out[module] = getattr(__import__(module), "__version__", "unknown")
        except ImportError:
            out[module] = "absent"
    return out


def device_description(device: Optional[object] = None) -> Dict[str, object]:
    info: Dict[str, object] = {"requested": str(device) if device is not None else "unset"}
    try:
        import torch
    except ImportError:
        return info
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
    info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return info


@dataclass
class RunProvenance:
    """Everything needed to attribute a number to a code state and a machine."""

    config_hash: str
    config_name: str
    seed: int
    git: Dict[str, str] = field(default_factory=git_state)
    libraries: Dict[str, str] = field(default_factory=library_versions)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    # Named with a suffix because a dataclass field called `platform` shadows
    # the module of that name inside the class body.
    platform_name: str = field(default_factory=lambda: platform.platform())
    machine: str = field(default_factory=lambda: platform.machine())
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 0)
    seeding: Dict[str, object] = field(default_factory=dict)
    device: Dict[str, object] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    wall_clock_seconds: Optional[float] = None

    def finish(self) -> "RunProvenance":
        self.wall_clock_seconds = round(time.time() - self.started_at, 3)
        return self

    def as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        payload["started_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at))
        return payload
