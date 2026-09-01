"""Determinism tests.

The claim the paper makes is that a config plus a seed determines the result.
These tests hold it to that on a tiny configuration: same seed, identical
metrics; different seed, different metrics (otherwise the seed is not doing
anything and the multi-seed error bars are fictitious).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flowconx.determinism import RunProvenance, git_state, library_versions, seed_everything


def test_seed_everything_reports_what_it_set():
    state = seed_everything(123)
    assert state["seed"] == 123
    assert "torch" in state
    # It must never claim a guarantee it failed to obtain.
    if state.get("deterministic_algorithms") is False:
        assert "deterministic_error" in state or state["deterministic_requested"] is False


def test_seeding_makes_numpy_reproducible():
    seed_everything(7)
    first = np.random.rand(5)
    seed_everything(7)
    assert np.array_equal(first, np.random.rand(5))


def test_provenance_captures_code_state():
    provenance = RunProvenance(config_hash="h", config_name="n", seed=0).finish().as_dict()
    for key in ("git", "libraries", "python", "platform_name", "machine", "wall_clock_seconds"):
        assert key in provenance
    assert "dirty" in provenance["git"], "a run on a dirty tree must be identifiable as such"
    assert "numpy" in provenance["libraries"]


def test_library_versions_marks_absent_rather_than_omitting():
    versions = library_versions()
    assert set(versions) >= {"numpy", "pandas", "torch"}
    assert all(isinstance(v, str) for v in versions.values())


def test_git_state_never_raises_outside_a_repo():
    state = git_state()
    assert set(state) == {"commit", "branch", "dirty", "dirty_files"}


# --------------------------------------------------------------------------
# End-to-end determinism. Needs the prepared dataset, so it skips when absent.
# --------------------------------------------------------------------------

SMOKE_CSV = Path("data/processed/cesnet_quic22.csv")


def _run(tmp_path: Path, seed: int, tag: str) -> dict:
    from flowconx.run import main

    code = main(
        [
            "--config",
            "configs/smoke.yaml",
            "--seed",
            str(seed),
            "--output-root",
            str(tmp_path / tag),
            "--overwrite",
        ]
    )
    assert code == 0
    metrics = next((tmp_path / tag).rglob("metrics.json"))
    return json.loads(metrics.read_text())


def _scores(metrics: dict) -> dict:
    return {
        head: {k: report[k] for k in ("macro_f1", "balanced_accuracy", "accuracy")}
        for head, report in metrics["closed_set"].items()
    }


@pytest.mark.slow
def test_same_seed_gives_identical_metrics(tmp_path):
    if not SMOKE_CSV.exists():
        pytest.skip(f"{SMOKE_CSV} not built; run `python -m flowconx.data.prepare --source cesnet`")
    first = _run(tmp_path, seed=0, tag="a")
    second = _run(tmp_path, seed=0, tag="b")
    assert first["config_hash"] == second["config_hash"]
    assert _scores(first) == _scores(second), "same config and seed must give identical metrics"


@pytest.mark.slow
def test_different_seed_changes_metrics(tmp_path):
    if not SMOKE_CSV.exists():
        pytest.skip(f"{SMOKE_CSV} not built; run `python -m flowconx.data.prepare --source cesnet`")
    first = _run(tmp_path, seed=0, tag="a")
    other = _run(tmp_path, seed=1, tag="c")
    assert first["config_hash"] == other["config_hash"], "the seed must not change experiment identity"
    assert _scores(first) != _scores(other), (
        "two seeds produced byte-identical metrics; the seed is not reaching the model, "
        "which would make every multi-seed error bar in the paper meaningless"
    )
