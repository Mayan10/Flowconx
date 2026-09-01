"""Baseline models run on the identical manifest-defined splits.

`deep.py` holds the non-pretrained neural baselines. The classical and
identifier-shortcut baselines live in `flowconx/audit/`, because their primary
job is auditing the dataset rather than competing with the model.

`WHY_NOT_RUN.md` records every baseline we could not run and precisely what
blocked it. No number for any model listed there appears anywhere as an
ours-run result.
"""

from .deep import (  # noqa: F401
    DEEP_BASELINES,
    BaselineSpec,
    build_baseline,
    predict,
    predict_scores,
    train_baseline,
)

__all__ = [
    "DEEP_BASELINES",
    "BaselineSpec",
    "build_baseline",
    "predict",
    "predict_scores",
    "train_baseline",
]
