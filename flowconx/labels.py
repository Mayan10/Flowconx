"""Label normalisation and rare-class handling policy.

The committed dataset has a service class (``xr_interactive``) with 23 rows
out of 112,121 -- roughly 0.02%. Leaving it in silently makes macro-F1 a
lottery on 4-5 test rows; dropping it silently hides a decision from the
reader. The policy is therefore explicit, configurable, and recorded in every
``metrics.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

RARE_CLASS_MODES = ("drop", "merge_into_parent", "keep_and_report")

# Where a rare class goes under ``merge_into_parent``. XR / metaverse traffic
# in this dataset is entirely Zepeto and Roblox, both interactive real-time
# rendered sessions, so gaming is the defensible parent. Stated in the paper.
DEFAULT_PARENT_CLASS: Dict[str, str] = {"xr_interactive": "gaming"}

# Below this many rows a class cannot support a meaningful held-out estimate:
# at a 20% test fraction it contributes fewer than 20 test rows, so its
# per-class F1 has a resolution worse than 5 percentage points.
DEFAULT_MIN_CLASS_COUNT = 100


@dataclass
class RareClassPolicy:
    mode: str = "drop"
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT
    parent_class: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PARENT_CLASS))
    label_column: str = "service"

    def __post_init__(self) -> None:
        if self.mode not in RARE_CLASS_MODES:
            raise ValueError(f"Unknown rare-class mode {self.mode!r}. Known: {list(RARE_CLASS_MODES)}")


def rare_classes(df: pd.DataFrame, policy: RareClassPolicy) -> List[str]:
    counts = df[policy.label_column].astype(str).value_counts()
    return sorted(counts[counts < policy.min_class_count].index.tolist())


def apply_rare_class_policy(
    df: pd.DataFrame, policy: Optional[RareClassPolicy] = None
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply the policy and return the table plus a record of what it did."""
    policy = policy or RareClassPolicy()
    column = policy.label_column
    before = df[column].astype(str).value_counts().to_dict()
    rare = rare_classes(df, policy)
    report: Dict[str, object] = {
        "mode": policy.mode,
        "min_class_count": policy.min_class_count,
        "label_column": column,
        "rare_classes": rare,
        "counts_before": {str(k): int(v) for k, v in before.items()},
        "rows_before": int(len(df)),
    }

    if not rare or policy.mode == "keep_and_report":
        out = df.copy()
        report["action"] = "kept" if rare else "no_rare_classes"
        if rare and policy.mode == "keep_and_report":
            report["warning"] = (
                "Rare classes retained. Their per-class F1 rests on a handful of test rows and "
                "must be read as an appendix result, not a headline number."
            )
    elif policy.mode == "drop":
        out = df[~df[column].astype(str).isin(rare)].reset_index(drop=True)
        report["action"] = "dropped"
        report["rows_dropped"] = int(len(df) - len(out))
    else:  # merge_into_parent
        out = df.copy()
        mapping = {name: policy.parent_class.get(name) for name in rare}
        unmapped = sorted(name for name, parent in mapping.items() if not parent)
        if unmapped:
            raise ValueError(
                f"Rare classes {unmapped} have no parent in RareClassPolicy.parent_class; "
                "either add one or use mode='drop'."
            )
        out[column] = out[column].astype(str).replace({k: v for k, v in mapping.items() if v})
        report["action"] = "merged"
        report["merges"] = {k: v for k, v in mapping.items() if v}

    report["counts_after"] = {str(k): int(v) for k, v in out[column].astype(str).value_counts().items()}
    report["rows_after"] = int(len(out))
    return out, report


def label_array(df: pd.DataFrame, column: str = "service") -> Tuple[np.ndarray, List[str]]:
    """Integer-encode a label column with a stable, sorted class order."""
    values = df[column].astype(str)
    classes = sorted(values.unique())
    index: Mapping[str, int] = {name: i for i, name in enumerate(classes)}
    return values.map(index).to_numpy(dtype=np.int64), classes
