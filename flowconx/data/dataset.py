"""Assembling a split into tensors.

Everything downstream of this module sees only what the config declares as a
model input. The provenance columns travel alongside as *labels* -- the
nuisance variable the adversary is asked to remove, and the group keys the
split protocols use -- but never enter the feature tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..features.packet import FeatureBundle, build_features

# Which provenance column plays the role of the nuisance variable the
# adversarial head is asked to remove. All of these are genuinely invisible to
# the model, unlike the previous `condition` label, which was a threshold on
# two of its own input features (AUDIT.md 3, L5).
NUISANCE_SOURCES = {
    "capture_id": "Which capture session the flow came from.",
    "week": "Which ISO week the flow was observed in (CESNET only).",
    "server_asn": "Autonomous system of the server (CESNET only).",
    "origin": "Which source dataset the flow came from.",
    "none": "No adversarial nuisance removal.",
}


@dataclass
class EncodedSplit:
    """One side of a split, as arrays."""

    features: FeatureBundle
    labels: np.ndarray
    app_labels: np.ndarray
    nuisance_labels: np.ndarray
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.features)


@dataclass
class LabelSpace:
    classes: List[str]
    apps: List[str]
    nuisance_values: List[str]
    nuisance_source: str

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_apps(self) -> int:
        return len(self.apps)

    @property
    def n_nuisance(self) -> int:
        return len(self.nuisance_values)

    def as_dict(self) -> Dict[str, object]:
        return {
            "classes": self.classes,
            "n_apps": self.n_apps,
            "nuisance_source": self.nuisance_source,
            "n_nuisance": self.n_nuisance,
        }


def nuisance_series(frame: pd.DataFrame, source: str) -> pd.Series:
    """The nuisance variable, derived from provenance the model never sees."""
    if source == "none":
        return pd.Series(["none"] * len(frame), index=frame.index)
    if source == "week":
        # capture_id is "W-2022-44/20221031" for CESNET; take the week part.
        if "capture_id" not in frame.columns:
            raise ValueError("nuisance source 'week' needs a capture_id column")
        return frame["capture_id"].astype(str).str.split("/").str[0]
    if source == "server_asn":
        if "dst_asn" not in frame.columns:
            raise ValueError("nuisance source 'server_asn' needs a dst_asn column")
        return frame["dst_asn"].astype(str).fillna("unknown").replace("", "unknown")
    if source not in frame.columns:
        raise ValueError(f"nuisance source {source!r} is not a column of this dataset")
    return frame[source].astype(str)


def encode_labels(values: Sequence[str], classes: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    order = list(classes) if classes is not None else sorted(set(str(v) for v in values))
    index: Mapping[str, int] = {name: i for i, name in enumerate(order)}
    # Values absent from the fitted order map to -1 rather than silently to 0,
    # so an unseen class in the test split is visible instead of being counted
    # as the first class.
    encoded = np.asarray([index.get(str(v), -1) for v in values], dtype=np.int64)
    return encoded, order


def build_label_space(
    frame: pd.DataFrame,
    label_column: str,
    nuisance_source: str,
) -> LabelSpace:
    return LabelSpace(
        classes=sorted(frame[label_column].astype(str).unique()),
        apps=sorted(frame["app"].astype(str).unique()) if "app" in frame.columns else [],
        nuisance_values=sorted(nuisance_series(frame, nuisance_source).unique()),
        nuisance_source=nuisance_source,
    )


def encode_split(
    frame: pd.DataFrame,
    label_space: LabelSpace,
    label_column: str,
    max_packets: int,
    observed_packets: Optional[int] = None,
) -> EncodedSplit:
    """Featurise one side of a split and align its labels to the kept rows."""
    bundle = build_features(frame, max_packets=max_packets, observed_packets=observed_packets)
    kept = frame.iloc[bundle.kept_index]
    labels, _ = encode_labels(kept[label_column].astype(str), label_space.classes)
    apps, _ = (
        encode_labels(kept["app"].astype(str), label_space.apps)
        if label_space.apps
        else (np.zeros(len(kept), dtype=np.int64), [])
    )
    nuisance, _ = encode_labels(
        nuisance_series(kept, label_space.nuisance_source), label_space.nuisance_values
    )
    # An unseen nuisance value in a held-out split is expected under a
    # session-disjoint protocol; it maps to a sentinel the adversary ignores.
    nuisance = np.where(nuisance < 0, 0, nuisance)
    return EncodedSplit(
        features=bundle,
        labels=labels,
        app_labels=apps,
        nuisance_labels=nuisance,
        frame=kept.reset_index(drop=True),
    )


def _torch_dataset_base():
    """Return ``torch.utils.data.Dataset`` if torch is installed, else object.

    The audit and metric layers must import this module without torch, so the
    base class is resolved lazily rather than at import time.
    """
    try:
        from torch.utils.data import Dataset

        return Dataset
    except ImportError:  # pragma: no cover
        return object


class TensorFlowDataset(_torch_dataset_base()):  # type: ignore[misc]
    """Torch Dataset over the encoded arrays.

    Deliberately not holding a DataFrame: the arrays are contiguous and small,
    and keeping pandas out of the worker processes avoids a large fork cost.
    """

    def __init__(self, split: EncodedSplit) -> None:
        import torch

        self.packet_seq = torch.from_numpy(np.ascontiguousarray(split.features.packet_seq))
        self.packet_mask = torch.from_numpy(np.ascontiguousarray(split.features.packet_mask))
        self.flow_features = torch.from_numpy(np.ascontiguousarray(split.features.flow_features))
        self.labels = torch.from_numpy(split.labels)
        self.app_labels = torch.from_numpy(split.app_labels)
        self.nuisance_labels = torch.from_numpy(split.nuisance_labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "packet_seq": self.packet_seq[idx],
            "packet_mask": self.packet_mask[idx],
            "flow_features": self.flow_features[idx],
            "label": self.labels[idx],
            "app_label": self.app_labels[idx],
            "nuisance_label": self.nuisance_labels[idx],
        }


def make_loader(split: EncodedSplit, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0):
    import torch
    from torch.utils.data import DataLoader

    from ..determinism import worker_init_fn

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorFlowDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers else None,
        drop_last=False,
    )
