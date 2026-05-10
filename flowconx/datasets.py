"""Dataset loading utilities for FlowCon-X."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import FlowConXConfig, infer_service, normalize_label
from .features import (
    augment_network_condition,
    condition_to_index,
    infer_condition,
    network_series_from_row,
    packet_sequence_from_row,
    row_get,
    stable_seed,
)


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass
class FlowRecord:
    """A single flow represented as model-ready arrays plus labels."""

    packet_seq: np.ndarray
    network_series: np.ndarray
    app: str
    service: str
    condition: str
    source: str = "unknown"


def detect_label_column(columns: Sequence[str]) -> str:
    candidates = ["service", "app", "application", "traffic_type", "label", "Label", "class", "category"]
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError("Could not detect a label column. Pass label_col explicitly.")


def load_csv_records(
    csv_path: str | Path,
    label_col: Optional[str] = None,
    app_col: Optional[str] = None,
    service_col: Optional[str] = None,
    source: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[FlowRecord]:
    """Load a generic flow CSV into FlowRecord objects."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    return records_from_dataframe(
        df,
        label_col=label_col,
        app_col=app_col,
        service_col=service_col,
        source=source or path.stem,
        limit=limit,
        seed_prefix=path.name,
    )


def records_from_dataframe(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    app_col: Optional[str] = None,
    service_col: Optional[str] = None,
    source: str = "dataframe",
    limit: Optional[int] = None,
    seed_prefix: str = "dataframe",
) -> List[FlowRecord]:
    """Convert an in-memory dataframe into FlowRecord objects."""
    if limit:
        df = df.head(limit)
    if label_col is None:
        label_col = detect_label_column(df.columns)
    records: List[FlowRecord] = []
    for row_idx, row in df.iterrows():
        row_dict = row.to_dict()
        raw_app = row_dict.get(app_col, row_dict.get(label_col, "unknown")) if app_col else row_dict.get(label_col, "unknown")
        app = normalize_label(raw_app)
        if service_col and service_col in row_dict:
            service = normalize_label(row_dict[service_col])
        else:
            service = infer_service(app)

        rtt = row_get(row_dict, ["rtt_ms", "rtt", "mean_rtt", "flow iat mean"], 35.0)
        jitter = row_get(row_dict, ["jitter_ms", "jitter", "flow iat std"], 5.0)
        loss = row_get(row_dict, ["loss_rate", "packet_loss", "loss"], 0.0)
        condition = normalize_label(row_dict.get("condition", infer_condition(rtt, jitter, loss)))
        rng = np.random.default_rng(stable_seed(seed_prefix, row_idx, app, service))
        packet_seq = packet_sequence_from_row(row_dict, rng=rng)
        network_series = network_series_from_row(row_dict, rng=rng)
        records.append(
            FlowRecord(
                packet_seq=packet_seq,
                network_series=network_series,
                app=app,
                service=service,
                condition=condition,
                source=source,
            )
        )
    return records


def build_label_maps(records: Sequence[FlowRecord], config: Optional[FlowConXConfig] = None) -> Dict[str, Dict[str, int]]:
    config = config or FlowConXConfig()
    apps = sorted({record.app for record in records})
    services = list(config.services)
    for service in sorted({record.service for record in records}):
        if service not in services:
            services.append(service)
    conditions = list(config.conditions)
    for condition in sorted({record.condition for record in records}):
        if condition not in conditions:
            conditions.append(condition)
    return {
        "app": {name: idx for idx, name in enumerate(apps)},
        "service": {name: idx for idx, name in enumerate(services)},
        "condition": {name: idx for idx, name in enumerate(conditions)},
    }


class FlowDataset(Dataset):
    """PyTorch dataset with optional counterfactual network augmentation."""

    def __init__(
        self,
        records: Sequence[FlowRecord],
        label_maps: Dict[str, Dict[str, int]],
        augment_count: int = 0,
        augment_seed: int = 7,
    ) -> None:
        self.samples: List[FlowRecord] = list(records)
        rng = np.random.default_rng(augment_seed)
        if augment_count > 0:
            augmented: List[FlowRecord] = []
            for record in records:
                for _ in range(augment_count):
                    pkt_aug, net_aug, condition = augment_network_condition(record.packet_seq, record.network_series, rng)
                    augmented.append(
                        FlowRecord(
                            packet_seq=pkt_aug,
                            network_series=net_aug,
                            app=record.app,
                            service=record.service,
                            condition=condition,
                            source=record.source + "_aug",
                        )
                    )
            self.samples.extend(augmented)
        self.label_maps = label_maps

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        if torch is None:
            raise RuntimeError("PyTorch is required to use FlowDataset.")
        record = self.samples[idx]
        app_id = self.label_maps["app"].get(record.app, 0)
        service_id = self.label_maps["service"].get(record.service, self.label_maps["service"].get("unknown", 0))
        condition_id = self.label_maps["condition"].get(record.condition, self.label_maps["condition"].get("unknown", 0))
        pkt_mask = np.isclose(record.packet_seq.sum(axis=1), 0.0)
        return {
            "packet_seq": torch.tensor(record.packet_seq, dtype=torch.float32),
            "network_series": torch.tensor(record.network_series, dtype=torch.float32),
            "packet_mask": torch.tensor(pkt_mask, dtype=torch.bool),
            "app_label": torch.tensor(app_id, dtype=torch.long),
            "service_label": torch.tensor(service_id, dtype=torch.long),
            "condition_label": torch.tensor(condition_id, dtype=torch.long),
            "app": record.app,
            "service": record.service,
            "condition": record.condition,
        }


def split_records(
    records: Sequence[FlowRecord],
    test_fraction: float = 0.2,
    seed: int = 42,
    stratify_by: str = "service",
) -> Tuple[List[FlowRecord], List[FlowRecord]]:
    """Small stratified split without requiring scikit-learn."""
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[FlowRecord]] = {}
    for record in records:
        key = getattr(record, stratify_by)
        groups.setdefault(key, []).append(record)
    train: List[FlowRecord] = []
    test: List[FlowRecord] = []
    for group_records in groups.values():
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_fraction))) if len(shuffled) > 1 else 0
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
