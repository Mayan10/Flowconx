"""Split protocols and committed split manifests.

The field's usual protocol -- a stratified random split over flows -- is
implemented here as ``random_flow`` so that it can be *reported as a
contrast*, not used as the headline. Anything that groups correlated flows
(same capture session, same server, same application, same source dataset)
into a single side of the split is a stricter protocol and is what the
headline table must use.

Every split is materialised as a manifest: an explicit list of flow IDs per
side plus a SHA256 over that list, so a reviewer can verify that the numbers
in the paper were produced on the splits committed to the repository.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Columns a dataset loader should emit so that strict split protocols are
# possible. Absence of one of these does not corrupt the data; it makes a
# whole family of split protocols unavailable, which is a finding in itself.
PROVENANCE_COLUMNS = {
    "flow_id": "Stable unique identifier for one flow / observation window.",
    "origin": "Name of the source dataset the row came from.",
    "capture_id": "Capture session, pcap file, or trace the row was extracted from.",
    "flow_start_ts": "Unix timestamp (seconds) of the first packet of the flow.",
    "server_ip": "Server-side address, used only for server-disjoint splitting.",
}

SPLIT_PROTOCOLS = (
    "random_flow",
    "session_disjoint",
    "temporal",
    "server_disjoint",
    "app_disjoint",
    "origin_disjoint",
)

# Which provenance column each protocol groups on. ``random_flow`` groups on
# nothing, which is exactly the problem with it.
_GROUP_COLUMN = {
    "random_flow": None,
    "session_disjoint": "capture_id",
    "temporal": "flow_start_ts",
    "server_disjoint": "server_ip",
    "app_disjoint": "app",
    "origin_disjoint": "origin",
}


class SplitUnavailable(RuntimeError):
    """Raised when a split protocol cannot be honoured by the given table.

    This is deliberately a hard error. Silently degrading a session-disjoint
    request to a random split is precisely the failure mode that produces an
    unreproducible headline number.
    """


@dataclass
class SplitManifest:
    protocol: str
    seed: int
    val_fraction: float
    test_fraction: float
    dataset_path: str
    dataset_sha256: str
    n_rows: int
    group_column: Optional[str]
    flow_id_synthesized: bool
    train: List[str] = field(default_factory=list)
    val: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)
    class_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def sha256_of_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_flow_ids(df: pd.DataFrame) -> pd.Series:
    """Deterministic per-row IDs derived from row content.

    Used only when the table carries no ``flow_id``. Two byte-identical rows
    collapse to the same ID, which is intentional: it makes exact duplicates
    visible to the leakage checks rather than hiding them behind row numbers.
    """
    payload = df.astype(str).agg("\x1f".join, axis=1)
    return payload.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()[:32])


def ensure_flow_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Return ``df`` with a usable ``flow_id`` column and whether it was made up."""
    if "flow_id" in df.columns and df["flow_id"].notna().all():
        out = df.copy()
        out["flow_id"] = out["flow_id"].astype(str)
        return out, False
    out = df.copy()
    base = content_flow_ids(out)
    # Disambiguate genuine duplicates with an occurrence counter so that
    # flow_id stays unique while the shared prefix still exposes the duplicate.
    occurrence = base.groupby(base).cumcount().astype(str)
    out["flow_id"] = base.str.cat(occurrence, sep="#")
    return out, True


def describe_provenance(df: pd.DataFrame) -> Dict[str, bool]:
    """Which provenance columns this table actually carries."""
    return {name: name in df.columns and df[name].notna().any() for name in PROVENANCE_COLUMNS}


def available_protocols(df: pd.DataFrame) -> Dict[str, bool]:
    present = describe_provenance(df)
    out: Dict[str, bool] = {}
    for protocol in SPLIT_PROTOCOLS:
        column = _GROUP_COLUMN[protocol]
        if column is None:
            out[protocol] = True
        elif column == "app":
            out[protocol] = "app" in df.columns
        else:
            out[protocol] = bool(present.get(column, False))
    return out


def _stratified_random(
    labels: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_fraction))
        n_val = int(round(len(idx) * val_fraction))
        # A class with a single member cannot be present on both sides; it
        # goes to train so that the class exists for the model at all.
        if len(idx) == 1:
            n_test = n_val = 0
        test.extend(idx[:n_test].tolist())
        val.extend(idx[n_test : n_test + n_val].tolist())
        train.extend(idx[n_test + n_val :].tolist())
    return (
        np.asarray(sorted(train), dtype=int),
        np.asarray(sorted(val), dtype=int),
        np.asarray(sorted(test), dtype=int),
    )


def _group_disjoint(
    groups: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign whole groups to sides, greedily balancing the label mix.

    Groups are visited largest-first and each is placed on whichever side is
    furthest below its target row count. This keeps the split sizes close to
    the requested fractions without ever splitting a group, which is the only
    property that matters for leakage.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise SplitUnavailable(
            f"Group-disjoint splitting needs at least 3 distinct groups to fill train/val/test, "
            f"but the grouping column has only {len(unique_groups)}: {list(unique_groups)[:5]}. "
            "A single-origin table cannot be split by origin."
        )
    order = sorted(unique_groups, key=lambda g: (-int(np.sum(groups == g)), str(g)))
    total = len(groups)
    targets = {
        "train": total * (1.0 - val_fraction - test_fraction),
        "val": total * val_fraction,
        "test": total * test_fraction,
    }
    assigned: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    # Break ties in a seeded but stable way so equal-size groups do not always
    # land on the same side across seeds.
    jitter = {str(g): float(rng.random()) for g in order}
    for group in order:
        idx = np.flatnonzero(groups == group)
        deficits = {
            side: (targets[side] - len(assigned[side])) / max(targets[side], 1.0) + 1e-6 * jitter[str(group)]
            for side in assigned
        }
        side = max(deficits, key=lambda s: deficits[s])
        if targets[side] <= 0:
            side = "train"
        assigned[side].extend(idx.tolist())
    train = np.asarray(sorted(assigned["train"]), dtype=int)
    val = np.asarray(sorted(assigned["val"]), dtype=int)
    test = np.asarray(sorted(assigned["test"]), dtype=int)
    _ = labels  # label mix is reported by the manifest, not enforced here
    return train, val, test


def _temporal(
    timestamps: np.ndarray,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(timestamps, kind="stable")
    n = len(order)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    n_train = n - n_val - n_test
    return (
        np.sort(order[:n_train]).astype(int),
        np.sort(order[n_train : n_train + n_val]).astype(int),
        np.sort(order[n_train + n_val :]).astype(int),
    )


def build_split(
    df: pd.DataFrame,
    protocol: str,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    label_column: str = "service",
    dataset_path: str | Path | None = None,
) -> Tuple[SplitManifest, Dict[str, np.ndarray]]:
    """Build a split under ``protocol`` and return its manifest and indices."""
    if protocol not in SPLIT_PROTOCOLS:
        raise ValueError(f"Unknown split protocol {protocol!r}. Known: {list(SPLIT_PROTOCOLS)}")

    frame, synthesized = ensure_flow_ids(df)
    rng = np.random.default_rng(seed)
    labels = frame[label_column].astype(str).to_numpy()
    group_column = _GROUP_COLUMN[protocol]
    notes: List[str] = []

    if protocol == "random_flow":
        notes.append(
            "Stratified random split over individual flows. Reported only as a contrast column: "
            "correlated flows from the same capture session appear on both sides."
        )
        train, val, test = _stratified_random(labels, val_fraction, test_fraction, rng)
    elif protocol == "temporal":
        if "flow_start_ts" not in frame.columns or frame["flow_start_ts"].isna().all():
            raise SplitUnavailable(
                "Temporal splitting needs a 'flow_start_ts' column. The current preprocessed CSV "
                "does not retain packet timestamps; regenerate it with scripts/prepare_*.py after "
                "the provenance columns are emitted."
            )
        timestamps = pd.to_numeric(frame["flow_start_ts"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(timestamps).all():
            raise SplitUnavailable("'flow_start_ts' contains non-numeric or missing values.")
        notes.append("Train on the earliest flows, test on the latest. No shuffling.")
        train, val, test = _temporal(timestamps, val_fraction, test_fraction)
    else:
        assert group_column is not None
        if group_column not in frame.columns or frame[group_column].isna().all():
            raise SplitUnavailable(
                f"Split protocol {protocol!r} groups on column {group_column!r}, which this table "
                f"does not carry. Available provenance columns: "
                f"{sorted(k for k, v in describe_provenance(frame).items() if v)}."
            )
        groups = frame[group_column].astype(str).to_numpy()
        notes.append(f"Whole {group_column!r} groups are assigned to exactly one side.")
        train, val, test = _group_disjoint(groups, labels, val_fraction, test_fraction, rng)

    indices = {"train": train, "val": val, "test": test}
    flow_ids = frame["flow_id"].to_numpy()
    manifest = SplitManifest(
        protocol=protocol,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        dataset_path=str(dataset_path) if dataset_path else "<in-memory>",
        dataset_sha256=sha256_of_file(dataset_path) if dataset_path else "",
        n_rows=len(frame),
        group_column=group_column,
        flow_id_synthesized=synthesized,
        train=flow_ids[train].tolist(),
        val=flow_ids[val].tolist(),
        test=flow_ids[test].tolist(),
        notes=notes,
    )
    manifest.checksums = {side: sha256_of_strings(getattr(manifest, side)) for side in ("train", "val", "test")}
    manifest.class_counts = {
        side: {
            str(label): int(count)
            for label, count in zip(*np.unique(labels[indices[side]], return_counts=True))
        }
        for side in ("train", "val", "test")
    }
    if synthesized:
        manifest.notes.append(
            "flow_id was synthesized from row content because the table carries none. "
            "Rows with identical content share an ID prefix, which the leakage tests use "
            "to detect cross-split duplicates."
        )
    return manifest, indices


def write_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    """Write a manifest, gzipping when the path ends in ``.gz``.

    A manifest for 112k flows is ~4 MB of JSON, which is more than a reviewer
    wants in a git diff. Gzipped it is ~1 MB and still plain ``zcat``-able,
    so the committed split stays inspectable without bloating the history.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.as_dict(), indent=2, sort_keys=True)
    if out.suffix == ".gz":
        with gzip.open(out, "wt", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        out.write_text(payload, encoding="utf-8")
    return out


def load_manifest(path: str | Path) -> SplitManifest:
    target = Path(path)
    if target.suffix == ".gz":
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            raw = handle.read()
    else:
        raw = target.read_text(encoding="utf-8")
    payload: Mapping[str, object] = json.loads(raw)
    known = {f for f in SplitManifest.__dataclass_fields__}
    return SplitManifest(**{k: v for k, v in payload.items() if k in known})


def indices_from_manifest(df: pd.DataFrame, manifest: SplitManifest) -> Dict[str, np.ndarray]:
    """Recover row indices for a committed manifest, verifying the checksums."""
    frame, _ = ensure_flow_ids(df)
    position = {flow_id: i for i, flow_id in enumerate(frame["flow_id"].astype(str))}
    out: Dict[str, np.ndarray] = {}
    for side in ("train", "val", "test"):
        ids = getattr(manifest, side)
        expected = manifest.checksums.get(side)
        if expected and sha256_of_strings(ids) != expected:
            raise ValueError(f"Manifest checksum mismatch for split side {side!r}.")
        missing = [flow_id for flow_id in ids if flow_id not in position]
        if missing:
            raise ValueError(
                f"{len(missing)} flow IDs from the {side!r} manifest are absent from the table; "
                "the dataset does not match the committed split."
            )
        out[side] = np.asarray([position[flow_id] for flow_id in ids], dtype=int)
    return out
