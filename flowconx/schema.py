"""The canonical Flow schema shared by every dataset preparer.

Two groups of columns:

*Behaviour* -- what the model is allowed to see. Packet size, direction and
inter-arrival series plus flow-level scalars.

*Provenance* -- what the model must never see, but which the split protocols
need in order to keep correlated observations on one side of the split. The
original preparers emitted none of these, which made session-disjoint and
temporal evaluation impossible; that is the finding recorded in AUDIT.md and
this module is the fix.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

# Identifiers and provenance. Never model inputs.
PROVENANCE_COLUMNS: List[str] = [
    "flow_id",
    "origin",
    "capture_id",
    "flow_start_ts",
    "server_ip",
]

# Labels and declared nuisance variable.
LABEL_COLUMNS: List[str] = ["app", "service", "condition"]

# Per-packet series, ';'-separated, truncated to the preparer's max_packets.
SEQUENCE_COLUMNS: List[str] = ["packet_lengths", "iat_values", "directions"]

# Flow-level scalars.
SCALAR_COLUMNS: List[str] = [
    "rtt_ms",
    "jitter_ms",
    "loss_rate",
    "total packets",
    "total fwd packets",
    "total backward packets",
    "packet length mean",
    "packet length std",
    "flow iat mean",
    "flow iat std",
    "flow duration",
    "flow bytes/s",
    "flow packets/s",
    "protocol",
]

CANONICAL_COLUMNS: List[str] = PROVENANCE_COLUMNS + LABEL_COLUMNS + SEQUENCE_COLUMNS + SCALAR_COLUMNS

# Columns a model may legitimately consume. Anything outside this set that
# reaches a feature extractor is a leak.
MODEL_INPUT_COLUMNS: List[str] = SEQUENCE_COLUMNS + SCALAR_COLUMNS

# Legacy tables written before provenance existed. Loaders accept these but
# the audit records that strict split protocols are unavailable for them.
LEGACY_COLUMNS: List[str] = LABEL_COLUMNS + SEQUENCE_COLUMNS + SCALAR_COLUMNS


def make_flow_id(origin: str, capture_id: str, index: int) -> str:
    """Stable, short, collision-resistant flow identifier."""
    payload = f"{origin}\x1f{capture_id}\x1f{index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def blank_provenance() -> Dict[str, object]:
    return {name: "" for name in PROVENANCE_COLUMNS}


def validate_row(row: Dict[str, object], columns: Sequence[str] = CANONICAL_COLUMNS) -> Dict[str, object]:
    """Fill in absent canonical columns and reject unknown ones."""
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ValueError(f"Row carries columns outside the canonical schema: {unknown}")
    return {name: row.get(name, "") for name in columns}
