"""Deterministic feature extraction.

`legacy` holds the original row-to-tensor path, kept so that results produced
before the restructure remain reproducible. `packet` holds the rewritten,
unit-tested extractor used by everything new.
"""

from .legacy import (  # noqa: F401
    augment_network_condition,
    condition_to_index,
    infer_condition,
    network_series_from_row,
    packet_sequence_from_row,
    pad_or_trim,
    parse_series,
    row_get,
    row_text,
    stable_seed,
)

__all__ = [
    "augment_network_condition",
    "condition_to_index",
    "infer_condition",
    "network_series_from_row",
    "packet_sequence_from_row",
    "pad_or_trim",
    "parse_series",
    "row_get",
    "row_text",
    "stable_seed",
]
