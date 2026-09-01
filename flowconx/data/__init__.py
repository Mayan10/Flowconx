"""Dataset loaders. One module per source, all returning the common Flow schema.

Every loader streams from the compressed archive in ``data/raw/`` and never
expands it. The archives are 3.2 GB and 21 GB; extracting them is not an
option on a normal machine, and streaming keeps the pipeline honest about
what it read.
"""

from .schema import (  # noqa: F401
    CANONICAL_COLUMNS,
    LABEL_COLUMNS,
    LEGACY_COLUMNS,
    MODEL_INPUT_COLUMNS,
    PROVENANCE_COLUMNS,
    SCALAR_COLUMNS,
    SEQUENCE_COLUMNS,
    make_flow_id,
    validate_row,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "LABEL_COLUMNS",
    "LEGACY_COLUMNS",
    "MODEL_INPUT_COLUMNS",
    "PROVENANCE_COLUMNS",
    "SCALAR_COLUMNS",
    "SEQUENCE_COLUMNS",
    "make_flow_id",
    "validate_row",
]
