"""Shortcut, leakage and split-protocol auditing for FlowCon-X.

This package exists to answer one question before any headline number is
believed: *is the model learning application behaviour, or is it learning a
property of how the dataset was assembled?*

Nothing in here imports :mod:`torch`. The audit must be runnable (and must
fail CI) on a machine with no GPU and no deep-learning stack.
"""

from .splits import (
    SPLIT_PROTOCOLS,
    SplitManifest,
    SplitUnavailable,
    build_split,
    load_manifest,
    write_manifest,
)

__all__ = [
    "SPLIT_PROTOCOLS",
    "SplitManifest",
    "SplitUnavailable",
    "build_split",
    "load_manifest",
    "write_manifest",
]
