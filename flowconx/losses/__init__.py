"""Training objectives. Every term is independently toggleable from config."""

from .objectives import (  # noqa: F401
    CrossCovarianceDisentanglement,
    FlowConXLoss,
    PairwiseEmbeddingMarginLoss,
    PrototypeAlignmentLoss,
    SupervisedContrastiveLoss,
)

__all__ = [
    "CrossCovarianceDisentanglement",
    "FlowConXLoss",
    "PairwiseEmbeddingMarginLoss",
    "PrototypeAlignmentLoss",
    "SupervisedContrastiveLoss",
]
