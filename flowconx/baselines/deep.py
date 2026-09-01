"""Deep, non-pretrained baselines, trained on the identical splits.

These are re-implementations against our Flow schema, not the authors' code.
That is a deviation and it is recorded here and in the results JSON rather
than in a footnote: each class documents what it follows from the original and
what it necessarily changes, because our schema retains packet-size,
direction and inter-arrival series but not raw payload bytes.

All four consume the same :class:`~flowconx.features.packet.FeatureBundle` as
FlowCon-X and are trained with the same optimiser, batch size, epoch budget
and early-stopping rule, so a difference in the table is a difference in the
model rather than in the training budget. Reviewers punish unfair baseline
tuning harder than anything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineSpec:
    name: str
    reference: str
    deviation: str


class DeepPacketCNN(nn.Module):
    """DeepPacket-style 1D CNN (Lotfollahi et al., Soft Computing 2020).

    The original consumes the first 1500 raw bytes of a packet. We do not
    retain payload -- deliberately, since it is unavailable under QUIC with
    encrypted headers -- so the convolution runs over the per-packet feature
    sequence instead. This is the standard adaptation for header-only
    settings and is the same input FlowCon-X gets.
    """

    SPEC = BaselineSpec(
        name="deeppacket_cnn",
        reference="Lotfollahi et al., Deep Packet, Soft Computing 2020",
        deviation=(
            "Original consumes 1500 raw payload bytes per packet. Our schema retains no payload, so "
            "the 1D convolution runs over the same per-packet feature sequence FlowCon-X uses. "
            "Both models therefore see identical inputs."
        ),
    )

    MIN_LENGTH = 16

    def __init__(self, in_dim: int, n_classes: int, flow_dim: int = 0, hidden: int = 200, dropout: float = 0.05) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + flow_dim, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, n_classes),
        )
        self.flow_dim = flow_dim

    def forward(self, packet_seq, flow_features, packet_mask=None):
        x = packet_seq.masked_fill(packet_mask.unsqueeze(-1), 0.0) if packet_mask is not None else packet_seq
        # conv1 (k=4, s=3) then conv2 (k=5) needs floor((L-4)/3)+1 >= 5, i.e.
        # L >= 16. Short budgets are zero-padded rather than failing, so the
        # early-classification sweep can score this baseline at packet 1 like
        # everything else. The padding is masked-equivalent: padded positions
        # are already zeroed above.
        if x.shape[1] < self.MIN_LENGTH:
            x = F.pad(x, (0, 0, 0, self.MIN_LENGTH - x.shape[1]))
        pooled = self.conv(x.transpose(1, 2)).squeeze(-1)
        if self.flow_dim:
            pooled = torch.cat([pooled, flow_features], dim=-1)
        return self.head(pooled)


class FSNet(nn.Module):
    """FS-Net-style stacked bidirectional GRU (Liu et al., INFOCOM 2019).

    FS-Net embeds discretised packet lengths, encodes them with a stacked
    bi-GRU, and classifies from the concatenated final states. We keep the
    encoder and the classifier; we drop the reconstruction decoder, which is
    an auxiliary regulariser rather than part of the classification path.
    """

    SPEC = BaselineSpec(
        name="fsnet",
        reference="Liu et al., FS-Net, IEEE INFOCOM 2019",
        deviation=(
            "Encoder and classifier only. The original's autoencoder reconstruction branch is omitted; "
            "it regularises the length embedding and does not participate at inference."
        ),
    )

    def __init__(self, in_dim: int, n_classes: int, flow_dim: int = 0, hidden: int = 128, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(
            hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 * layers + flow_dim, hidden * 2),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )
        self.flow_dim = flow_dim

    def forward(self, packet_seq, flow_features, packet_mask=None):
        x = self.proj(packet_seq)
        _, hidden = self.gru(x)
        pooled = hidden.permute(1, 0, 2).reshape(x.shape[0], -1)
        if self.flow_dim:
            pooled = torch.cat([pooled, flow_features], dim=-1)
        return self.head(pooled)


class LSTMAttention(nn.Module):
    """Bi-LSTM with additive attention pooling, the standard sequence baseline."""

    SPEC = BaselineSpec(
        name="lstm_attention",
        reference="Standard bi-LSTM + attention sequence classifier",
        deviation="No published reference implementation; a conventional architecture at matched capacity.",
    )

    def __init__(self, in_dim: int, n_classes: int, flow_dim: int = 0, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + flow_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )
        self.flow_dim = flow_dim

    def forward(self, packet_seq, flow_features, packet_mask=None):
        tokens, _ = self.lstm(packet_seq)
        scores = self.attention(tokens).squeeze(-1)
        if packet_mask is not None:
            scores = scores.masked_fill(packet_mask, -1e4)
        pooled = torch.sum(tokens * torch.softmax(scores, dim=-1).unsqueeze(-1), dim=1)
        if self.flow_dim:
            pooled = torch.cat([pooled, flow_features], dim=-1)
        return self.head(pooled)


class MLPStats(nn.Module):
    """Flow-statistics MLP: the deep counterpart of the five-feature forest."""

    SPEC = BaselineSpec(
        name="mlp_stats",
        reference="Flow-statistics multilayer perceptron",
        deviation="Sees only the flow-level statistics vector, no packet sequence. A capacity control.",
    )

    def __init__(self, in_dim: int, n_classes: int, flow_dim: int = 0, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        _ = in_dim
        self.net = nn.Sequential(
            nn.Linear(flow_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, packet_seq, flow_features, packet_mask=None):
        return self.net(flow_features)


DEEP_BASELINES = {
    "deeppacket_cnn": DeepPacketCNN,
    "fsnet": FSNet,
    "lstm_attention": LSTMAttention,
    "mlp_stats": MLPStats,
}


def build_baseline(name: str, in_dim: int, flow_dim: int, n_classes: int) -> nn.Module:
    if name not in DEEP_BASELINES:
        raise ValueError(f"Unknown deep baseline {name!r}. Known: {sorted(DEEP_BASELINES)}")
    return DEEP_BASELINES[name](in_dim=in_dim, flow_dim=flow_dim, n_classes=n_classes)


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, matching the class_weight='balanced' rule
    every classical baseline uses. Without this the comparison would reward
    whichever model happened to have imbalance handling."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(labels) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_baseline(
    name: str,
    train_split,
    val_split,
    n_classes: int,
    seed: int = 0,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    early_stop_patience: Optional[int] = 5,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """Train one baseline under the same budget FlowCon-X gets."""
    from ..data.dataset import make_loader
    from ..determinism import seed_everything
    from ..metrics import macro_f1

    seed_everything(seed)
    device = device or torch.device("cpu")
    model = build_baseline(
        name,
        in_dim=train_split.features.packet_seq.shape[-1],
        flow_dim=train_split.features.flow_features.shape[-1],
        n_classes=n_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_split.labels, n_classes).to(device))
    loader = make_loader(train_split, batch_size=batch_size, shuffle=True, seed=seed)

    history: List[Dict[str, float]] = []
    best_score, best_state, patience = -1.0, None, early_stop_patience
    for epoch in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in loader:
            logits = model(
                batch["packet_seq"].to(device), batch["flow_features"].to(device), batch["packet_mask"].to(device)
            )
            loss = criterion(logits, batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan")}
        if val_split is not None and len(val_split) > 0:
            predictions = predict(model, val_split, device)
            score = macro_f1(predictions, val_split.labels, np.arange(n_classes))
            record["val_macro_f1"] = score
            if score > best_score:
                best_score, patience = score, early_stop_patience
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif patience is not None:
                patience -= 1
        history.append(record)
        if patience is not None and patience <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model: nn.Module, split, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    packet_seq = torch.from_numpy(split.features.packet_seq)
    packet_mask = torch.from_numpy(split.features.packet_mask)
    flow_features = torch.from_numpy(split.features.flow_features)
    out: List[np.ndarray] = []
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        logits = model(
            packet_seq[start:stop].to(device),
            flow_features[start:stop].to(device),
            packet_mask[start:stop].to(device),
        )
        out.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


@torch.no_grad()
def predict_scores(model: nn.Module, split, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Softmax probabilities, for the open-set rejection comparison."""
    model.eval()
    packet_seq = torch.from_numpy(split.features.packet_seq)
    packet_mask = torch.from_numpy(split.features.packet_mask)
    flow_features = torch.from_numpy(split.features.flow_features)
    out: List[np.ndarray] = []
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        logits = model(
            packet_seq[start:stop].to(device),
            flow_features[start:stop].to(device),
            packet_mask[start:stop].to(device),
        )
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 1))
