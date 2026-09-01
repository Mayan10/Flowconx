"""FlowCon-X encoders, fusion and heads.

Every structural choice the ablation table needs is a constructor argument, so
that a row of that table is a config override rather than a code branch:

* ``dual_encoder=False`` collapses to one shared encoder over the sequence
  with the flow-context vector appended to every position.
* ``fusion`` selects cross-attention, concatenation, a gated sum, or late
  fusion, so the "minus cross-attention" rows are real alternatives rather
  than the component simply deleted.
* ``adversarial_head=False`` removes the gradient-reversal branch entirely.
* ``width_multiplier`` scales hidden widths so ablations can be matched on
  parameter count -- a win that is only more parameters is not a win.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.packet import FLOW_FEATURE_DIM, PKT_FEATURE_DIM


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:  # type: ignore[override]
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReverseFn.apply(x, scale)


def scaled(dim: int, multiplier: float) -> int:
    """Scale a hidden width, keeping it a multiple of 8 for head divisibility."""
    return max(8, int(round(dim * multiplier / 8.0)) * 8)


class TemporalConvBlock(nn.Module):
    """Gated depthwise-separable convolution over the packet axis."""

    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.silu(self.pointwise(y)).transpose(1, 2)
        return residual + self.dropout(y * torch.sigmoid(self.gate(residual)))


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(tokens).squeeze(-1)
        if mask is not None:
            # -inf would produce NaN for an all-padded row; a large finite
            # value keeps the softmax defined even in that degenerate case.
            scores = scores.masked_fill(mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(tokens * weights, dim=1)


class SequenceEncoder(nn.Module):
    """Per-packet sequence -> embedding. Convolution stack then a transformer."""

    def __init__(
        self,
        in_dim: int = PKT_FEATURE_DIM,
        hidden_dim: int = 192,
        out_dim: int = 256,
        n_conv: int = 3,
        n_heads: int = 6,
        n_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        # Sinusoidal positions: packet order is the signal, and a learned table
        # would tie the model to one sequence length, which the input-budget
        # sweep varies.
        self.register_buffer("positions", _sinusoidal(max_len, hidden_dim), persistent=False)
        self.conv_blocks = nn.ModuleList([TemporalConvBlock(hidden_dim, dropout=dropout) for _ in range(n_conv)])
        while hidden_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = AttentionPooling(hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(sequence) + self.positions[: sequence.shape[1]].unsqueeze(0)
        for block in self.conv_blocks:
            x = block(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        pooled = self.pool(x, mask)
        return x, F.normalize(self.output(pooled), dim=-1)


class ContextEncoder(nn.Module):
    """Flow-level statistics -> embedding.

    The original network encoder read a constant vector with Gaussian drift
    stamped over it, which is not a time series and carried the nuisance label
    in one of its channels. This reads the flow-level statistics directly, as
    a second genuine view of the same flow.
    """

    def __init__(self, in_dim: int = FLOW_FEATURE_DIM, hidden_dim: int = 128, out_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))

    def forward(self, flow_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(flow_features)
        return hidden.unsqueeze(1), F.normalize(self.output(hidden), dim=-1)


class Fusion(nn.Module):
    """Combine the sequence and context views into the deployed embedding."""

    def __init__(
        self,
        mode: str,
        seq_hidden_dim: int,
        ctx_hidden_dim: int,
        seq_emb_dim: int,
        ctx_emb_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "cross_attention":
            while out_dim % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            self.query_proj = nn.Linear(seq_hidden_dim, out_dim)
            self.key_proj = nn.Linear(ctx_hidden_dim, out_dim)
            self.value_proj = nn.Linear(ctx_hidden_dim, out_dim)
            self.cross_attn = nn.MultiheadAttention(out_dim, n_heads, dropout=dropout, batch_first=True)
            self.pool = AttentionPooling(out_dim)
            fused_in = out_dim + seq_emb_dim
        elif mode == "concat":
            fused_in = seq_emb_dim + ctx_emb_dim
        elif mode == "gated_sum":
            self.gate = nn.Linear(seq_emb_dim + ctx_emb_dim, seq_emb_dim)
            self.ctx_to_seq = nn.Linear(ctx_emb_dim, seq_emb_dim)
            fused_in = seq_emb_dim
        elif mode == "late":
            # Late fusion: the two embeddings are never mixed before the head;
            # z_flow is their concatenation with no learned interaction.
            fused_in = seq_emb_dim + ctx_emb_dim
        elif mode == "none":
            fused_in = seq_emb_dim
        else:
            raise ValueError(f"Unknown fusion mode {mode!r}")

        if mode == "late":
            self.output = nn.Identity()
            self.out_dim = fused_in
        else:
            self.output = nn.Sequential(
                nn.LayerNorm(fused_in),
                nn.Linear(fused_in, out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )
            self.out_dim = out_dim

    def forward(
        self,
        seq_tokens: torch.Tensor,
        z_seq: torch.Tensor,
        ctx_tokens: Optional[torch.Tensor],
        z_ctx: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "none" or z_ctx is None:
            return F.normalize(self.output(z_seq), dim=-1)
        if self.mode == "cross_attention":
            assert ctx_tokens is not None
            context, _ = self.cross_attn(
                self.query_proj(seq_tokens), self.key_proj(ctx_tokens), self.value_proj(ctx_tokens)
            )
            pooled = self.pool(context, mask)
            return F.normalize(self.output(torch.cat([z_seq, pooled], dim=-1)), dim=-1)
        if self.mode in ("concat", "late"):
            return F.normalize(self.output(torch.cat([z_seq, z_ctx], dim=-1)), dim=-1)
        # gated_sum
        gate = torch.sigmoid(self.gate(torch.cat([z_seq, z_ctx], dim=-1)))
        return F.normalize(self.output(gate * z_seq + (1.0 - gate) * self.ctx_to_seq(z_ctx)), dim=-1)


class NuisanceAdversary(nn.Module):
    """Gradient-reversal head predicting a *real* nuisance variable.

    The previous adversary predicted a label that was a threshold on two of
    the model's own input features, so removing it was circular (AUDIT.md 3,
    L5). The nuisance here comes from provenance the model never sees --
    capture session, week, or server AS -- which makes removal a substantive
    claim and the Phase 3.2 probe a real measurement.
    """

    def __init__(self, emb_dim: int, n_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))

    def forward(self, embedding: torch.Tensor, grl_scale: float = 1.0) -> torch.Tensor:
        return self.net(gradient_reverse(embedding, grl_scale))


def _sinusoidal(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    out = torch.zeros(length, dim)
    out[:, 0::2] = torch.sin(position * div)
    out[:, 1::2] = torch.cos(position * div[: out[:, 1::2].shape[1]])
    return out


class FlowConX(nn.Module):
    """The full model, assembled from an :class:`~flowconx.experiment.ModelConfig`."""

    def __init__(
        self,
        packet_dim: int = PKT_FEATURE_DIM,
        flow_dim: int = FLOW_FEATURE_DIM,
        n_nuisance: int = 0,
        seq_hidden_dim: int = 192,
        ctx_hidden_dim: int = 128,
        seq_emb_dim: int = 256,
        ctx_emb_dim: int = 128,
        flow_emb_dim: int = 256,
        dropout: float = 0.1,
        n_conv: int = 3,
        n_heads: int = 6,
        n_layers: int = 1,
        dual_encoder: bool = True,
        fusion: str = "cross_attention",
        adversarial_head: bool = True,
        width_multiplier: float = 1.0,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        seq_hidden_dim = scaled(seq_hidden_dim, width_multiplier)
        ctx_hidden_dim = scaled(ctx_hidden_dim, width_multiplier)
        self.dual_encoder = dual_encoder
        self.fusion_mode = fusion if dual_encoder else "none"

        # Without the dual encoder there is one encoder, and the flow-level
        # context is appended to every packet position instead of being read
        # by a second tower. Same information, one pathway.
        encoder_in = packet_dim if dual_encoder else packet_dim + flow_dim
        self.sequence_encoder = SequenceEncoder(
            in_dim=encoder_in,
            hidden_dim=seq_hidden_dim,
            out_dim=seq_emb_dim,
            n_conv=n_conv,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.context_encoder = (
            ContextEncoder(in_dim=flow_dim, hidden_dim=ctx_hidden_dim, out_dim=ctx_emb_dim, dropout=dropout)
            if dual_encoder
            else None
        )
        self.fusion = Fusion(
            mode=self.fusion_mode,
            seq_hidden_dim=seq_hidden_dim,
            ctx_hidden_dim=ctx_hidden_dim,
            seq_emb_dim=seq_emb_dim,
            ctx_emb_dim=ctx_emb_dim,
            out_dim=flow_emb_dim,
            dropout=dropout,
        )
        self.flow_emb_dim = self.fusion.out_dim
        self.adversary = (
            NuisanceAdversary(self.flow_emb_dim, n_nuisance) if adversarial_head and n_nuisance > 1 else None
        )

    def forward(
        self,
        packet_seq: torch.Tensor,
        flow_features: torch.Tensor,
        packet_mask: Optional[torch.Tensor] = None,
        grl_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if self.dual_encoder:
            seq_input = packet_seq
        else:
            broadcast = flow_features.unsqueeze(1).expand(-1, packet_seq.shape[1], -1)
            seq_input = torch.cat([packet_seq, broadcast], dim=-1)

        seq_tokens, z_app = self.sequence_encoder(seq_input, packet_mask)
        if self.context_encoder is not None:
            ctx_tokens, z_network = self.context_encoder(flow_features)
        else:
            ctx_tokens, z_network = None, None
        z_flow = self.fusion(seq_tokens, z_app, ctx_tokens, z_network, packet_mask)

        out: Dict[str, torch.Tensor] = {"z_app": z_app, "z_flow": z_flow}
        if z_network is not None:
            out["z_network"] = z_network
            out["z_concat"] = F.normalize(torch.cat([z_app, z_network], dim=-1), dim=-1)
        if self.adversary is not None:
            out["nuisance_logits"] = self.adversary(z_flow, grl_scale=grl_scale)
        return out

    def embedding(self, outputs: Dict[str, torch.Tensor], which: str = "z_flow") -> torch.Tensor:
        if which not in outputs:
            raise KeyError(
                f"Embedding {which!r} is not produced by this configuration "
                f"(available: {sorted(outputs)}). A z_network or z_concat head needs dual_encoder=true."
            )
        return outputs[which]

    @torch.no_grad()
    def encode(
        self,
        packet_seq: torch.Tensor,
        flow_features: torch.Tensor,
        packet_mask: Optional[torch.Tensor] = None,
        which: str = "z_flow",
    ) -> torch.Tensor:
        return self.embedding(self.forward(packet_seq, flow_features, packet_mask, grl_scale=0.0), which)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(model_config, n_nuisance: int, max_len: int) -> FlowConX:
    """Instantiate from an :class:`~flowconx.experiment.ModelConfig`."""
    return FlowConX(
        n_nuisance=n_nuisance,
        seq_hidden_dim=model_config.app_hidden_dim,
        ctx_hidden_dim=model_config.net_hidden_dim,
        seq_emb_dim=model_config.app_emb_dim,
        ctx_emb_dim=model_config.net_emb_dim,
        flow_emb_dim=model_config.flow_emb_dim,
        dropout=model_config.dropout,
        n_conv=model_config.n_conv_blocks,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_transformer_layers,
        dual_encoder=model_config.dual_encoder,
        fusion=model_config.fusion,
        adversarial_head=model_config.adversarial_head,
        width_multiplier=model_config.width_multiplier,
        max_len=max_len,
    )
