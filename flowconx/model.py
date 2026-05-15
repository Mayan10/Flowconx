from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import APP_EMB_DIM, FLOW_EMB_DIM, NET_EMB_DIM, NET_FEAT_DIM, PKT_FEAT_DIM


class GradientReverseFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReverseFn.apply(x, scale)


class TemporalConvBlock(nn.Module):

    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, groups=hidden_dim)
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.depthwise(y)
        y = F.silu(self.pointwise(y))
        y = y.transpose(1, 2)
        gate = torch.sigmoid(self.gate(residual))
        return residual + self.dropout(y * gate)


class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(tokens).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(tokens * weights, dim=1)


class ApplicationIdentityEncoder(nn.Module):

    def __init__(
        self,
        pkt_feat_dim: int = PKT_FEAT_DIM,
        hidden_dim: int = 192,
        out_dim: int = APP_EMB_DIM,
        n_conv: int = 3,
        n_heads: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(pkt_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.conv_blocks = nn.ModuleList([TemporalConvBlock(hidden_dim, dropout=dropout) for _ in range(n_conv)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = AttentionPooling(hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, packet_seq: torch.Tensor, packet_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(packet_seq)
        for block in self.conv_blocks:
            x = block(x)
        x = self.transformer(x, src_key_padding_mask=packet_mask)
        pooled = self.pool(x, packet_mask)
        z_app = self.output(pooled)
        return x, F.normalize(z_app, dim=-1)


class NetworkConditionEncoder(nn.Module):

    def __init__(
        self,
        net_feat_dim: int = NET_FEAT_DIM,
        hidden_dim: int = 128,
        out_dim: int = NET_EMB_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(net_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentionPooling(hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, network_series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(network_series)
        tokens, _ = self.gru(x)
        pooled = self.pool(tokens)
        z_net = self.output(pooled)
        return tokens, F.normalize(z_net, dim=-1)


class ContextFusion(nn.Module):

    def __init__(
        self,
        app_hidden_dim: int = 192,
        net_hidden_dim: int = 128,
        app_emb_dim: int = APP_EMB_DIM,
        fused_dim: int = FLOW_EMB_DIM,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(app_hidden_dim, fused_dim)
        self.key_proj = nn.Linear(net_hidden_dim, fused_dim)
        self.value_proj = nn.Linear(net_hidden_dim, fused_dim)
        self.cross_attn = nn.MultiheadAttention(fused_dim, n_heads, dropout=dropout, batch_first=True)
        self.pool = AttentionPooling(fused_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(fused_dim + app_emb_dim),
            nn.Linear(fused_dim + app_emb_dim, fused_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(
        self,
        app_tokens: torch.Tensor,
        z_app: torch.Tensor,
        net_tokens: torch.Tensor,
        packet_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query_proj(app_tokens)
        k = self.key_proj(net_tokens)
        v = self.value_proj(net_tokens)
        ctx, _ = self.cross_attn(q, k, v)
        pooled_ctx = self.pool(ctx, packet_mask)
        z_flow = self.output(torch.cat([z_app, pooled_ctx], dim=-1))
        return F.normalize(z_flow, dim=-1)


class ConditionAdversary(nn.Module):

    def __init__(self, emb_dim: int = APP_EMB_DIM, n_conditions: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_conditions),
        )

    def forward(self, z_app: torch.Tensor, grl_scale: float = 1.0) -> torch.Tensor:
        return self.net(gradient_reverse(z_app, grl_scale))


class FlowConX(nn.Module):

    def __init__(
        self,
        pkt_feat_dim: int = PKT_FEAT_DIM,
        net_feat_dim: int = NET_FEAT_DIM,
        n_conditions: int = 5,
        app_hidden_dim: int = 192,
        net_hidden_dim: int = 128,
        app_emb_dim: int = APP_EMB_DIM,
        net_emb_dim: int = NET_EMB_DIM,
        flow_emb_dim: int = FLOW_EMB_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.app_encoder = ApplicationIdentityEncoder(
            pkt_feat_dim=pkt_feat_dim,
            hidden_dim=app_hidden_dim,
            out_dim=app_emb_dim,
            dropout=dropout,
        )
        self.net_encoder = NetworkConditionEncoder(
            net_feat_dim=net_feat_dim,
            hidden_dim=net_hidden_dim,
            out_dim=net_emb_dim,
            dropout=dropout,
        )
        self.fusion = ContextFusion(
            app_hidden_dim=app_hidden_dim,
            net_hidden_dim=net_hidden_dim,
            app_emb_dim=app_emb_dim,
            fused_dim=flow_emb_dim,
            dropout=dropout,
        )
        self.condition_adversary = ConditionAdversary(app_emb_dim, n_conditions=n_conditions)

    def forward(
        self,
        packet_seq: torch.Tensor,
        network_series: torch.Tensor,
        packet_mask: Optional[torch.Tensor] = None,
        grl_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        app_tokens, z_app = self.app_encoder(packet_seq, packet_mask)
        net_tokens, z_net = self.net_encoder(network_series)
        z_flow = self.fusion(app_tokens, z_app, net_tokens, packet_mask)
        condition_logits = self.condition_adversary(z_app, grl_scale=grl_scale)
        return {
            "z_app": z_app,
            "z_net": z_net,
            "z_flow": z_flow,
            "condition_logits": condition_logits,
        }

    @torch.no_grad()
    def encode(
        self,
        packet_seq: torch.Tensor,
        network_series: torch.Tensor,
        packet_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(packet_seq, network_series, packet_mask, grl_scale=0.0)["z_flow"]

