from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        contrast = embeddings
        contrast_labels = labels
        if memory is not None:
            mem_emb, mem_labels = memory
            if mem_emb.numel() > 0:
                contrast = torch.cat([contrast, F.normalize(mem_emb.to(embeddings.device), dim=-1)], dim=0)
                contrast_labels = torch.cat([contrast_labels, mem_labels.to(labels.device)], dim=0)

        logits = torch.matmul(embeddings, contrast.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positive_mask = labels[:, None].eq(contrast_labels[None, :])
        self_mask = torch.zeros_like(positive_mask)
        self_mask[:, : embeddings.shape[0]] = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        positive_mask = positive_mask & ~self_mask
        logits = logits.masked_fill(self_mask, -1e9)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not torch.any(valid):
            return embeddings.sum() * 0.0
        loss = -(log_prob * positive_mask).sum(dim=1) / positive_count.clamp(min=1)
        return loss[valid].mean()


class CrossCovarianceDisentanglement(nn.Module):

    def forward(self, z_app: torch.Tensor, z_net: torch.Tensor) -> torch.Tensor:
        z_app = z_app - z_app.mean(dim=0, keepdim=True)
        z_net = z_net - z_net.mean(dim=0, keepdim=True)
        denom = max(z_app.shape[0] - 1, 1)
        cov = torch.matmul(z_app.T, z_net) / denom
        return cov.pow(2).mean()


class PrototypeAlignmentLoss(nn.Module):

    def __init__(self, n_classes: int, emb_dim: int, temperature: float = 0.07) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, emb_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(F.normalize(embeddings, dim=-1), prototypes.T) / self.temperature
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def normalized(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1)


class FlowConXLoss(nn.Module):

    def __init__(
        self,
        n_services: int,
        n_apps: int,
        n_conditions: int,
        emb_dim: int = 256,
        temperature: float = 0.07,
        lambda_app: float = 0.35,
        lambda_proto: float = 0.10,
        lambda_dis: float = 0.25,
        lambda_adv: float = 0.15,
    ) -> None:
        super().__init__()
        self.service_supcon = SupervisedContrastiveLoss(temperature)
        self.app_supcon = SupervisedContrastiveLoss(temperature)
        self.prototype = PrototypeAlignmentLoss(n_services, emb_dim, temperature)
        self.disentangle = CrossCovarianceDisentanglement()
        self.lambda_app = lambda_app
        self.lambda_proto = lambda_proto
        self.lambda_dis = lambda_dis
        self.lambda_adv = lambda_adv
        self.n_apps = n_apps
        self.n_conditions = n_conditions

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        service_labels: torch.Tensor,
        app_labels: Optional[torch.Tensor] = None,
        condition_labels: Optional[torch.Tensor] = None,
        memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        z_app = outputs["z_app"]
        z_flow = outputs["z_flow"]
        z_net = outputs["z_net"]

        service_loss = self.service_supcon(z_app, service_labels, memory=memory)
        app_loss = z_app.sum() * 0.0
        if app_labels is not None and self.n_apps > 1:
            app_loss = self.app_supcon(z_flow, app_labels)
        proto_loss = self.prototype(z_app, service_labels)
        dis_loss = self.disentangle(z_app, z_net)
        adv_loss = z_app.sum() * 0.0
        if condition_labels is not None and self.n_conditions > 1:
            adv_loss = F.cross_entropy(outputs["condition_logits"], condition_labels)

        total = (
            service_loss
            + self.lambda_app * app_loss
            + self.lambda_proto * proto_loss
            + self.lambda_dis * dis_loss
            + self.lambda_adv * adv_loss
        )
        return total, {
            "total": float(total.detach().cpu()),
            "service_supcon": float(service_loss.detach().cpu()),
            "app_supcon": float(app_loss.detach().cpu()),
            "prototype": float(proto_loss.detach().cpu()),
            "disentangle": float(dis_loss.detach().cpu()),
            "condition_adv": float(adv_loss.detach().cpu()),
        }

