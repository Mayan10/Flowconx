"""Training objectives.

Every term is weighted from config and every weight may be zero, which is how
the loss-term ablations are expressed. The returned log dict always contains
every term, including the disabled ones at 0.0, so that a results table can be
built without knowing which configuration produced which file.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """SupCon (Khosla et al., NeurIPS 2020), optionally against a memory bank."""

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
        contrast, contrast_labels = embeddings, labels
        if memory is not None:
            mem_emb, mem_labels = memory
            if mem_emb.numel() > 0 and mem_emb.shape[-1] == embeddings.shape[-1]:
                contrast = torch.cat([contrast, F.normalize(mem_emb.to(embeddings.device), dim=-1)], dim=0)
                contrast_labels = torch.cat([contrast_labels, mem_labels.to(labels.device)], dim=0)

        logits = torch.matmul(embeddings, contrast.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positives = labels[:, None].eq(contrast_labels[None, :])
        self_mask = torch.zeros_like(positives)
        self_mask[:, : embeddings.shape[0]] = torch.eye(
            embeddings.shape[0], dtype=torch.bool, device=embeddings.device
        )
        positives = positives & ~self_mask
        logits = logits.masked_fill(self_mask, -1e4)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        counts = positives.sum(dim=1)
        valid = counts > 0
        if not torch.any(valid):
            # A batch where every row is its own class contributes nothing;
            # returning a zero that is still attached to the graph keeps the
            # optimiser step well-defined.
            return embeddings.sum() * 0.0
        loss = -(log_prob * positives).sum(dim=1) / counts.clamp(min=1)
        return loss[valid].mean()


class PairwiseEmbeddingMarginLoss(nn.Module):
    """Push same-class cosine above a target and different-class below a margin.

    This is the term that makes the deployed embedding metric-trained, which
    is what the open-set and few-shot claims rest on, so it is the one the
    headline ablation switches off.
    """

    def __init__(self, negative_margin: float = 0.2, positive_target: float = 0.75) -> None:
        super().__init__()
        self.negative_margin = negative_margin
        self.positive_target = positive_target

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        similarity = torch.matmul(embeddings, embeddings.T)
        self_mask = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        same = labels[:, None].eq(labels[None, :]) & ~self_mask
        different = ~labels[:, None].eq(labels[None, :])
        loss = embeddings.sum() * 0.0
        if torch.any(same):
            loss = loss + F.relu(self.positive_target - similarity[same]).pow(2).mean()
        if torch.any(different):
            loss = loss + F.relu(similarity[different] - self.negative_margin).pow(2).mean()
        return loss


class PrototypeAlignmentLoss(nn.Module):
    """Learned class prototypes with a cosine-softmax objective."""

    def __init__(self, n_classes: int, emb_dim: int, temperature: float = 0.07) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(F.normalize(embeddings, dim=-1), prototypes.T) / self.temperature
        return F.cross_entropy(logits, labels)


class CrossCovarianceDisentanglement(nn.Module):
    """Penalise linear dependence between the two views' embeddings."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a - a.mean(dim=0, keepdim=True)
        b = b - b.mean(dim=0, keepdim=True)
        cov = torch.matmul(a.T, b) / max(a.shape[0] - 1, 1)
        return cov.pow(2).mean()


class FlowConXLoss(nn.Module):
    """The full objective, assembled from a :class:`~flowconx.experiment.LossConfig`."""

    TERMS = (
        "service_supcon",
        "flow_supcon",
        "app_supcon",
        "prototype",
        "disentangle",
        "adversarial",
        "pair_margin",
        "flow_pair_margin",
    )

    def __init__(self, loss_config, n_classes: int, n_apps: int, flow_emb_dim: int, app_emb_dim: int) -> None:
        super().__init__()
        self.cfg = loss_config
        self.supcon = SupervisedContrastiveLoss(loss_config.temperature)
        self.pair = PairwiseEmbeddingMarginLoss(loss_config.pair_negative_margin, loss_config.pair_positive_target)
        self.disentangle = CrossCovarianceDisentanglement()
        self.prototype = (
            PrototypeAlignmentLoss(n_classes, app_emb_dim, loss_config.temperature)
            if loss_config.lambda_prototype > 0
            else None
        )
        self.n_apps = n_apps
        self.flow_emb_dim = flow_emb_dim

    def active_terms(self) -> Dict[str, float]:
        return {
            "service_supcon": self.cfg.lambda_service_supcon,
            "flow_supcon": self.cfg.lambda_flow_supcon,
            "app_supcon": self.cfg.lambda_app_supcon,
            "prototype": self.cfg.lambda_prototype,
            "disentangle": self.cfg.lambda_disentangle,
            "adversarial": self.cfg.lambda_adversarial,
            "pair_margin": self.cfg.lambda_pair_margin,
            "flow_pair_margin": self.cfg.lambda_flow_pair_margin,
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        app_labels: Optional[torch.Tensor] = None,
        nuisance_labels: Optional[torch.Tensor] = None,
        memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        z_app = outputs["z_app"]
        z_flow = outputs["z_flow"]
        z_network = outputs.get("z_network")
        zero = z_app.sum() * 0.0
        weights = self.active_terms()
        terms: Dict[str, torch.Tensor] = {name: zero for name in self.TERMS}

        if weights["service_supcon"]:
            terms["service_supcon"] = self.supcon(z_app, labels, memory=memory)
        if weights["flow_supcon"]:
            terms["flow_supcon"] = self.supcon(z_flow, labels)
        if weights["app_supcon"] and app_labels is not None and self.n_apps > 1:
            terms["app_supcon"] = self.supcon(z_flow, app_labels)
        if weights["prototype"] and self.prototype is not None:
            terms["prototype"] = self.prototype(z_app, labels)
        if weights["disentangle"] and z_network is not None:
            terms["disentangle"] = self.disentangle(z_app, z_network)
        if weights["adversarial"] and nuisance_labels is not None and "nuisance_logits" in outputs:
            terms["adversarial"] = F.cross_entropy(outputs["nuisance_logits"], nuisance_labels)
        if weights["pair_margin"]:
            terms["pair_margin"] = self.pair(z_app, labels)
        if weights["flow_pair_margin"]:
            terms["flow_pair_margin"] = self.pair(z_flow, labels)

        total = zero
        for name, value in terms.items():
            total = total + weights[name] * value
        log = {name: float(value.detach().cpu()) for name, value in terms.items()}
        log["total"] = float(total.detach().cpu())
        return total, log
