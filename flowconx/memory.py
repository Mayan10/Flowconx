"""Memory and prototype banks for adaptive FlowCon-X learning."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


class EmbeddingMemoryBank:
    """Class-balanced queue used as extra contrastive examples."""

    def __init__(self, max_per_class: int = 512) -> None:
        self.max_per_class = max_per_class
        self.storage: Dict[int, Deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=max_per_class))

    @torch.no_grad()
    def add(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = F.normalize(embeddings.detach().cpu(), dim=-1)
        labels = labels.detach().cpu()
        for emb, label in zip(embeddings, labels):
            self.storage[int(label)].append(emb.clone())

    def sample(self, device: torch.device, max_total: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
        chunks = []
        labels = []
        for label, queue in self.storage.items():
            for emb in list(queue):
                chunks.append(emb)
                labels.append(label)
                if len(chunks) >= max_total:
                    break
            if len(chunks) >= max_total:
                break
        if not chunks:
            return torch.empty(0, 1, device=device), torch.empty(0, dtype=torch.long, device=device)
        return torch.stack(chunks).to(device), torch.tensor(labels, dtype=torch.long, device=device)


class PrototypeBank:
    """Trust-gated exponential moving average service prototypes."""

    def __init__(self, n_classes: int, emb_dim: int, momentum: float = 0.95, high_confidence: float = 0.75) -> None:
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.momentum = momentum
        self.high_confidence = high_confidence
        self.prototypes = torch.zeros(n_classes, emb_dim)
        self.counts = torch.zeros(n_classes, dtype=torch.long)

    @torch.no_grad()
    def bootstrap(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = F.normalize(embeddings.detach().cpu(), dim=-1)
        labels = labels.detach().cpu()
        for class_id in range(self.n_classes):
            mask = labels == class_id
            if torch.any(mask):
                proto = F.normalize(embeddings[mask].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
                self.prototypes[class_id] = proto
                self.counts[class_id] = int(mask.sum())

    @torch.no_grad()
    def nearest(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = F.normalize(embeddings.detach().cpu(), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        valid = self.counts > 0
        if not torch.any(valid):
            sims = torch.zeros(embeddings.shape[0], self.n_classes)
        else:
            sims = torch.matmul(embeddings, prototypes.T)
            sims[:, ~valid] = -1.0
        score, label = sims.max(dim=1)
        return label, score

    @torch.no_grad()
    def update_trusted(self, embeddings: torch.Tensor, labels: torch.Tensor) -> int:
        embeddings = F.normalize(embeddings.detach().cpu(), dim=-1)
        labels = labels.detach().cpu()
        updated = 0
        predicted, score = self.nearest(embeddings)
        for emb, label, pred, sim in zip(embeddings, labels, predicted, score):
            class_id = int(label)
            if self.counts[class_id] == 0 or (int(pred) == class_id and float(sim) >= self.high_confidence):
                if self.counts[class_id] == 0:
                    self.prototypes[class_id] = emb
                else:
                    self.prototypes[class_id] = F.normalize(
                        self.momentum * self.prototypes[class_id] + (1.0 - self.momentum) * emb,
                        dim=0,
                    )
                self.counts[class_id] += 1
                updated += 1
        return updated

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"prototypes": self.prototypes, "counts": self.counts}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.prototypes = state["prototypes"].detach().cpu()
        self.counts = state["counts"].detach().cpu()

