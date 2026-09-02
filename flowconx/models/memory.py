from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class EmbeddingMemoryBank:
    """Per-class ring buffer of recent embeddings, for the contrastive loss.

    Implemented as one preallocated tensor per class rather than a deque of
    individual tensors. The naive version stacked up to 9,216 one-row tensors
    on every training batch, which cost more than the forward and backward
    passes combined -- about 3 minutes per epoch on 137k rows, against 25
    seconds for the same epoch with this version. The sampled view is also
    cached and invalidated on write, because the loss reads it once per batch
    and it only changes when something is added.
    """

    def __init__(self, max_per_class: int = 512) -> None:
        self.max_per_class = max_per_class
        self._buffer: Optional[torch.Tensor] = None      # (n_classes, max_per_class, dim)
        self._labels: Optional[torch.Tensor] = None      # (n_classes,) class ids
        self._write: Dict[int, int] = {}                 # next slot per class
        self._filled: Dict[int, int] = {}                # rows written per class
        self._class_index: Dict[int, int] = {}           # class id -> buffer row
        self._cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._cache_device: Optional[torch.device] = None

    def _ensure(self, dim: int, class_id: int) -> int:
        if self._buffer is None:
            self._buffer = torch.zeros(1, self.max_per_class, dim)
            self._labels = torch.tensor([class_id], dtype=torch.long)
            self._class_index = {class_id: 0}
            self._write = {class_id: 0}
            self._filled = {class_id: 0}
            return 0
        if class_id in self._class_index:
            return self._class_index[class_id]
        row = self._buffer.shape[0]
        self._buffer = torch.cat([self._buffer, torch.zeros(1, self.max_per_class, dim)], dim=0)
        self._labels = torch.cat([self._labels, torch.tensor([class_id], dtype=torch.long)])
        self._class_index[class_id] = row
        self._write[class_id] = 0
        self._filled[class_id] = 0
        return row

    @torch.no_grad()
    def add(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = F.normalize(embeddings.detach(), dim=-1).cpu()
        labels = labels.detach().cpu()
        dim = embeddings.shape[-1]
        # Grouped by class so each class is a single slice assignment rather
        # than one assignment per example.
        for class_id in torch.unique(labels).tolist():
            rows = embeddings[labels == class_id]
            row = self._ensure(dim, int(class_id))
            start = self._write[int(class_id)]
            for offset in range(0, rows.shape[0], self.max_per_class):
                chunk = rows[offset : offset + self.max_per_class]
                n = chunk.shape[0]
                end = start + n
                if end <= self.max_per_class:
                    self._buffer[row, start:end] = chunk
                else:
                    split = self.max_per_class - start
                    self._buffer[row, start:] = chunk[:split]
                    self._buffer[row, : n - split] = chunk[split:]
                start = end % self.max_per_class
            self._write[int(class_id)] = start
            self._filled[int(class_id)] = min(self.max_per_class, self._filled[int(class_id)] + rows.shape[0])
        self._cache = None

    def sample(self, device: torch.device, max_total: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
        """A flat view of the bank, cached until the next write."""
        if self._cache is not None and self._cache_device == device:
            return self._cache
        if self._buffer is None:
            empty = (torch.empty(0, 1, device=device), torch.empty(0, dtype=torch.long, device=device))
            self._cache, self._cache_device = empty, device
            return empty
        chunks = []
        labels = []
        # Even share per class, so a frequent class cannot crowd the bank.
        per_class = max(1, max_total // max(len(self._class_index), 1))
        for class_id, row in sorted(self._class_index.items()):
            filled = self._filled[class_id]
            if filled == 0:
                continue
            take = min(filled, per_class)
            chunks.append(self._buffer[row, :take])
            labels.append(torch.full((take,), class_id, dtype=torch.long))
        if not chunks:
            empty = (torch.empty(0, 1, device=device), torch.empty(0, dtype=torch.long, device=device))
            self._cache, self._cache_device = empty, device
            return empty
        out = (torch.cat(chunks).to(device), torch.cat(labels).to(device))
        self._cache, self._cache_device = out, device
        return out

    def __len__(self) -> int:
        return int(sum(self._filled.values()))


class PrototypeBank:

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

