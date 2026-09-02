"""The training loop.

One loop, driven entirely by :class:`~flowconx.experiment.ExperimentConfig`.
Two schedules are supported:

``two_stage``  Stage 1 trains the sequence encoder with the representation
               losses only. Stage 2 switches the flow-level objectives and the
               adversary on. This is the schedule the original pipeline used
               implicitly by resuming from a tuned checkpoint; here it is
               explicit and reproducible.
``joint``      Everything from epoch one. The two-stage-versus-joint ablation
               is a config override between these.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.dataset import EncodedSplit, LabelSpace, make_loader
from ..eval.closed_set import evaluate_heads
from ..losses import FlowConXLoss
from ..models import build_model
from ..models.memory import EmbeddingMemoryBank


# Training rows per class used to build validation prototypes. A class mean is
# stable well below this; the cost of the full training set is not.
VALIDATION_PROTOTYPE_EXAMPLES = 256


def select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device: cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device: mps was requested but MPS is not available.")
        return torch.device("mps")
    if choice != "auto":
        raise ValueError(f"Unknown device {choice!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingOutcome:
    model: torch.nn.Module
    history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_score: Optional[float] = None
    device: str = "cpu"
    n_parameters: int = 0
    stage_boundaries: Dict[str, int] = field(default_factory=dict)


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    split: EncodedSplit,
    device: torch.device,
    which: str = "z_flow",
    batch_size: int = 512,
) -> np.ndarray:
    """Embeddings for a whole split, in source order."""
    model.eval()
    packet_seq = torch.from_numpy(split.features.packet_seq)
    packet_mask = torch.from_numpy(split.features.packet_mask)
    flow_features = torch.from_numpy(split.features.flow_features)
    chunks: List[np.ndarray] = []
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        outputs = model(
            packet_seq[start:stop].to(device),
            flow_features[start:stop].to(device),
            packet_mask[start:stop].to(device),
            grl_scale=0.0,
        )
        chunks.append(model.embedding(outputs, which).detach().cpu().numpy())
    return np.vstack(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)


def _stage_loss_config(base, stage: int, schedule: str):
    """Loss weights for a training stage.

    In stage 1 of the two-stage schedule the flow-level and adversarial terms
    are held at zero, so that the sequence encoder settles before the fused
    embedding and the adversary start pulling on it.
    """
    if schedule == "joint" or stage == 2:
        return base
    staged = copy.deepcopy(base)
    staged.lambda_flow_supcon = 0.0
    staged.lambda_flow_pair_margin = 0.0
    staged.lambda_adversarial = 0.0
    return staged


def train(
    config,
    train_split: EncodedSplit,
    val_split: Optional[EncodedSplit],
    label_space: LabelSpace,
    log: Optional[Sequence] = None,
) -> TrainingOutcome:
    device = select_device(config.train.device)
    model = build_model(
        config.model,
        n_nuisance=label_space.n_nuisance,
        max_len=train_split.features.observed_packets,
    ).to(device)

    loader = make_loader(
        train_split,
        batch_size=config.train.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.train.num_workers,
    )
    memory = EmbeddingMemoryBank(max_per_class=config.train.memory_per_class)

    outcome = TrainingOutcome(
        model=model,
        device=str(device),
        n_parameters=model.n_parameters(),
        stage_boundaries={"stage1_epochs": config.train.stage1_epochs if config.train.schedule == "two_stage" else 0},
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = config.train.early_stop_patience
    optimizer: Optional[torch.optim.Optimizer] = None
    last_stage: Optional[int] = None
    validation_cache: Dict[str, object] = {}

    for epoch in range(1, config.train.epochs + 1):
        stage = 1 if (config.train.schedule == "two_stage" and epoch <= config.train.stage1_epochs) else 2
        loss_cfg = _stage_loss_config(config.loss, stage, config.train.schedule)
        loss_fn = FlowConXLoss(
            loss_cfg,
            n_classes=label_space.n_classes,
            n_apps=label_space.n_apps,
            flow_emb_dim=model.flow_emb_dim,
            app_emb_dim=config.model.app_emb_dim,
        ).to(device)
        if optimizer is None or stage != last_stage:
            # Rebuilt when the stage changes, because the loss module owns
            # learnable prototypes that only exist in some stages. Held in a
            # local rather than on the function object, so that two runs in
            # one process cannot inherit each other's state.
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(loss_fn.parameters()),
                lr=config.train.lr,
                weight_decay=config.train.weight_decay,
            )
            last_stage = stage

        model.train()
        grl_scale = min(1.0, epoch / max(config.loss.grl_warmup_epochs, 1)) if stage == 2 else 0.0
        epoch_logs: List[Dict[str, float]] = []
        for batch in loader:
            packet_seq = batch["packet_seq"].to(device)
            flow_features = batch["flow_features"].to(device)
            packet_mask = batch["packet_mask"].to(device)
            labels = batch["label"].to(device)
            app_labels = batch["app_label"].to(device)
            nuisance = batch["nuisance_label"].to(device)

            outputs = model(packet_seq, flow_features, packet_mask, grl_scale=grl_scale)
            loss, info = loss_fn(
                outputs,
                labels=labels,
                app_labels=app_labels,
                nuisance_labels=nuisance,
                memory=memory.sample(device=device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(loss_fn.parameters()), config.train.grad_clip
            )
            optimizer.step()
            with torch.no_grad():
                memory.add(outputs["z_app"], labels)
            epoch_logs.append(info)

        record = {"epoch": epoch, "stage": stage, "grl_scale": grl_scale}
        record.update({k: float(np.mean([e[k] for e in epoch_logs])) for k in epoch_logs[0]} if epoch_logs else {})

        if val_split is not None and len(val_split) > 0:
            score = _validation_score(
                model, train_split, val_split, label_space, config, device, cache=validation_cache
            )
            record["val_macro_f1"] = score
            if outcome.best_score is None or score > outcome.best_score:
                outcome.best_score, outcome.best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = config.train.early_stop_patience
            elif patience_left is not None:
                patience_left -= 1
        outcome.history.append(record)
        if log is not None:
            log.append(record)
        _print_epoch(record)
        if patience_left is not None and patience_left <= 0:
            print(f"  early stop at epoch {epoch} (no val improvement for {config.train.early_stop_patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return outcome


def stratified_subsample(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Indices of at most ``per_class`` rows per class, seeded."""
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for cls in np.unique(labels):
        pool = np.flatnonzero(labels == cls)
        take = min(per_class, pool.size)
        keep.extend(rng.choice(pool, size=take, replace=False).tolist())
    return np.asarray(sorted(keep), dtype=int)


def _validation_score(
    model, train_split, val_split, label_space, config, device, cache: Optional[Dict[str, object]] = None
) -> float:
    """Macro-F1 of the prototype head on validation.

    The prototype head is used for model selection because it is the head the
    deployment claims rest on, and because it needs no extra fitting.

    The training embeddings behind the prototypes are a seeded stratified
    subsample. Embedding all 137k training rows every epoch to compute a class
    mean costs more than the epoch itself, and a class mean is already stable
    at a few hundred examples. The subsample is fixed across epochs so the
    validation score moves with the model rather than with the draw.
    """
    if cache is not None and "train_idx" in cache:
        train_idx = cache["train_idx"]
    else:
        train_idx = stratified_subsample(
            train_split.labels, per_class=VALIDATION_PROTOTYPE_EXAMPLES, seed=config.seed
        )
        if cache is not None:
            cache["train_idx"] = train_idx

    from ..run import _subset

    train_view = _subset(train_split, train_idx)
    train_emb = extract_embeddings(model, train_view, device, which=config.model.classify_from)
    val_emb = extract_embeddings(model, val_split, device, which=config.model.classify_from)
    report = evaluate_heads(
        train_emb,
        train_view.labels,
        val_emb,
        val_split.labels,
        label_space.classes,
        heads=("prototype",),
        k=config.eval.knn_k,
        seed=config.seed,
        bootstrap_resamples=0,
    )
    return float(report["prototype"]["macro_f1"])


def _print_epoch(record: Dict[str, float]) -> None:
    parts = [f"epoch {int(record['epoch']):03d}", f"stage {int(record['stage'])}"]
    for key in ("total", "service_supcon", "flow_supcon", "flow_pair_margin", "adversarial"):
        if key in record:
            parts.append(f"{key}={record[key]:.4f}")
    if "val_macro_f1" in record:
        parts.append(f"val_macroF1={record['val_macro_f1']:.4f}")
    print("  " + "  ".join(parts))
