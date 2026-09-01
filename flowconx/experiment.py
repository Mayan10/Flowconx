"""Experiment configuration.

Every experiment is a YAML file. Nothing important is reachable only from the
command line, and nothing is reachable only from a notebook: the ablation
runner works by overriding fields of this tree, so any component that an
ablation needs to switch off has to be a field here.

The config is hashed into every ``metrics.json``. Two runs whose config hash
and seed match must produce identical metrics; ``tests/test_determinism.py``
asserts that on a tiny config.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# --------------------------------------------------------------------------
# Enumerations, kept as string literals so a YAML file is readable
# --------------------------------------------------------------------------

FUSION_MODES = ("cross_attention", "concat", "gated_sum", "late", "none")
CLASSIFY_FROM = ("z_flow", "z_app", "z_network", "z_concat")
TRAIN_SCHEDULES = ("two_stage", "joint")
CLASSIFIER_HEADS = ("knn", "prototype", "linear", "svm")


@dataclass
class DataConfig:
    """Which rows the model sees, and how they are partitioned."""

    dataset: str = "cesnet_quic22"
    csv: str = "data/processed/cesnet_quic22.csv"
    label_column: str = "service"
    split_protocol: str = "session_disjoint"
    split_seed: int = 42
    val_fraction: float = 0.1
    test_fraction: float = 0.2
    rare_class_mode: str = "drop"
    min_class_count: int = 100
    max_packets: int = 32
    # Number of leading packets the encoder may observe. Distinct from
    # max_packets, which is what the CSV stores: the input-budget sweep in the
    # ablations varies this without re-reading the data.
    observed_packets: Optional[int] = None
    limit: Optional[int] = None
    # Hold these apps out of training entirely, for open-set evaluation.
    unknown_apps: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Architecture switches. Every component here is independently ablatable."""

    app_hidden_dim: int = 192
    net_hidden_dim: int = 128
    app_emb_dim: int = 256
    net_emb_dim: int = 128
    flow_emb_dim: int = 256
    dropout: float = 0.1
    n_conv_blocks: int = 3
    n_heads: int = 6
    n_transformer_layers: int = 1
    # Minus dual encoder: one shared encoder over the concatenated inputs.
    dual_encoder: bool = True
    fusion: str = "cross_attention"
    # Minus adversarial condition removal.
    adversarial_head: bool = True
    classify_from: str = "z_flow"
    # Capacity control: scale hidden widths so ablations can be matched on
    # parameter count rather than on layer count.
    width_multiplier: float = 1.0


@dataclass
class LossConfig:
    """Objective weights. Zero disables a term; the runner records which."""

    temperature: float = 0.07
    lambda_service_supcon: float = 1.0
    lambda_flow_supcon: float = 1.0
    lambda_app_supcon: float = 0.0
    lambda_prototype: float = 0.1
    lambda_disentangle: float = 0.25
    lambda_adversarial: float = 0.15
    lambda_pair_margin: float = 0.0
    lambda_flow_pair_margin: float = 1.0
    pair_negative_margin: float = 0.2
    pair_positive_target: float = 0.75
    grl_warmup_epochs: int = 5


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    schedule: str = "two_stage"
    stage1_epochs: int = 10
    memory_per_class: int = 512
    num_workers: int = 0
    device: str = "auto"
    early_stop_patience: Optional[int] = None
    amp: bool = False


@dataclass
class EvalConfig:
    knn_k: int = 5
    classifier_heads: List[str] = field(default_factory=lambda: ["knn", "prototype"])
    bootstrap_resamples: int = 1000
    open_set: bool = False
    few_shot: bool = False
    few_shot_k: List[int] = field(default_factory=lambda: [1, 5, 10, 25, 50, 100])
    drift: bool = False
    robustness: bool = False
    early_classification: bool = False
    early_packet_budgets: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    cost: bool = True
    probes: bool = False


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    seed: int = 0
    output_root: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---------------------------------------------------------------- checks
    def validate(self) -> None:
        if self.model.fusion not in FUSION_MODES:
            raise ValueError(f"model.fusion must be one of {FUSION_MODES}, got {self.model.fusion!r}")
        if self.model.classify_from not in CLASSIFY_FROM:
            raise ValueError(f"model.classify_from must be one of {CLASSIFY_FROM}, got {self.model.classify_from!r}")
        if self.train.schedule not in TRAIN_SCHEDULES:
            raise ValueError(f"train.schedule must be one of {TRAIN_SCHEDULES}, got {self.train.schedule!r}")
        for head in self.eval.classifier_heads:
            if head not in CLASSIFIER_HEADS:
                raise ValueError(f"eval.classifier_heads entries must be in {CLASSIFIER_HEADS}, got {head!r}")
        if self.train.schedule == "two_stage" and self.train.stage1_epochs >= self.train.epochs:
            raise ValueError("train.stage1_epochs must be smaller than train.epochs for a two-stage schedule.")
        if not self.model.dual_encoder and self.model.fusion != "none":
            raise ValueError(
                "model.dual_encoder=false means there is no second encoder to fuse with; "
                "set model.fusion=none for that ablation."
            )
        if self.model.classify_from == "z_network" and not self.model.dual_encoder:
            raise ValueError("model.classify_from='z_network' requires the dual encoder.")
        observed = self.data.observed_packets
        if observed is not None and observed > self.data.max_packets:
            raise ValueError(
                f"data.observed_packets ({observed}) exceeds data.max_packets ({self.data.max_packets}); "
                "the extra packets were never read from the CSV."
            )

    # ------------------------------------------------------------ conversion
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        """Stable hash of the config, excluding fields that cannot change results.

        ``seed`` is excluded on purpose: the point of the hash is to identify
        the *experiment*, so that runs of the same experiment at different
        seeds group together in ``results/``.
        """
        payload = self.as_dict()
        payload.pop("seed", None)
        payload.pop("output_root", None)
        payload.pop("description", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    @property
    def run_dir(self) -> Path:
        """``results/<experiment>/<dataset>/<split>/<seed>/``."""
        return (
            Path(self.output_root)
            / self.name
            / self.data.dataset
            / self.data.split_protocol
            / f"seed{self.seed}"
        )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _from_mapping(cls: type, payload: Mapping[str, Any], path: str = "") -> Any:
    """Build a dataclass from a mapping, rejecting unknown keys.

    Rejecting unknown keys matters more than it looks: a typo in a YAML field
    name would otherwise silently leave the default in place, and the run
    would be recorded as the ablation it was not.
    """
    if not is_dataclass(cls):
        return payload
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(payload) - set(known))
    if unknown:
        where = f" at {path}" if path else ""
        raise ValueError(f"Unknown config key(s){where}: {unknown}. Known keys: {sorted(known)}")
    kwargs: Dict[str, Any] = {}
    for key, value in payload.items():
        field_type = known[key].type
        if isinstance(value, Mapping) and is_dataclass(_resolve(field_type)):
            kwargs[key] = _from_mapping(_resolve(field_type), value, f"{path}.{key}" if path else key)
        else:
            kwargs[key] = value
    return cls(**kwargs)


_NESTED = {
    "data": DataConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def _resolve(annotation: Any) -> Any:
    """Map a field annotation to its dataclass, tolerating string annotations."""
    if is_dataclass(annotation):
        return annotation
    name = getattr(annotation, "__name__", str(annotation))
    for cls in _NESTED.values():
        if cls.__name__ in name:
            return cls
    return annotation


def from_dict(payload: Mapping[str, Any]) -> ExperimentConfig:
    top = {k: v for k, v in payload.items() if k not in _NESTED}
    config = _from_mapping(ExperimentConfig, top)
    for key, cls in _NESTED.items():
        if key in payload:
            setattr(config, key, _from_mapping(cls, payload[key], key))
    config.validate()
    return config


def deep_update(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, overrides: Optional[Sequence[str]] = None) -> ExperimentConfig:
    """Load a YAML config, following a ``defaults:`` chain, then apply overrides.

    ``defaults`` is a list of sibling config paths merged in order before this
    file's own keys, which is how the ablation configs stay to three lines
    each instead of restating the whole tree.
    """
    import yaml

    target = Path(path)
    payload: Dict[str, Any] = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    merged: Dict[str, Any] = {}
    for parent in payload.pop("defaults", []) or []:
        parent_path = (target.parent / parent).resolve()
        parent_payload = yaml.safe_load(Path(parent_path).read_text(encoding="utf-8")) or {}
        parent_payload.pop("defaults", None)
        merged = deep_update(merged, parent_payload)
    merged = deep_update(merged, payload)
    for override in overrides or []:
        merged = deep_update(merged, _parse_override(override))
    return from_dict(merged)


def _parse_override(text: str) -> Dict[str, Any]:
    """``model.fusion=concat`` -> ``{"model": {"fusion": "concat"}}``."""
    import yaml

    if "=" not in text:
        raise ValueError(f"Override {text!r} must look like key.path=value")
    dotted, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    out: Dict[str, Any] = value
    for key in reversed(dotted.strip().split(".")):
        out = {key: out}
    return out


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    import yaml

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8")
    return out
