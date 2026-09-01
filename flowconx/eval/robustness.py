"""Robustness to network conditions and to traffic-analysis countermeasures.

This is where the adversarial component has to earn its place. Each transform
is applied to the *test* features only, with the encoder frozen, so the
question asked is exactly the deployment question: a model trained on clean
traffic meets traffic that has been padded, delayed, or reshaped.

Every countermeasure reports the overhead it imposes alongside the accuracy it
costs. A defence that halves accuracy by tripling bandwidth is not the same
result as one that halves it for free, and a table that omits overhead cannot
tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..features.packet import MAX_PACKET_SIZE, build_packet_features
from ..metrics import balanced_accuracy, macro_f1
from .closed_set import class_prototypes
from .few_shot import predict_from_prototypes

# Bucket sizes used by padding defences, in bytes.
PADDING_BUCKETS = np.array([64, 128, 256, 512, 1024, 1500], dtype=np.float64)


@dataclass
class Perturbation:
    name: str
    kind: str
    apply: Callable[[np.ndarray, np.ndarray, np.ndarray, np.random.Generator], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    description: str


def _pad_to_mtu(lengths, iats, directions, rng):
    _ = rng
    return np.full_like(lengths, MAX_PACKET_SIZE), iats, directions


def _pad_to_buckets(lengths, iats, directions, rng):
    _ = rng
    idx = np.searchsorted(PADDING_BUCKETS, lengths, side="left")
    idx = np.clip(idx, 0, len(PADDING_BUCKETS) - 1)
    return PADDING_BUCKETS[idx], iats, directions


def _quantize_128(lengths, iats, directions, rng):
    _ = rng
    return np.ceil(lengths / 128.0) * 128.0, iats, directions


def _random_pad(lengths, iats, directions, rng):
    extra = rng.integers(0, 256, size=len(lengths)).astype(np.float64)
    return np.minimum(lengths + extra, MAX_PACKET_SIZE), iats, directions


def _constant_rate(lengths, iats, directions, rng):
    """Constant-rate cover traffic: fixed size and fixed inter-arrival."""
    _ = rng
    return (
        np.full_like(lengths, MAX_PACKET_SIZE),
        np.full_like(iats, float(np.median(iats)) if len(iats) else 0.0),
        directions,
    )


def _jitter(scale: float):
    def apply(lengths, iats, directions, rng):
        noise = rng.normal(0.0, scale, size=len(iats))
        return lengths, np.clip(iats + noise, 0.0, None), directions

    return apply


def _packet_loss(rate: float):
    def apply(lengths, iats, directions, rng):
        keep = rng.random(len(lengths)) >= rate
        if not keep.any():
            keep[0] = True
        # A dropped packet folds its inter-arrival into the next one, which is
        # what an observer downstream of the loss actually sees.
        merged_iats = np.zeros(int(keep.sum()))
        carry = 0.0
        j = 0
        for i in range(len(iats)):
            carry += iats[i]
            if keep[i]:
                merged_iats[j] = carry
                carry = 0.0
                j += 1
        return lengths[keep], merged_iats, directions[keep]

    return apply


def _dummy_injection(rate: float):
    def apply(lengths, iats, directions, rng):
        n_extra = max(1, int(len(lengths) * rate))
        positions = rng.integers(0, len(lengths) + 1, size=n_extra)
        sizes = rng.choice(PADDING_BUCKETS, size=n_extra)
        dirs = rng.choice([1, -1], size=n_extra)
        out_lengths = np.insert(lengths, positions, sizes)
        out_iats = np.insert(iats, positions, 0.0)
        out_dirs = np.insert(directions, positions, dirs)
        return out_lengths, out_iats, out_dirs

    return apply


PERTURBATIONS: List[Perturbation] = [
    Perturbation("clean", "reference", lambda ln, it, dr, rng: (ln, it, dr), "Unmodified test traffic."),
    Perturbation("jitter_5ms", "condition", _jitter(5.0), "Gaussian delay noise, sigma = 5 ms."),
    Perturbation("jitter_25ms", "condition", _jitter(25.0), "Gaussian delay noise, sigma = 25 ms."),
    Perturbation("loss_1pct", "condition", _packet_loss(0.01), "1% of packets dropped, IATs merged."),
    Perturbation("loss_5pct", "condition", _packet_loss(0.05), "5% of packets dropped, IATs merged."),
    Perturbation("pad_mtu", "defence", _pad_to_mtu, "Every packet padded to the MTU."),
    Perturbation("pad_buckets", "defence", _pad_to_buckets, "Packets padded up to the next bucket size."),
    Perturbation("quantize_128", "defence", _quantize_128, "Sizes rounded up to a multiple of 128 bytes."),
    Perturbation("random_pad", "defence", _random_pad, "Uniform 0-255 bytes of random padding per packet."),
    Perturbation("constant_rate", "defence", _constant_rate, "Constant packet size and constant rate."),
    Perturbation("dummy_20pct", "defence", _dummy_injection(0.2), "20% dummy packets injected at random positions."),
]


def _overhead(original_lengths: List[np.ndarray], perturbed_lengths: List[np.ndarray]) -> Dict[str, float]:
    before = float(sum(float(a.sum()) for a in original_lengths))
    after = float(sum(float(a.sum()) for a in perturbed_lengths))
    packets_before = float(sum(len(a) for a in original_lengths))
    packets_after = float(sum(len(a) for a in perturbed_lengths))
    return {
        "byte_overhead_ratio": float(after / before) if before else float("nan"),
        "packet_overhead_ratio": float(packets_after / packets_before) if packets_before else float("nan"),
    }


def _reparse(frame) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Recover the raw packet series so a perturbation can act on real bytes.

    Perturbing the already-normalised feature tensor would be wrong: padding
    to the MTU is a statement about byte counts, not about a log-scaled
    channel, and the flow-level summary has to be recomputed from the
    perturbed packets anyway.
    """
    from ..features.packet import parse_float_series

    return [
        (parse_float_series(lengths), parse_float_series(iats), parse_float_series(directions))
        for lengths, iats, directions in zip(
            frame["packet_lengths"], frame["iat_values"], frame["directions"]
        )
    ]


def evaluate_robustness(
    model,
    splits,
    config,
    label_space,
    device,
    perturbations: Optional[List[Perturbation]] = None,
    max_test_rows: int = 4000,
) -> Dict[str, object]:
    """Accuracy under each perturbation, with the overhead it imposes."""
    import torch

    from ..features.packet import PKT_FEATURE_DIM
    from ..train.loop import extract_embeddings

    selected = perturbations or PERTURBATIONS
    train_split, test_split = splits["train"], splits["test"]
    n_classes = label_space.n_classes
    labels = np.arange(n_classes)

    train_emb = extract_embeddings(model, train_split, device, which=config.model.classify_from)
    prototypes = class_prototypes(train_emb, train_split.labels, n_classes)

    rng_master = np.random.default_rng(config.seed)
    rows = np.arange(len(test_split))
    if len(rows) > max_test_rows:
        rows = np.sort(rng_master.choice(rows, size=max_test_rows, replace=False))
    frame = test_split.frame.iloc[rows]
    truth = test_split.labels[rows]
    budget = test_split.features.observed_packets
    parsed = _reparse(frame)

    results: Dict[str, object] = {"status": "ok", "n_test_rows": int(len(rows)), "perturbations": []}
    for perturbation in selected:
        rng = np.random.default_rng(config.seed * 31 + abs(hash(perturbation.name)) % 10000)
        packet_seq = np.zeros((len(parsed), budget, PKT_FEATURE_DIM), dtype=np.float32)
        packet_mask = np.ones((len(parsed), budget), dtype=bool)
        originals: List[np.ndarray] = []
        modified: List[np.ndarray] = []
        for i, (lengths, iats, directions) in enumerate(parsed):
            originals.append(lengths)
            new_lengths, new_iats, new_dirs = perturbation.apply(lengths, iats, directions, rng)
            modified.append(new_lengths)
            features = build_packet_features(new_lengths, new_iats, new_dirs, budget)
            if features is None:
                continue
            packet_seq[i, : features.shape[0]] = features
            packet_mask[i, : features.shape[0]] = False

        # Flow-level context is recomputed from the perturbed packets, because
        # a defence that changes the packets changes the summary too; leaving
        # the original summary in place would leak the undefended traffic.
        flow_features = _recompute_flow_features(modified, parsed, budget)

        with torch.no_grad():
            outputs = model(
                torch.from_numpy(packet_seq).to(device),
                torch.from_numpy(flow_features).to(device),
                torch.from_numpy(packet_mask).to(device),
                grl_scale=0.0,
            )
            embedding = model.embedding(outputs, config.model.classify_from).cpu().numpy()

        predictions = predict_from_prototypes(prototypes, embedding)
        results["perturbations"].append(
            {
                "name": perturbation.name,
                "kind": perturbation.kind,
                "description": perturbation.description,
                "macro_f1": macro_f1(predictions, truth, labels),
                "balanced_accuracy": balanced_accuracy(predictions, truth, labels),
                **_overhead(originals, modified),
            }
        )

    clean = next((p for p in results["perturbations"] if p["name"] == "clean"), None)
    if clean:
        for entry in results["perturbations"]:
            entry["macro_f1_drop_vs_clean"] = float(clean["macro_f1"] - entry["macro_f1"])
    return results


def _recompute_flow_features(modified: List[np.ndarray], parsed, budget: int) -> np.ndarray:
    from ..features.packet import FLOW_FEATURE_DIM, build_flow_features

    out = np.zeros((len(modified), FLOW_FEATURE_DIM), dtype=np.float32)
    for i, lengths in enumerate(modified):
        iats = parsed[i][1]
        duration_ms = float(iats.sum())
        n = max(len(lengths), 1)
        out[i] = build_flow_features(
            {
                "total packets": n,
                "total fwd packets": float((parsed[i][2] > 0).sum()),
                "total backward packets": float((parsed[i][2] <= 0).sum()),
                "packet length mean": float(lengths.mean()) if len(lengths) else 0.0,
                "packet length std": float(lengths.std()) if len(lengths) else 0.0,
                "flow iat mean": float(iats.mean()) if len(iats) else 0.0,
                "flow iat std": float(iats.std()) if len(iats) else 0.0,
                "flow duration": duration_ms,
                "flow bytes/s": float(lengths.sum() / max(duration_ms / 1000.0, 1e-3)),
                "flow packets/s": float(n / max(duration_ms / 1000.0, 1e-3)),
            },
            min(n, budget),
        )
    return out
