"""Evaluation utilities for FlowCon-X KPIs."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import FlowDataset, FlowRecord
from .features import augment_network_condition


@torch.no_grad()
def extract_embeddings(model, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    z_flow = []
    z_app = []
    services = []
    apps = []
    conditions = []
    for batch in loader:
        packet_seq = batch["packet_seq"].to(device)
        network_series = batch["network_series"].to(device)
        packet_mask = batch["packet_mask"].to(device)
        outputs = model(packet_seq, network_series, packet_mask, grl_scale=0.0)
        z_flow.append(outputs["z_flow"].cpu().numpy())
        z_app.append(outputs["z_app"].cpu().numpy())
        services.append(batch["service_label"].cpu().numpy())
        apps.append(batch["app_label"].cpu().numpy())
        conditions.append(batch["condition_label"].cpu().numpy())
    return {
        "z_flow": np.vstack(z_flow),
        "z_app": np.vstack(z_app),
        "service": np.concatenate(services),
        "app": np.concatenate(apps),
        "condition": np.concatenate(conditions),
    }


def cosine_matrix(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    b = a if b is None else b
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a_norm @ b_norm.T


def embedding_similarity(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    sims = cosine_matrix(embeddings)
    intra = []
    inter = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra.append(float(sims[i, j]))
            else:
                inter.append(float(sims[i, j]))
    return {
        "intra": float(np.mean(intra)) if intra else 0.0,
        "inter": float(np.mean(inter)) if inter else 0.0,
    }


def knn_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int = 5) -> np.ndarray:
    sims = cosine_matrix(test_x, train_x)
    top = np.argsort(-sims, axis=1)[:, :k]
    preds = []
    for row in top:
        values, counts = np.unique(train_y[row], return_counts=True)
        preds.append(values[np.argmax(counts)])
    return np.asarray(preds)


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(pred == target)) if len(target) else 0.0


def macro_f1(pred: np.ndarray, target: np.ndarray) -> float:
    labels = np.unique(np.concatenate([pred, target]))
    scores = []
    for label in labels:
        tp = np.sum((pred == label) & (target == label))
        fp = np.sum((pred == label) & (target != label))
        fn = np.sum((pred != label) & (target == label))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(float(2 * precision * recall / (precision + recall)))
    return float(np.mean(scores)) if scores else 0.0


def closed_set_classification(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    preds = knn_predict(train_embeddings, train_labels, test_embeddings, k=k)
    results = {
        "knn_accuracy": accuracy(preds, test_labels),
        "knn_macro_f1": macro_f1(preds, test_labels),
    }
    try:
        from sklearn.svm import SVC

        svm = SVC(kernel="rbf", C=10, gamma="scale")
        svm.fit(train_embeddings, train_labels)
        svm_preds = svm.predict(test_embeddings)
        results["svm_accuracy"] = accuracy(svm_preds, test_labels)
        results["svm_macro_f1"] = macro_f1(svm_preds, test_labels)
    except ImportError:
        results["svm_accuracy"] = float("nan")
        results["svm_macro_f1"] = float("nan")
    return results


def prototype_generalization(
    train_embeddings: np.ndarray,
    train_services: np.ndarray,
    test_embeddings: np.ndarray,
    test_services: np.ndarray,
) -> Dict[str, float]:
    prototypes = []
    labels = []
    for service in np.unique(train_services):
        mask = train_services == service
        proto = train_embeddings[mask].mean(axis=0)
        proto = proto / max(np.linalg.norm(proto), 1e-12)
        prototypes.append(proto)
        labels.append(service)
    if not prototypes:
        return {"prototype_accuracy": 0.0}
    sims = cosine_matrix(test_embeddings, np.vstack(prototypes))
    pred = np.asarray([labels[idx] for idx in np.argmax(sims, axis=1)])
    return {"prototype_accuracy": accuracy(pred, test_services), "prototype_macro_f1": macro_f1(pred, test_services)}


def leave_one_app_out_generalization(
    embeddings: np.ndarray,
    app_labels: np.ndarray,
    service_labels: np.ndarray,
) -> Dict[str, float]:
    """Assign each held-out app to a service prototype built from other apps."""
    preds = []
    targets = []
    skipped = 0
    for app in np.unique(app_labels):
        test_mask = app_labels == app
        train_mask = ~test_mask
        train_services = np.unique(service_labels[train_mask])
        prototypes = []
        labels = []
        for service in train_services:
            service_mask = train_mask & (service_labels == service)
            if not np.any(service_mask):
                continue
            proto = embeddings[service_mask].mean(axis=0)
            proto = proto / max(np.linalg.norm(proto), 1e-12)
            prototypes.append(proto)
            labels.append(service)
        if not prototypes:
            skipped += int(np.sum(test_mask))
            continue
        sims = cosine_matrix(embeddings[test_mask], np.vstack(prototypes))
        app_preds = np.asarray([labels[idx] for idx in np.argmax(sims, axis=1)])
        preds.append(app_preds)
        targets.append(service_labels[test_mask])
    if not preds:
        return {"leave_one_app_accuracy": 0.0, "leave_one_app_macro_f1": 0.0, "leave_one_app_skipped": float(skipped)}
    pred_arr = np.concatenate(preds)
    target_arr = np.concatenate(targets)
    return {
        "leave_one_app_accuracy": accuracy(pred_arr, target_arr),
        "leave_one_app_macro_f1": macro_f1(pred_arr, target_arr),
        "leave_one_app_skipped": float(skipped),
    }


@torch.no_grad()
def cist_score(
    model,
    records: Sequence[FlowRecord],
    label_maps: Dict[str, Dict[str, int]],
    device: torch.device,
    max_records: int = 256,
    seed: int = 99,
) -> float:
    """Context Invariance Stress Test score on z_app."""
    rng = np.random.default_rng(seed)
    base_records = list(records[:max_records])
    augmented: List[FlowRecord] = []
    for record in base_records:
        pkt_aug, net_aug, condition = augment_network_condition(record.packet_seq, record.network_series, rng)
        augmented.append(
            FlowRecord(
                packet_seq=pkt_aug,
                network_series=net_aug,
                app=record.app,
                service=record.service,
                condition=condition,
                source=record.source + "_cist",
            )
        )
    base_loader = DataLoader(FlowDataset(base_records, label_maps), batch_size=64, shuffle=False)
    aug_loader = DataLoader(FlowDataset(augmented, label_maps), batch_size=64, shuffle=False)
    base = extract_embeddings(model, base_loader, device)["z_app"]
    aug = extract_embeddings(model, aug_loader, device)["z_app"]
    return float(np.mean(np.sum(base * aug, axis=1)))


@torch.no_grad()
def benchmark_latency(model, device: torch.device, runs: int = 100) -> Dict[str, float]:
    from .config import MAX_PACKETS, NET_FEAT_DIM, NET_TIMESTEPS, PKT_FEAT_DIM

    model.eval()
    packet_seq = torch.randn(1, MAX_PACKETS, PKT_FEAT_DIM, device=device)
    network_series = torch.randn(1, NET_TIMESTEPS, NET_FEAT_DIM, device=device)
    packet_mask = torch.zeros(1, MAX_PACKETS, dtype=torch.bool, device=device)
    for _ in range(10):
        _ = model.encode(packet_seq, network_series, packet_mask)
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.encode(packet_seq, network_series, packet_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(latencies)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }
