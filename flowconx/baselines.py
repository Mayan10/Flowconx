from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .datasets import FlowRecord, build_label_maps, load_csv_records, records_from_dataframe, split_records
from .synthetic import generate_synthetic_dataframe


def handcrafted_embedding(record: FlowRecord) -> np.ndarray:
    pkt = record.packet_seq
    net = record.network_series
    pkt_valid = pkt[np.abs(pkt).sum(axis=1) > 0]
    if len(pkt_valid) == 0:
        pkt_valid = pkt[:1]
    parts = [
        pkt_valid.mean(axis=0),
        pkt_valid.std(axis=0),
        pkt_valid.min(axis=0),
        pkt_valid.max(axis=0),
        net.mean(axis=0),
        net.std(axis=0),
        net.min(axis=0),
        net.max(axis=0),
    ]
    emb = np.concatenate(parts).astype(np.float32)
    emb = emb - emb.mean()
    emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
    return emb


def embed_records(records: Sequence[FlowRecord]) -> np.ndarray:
    return np.vstack([handcrafted_embedding(record) for record in records])


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a_norm @ b_norm.T


def embedding_similarity_np(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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


def closed_set_classification_np(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    pred = knn_predict(train_x, train_y, test_x, k=k)
    return {"knn_accuracy": accuracy(pred, test_y), "knn_macro_f1": macro_f1(pred, test_y)}


def prototype_generalization_np(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> Dict[str, float]:
    prototypes = []
    labels = []
    for label in np.unique(train_y):
        mask = train_y == label
        proto = train_x[mask].mean(axis=0)
        proto = proto / max(float(np.linalg.norm(proto)), 1e-12)
        prototypes.append(proto)
        labels.append(label)
    sims = cosine_matrix(test_x, np.vstack(prototypes))
    pred = np.asarray([labels[idx] for idx in np.argmax(sims, axis=1)])
    return {"prototype_accuracy": accuracy(pred, test_y), "prototype_macro_f1": macro_f1(pred, test_y)}


def leave_one_app_out_np(embeddings: np.ndarray, app_labels: np.ndarray, service_labels: np.ndarray) -> Dict[str, float]:
    preds = []
    targets = []
    skipped = 0
    for app in np.unique(app_labels):
        test_mask = app_labels == app
        train_mask = ~test_mask
        prototypes = []
        labels = []
        for service in np.unique(service_labels[train_mask]):
            mask = train_mask & (service_labels == service)
            proto = embeddings[mask].mean(axis=0)
            proto = proto / max(float(np.linalg.norm(proto)), 1e-12)
            prototypes.append(proto)
            labels.append(service)
        if not prototypes:
            skipped += int(np.sum(test_mask))
            continue
        sims = cosine_matrix(embeddings[test_mask], np.vstack(prototypes))
        preds.append(np.asarray([labels[idx] for idx in np.argmax(sims, axis=1)]))
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


def evaluate_baseline(records: Sequence[FlowRecord]) -> Dict[str, object]:
    label_maps = build_label_maps(records)
    train_records, test_records = split_records(records, test_fraction=0.2, seed=42, stratify_by="service")
    train_x = embed_records(train_records)
    test_x = embed_records(test_records)
    train_y = np.asarray([label_maps["service"][record.service] for record in train_records])
    test_y = np.asarray([label_maps["service"][record.service] for record in test_records])
    all_x = embed_records(records)
    all_apps = np.asarray([label_maps["app"][record.app] for record in records])
    all_services = np.asarray([label_maps["service"][record.service] for record in records])
    return {
        "service_similarity": embedding_similarity_np(test_x, test_y),
        "classification": closed_set_classification_np(train_x, train_y, test_x, test_y, k=5),
        "prototype_generalization": prototype_generalization_np(train_x, train_y, test_x, test_y),
        "leave_one_app_out": leave_one_app_out_np(all_x, all_apps, all_services),
        "label_maps": label_maps,
        "train_records": len(train_records),
        "test_records": len(test_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a NumPy handcrafted baseline.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a real flow CSV.")
    parser.add_argument("--synthetic", action="store_true", help="Use generated synthetic data. For smoke tests only.")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--app-col", type=str, default=None)
    parser.add_argument("--service-col", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flows-per-app", type=int, default=80)
    parser.add_argument("--output", type=str, default="outputs/baseline_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv:
        records = load_csv_records(
            args.csv,
            label_col=args.label_col,
            app_col=args.app_col,
            service_col=args.service_col,
            limit=args.limit,
        )
    elif args.synthetic:
        df = generate_synthetic_dataframe(flows_per_app=args.flows_per_app, include_xr=True, seed=123)
        records = records_from_dataframe(df, label_col="app", app_col="app", service_col="service", source="synthetic_baseline")
    else:
        raise ValueError("Real baseline data is required. Pass --csv path/to/real_flows.csv, or use --synthetic only for smoke tests.")
    metrics = evaluate_baseline(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved baseline metrics to {output}")


if __name__ == "__main__":
    main()