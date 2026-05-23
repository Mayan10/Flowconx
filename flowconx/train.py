from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .config import FlowConXConfig
from .datasets import FlowDataset, build_label_maps, load_csv_records, split_records
from .evaluate import (
    benchmark_latency,
    cist_score,
    closed_set_classification,
    embedding_similarity,
    extract_embeddings,
    leave_one_app_out_generalization,
    prototype_generalization,
)
from .losses import FlowConXLoss
from .memory import EmbeddingMemoryBank, PrototypeBank
from .model import FlowConX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FlowCon-X.")
    parser.add_argument("--csv", type=str, required=True, help="Path to a real flow CSV for training.")
    parser.add_argument("--label-col", type=str, default=None, help="Column containing app or service labels.")
    parser.add_argument("--app-col", type=str, default=None, help="Optional app label column.")
    parser.add_argument("--service-col", type=str, default=None, help="Optional service label column.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick experiments.")
    parser.add_argument("--output-dir", type=str, default="outputs/run", help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augment-count", type=int, default=0, help="Optional counterfactual network augmentation per real flow. Default is 0 for real-data-only training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--memory-per-class", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07, help="SupCon and prototype temperature.")
    parser.add_argument("--lambda-app", type=float, default=0.35, help="Weight for app-level SupCon loss.")
    parser.add_argument("--lambda-proto", type=float, default=0.10, help="Weight for service prototype alignment loss.")
    parser.add_argument("--lambda-dis", type=float, default=0.25, help="Weight for app/network disentanglement loss.")
    parser.add_argument("--lambda-adv", type=float, default=0.15, help="Weight for network-condition adversary loss.")
    parser.add_argument("--lambda-pair", type=float, default=0.0, help="Weight for pairwise service margin loss.")
    parser.add_argument("--lambda-flow-service", type=float, default=0.0, help="Weight for service SupCon loss on fused z_flow embeddings.")
    parser.add_argument("--lambda-flow-pair", type=float, default=0.0, help="Weight for pairwise service margin loss on fused z_flow embeddings.")
    parser.add_argument("--pair-negative-margin", type=float, default=0.20, help="Maximum target cosine for different service labels.")
    parser.add_argument("--pair-positive-target", type=float, default=0.75, help="Minimum target cosine for matching service labels.")
    parser.add_argument("--eval-max-train", type=int, default=None, help="Optional stratified cap for train embeddings used by heavy eval metrics.")
    parser.add_argument("--eval-max-test", type=int, default=None, help="Optional stratified cap for test embeddings used by heavy eval metrics.")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Optional checkpoint to continue training from.")
    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_records_from_args(args: argparse.Namespace):
    return load_csv_records(
        args.csv,
        label_col=args.label_col,
        app_col=args.app_col,
        service_col=args.service_col,
        limit=args.limit,
    )


def average_logs(logs: List[Dict[str, float]]) -> Dict[str, float]:
    if not logs:
        return {}
    keys = logs[0].keys()
    return {key: sum(item[key] for item in logs) / len(logs) for key in keys}


def capped_embeddings(embeddings: Dict[str, object], label_key: str, limit: Optional[int], seed: int) -> Dict[str, object]:
    if limit is None:
        return embeddings
    labels = embeddings[label_key]
    if len(labels) <= limit:
        return embeddings
    import numpy as np

    rng = np.random.default_rng(seed)
    selected = []
    unique = np.unique(labels)
    per_label = max(1, limit // max(len(unique), 1))
    for label in unique:
        idx = np.flatnonzero(labels == label)
        take = min(len(idx), per_label)
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
        extra = min(len(remaining), limit - len(selected))
        if extra > 0:
            selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    selected_arr = np.asarray(sorted(selected[:limit]))
    return {key: value[selected_arr] for key, value in embeddings.items()}


def train_one_epoch(
    model: FlowConX,
    loss_fn: FlowConXLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    memory: EmbeddingMemoryBank,
    prototype_bank: PrototypeBank,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    logs: List[Dict[str, float]] = []
    grl_scale = min(1.0, epoch / 5.0)
    for batch in loader:
        packet_seq = batch["packet_seq"].to(device)
        network_series = batch["network_series"].to(device)
        packet_mask = batch["packet_mask"].to(device)
        service_labels = batch["service_label"].to(device)
        app_labels = batch["app_label"].to(device)
        condition_labels = batch["condition_label"].to(device)

        memory_tuple = memory.sample(device=device)
        outputs = model(packet_seq, network_series, packet_mask, grl_scale=grl_scale)
        loss, info = loss_fn(
            outputs,
            service_labels=service_labels,
            app_labels=app_labels,
            condition_labels=condition_labels,
            memory=memory_tuple,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            memory.add(outputs["z_app"], service_labels)
            prototype_bank.update_trusted(outputs["z_app"], service_labels)
        logs.append(info)
    return average_logs(logs)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(f"Device: {device}")

    records = load_records_from_args(args)
    if len(records) < 20:
        raise ValueError("Need at least 20 flow records for a meaningful train/test split.")
    train_records, test_records = split_records(records, test_fraction=0.2, seed=args.seed, stratify_by="service")
    label_maps = build_label_maps(records, FlowConXConfig())

    train_ds = FlowDataset(train_records, label_maps, augment_count=args.augment_count, augment_seed=args.seed)
    test_ds = FlowDataset(test_records, label_maps, augment_count=0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FlowConX(n_conditions=len(label_maps["condition"])).to(device)
    loss_fn = FlowConXLoss(
        n_services=len(label_maps["service"]),
        n_apps=len(label_maps["app"]),
        n_conditions=len(label_maps["condition"]),
        temperature=args.temperature,
        lambda_app=args.lambda_app,
        lambda_proto=args.lambda_proto,
        lambda_dis=args.lambda_dis,
        lambda_adv=args.lambda_adv,
        lambda_pair=args.lambda_pair,
        lambda_flow_service=args.lambda_flow_service,
        lambda_flow_pair=args.lambda_flow_pair,
        pair_negative_margin=args.pair_negative_margin,
        pair_positive_target=args.pair_positive_target,
    ).to(device)
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "loss" in checkpoint:
            loss_fn.load_state_dict(checkpoint["loss"], strict=False)
        print(f"Resumed weights from {args.resume_checkpoint}")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    memory = EmbeddingMemoryBank(max_per_class=args.memory_per_class)
    prototype_bank = PrototypeBank(n_classes=len(label_maps["service"]), emb_dim=256)

    print(f"Records: {len(records)}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Services: {label_maps['service']}")
    print(f"Apps: {len(label_maps['app'])}")

    history = []
    for epoch in range(1, args.epochs + 1):
        info = train_one_epoch(model, loss_fn, train_loader, optimizer, memory, prototype_bank, device, epoch)
        history.append({"epoch": epoch, **info})
        print(
            "Epoch "
            f"{epoch:03d} total={info['total']:.4f} "
            f"service={info['service_supcon']:.4f} "
            f"flow_service={info['flow_service_supcon']:.4f} "
            f"app={info['app_supcon']:.4f} "
            f"proto={info['prototype']:.4f} "
            f"dis={info['disentangle']:.5f} "
            f"adv={info['condition_adv']:.4f} "
            f"pair={info['pair_margin']:.4f} "
            f"flow_pair={info['flow_pair_margin']:.4f}"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "loss": loss_fn.state_dict(),
            "label_maps": label_maps,
            "args": vars(args),
            "prototype_bank": prototype_bank.state_dict(),
        },
        output_dir / "flowconx_checkpoint.pt",
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    train_emb = extract_embeddings(model, train_loader, device)
    test_emb = extract_embeddings(model, test_loader, device)
    train_eval = capped_embeddings(train_emb, "service", args.eval_max_train, args.seed + 101)
    test_eval = capped_embeddings(test_emb, "service", args.eval_max_test, args.seed + 202)
    service_sim = embedding_similarity(test_eval["z_app"], test_eval["service"])
    app_sim = embedding_similarity(test_eval["z_flow"], test_eval["app"])
    clf = closed_set_classification(
        train_eval["z_flow"],
        train_eval["service"],
        test_eval["z_flow"],
        test_eval["service"],
        k=5,
    )
    proto = prototype_generalization(
        train_eval["z_app"],
        train_eval["service"],
        test_eval["z_app"],
        test_eval["service"],
    )
    loo_app = leave_one_app_out_generalization(test_eval["z_app"], test_eval["app"], test_eval["service"])
    cist = cist_score(model, test_records, label_maps, device)
    latency = benchmark_latency(model, device)

    metrics = {
        "service_similarity": service_sim,
        "app_similarity": app_sim,
        "classification": clf,
        "prototype_generalization": proto,
        "leave_one_app_out": loo_app,
        "cist_score": cist,
        "latency": latency,
        "label_maps": label_maps,
        "eval_caps": {
            "train_embeddings": len(train_eval["service"]),
            "test_embeddings": len(test_eval["service"]),
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("KPI summary:")
    print(f"  Service intra cosine: {service_sim['intra']:.4f}")
    print(f"  Service inter cosine: {service_sim['inter']:.4f}")
    print(f"  k-NN service accuracy: {clf['knn_accuracy'] * 100:.2f}%")
    if clf["svm_accuracy"] == clf["svm_accuracy"]:
        print(f"  SVM service accuracy: {clf['svm_accuracy'] * 100:.2f}%")
    print(f"  Prototype accuracy: {proto['prototype_accuracy'] * 100:.2f}%")
    print(f"  Leave-one-app accuracy: {loo_app['leave_one_app_accuracy'] * 100:.2f}%")
    print(f"  CIST score: {cist:.4f}")
    print(f"  Latency mean ms: {latency['mean_ms']:.2f}")
    print(f"Saved checkpoint and metrics to {output_dir}")


if __name__ == "__main__":
    main()
