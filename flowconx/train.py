from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .config import FlowConXConfig
from .datasets import FlowDataset, build_label_maps, load_csv_records, records_from_dataframe, split_records
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
from .synthetic import generate_synthetic_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FlowCon-X.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a real flow CSV for training.")
    parser.add_argument("--synthetic", action="store_true", help="Use generated synthetic data. For smoke tests only.")
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
    parser.add_argument("--flows-per-app", type=int, default=80, help="Synthetic flows per app.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--memory-per-class", type=int, default=512)
    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_records_from_args(args: argparse.Namespace):
    if args.csv:
        return load_csv_records(
            args.csv,
            label_col=args.label_col,
            app_col=args.app_col,
            service_col=args.service_col,
            limit=args.limit,
        )
    if args.synthetic:
        df = generate_synthetic_dataframe(flows_per_app=args.flows_per_app, include_xr=True, seed=args.seed)
        synthetic_path = Path(args.output_dir) / "synthetic_flows.csv"
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(synthetic_path, index=False)
        return records_from_dataframe(df, label_col="app", app_col="app", service_col="service", source="synthetic", limit=args.limit)
    raise ValueError("Real training data is required. Pass --csv path/to/real_flows.csv, or use --synthetic only for smoke tests.")


def average_logs(logs: List[Dict[str, float]]) -> Dict[str, float]:
    if not logs:
        return {}
    keys = logs[0].keys()
    return {key: sum(item[key] for item in logs) / len(logs) for key in keys}


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
    ).to(device)
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
            f"app={info['app_supcon']:.4f} "
            f"proto={info['prototype']:.4f} "
            f"dis={info['disentangle']:.5f} "
            f"adv={info['condition_adv']:.4f}"
        )

    train_emb = extract_embeddings(model, train_loader, device)
    test_emb = extract_embeddings(model, test_loader, device)
    service_sim = embedding_similarity(test_emb["z_app"], test_emb["service"])
    app_sim = embedding_similarity(test_emb["z_flow"], test_emb["app"])
    clf = closed_set_classification(
        train_emb["z_flow"],
        train_emb["service"],
        test_emb["z_flow"],
        test_emb["service"],
        k=5,
    )
    proto = prototype_generalization(
        train_emb["z_app"],
        train_emb["service"],
        test_emb["z_app"],
        test_emb["service"],
    )
    loo_app = leave_one_app_out_generalization(test_emb["z_app"], test_emb["app"], test_emb["service"])
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
    }

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
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

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
