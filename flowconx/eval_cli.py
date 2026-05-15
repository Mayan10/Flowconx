from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets import FlowDataset, load_csv_records, records_from_dataframe, split_records
from .evaluate import (
    benchmark_latency,
    cist_score,
    closed_set_classification,
    embedding_similarity,
    extract_embeddings,
    leave_one_app_out_generalization,
    prototype_generalization,
)
from .model import FlowConX
from .synthetic import generate_synthetic_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FlowCon-X.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, default=None, help="Path to a real evaluation CSV.")
    parser.add_argument("--synthetic", action="store_true", help="Use generated synthetic data. For smoke tests only.")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--app-col", type=str, default=None)
    parser.add_argument("--service-col", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/eval_metrics.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    label_maps = checkpoint["label_maps"]

    if args.csv:
        records = load_csv_records(
            args.csv,
            label_col=args.label_col,
            app_col=args.app_col,
            service_col=args.service_col,
            limit=args.limit,
        )
    elif args.synthetic:
        df = generate_synthetic_dataframe(flows_per_app=50, include_xr=True, seed=args.seed + 11)
        records = records_from_dataframe(df, label_col="app", app_col="app", service_col="service", source="synthetic_eval")
    else:
        raise ValueError("Real evaluation data is required. Pass --csv path/to/real_flows.csv, or use --synthetic only for smoke tests.")

    train_records, test_records = split_records(records, test_fraction=0.2, seed=args.seed, stratify_by="service")
    train_loader = DataLoader(FlowDataset(train_records, label_maps), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FlowDataset(test_records, label_maps), batch_size=args.batch_size, shuffle=False)

    model = FlowConX(n_conditions=len(label_maps["condition"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    train_emb = extract_embeddings(model, train_loader, device)
    test_emb = extract_embeddings(model, test_loader, device)
    service_sim = embedding_similarity(test_emb["z_app"], test_emb["service"])
    app_sim = embedding_similarity(test_emb["z_flow"], test_emb["app"])
    clf = closed_set_classification(train_emb["z_flow"], train_emb["service"], test_emb["z_flow"], test_emb["service"], k=5)
    proto = prototype_generalization(train_emb["z_app"], train_emb["service"], test_emb["z_app"], test_emb["service"])
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
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()