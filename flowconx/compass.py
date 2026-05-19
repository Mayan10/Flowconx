from __future__ import annotations

import argparse
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import FlowDataset, FlowRecord, load_csv_records
from .evaluate import extract_embeddings
from .model import FlowConX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream flows and print nearest service prototypes.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calibration-csv", type=str, required=True, help="Labeled CSV used to build service prototypes.")
    parser.add_argument("--stream-csv", type=str, required=True, help="CSV rows to classify.")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--app-col", type=str, default=None)
    parser.add_argument("--service-col", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
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


def id_to_name(mapping: Dict[str, int]) -> Dict[int, str]:
    return {idx: name for name, idx in mapping.items()}


def build_prototypes(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prototypes = []
    proto_labels = []
    for label in np.unique(labels):
        mask = labels == label
        proto = embeddings[mask].mean(axis=0)
        proto = proto / max(float(np.linalg.norm(proto)), 1e-12)
        prototypes.append(proto)
        proto_labels.append(label)
    return np.vstack(prototypes), np.asarray(proto_labels)


def cosine_scores(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix @ vector


@torch.no_grad()
def encode_one(model: FlowConX, record: FlowRecord, label_maps: Dict[str, Dict[str, int]], device: torch.device) -> np.ndarray:
    loader = DataLoader(FlowDataset([record], label_maps), batch_size=1, shuffle=False)
    batch = next(iter(loader))
    outputs = model(
        batch["packet_seq"].to(device),
        batch["network_series"].to(device),
        batch["packet_mask"].to(device),
        grl_scale=0.0,
    )
    return outputs["z_app"].cpu().numpy()[0]


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    label_maps = checkpoint["label_maps"]
    service_names = id_to_name(label_maps["service"])

    model = FlowConX(n_conditions=len(label_maps["condition"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    calibration_records = load_csv_records(
        args.calibration_csv,
        label_col=args.label_col,
        app_col=args.app_col,
        service_col=args.service_col,
    )
    calibration_loader = DataLoader(FlowDataset(calibration_records, label_maps), batch_size=64, shuffle=False)
    calibration = extract_embeddings(model, calibration_loader, device)
    prototypes, proto_labels = build_prototypes(calibration["z_app"], calibration["service"])

    stream_records = load_csv_records(
        args.stream_csv,
        label_col=args.label_col,
        app_col=args.app_col,
        service_col=args.service_col,
        limit=args.limit,
    )
    for idx, record in enumerate(stream_records, start=1):
        emb = encode_one(model, record, label_maps, device)
        scores = cosine_scores(emb, prototypes)
        order = np.argsort(-scores)[: args.top_k]
        pairs = []
        for proto_idx in order:
            service_id = int(proto_labels[proto_idx])
            pairs.append(f"{service_names.get(service_id, str(service_id))}:{scores[proto_idx]:.3f}")
        print(f"flow={idx} app={record.app} service={record.service} compass=" + ", ".join(pairs))
        if args.delay > 0:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
