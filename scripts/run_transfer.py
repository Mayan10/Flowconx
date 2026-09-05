#!/usr/bin/env python3
"""Cross-dataset transfer, using one trained encoder on both corpora.

    python scripts/run_transfer.py --checkpoint <model.pt> --out results/transfer

The encoder is trained on the source corpus and applied, frozen, to the target.
Both corpora reduce to the same feature schema, so one encoder consumes both
without retraining -- which is the whole point: if the representation is about
traffic behaviour rather than about a corpus, it should transfer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowconx.audit.splits import build_split, ensure_flow_ids  # noqa: E402
from flowconx.data.dataset import build_label_space, encode_split  # noqa: E402
from flowconx.determinism import seed_everything  # noqa: E402
from flowconx.eval.transfer import evaluate_transfer  # noqa: E402
from flowconx.experiment import load_config  # noqa: E402
from flowconx.labels import RareClassPolicy, apply_rare_class_policy  # noqa: E402
from flowconx.models import build_model  # noqa: E402
from flowconx.train.loop import extract_embeddings, select_device  # noqa: E402


def load_split(csv: str, protocol: str, max_packets: int, seed: int, side: str):
    frame = pd.read_csv(csv)
    frame, _ = apply_rare_class_policy(frame, RareClassPolicy(mode="drop", min_class_count=100))
    frame, _ = ensure_flow_ids(frame)
    _, indices = build_split(frame, protocol, seed=seed, label_column="service", dataset_path=csv)
    subset = frame.iloc[indices[side]].reset_index(drop=True)
    space = build_label_space(subset, "service", "none")
    return encode_split(subset, space, "service", max_packets=max_packets), subset


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-dataset transfer evaluation.")
    parser.add_argument("--config", default="configs/cesnet_main.yaml", help="Config the encoder was trained with.")
    parser.add_argument("--checkpoint", required=True, help="model.pt from a --save-model run.")
    parser.add_argument("--source-csv", default="data/processed/cesnet_quic22.csv")
    parser.add_argument("--source-name", default="cesnet_quic22")
    parser.add_argument("--target-csv", default="data/processed/fiveg_traffic.csv")
    parser.add_argument("--target-name", default="fiveg_traffic")
    parser.add_argument("--protocol", default="session_disjoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/transfer")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed_everything(args.seed)
    device = select_device(config.train.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(
        config.model,
        n_nuisance=int(checkpoint.get("label_space", {}).get("n_nuisance", 1)),
        max_len=config.data.max_packets,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Source rows come from the *training* side: those are what the encoder
    # saw, and their prototypes are what a deployment would carry over.
    source, source_frame = load_split(
        args.source_csv, args.protocol, config.data.max_packets, config.data.split_seed, "train"
    )
    target, target_frame = load_split(
        args.target_csv, args.protocol, config.data.max_packets, config.data.split_seed, "test"
    )

    which = config.model.classify_from
    source_emb = extract_embeddings(model, source, device, which)
    target_emb = extract_embeddings(model, target, device, which)

    result = evaluate_transfer(
        source_emb,
        source_frame["service"].iloc[source.features.kept_index].tolist(),
        args.source_name,
        target_emb,
        target_frame["service"].iloc[target.features.kept_index].tolist(),
        args.target_name,
        seed=args.seed,
    )
    result["checkpoint"] = args.checkpoint
    result["protocol"] = args.protocol
    result["seed"] = args.seed

    out = Path(args.out) / f"{args.source_name}_to_{args.target_name}" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "transfer.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"transfer {args.source_name} -> {args.target_name}: {result['status']}")
    if result["status"] == "ok":
        print(f"  shared classes: {result['shared_classes']}")
        print(f"  mapped rows: {result['n_source_rows_mapped']:,} source, {result['n_target_rows_mapped']:,} target")
        print(f"  dropped:     {result['source_rows_dropped']:,} source, {result['target_rows_dropped']:,} target")
        for point in result["enrollment_curve"]:
            tag = "zero-shot" if point["shots"] == 0 else f"k={point['shots']:<4d}"
            print(f"  {tag}  macro-F1 {point['macro_f1_mean']:.4f} ± {point['macro_f1_std']:.4f}")
    else:
        print(f"  reason: {result['reason']}")
    print(f"\nwrote {out / 'transfer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
