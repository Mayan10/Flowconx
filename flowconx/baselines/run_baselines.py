"""Run the deep baselines on the identical splits.

    python -m flowconx.baselines.run_baselines --config configs/cesnet_main.yaml --seed 0

Writes into the same ``results/<experiment>/<dataset>/<split>/seed<n>/``
layout as ``flowconx.run``, with ``experiment`` set to ``baseline_<name>``, so
the aggregator and the table generator treat baselines and the model
identically and no separate bookkeeping can drift out of sync.

The data path, the split, the rare-class policy, the feature budget, the
optimiser, the epoch budget and the early-stopping rule all come from the same
config the model was run with. That is the point: a difference in the table
must be a difference in the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ..audit.splits import SplitUnavailable, build_split
from ..data.dataset import build_label_space, encode_split
from ..determinism import RunProvenance, device_description, seed_everything
from ..eval.cost import measure_callable_latency
from ..experiment import load_config, save_config
from ..metrics import classification_report, top_confusions
from .deep import DEEP_BASELINES, predict, train_baseline


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the deep baselines.")
    parser.add_argument("--config", required=True, help="The same config the model was run with.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", nargs="*", default=sorted(DEEP_BASELINES))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.set)
    config.seed = args.seed
    if args.output_root:
        config.output_root = args.output_root

    from ..run import load_frame

    frame = load_frame(config)
    try:
        manifest, indices = build_split(
            frame,
            config.data.split_protocol,
            seed=config.data.split_seed,
            val_fraction=config.data.val_fraction,
            test_fraction=config.data.test_fraction,
            label_column=config.data.label_column,
            dataset_path=config.data.csv,
        )
    except SplitUnavailable as exc:
        print(f"Split protocol {config.data.split_protocol!r} unavailable: {exc}", file=sys.stderr)
        return 3

    label_space = build_label_space(frame, config.data.label_column, "none")
    splits = {
        side: encode_split(
            frame.iloc[indices[side]].reset_index(drop=True),
            label_space,
            config.data.label_column,
            max_packets=config.data.max_packets,
            observed_packets=config.data.observed_packets,
        )
        for side in ("train", "val", "test")
    }

    from ..train.loop import select_device

    device = select_device(config.train.device)
    print(f"baselines on {config.data.dataset} / {config.data.split_protocol} / seed {args.seed}")
    print("sizes: " + "  ".join(f"{k}={len(v):,}" for k, v in splits.items()))

    exit_code = 0
    for name in args.baselines:
        run_dir = (
            Path(config.output_root)
            / f"baseline_{name}"
            / config.data.dataset
            / config.data.split_protocol
            / f"seed{args.seed}"
        )
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not args.overwrite:
            print(f"  {name:18s} already present, skipping")
            continue

        seeding = seed_everything(args.seed)
        provenance = RunProvenance(
            config_hash=config.hash(),
            config_name=f"baseline_{name}",
            seed=args.seed,
            seeding=seeding,
            device=device_description(config.train.device),
        )
        started = time.perf_counter()
        model, history = train_baseline(
            name,
            splits["train"],
            splits["val"],
            n_classes=label_space.n_classes,
            seed=args.seed,
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
            device=device,
            early_stop_patience=config.train.early_stop_patience or 6,
        )
        train_seconds = time.perf_counter() - started

        predictions = predict(model, splits["test"], device)
        labels = np.arange(label_space.n_classes)
        report = classification_report(
            predictions,
            splits["test"].labels,
            labels=labels,
            bootstrap=config.eval.bootstrap_resamples > 0,
            seed=args.seed,
        )
        report["per_class_f1"] = {label_space.classes[int(k)]: v for k, v in report["per_class_f1"].items()}
        report["support"] = {label_space.classes[int(k)]: v for k, v in report["support"].items()}
        report["labels"] = list(label_space.classes)
        report["top_confusions"] = [
            {
                **item,
                "true": label_space.classes[int(item["true"])],
                "predicted": label_space.classes[int(item["predicted"])],
            }
            for item in top_confusions(predictions, splits["test"].labels, labels=labels, k=8)
        ]

        spec = DEEP_BASELINES[name].SPEC
        metrics: Dict[str, object] = {
            "experiment": f"baseline_{name}",
            "baseline": name,
            "reference": spec.reference,
            "deviation_from_original": spec.deviation,
            "config_hash": config.hash(),
            "seed": args.seed,
            "config": config.as_dict(),
            "provenance": provenance.finish().as_dict(),
            "data": {
                "csv": config.data.csv,
                "split_protocol": config.data.split_protocol,
                "split_checksums": manifest.checksums,
                "split_sizes": {side: int(len(splits[side])) for side in splits},
                "label_space": label_space.as_dict(),
            },
            "training": {
                "n_parameters": int(sum(p.numel() for p in model.parameters())),
                "device": str(device),
                "epochs_run": len(history),
                "seconds": round(train_seconds, 2),
                # Recorded so the cost table can state honestly that a baseline
                # needs a fine-tuning run to add a class, whereas the prototype
                # head needs k forward passes.
                "enrollment_requires_retraining": True,
            },
            # Same key as flowconx.run, so the aggregator needs no special case.
            "closed_set": {"softmax": report},
            "cost": _baseline_cost(model, splits["test"], device),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_config(config, run_dir / "config.yaml")
        print(
            f"  {name:18s} macro-F1={report['macro_f1']:.4f}  bal-acc={report['balanced_accuracy']:.4f}  "
            f"params={metrics['training']['n_parameters']:,}  {train_seconds:.0f}s"
        )
    return exit_code


def _baseline_cost(model, split, device) -> Dict[str, object]:
    """Latency and size, measured the same way as for the model."""
    import torch

    from ..eval.cost import model_size_bytes

    budget = split.features.observed_packets
    packet_dim = split.features.packet_seq.shape[-1]
    flow_dim = split.features.flow_features.shape[-1]
    packet_seq = torch.zeros(1, budget, packet_dim, device=device)
    flow_features = torch.zeros(1, flow_dim, device=device)
    packet_mask = torch.zeros(1, budget, dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        forward = measure_callable_latency(lambda: model(packet_seq, flow_features, packet_mask), runs=200)
    return {
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "model_size_bytes": model_size_bytes(model),
        "observed_packets": int(budget),
        "device": str(device),
        "forward_batch1": forward,
        "note": "Forward pass only. The end-to-end path is measured for the model in flowconx/eval/cost.py.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
