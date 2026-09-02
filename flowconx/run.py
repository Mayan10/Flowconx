"""The single entry point.

    python -m flowconx.run --config configs/<name>.yaml --seed 0

Everything the paper reports comes through here. There is no second training
script and nothing important lives in a notebook. The run writes

    results/<experiment>/<dataset>/<split>/seed<n>/
        metrics.json     every number, plus the provenance to attribute it
        config.yaml      the fully resolved config, after defaults and overrides
        history.json     per-epoch training log
        model.pt         weights, when --save-model is passed

and refuses to overwrite an existing metrics.json unless --overwrite is given,
so a sweep cannot silently clobber a result that a table already cites.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .audit.splits import SplitUnavailable, build_split, ensure_flow_ids, write_manifest
from .data.dataset import build_label_space, encode_split
from .determinism import RunProvenance, device_description, seed_everything
from .experiment import ExperimentConfig, load_config, save_config
from .labels import RareClassPolicy, apply_rare_class_policy


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FlowCon-X experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--seed", type=int, default=None, help="Overrides config.seed.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, e.g. --set model.fusion=concat. Repeatable.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing metrics.json.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the config, print it, and stop.")
    return parser.parse_args(argv)


def load_frame(config: ExperimentConfig) -> pd.DataFrame:
    path = Path(config.data.csv)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Build it with `python -m flowconx.data.prepare --source all`."
        )
    frame = pd.read_csv(path, nrows=config.data.limit)
    if config.data.subsample_rows and len(frame) > config.data.subsample_rows:
        frame = _stratified_rows(
            frame, config.data.label_column, config.data.subsample_rows, config.data.split_seed
        )
    policy = RareClassPolicy(
        mode=config.data.rare_class_mode,
        min_class_count=config.data.min_class_count,
        label_column=config.data.label_column,
    )
    frame, _ = apply_rare_class_policy(frame, policy)
    frame, _ = ensure_flow_ids(frame)
    return frame.reset_index(drop=True)


def _stratified_rows(frame: pd.DataFrame, label_column: str, target: int, seed: int) -> pd.DataFrame:
    """Seeded stratified subsample, preserving class balance and capture spread.

    Keyed on the split seed rather than the run seed so that every seed of one
    experiment sees the same rows and the seeds differ only in initialisation
    and batch order -- otherwise the error bars would mix two sources of
    variance and mean nothing.
    """
    rng = np.random.default_rng(seed)
    labels = frame[label_column].astype(str).to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    per_class = max(1, target // len(classes))
    keep: List[int] = []
    for cls in classes:
        pool = np.flatnonzero(labels == cls)
        keep.extend(rng.choice(pool, size=min(per_class, pool.size), replace=False).tolist())
    return frame.iloc[sorted(keep)].reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.set)
    if args.seed is not None:
        config.seed = args.seed
    if args.output_root is not None:
        config.output_root = args.output_root
    config.validate()

    if args.dry_run:
        print(json.dumps(config.as_dict(), indent=2))
        print(f"\nconfig hash: {config.hash()}\nrun dir:     {config.run_dir}")
        return 0

    run_dir = config.run_dir
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(f"{metrics_path} already exists. Pass --overwrite to replace it.", file=sys.stderr)
        return 2

    seeding = seed_everything(config.seed)
    provenance = RunProvenance(
        config_hash=config.hash(),
        config_name=config.name,
        seed=config.seed,
        seeding=seeding,
        device=device_description(config.train.device),
    )

    # ------------------------------------------------------------------ data
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
        print(f"Split protocol {config.data.split_protocol!r} is unavailable: {exc}", file=sys.stderr)
        return 3

    label_space = build_label_space(frame, config.data.label_column, _nuisance_source(config))
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
    held_out = _hold_out_unknown_apps(splits, config, label_space)

    print(f"experiment: {config.name}  hash={config.hash()}  seed={config.seed}")
    print(f"dataset:    {config.data.csv}  split={config.data.split_protocol}")
    print("sizes:      " + "  ".join(f"{side}={len(splits[side]):,}" for side in ("train", "val", "test")))
    print(f"classes:    {label_space.classes}")
    print(f"nuisance:   {label_space.nuisance_source} ({label_space.n_nuisance} values)")

    # --------------------------------------------------------------- training
    from .train.loop import extract_embeddings, select_device, train

    started = time.perf_counter()
    outcome = train(config, splits["train"], splits["val"], label_space)
    train_seconds = time.perf_counter() - started
    device = select_device(config.train.device)

    # ------------------------------------------------------------- evaluation
    from .eval.closed_set import embedding_geometry, evaluate_heads

    which = config.model.classify_from
    embeddings = {side: extract_embeddings(outcome.model, splits[side], device, which) for side in splits}

    from .train.loop import stratified_subsample

    reference_idx = (
        stratified_subsample(
            splits["train"].labels,
            per_class=max(1, config.eval.max_reference_rows // max(label_space.n_classes, 1)),
            seed=config.seed,
        )
        if config.eval.max_reference_rows and len(splits["train"]) > config.eval.max_reference_rows
        else np.arange(len(splits["train"]))
    )
    closed_set = evaluate_heads(
        embeddings["train"][reference_idx],
        splits["train"].labels[reference_idx],
        embeddings["test"],
        splits["test"].labels,
        label_space.classes,
        heads=config.eval.classifier_heads,
        k=config.eval.knn_k,
        seed=config.seed,
        bootstrap_resamples=config.eval.bootstrap_resamples,
    )

    metrics: Dict[str, object] = {
        "experiment": config.name,
        "config_hash": config.hash(),
        "seed": config.seed,
        "config": config.as_dict(),
        "provenance": provenance.finish().as_dict(),
        "data": {
            "csv": config.data.csv,
            "n_rows": int(len(frame)),
            "subsample_rows": config.data.subsample_rows,
            "split_protocol": config.data.split_protocol,
            "split_checksums": manifest.checksums,
            "split_sizes": {side: int(len(splits[side])) for side in splits},
            "label_space": label_space.as_dict(),
            "features": splits["train"].features.describe(),
            "held_out_apps": held_out,
            "classifier_reference_rows": int(len(reference_idx)),
            "classifier_reference_is_subsample": bool(len(reference_idx) < len(splits["train"])),
        },
        "training": {
            "n_parameters": outcome.n_parameters,
            "device": outcome.device,
            "epochs_run": len(outcome.history),
            "best_epoch": outcome.best_epoch,
            "best_val_macro_f1": outcome.best_score,
            "seconds": round(train_seconds, 2),
            "stage_boundaries": outcome.stage_boundaries,
            "active_loss_terms": {k: v for k, v in _active_terms(config).items() if v},
        },
        "closed_set": closed_set,
        "embedding_geometry": {
            side: embedding_geometry(embeddings[side], splits[side].labels, seed=config.seed)
            for side in ("train", "test")
        },
    }

    _run_optional_evaluations(metrics, config, outcome, splits, embeddings, label_space, device)

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(outcome.history, indent=2), encoding="utf-8")
    save_config(config, run_dir / "config.yaml")
    write_manifest(manifest, run_dir / "split_manifest.json.gz")
    if args.save_model:
        import torch

        torch.save(
            {"model": outcome.model.state_dict(), "config": config.as_dict(), "label_space": label_space.as_dict()},
            run_dir / "model.pt",
        )

    print(f"\nwrote {metrics_path}")
    for head, report in closed_set.items():
        ci = report.get("macro_f1_ci95", {})
        interval = f" [{ci['lo']:.4f}, {ci['hi']:.4f}]" if ci else ""
        print(
            f"  {head:10s} macro-F1={report['macro_f1']:.4f}{interval}  "
            f"bal-acc={report['balanced_accuracy']:.4f}  acc={report['accuracy']:.4f}"
        )
    return 0


def _nuisance_source(config: ExperimentConfig) -> str:
    """Which provenance column the adversary is asked to remove."""
    if not config.model.adversarial_head or config.loss.lambda_adversarial <= 0:
        return "none"
    return "week" if config.data.dataset == "cesnet_quic22" else "capture_id"


def _active_terms(config: ExperimentConfig) -> Dict[str, float]:
    return {
        "service_supcon": config.loss.lambda_service_supcon,
        "flow_supcon": config.loss.lambda_flow_supcon,
        "app_supcon": config.loss.lambda_app_supcon,
        "prototype": config.loss.lambda_prototype,
        "disentangle": config.loss.lambda_disentangle,
        "adversarial": config.loss.lambda_adversarial,
        "pair_margin": config.loss.lambda_pair_margin,
        "flow_pair_margin": config.loss.lambda_flow_pair_margin,
    }


def _hold_out_unknown_apps(splits, config: ExperimentConfig, label_space) -> List[str]:
    """Remove configured apps from train and val, leaving them only in test.

    This is what makes the open-set evaluation honest: the "unknown" apps must
    never have been seen, including in validation, or model selection leaks
    them back in.
    """
    unknown = [app.lower() for app in config.data.unknown_apps]
    if not unknown:
        return []
    import numpy as np

    for side in ("train", "val"):
        split = splits[side]
        if "app" not in split.frame.columns:
            continue
        keep = ~split.frame["app"].astype(str).str.lower().isin(unknown)
        idx = np.flatnonzero(keep.to_numpy())
        splits[side] = _subset(split, idx)
    _ = label_space
    return unknown


def _subset(split, idx: "np.ndarray"):
    from .data.dataset import EncodedSplit
    from .features.packet import FeatureBundle

    features = split.features
    return EncodedSplit(
        features=FeatureBundle(
            packet_seq=features.packet_seq[idx],
            packet_mask=features.packet_mask[idx],
            flow_features=features.flow_features[idx],
            kept_index=features.kept_index[idx],
            n_dropped=features.n_dropped,
            observed_packets=features.observed_packets,
            packet_feature_names=features.packet_feature_names,
            flow_feature_names=features.flow_feature_names,
        ),
        labels=split.labels[idx],
        app_labels=split.app_labels[idx],
        nuisance_labels=split.nuisance_labels[idx],
        frame=split.frame.iloc[idx].reset_index(drop=True),
    )


def _run_optional_evaluations(metrics, config, outcome, splits, embeddings, label_space, device) -> None:
    """Phase 4 evaluation modes, each behind its own config flag."""
    if config.eval.cost:
        from .eval.cost import measure_cost

        metrics["cost"] = measure_cost(outcome.model, splits["test"], device, config)
    if config.eval.open_set:
        from .eval.open_set import evaluate_open_set

        metrics["open_set"] = evaluate_open_set(
            embeddings["train"], splits["train"], embeddings["test"], splits["test"], config, label_space
        )
    if config.eval.few_shot:
        from .eval.few_shot import evaluate_few_shot

        metrics["few_shot"] = evaluate_few_shot(
            embeddings["train"], splits["train"], embeddings["test"], splits["test"], config, label_space
        )
    if config.eval.drift:
        from .eval.drift import evaluate_drift

        metrics["drift"] = evaluate_drift(
            embeddings["train"], splits["train"], embeddings["test"], splits["test"], config, label_space
        )
    if config.eval.early_classification:
        from .eval.early import evaluate_early_classification

        metrics["early_classification"] = evaluate_early_classification(
            outcome.model, splits, config, label_space, device
        )
    if config.eval.robustness:
        from .eval.robustness import evaluate_robustness

        metrics["robustness"] = evaluate_robustness(outcome.model, splits, config, label_space, device)
    if config.eval.probes:
        from .eval.probes import evaluate_nuisance_probes

        metrics["probes"] = evaluate_nuisance_probes(embeddings, splits, config, label_space)


if __name__ == "__main__":
    raise SystemExit(main())
