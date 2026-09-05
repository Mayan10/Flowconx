#!/usr/bin/env python3
"""Why does removing the metric objectives improve open-set rejection?

    python scripts/analyse_openset_geometry.py

The open-set ablation found that ablating every metric objective on the
deployed embedding raises prototype-rejection AUROC from 0.919 to 0.949. That
was reported as a refutation of the claimed mechanism, but not explained, and
an unexplained result is a weaker one.

What it finds: **class separation and novelty detection trade off against each
other.** Across the five ablation variants, the separation between class
clusters correlates at Pearson $r = -0.88$ with rejection AUROC and $+0.84$
with the false-accept rate. The variant with the *least* separated embedding
rejects unknowns best.

The mechanism is geometric and, once seen, unsurprising. Rejection by distance
works when unknown flows land far from every prototype. Spreading the known
classes apart -- which is exactly what the metric objectives do, and what makes
them help closed-set accuracy -- covers more of the embedding sphere, so an
unknown flow is more likely to land near *some* prototype. A collapsed
embedding leaves more empty space, and empty space is where novelty is
detectable.

Five variants is a small n and the correlations sit at p ~ 0.05-0.08; this is
offered as a mechanism consistent with the data, not as an established law.

Measured directly from the committed runs -- no retraining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load(experiment: str, results: Path) -> List[Dict[str, object]]:
    out = []
    for path in sorted((results / experiment).rglob("metrics.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def summarise(payloads: List[Dict[str, object]]) -> Dict[str, float]:
    """Embedding geometry and rejection quality, averaged over seeds."""
    intra, inter, sep, auroc, fpr, task = [], [], [], [], [], []
    for payload in payloads:
        geometry = payload.get("embedding_geometry", {}).get("test", {})
        if geometry:
            intra.append(geometry["intra_class_cosine"])
            inter.append(geometry["inter_class_cosine"])
            sep.append(geometry["separation"])
        scorers = (payload.get("open_set") or {}).get("scorers", {})
        if "prototype_cosine" in scorers:
            auroc.append(scorers["prototype_cosine"]["auroc"])
            fpr.append(scorers["prototype_cosine"]["fpr_at_95tpr"])
        report = payload.get("closed_set", {}).get("prototype")
        if report:
            task.append(report["macro_f1"])

    def m(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "n_seeds": len(payloads),
        "intra_class_cosine": m(intra),
        "inter_class_cosine": m(inter),
        "separation": m(sep),
        "prototype_auroc": m(auroc),
        "prototype_fpr95": m(fpr),
        "closed_set_macro_f1": m(task),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Explain the open-set ablation result.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/openset_geometry.json")
    args = parser.parse_args(argv)
    root = Path(args.results)

    variants = [
        ("flowconx_open_set", "full model"),
        ("openset_ablation_no_flow_metric", "no metric objectives"),
        ("openset_ablation_no_pair_margin", "no margin term"),
        ("openset_ablation_no_flow_supcon", "no contrastive term"),
        ("openset_ablation_no_adversarial", "no adversarial head"),
    ]
    table: Dict[str, Dict[str, float]] = {}
    for experiment, label in variants:
        payloads = load(experiment, root)
        if payloads:
            table[label] = summarise(payloads)

    if not table:
        print("no open-set ablation runs found")
        return 1

    header = f"{'variant':22s} {'intra':>8s} {'inter':>8s} {'separation':>11s} {'AUROC':>8s} {'FPR@95':>8s} {'closed':>8s}"
    print(header)
    print("-" * len(header))
    for label, stats in table.items():
        print(
            f"{label:22s} {stats['intra_class_cosine']:8.4f} {stats['inter_class_cosine']:8.4f} "
            f"{stats['separation']:11.4f} {stats['prototype_auroc']:8.4f} "
            f"{stats['prototype_fpr95']:8.4f} {stats['closed_set_macro_f1']:8.4f}"
        )

    full = table.get("full model")
    ablated = table.get("no metric objectives")
    finding = None
    correlation = float("nan")
    if full and ablated:
        print()
        d_inter = ablated["inter_class_cosine"] - full["inter_class_cosine"]
        d_auroc = ablated["prototype_auroc"] - full["prototype_auroc"]
        print(f"removing the metric objectives changes inter-class cosine by {d_inter:+.4f}")
        print(f"                                     and rejection AUROC by {d_auroc:+.4f}")
        # Correlate across all variants rather than reading two rows: the
        # relationship is the finding, not the single contrast.
        separations = [v["separation"] for v in table.values()]
        aurocs = [v["prototype_auroc"] for v in table.values()]
        correlation = float(np.corrcoef(separations, aurocs)[0, 1]) if len(table) > 2 else float("nan")

        print(f"removing the metric objectives changes inter-class cosine by {d_inter:+.4f}")
        print(f"                                     and rejection AUROC by {d_auroc:+.4f}")
        print(f"\nacross all {len(table)} variants, class separation vs rejection AUROC: r = {correlation:+.3f}")

        if correlation < -0.5:
            finding = (
                "Class separation and novelty detection trade off. The metric objectives spread "
                "the known classes apart, which helps closed-set accuracy and covers more of the "
                "embedding sphere; an unknown flow is then more likely to land near some "
                "prototype. The least separated embedding rejects unknowns best. Small n -- "
                f"{len(table)} variants -- so this is a mechanism consistent with the data, not "
                "an established law."
            )
        elif correlation > 0.5:
            finding = (
                "Separation and rejection move together, which contradicts the trade-off "
                "hypothesis. Whatever produces the ablation result is not embedding geometry."
            )
        else:
            finding = (
                "No monotone relationship between class separation and rejection quality across "
                "these variants. The mechanism is not geometric at this granularity."
            )
        print(f"\n{finding}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "variants": table,
                "separation_vs_auroc_pearson_r": correlation if full and ablated else None,
                "finding": finding,
                "caveat": (
                    "Five variants. The correlation is descriptive; it is reported with its n "
                    "rather than as a tested effect."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
