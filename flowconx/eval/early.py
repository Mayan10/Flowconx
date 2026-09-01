"""Early classification: accuracy against the number of observed packets.

A deployment does not get the whole flow. It gets the first few packets and
has to decide. The curve from packet 1 to packet 20 is the quantity that
actually determines whether a classifier can sit inline, and it is more
informative than any single full-flow number.

The encoder is frozen; only the input budget changes. Prototypes are rebuilt
at each budget from the training split at that same budget, because a
prototype computed on 32 packets does not describe an embedding computed on 3.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..features.packet import truncate
from ..metrics import balanced_accuracy, macro_f1
from .closed_set import class_prototypes
from .few_shot import predict_from_prototypes


def evaluate_early_classification(model, splits, config, label_space, device) -> Dict[str, object]:
    from ..data.dataset import EncodedSplit
    from ..train.loop import extract_embeddings

    budgets = sorted({b for b in config.eval.early_packet_budgets if b > 0})
    available = splits["train"].features.observed_packets
    budgets = [b for b in budgets if b <= available]
    if not budgets:
        return {"status": "skipped", "reason": f"no configured budget is <= observed_packets ({available})"}

    labels = np.arange(label_space.n_classes)
    curve: List[Dict[str, object]] = []
    for budget in budgets:
        views = {
            side: EncodedSplit(
                features=truncate(splits[side].features, budget),
                labels=splits[side].labels,
                app_labels=splits[side].app_labels,
                nuisance_labels=splits[side].nuisance_labels,
                frame=splits[side].frame,
            )
            for side in ("train", "test")
        }
        train_emb = extract_embeddings(model, views["train"], device, which=config.model.classify_from)
        test_emb = extract_embeddings(model, views["test"], device, which=config.model.classify_from)
        prototypes = class_prototypes(train_emb, views["train"].labels, label_space.n_classes)
        predictions = predict_from_prototypes(prototypes, test_emb)
        curve.append(
            {
                "observed_packets": int(budget),
                "macro_f1": macro_f1(predictions, views["test"].labels, labels),
                "balanced_accuracy": balanced_accuracy(predictions, views["test"].labels, labels),
            }
        )

    full = curve[-1]["macro_f1"] if curve else 0.0
    for entry in curve:
        entry["fraction_of_full_budget_macro_f1"] = float(entry["macro_f1"] / full) if full else float("nan")
    return {"status": "ok", "trained_on_packets": int(available), "curve": curve}
