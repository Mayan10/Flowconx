"""Cross-dataset transfer: train on one corpus, test on another.

The natural test of whether anything here generalises. Our two corpora label
different taxonomies -- 5G Traffic uses six service categories from its own
directory structure, CESNET-QUIC22 eighteen from its publishers' catalogue --
so a transfer evaluation needs an explicit mapping onto shared classes, and
that mapping is a judgement call rather than a fact. It is written down here,
in one place, with its reasoning, so a reader can disagree with a specific line
rather than with an opaque result.

Classes that do not map are dropped, not forced. Forcing `mail` or
`authentication_services` into a taxonomy that has no such notion would
manufacture a transfer result out of a labelling decision.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..metrics import balanced_accuracy, classification_report, macro_f1

# Shared classes between the two corpora, with the reasoning for each line.
# Only categories whose *behaviour* should plausibly transfer are included.
SHARED_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "streaming": {
        # Stored and live video. CESNET's `streaming_media` covers both; 5G
        # separates them, and both 5G categories are long-lived high-volume
        # downstream flows, which is the behaviour being transferred.
        "fiveg_traffic": ["live_streaming", "stored_streaming"],
        "cesnet_quic22": ["streaming_media"],
    },
    "gaming": {
        # 5G separates cloud game streaming from native online games. The
        # former is really video streaming and is excluded on purpose; only
        # `online_game` maps, because CESNET's `games` category is native
        # game traffic.
        "fiveg_traffic": ["online_game"],
        "cesnet_quic22": ["games"],
    },
    "conferencing": {
        "fiveg_traffic": ["video_conferencing"],
        "cesnet_quic22": ["videoconferencing"],
    },
}

# Deliberately unmapped, recorded so the omission is a decision rather than an
# oversight.
UNMAPPED = {
    "fiveg_traffic": {
        "game_streaming": "cloud game streaming is video, not game traffic; CESNET has no equivalent",
        "metaverse": "no CESNET category corresponds",
    },
    "cesnet_quic22": {
        "advertising": "no 5G equivalent",
        "analytics_&_telemetry": "no 5G equivalent",
        "antivirus": "no 5G equivalent",
        "authentication_services": "no 5G equivalent",
        "blogs_&_news": "no 5G equivalent",
        "default": "catch-all, not a behaviour class",
        "e_commerce": "no 5G equivalent",
        "file_sharing": "5G has no bulk-transfer capture",
        "information_systems": "no 5G equivalent",
        "instant_messaging": "no 5G equivalent",
        "mail": "no 5G equivalent",
        "music": "audio-only streaming; 5G captures are video, so behaviour differs",
        "other_services_and_apis": "catch-all, not a behaviour class",
        "search": "no 5G equivalent",
        "social": "no 5G equivalent",
    },
}


def map_to_shared(labels: Sequence[str], dataset: str) -> np.ndarray:
    """Map a corpus's own labels onto the shared taxonomy; -1 where unmapped."""
    lookup: Dict[str, int] = {}
    for index, (_, sources) in enumerate(sorted(SHARED_TAXONOMY.items())):
        for name in sources.get(dataset, []):
            lookup[name] = index
    return np.asarray([lookup.get(str(label), -1) for label in labels], dtype=np.int64)


def shared_class_names() -> List[str]:
    return sorted(SHARED_TAXONOMY)


def evaluate_transfer(
    source_embeddings: np.ndarray,
    source_labels: Sequence[str],
    source_dataset: str,
    target_embeddings: np.ndarray,
    target_labels: Sequence[str],
    target_dataset: str,
    shots: Sequence[int] = (0, 1, 5, 25, 100),
    seed: int = 0,
    repeats: int = 5,
) -> Dict[str, object]:
    """Zero-shot transfer and k-shot re-enrollment onto the shared taxonomy.

    Zero-shot means prototypes built entirely from the source corpus. k-shot
    means prototypes rebuilt from k target flows per class, encoder frozen --
    the deployment question of what it costs to move a model to a new network.
    """
    from .closed_set import class_prototypes
    from .few_shot import enroll_prototypes, predict_from_prototypes

    names = shared_class_names()
    source_y = map_to_shared(source_labels, source_dataset)
    target_y = map_to_shared(target_labels, target_dataset)
    source_keep = source_y >= 0
    target_keep = target_y >= 0

    if source_keep.sum() == 0 or target_keep.sum() == 0:
        return {
            "status": "skipped",
            "reason": (
                f"no shared classes between {source_dataset} and {target_dataset}: "
                f"{int(source_keep.sum())} source and {int(target_keep.sum())} target rows map"
            ),
        }
    if source_embeddings.shape[1] != target_embeddings.shape[1]:
        return {
            "status": "skipped",
            "reason": (
                f"embedding dimensions differ ({source_embeddings.shape[1]} vs "
                f"{target_embeddings.shape[1]}); transfer needs one encoder applied to both"
            ),
        }

    sx, sy = source_embeddings[source_keep], source_y[source_keep]
    tx, ty = target_embeddings[target_keep], target_y[target_keep]
    labels = np.arange(len(names))

    curve: List[Dict[str, object]] = []
    for k in shots:
        if k == 0:
            prototypes = class_prototypes(sx, sy, len(names))
            predictions = predict_from_prototypes(prototypes, tx)
            curve.append(
                {
                    "shots": 0,
                    "macro_f1_mean": macro_f1(predictions, ty, labels),
                    "macro_f1_std": 0.0,
                    "balanced_accuracy_mean": balanced_accuracy(predictions, ty, labels),
                }
            )
            continue
        scores, balanced = [], []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed * 977 + k * 13 + repeat)
            # Re-enrol from target data only: this measures what it costs to
            # move to the new network, not how well the source prototypes did.
            prototypes = enroll_prototypes(tx, ty, len(names), k, rng)
            predictions = predict_from_prototypes(prototypes, tx)
            scores.append(macro_f1(predictions, ty, labels))
            balanced.append(balanced_accuracy(predictions, ty, labels))
        curve.append(
            {
                "shots": int(k),
                "macro_f1_mean": float(np.mean(scores)),
                "macro_f1_std": float(np.std(scores)),
                "balanced_accuracy_mean": float(np.mean(balanced)),
            }
        )

    zero_shot = curve[0]["macro_f1_mean"] if curve and curve[0]["shots"] == 0 else None
    prototypes = class_prototypes(sx, sy, len(names))
    report = classification_report(
        predict_from_prototypes(prototypes, tx), ty, labels=labels, bootstrap=False
    )
    report["per_class_f1"] = {names[int(k)]: v for k, v in report["per_class_f1"].items()}
    report["support"] = {names[int(k)]: v for k, v in report["support"].items()}

    return {
        "status": "ok",
        "source": source_dataset,
        "target": target_dataset,
        "shared_classes": names,
        "n_source_rows_mapped": int(source_keep.sum()),
        "n_target_rows_mapped": int(target_keep.sum()),
        "source_rows_dropped": int((~source_keep).sum()),
        "target_rows_dropped": int((~target_keep).sum()),
        "unmapped_classes": UNMAPPED,
        "zero_shot_macro_f1": zero_shot,
        "enrollment_curve": curve,
        "zero_shot_report": report,
    }
