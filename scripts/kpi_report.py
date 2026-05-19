#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def get_nested(data: Dict[str, Any], path: str) -> Optional[float]:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    return float(value)


def fmt(value: Optional[float], percent: bool = False) -> str:
    if value is None:
        return "missing"
    if percent:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def status(value: Optional[float], target: float, direction: str) -> str:
    if value is None:
        return "FAIL"
    if direction == "gte":
        return "PASS" if value >= target else "FAIL"
    if direction == "lt":
        return "PASS" if value < target else "FAIL"
    raise ValueError(f"Unknown direction: {direction}")


def build_rows(metrics: Dict[str, Any]) -> list[tuple[str, str, str, str]]:
    intra = get_nested(metrics, "service_similarity.intra")
    inter = get_nested(metrics, "service_similarity.inter")
    knn = get_nested(metrics, "classification.knn_accuracy")
    svm = get_nested(metrics, "classification.svm_accuracy")
    prototype = get_nested(metrics, "prototype_generalization.prototype_accuracy")
    leave_one_app = get_nested(metrics, "leave_one_app_out.leave_one_app_accuracy")
    latency = get_nested(metrics, "latency.mean_ms")
    cist = get_nested(metrics, "cist_score")

    return [
        ("Embedding intra cosine", "> 0.70", fmt(intra), status(intra, 0.70, "gte")),
        ("Embedding inter cosine", "< 0.30", fmt(inter), status(inter, 0.30, "lt")),
        ("k-NN classification accuracy", ">= 90%", fmt(knn, percent=True), status(knn, 0.90, "gte")),
        ("SVM classification accuracy", ">= 90%", fmt(svm, percent=True), status(svm, 0.90, "gte")),
        ("Prototype generalization", ">= 85%", fmt(prototype, percent=True), status(prototype, 0.85, "gte")),
        ("Leave-one-app generalization", ">= 85%", fmt(leave_one_app, percent=True), status(leave_one_app, 0.85, "gte")),
        ("Context invariance score", "higher is better", fmt(cist), "INFO"),
        ("Mean latency per flow", "< 100 ms", f"{latency:.2f} ms" if latency is not None else "missing", status(latency, 100.0, "lt")),
    ]


def render_markdown(rows: list[tuple[str, str, str, str]]) -> str:
    lines = [
        "| KPI | Target | Value | Status |",
        "| --- | --- | --- | --- |",
    ]
    for name, target, value, row_status in rows:
        lines.append(f"| {name} | {target} | {value} | {row_status} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Samsung KPI pass/fail report from FlowCon-X metrics.")
    parser.add_argument("--metrics", required=True, help="Path to a FlowCon-X metrics.json file.")
    parser.add_argument("--output", default=None, help="Optional markdown output path.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = render_markdown(build_rows(metrics))
    print(report)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
