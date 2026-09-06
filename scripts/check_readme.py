"""Verify that the numbers in README.md match the generated tables.

The project's first standing rule is that no number is ever hand-copied: every
figure in the paper is rendered into ``paper/tables/*.tex`` by
``make_paper_assets.py`` from a ``metrics.json`` produced by a real run.

The README was the hole in that rule. It quotes headline numbers in prose,
where no generator can reach them, and they went stale -- the 5G split-contrast
figures, the open-set false-accept rates, the latency and the test count were
all left at pre-certification values while the tables moved on. A reader
comparing the README to the paper would have found them disagreeing.

This script closes the hole from the other side. Each check re-derives a number
from the same generated table the paper cites, formats it exactly as the README
should state it, and fails if that string is absent. It does not attempt to
parse English -- it asserts that the correct value is present somewhere in the
prose, which is enough to catch a number that has drifted.

Run by CI. If it fails after a rerun, the fix is to update the README to the
new value, never to relax the check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
TABLES = ROOT / "paper" / "tables"


def _table(name: str) -> str:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing -- run `make paper` first")
    return path.read_text(encoding="utf-8")


def _cell(table: str, row_label: str, column: int) -> str:
    """The `column`-th cell (0-indexed, after the label) of the row starting with `row_label`."""
    for line in table.splitlines():
        stripped = line.strip()
        if not stripped.startswith(row_label):
            continue
        cells = [c.strip() for c in stripped.rstrip("\\").split("&")]
        if len(cells) > column + 1:
            return cells[column + 1]
    raise LookupError(f"row {row_label!r} not found")


def _number(cell: str) -> float:
    """The leading numeric value of a cell, ignoring any $\\pm$ spread and bold markup."""
    cleaned = cell.replace("\\textbf{", "").replace("}", "").replace("\\,", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    if not match:
        raise ValueError(f"no number in cell {cell!r}")
    return float(match.group())


Check = Tuple[str, Callable[[], str]]


def _checks() -> List[Check]:
    split = _table("split_contrast.tex")
    fiveg = _table("main_comparison_fiveg_traffic.tex")
    cesnet = _table("main_comparison_cesnet_quic22.tex")
    openset = _table("open_set.tex")
    cost = _table("cost.tex")

    def fmt(value: float, places: int = 3) -> str:
        return f"{value:.{places}f}"

    return [
        # Split contrast. Column 0 is CESNET, column 1 is 5G.
        ("5G random-flow macro-F1", lambda: fmt(_number(_cell(split, "Random flow", 1)))),
        ("5G session-disjoint macro-F1", lambda: fmt(_number(_cell(split, "Session-disjoint", 1)))),
        ("5G temporal macro-F1", lambda: fmt(_number(_cell(split, "Temporal", 1)))),
        ("CESNET random-flow macro-F1", lambda: fmt(_number(_cell(split, "Random flow", 0)))),
        ("CESNET server-disjoint macro-F1", lambda: fmt(_number(_cell(split, "Server-disjoint", 0)))),
        # Headline model and baseline numbers.
        ("FlowCon-X on CESNET", lambda: fmt(_number(_cell(cesnet, "\\textbf{FlowCon-X}", 0)))),
        ("best CESNET baseline (first 20 sizes)", lambda: fmt(_number(_cell(cesnet, "First 20 packet sizes", 0)))),
        ("FlowCon-X on 5G", lambda: fmt(_number(_cell(fiveg, "\\textbf{FlowCon-X}", 0)))),
        ("best 5G baseline (all flow scalars)", lambda: fmt(_number(_cell(fiveg, "All flow scalars", 0)))),
        # Identifier shortcuts -- the two that beat the model.
        ("SNI-only on CESNET", lambda: fmt(_number(_cell(cesnet, "SNI string only", 0)))),
        # Open-set rejection: the FPR@95TPR column is the one the paper reads.
        ("prototype FPR@95TPR", lambda: fmt(_number(_cell(openset, "Prototype cosine", 1)))),
        ("max-softmax FPR@95TPR", lambda: fmt(_number(_cell(openset, "Max softmax probability", 1)))),
        # Deployment cost.
        ("FlowCon-X p50 latency (ms)", lambda: fmt(_number(_cell(cost, "FlowCon-X", 0)), 2)),
    ]


def main() -> int:
    if not README.exists():
        print("error: README.md not found", file=sys.stderr)
        return 1
    text = README.read_text(encoding="utf-8")

    try:
        checks = _checks()
    except (FileNotFoundError, LookupError, ValueError) as exc:
        print(f"error: could not read the generated tables: {exc}", file=sys.stderr)
        return 1

    missing = [(label, fn()) for label, fn in checks if fn() not in text]

    for label, value in missing:
        print(f"  README does not state the current value for {label}: expected {value}", file=sys.stderr)

    if missing:
        print(
            f"\n{len(missing)} of {len(checks)} README numbers are stale or absent.\n"
            "Update README.md to match paper/tables/, which are generated from results/.",
            file=sys.stderr,
        )
        return 1

    print(f"all {len(checks)} README numbers match the generated tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
