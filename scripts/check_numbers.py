"""Verify that prose in README.md and the paper states the generated numbers.

The project's first standing rule is that no number is ever hand-copied: every
figure the paper reports is rendered into ``paper/tables/*.tex`` by
``make_paper_assets.py`` from a ``metrics.json`` produced by a real run.

Prose was the hole in that rule. Both the README and the paper's section files
quote headline numbers in running text, where no generator reaches them, and
both drifted. The README had ten stale values. The paper had drifted against
*itself*: section 1 said 0.574 where section 9 said 0.567 for the same
quantity, and section 6 carried a hand-written results table whose numbers
contradicted the generated table beside it.

Two kinds of check run here.

``Present`` checks assert a value appears somewhere in a document. They suit
the README, which is short and states each headline number once.

``Anchored`` checks bind a value to the sentence that reports it, via a regex
whose capture group is the number. They suit the paper, where the same
three-digit value may legitimately appear as an unrelated quantity -- an
earlier version of this script matched on numeric proximity alone and flagged
a drift figure of 0.791 as a stale copy of a baseline's 0.790. Anchoring to
the surrounding words removes that ambiguity: the check knows *which* number
in the sentence it is reading.

Run by CI. If a check fails after a rerun, the fix is to update the prose to
the new value, never to relax the check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple, Tuple

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
SECTIONS = ROOT / "paper" / "sections"
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


def _number(cell: str, index: int = 0) -> float:
    """The `index`-th numeric value in a cell, ignoring bold markup and $\\pm$."""
    cleaned = cell.replace("\\textbf{", "").replace("}", "").replace("\\,", "")
    found = re.findall(r"-?\d+\.?\d*", cleaned)
    if len(found) <= index:
        raise ValueError(f"no number {index} in cell {cell!r}")
    return float(found[index])


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


class Anchored(NamedTuple):
    """A number bound to the sentence that reports it."""

    section: str
    pattern: str
    expected: Callable[[], List[str]]
    what: str


def _present_checks() -> List[Tuple[str, str]]:
    """(label, value) pairs that must appear somewhere in the README."""
    split = _table("split_contrast.tex")
    fiveg = _table("main_comparison_fiveg_traffic.tex")
    cesnet = _table("main_comparison_cesnet_quic22.tex")
    openset = _table("open_set.tex")
    cost = _table("cost.tex")
    return [
        ("5G random-flow macro-F1", fmt(_number(_cell(split, "Random flow", 1)))),
        ("5G session-disjoint macro-F1", fmt(_number(_cell(split, "Session-disjoint", 1)))),
        ("5G temporal macro-F1", fmt(_number(_cell(split, "Temporal", 1)))),
        ("CESNET random-flow macro-F1", fmt(_number(_cell(split, "Random flow", 0)))),
        ("CESNET server-disjoint macro-F1", fmt(_number(_cell(split, "Server-disjoint", 0)))),
        ("FlowCon-X on CESNET", fmt(_number(_cell(cesnet, "\\textbf{FlowCon-X}", 0)))),
        ("best CESNET baseline", fmt(_number(_cell(cesnet, "First 20 packet sizes", 0)))),
        ("FlowCon-X on 5G", fmt(_number(_cell(fiveg, "\\textbf{FlowCon-X}", 0)))),
        ("best 5G baseline", fmt(_number(_cell(fiveg, "All flow scalars", 0)))),
        ("SNI-only on CESNET", fmt(_number(_cell(cesnet, "SNI string only", 0)))),
        ("prototype FPR@95TPR", fmt(_number(_cell(openset, "Prototype cosine", 1)))),
        ("max-softmax FPR@95TPR", fmt(_number(_cell(openset, "Max softmax probability", 1)))),
        ("FlowCon-X p50 latency", fmt(_number(_cell(cost, "FlowCon-X", 0)), 2)),
    ]


def _anchored_checks() -> List[Anchored]:
    """Numbers in the paper, each bound to the phrasing that reports it."""
    split = _table("split_contrast.tex")
    fiveg = _table("main_comparison_fiveg_traffic.tex")
    cesnet = _table("main_comparison_cesnet_quic22.tex")
    cost = _table("cost.tex")

    fiveg_random = lambda: [fmt(_number(_cell(split, "Random flow", 1)))]  # noqa: E731
    fiveg_session = lambda: [fmt(_number(_cell(split, "Session-disjoint", 1)))]  # noqa: E731

    return [
        Anchored(
            "01-introduction.tex",
            r"macro-F1 \\textbf\{(0\.\d{3})\}\s*\n?\s*under a stratified",
            fiveg_random,
            "5G macro-F1 under the random split",
        ),
        Anchored(
            "01-introduction.tex",
            r"\\textbf\{(0\.\d{3})\} when flows from",
            fiveg_session,
            "5G macro-F1 under session-disjoint splitting",
        ),
        Anchored(
            "04-protocols.tex",
            r"our model from (0\.\d{3}) to (0\.\d{3})",
            lambda: fiveg_random() + fiveg_session(),
            "5G random-to-session drop",
        ),
        Anchored(
            "06-results.tex",
            r"reaches (0\.\d{3}) \$\\pm\$ (0\.\d{3}) over eight seeds",
            lambda: [
                fmt(_number(_cell(cesnet, "\\textbf{FlowCon-X}", 0))),
                fmt(_number(_cell(cesnet, "\\textbf{FlowCon-X}", 0), index=1)),
            ],
            "FlowCon-X on CESNET",
        ),
        Anchored(
            "06-results.tex",
            r"scores (0\.\d{3}), and matches the sequence models",
            lambda: [fmt(_number(_cell(fiveg, "Flow-statistics MLP", 0)))],
            "flow-statistics MLP on 5G",
        ),
        Anchored(
            "06-results.tex",
            r"batch size one: (\d+\.\d+)\\,ms median, (\d+\.\d+)\\,ms\s*\n?"
            r"at the 95th and (\d+\.\d+)\\,ms at the 99th",
            lambda: [
                fmt(_number(_cell(cost, "FlowCon-X", i)), 2) for i in (0, 1, 2)
            ],
            "end-to-end latency percentiles",
        ),
        Anchored(
            "09-conclusion.tex",
            r"from \$(0\.\d{3})\$ under a stratified random split to \$(0\.\d{3})\$",
            lambda: fiveg_random() + fiveg_session(),
            "5G swing in the conclusion",
        ),
    ]


def main() -> int:
    try:
        present = _present_checks()
        anchored = _anchored_checks()
    except (FileNotFoundError, LookupError, ValueError) as exc:
        print(f"error: could not read the generated tables: {exc}", file=sys.stderr)
        return 1

    failures = 0

    if not README.exists():
        print("error: README.md not found", file=sys.stderr)
        return 1
    readme = README.read_text(encoding="utf-8")
    for label, value in present:
        if value not in readme:
            print(f"  README.md: {label} should be {value}, and is absent", file=sys.stderr)
            failures += 1

    for check in anchored:
        path = SECTIONS / check.section
        if not path.exists():
            print(f"  {check.section}: missing", file=sys.stderr)
            failures += 1
            continue
        match = re.search(check.pattern, path.read_text(encoding="utf-8"))
        if not match:
            print(
                f"  {check.section}: could not locate the sentence reporting "
                f"{check.what} -- if it was reworded, update the anchor in this script",
                file=sys.stderr,
            )
            failures += 1
            continue
        found = list(match.groups())
        want = check.expected()
        if found != want:
            print(
                f"  {check.section}: {check.what} is {found}, tables say {want}",
                file=sys.stderr,
            )
            failures += 1

    if failures:
        print(
            f"\n{failures} problem(s). Update the prose to match paper/tables/, "
            "which are generated from results/.",
            file=sys.stderr,
        )
        return 1

    print(
        f"prose agrees with the generated tables "
        f"({len(present)} README values, {len(anchored)} anchored paper claims)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
