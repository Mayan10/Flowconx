#!/usr/bin/env python3
"""Structural checks on the paper source, without needing a TeX distribution.

    python scripts/check_paper.py

Catches the failures that waste a compile cycle or, worse, survive one:
missing \\input targets, unbalanced environments, undefined \\ref and \\cite,
and -- the one specific to this project -- **TODO cells reaching the paper**.
A generated table with a TODO in it means a run has not happened, and a draft
that cites such a table is citing a number that does not exist.

Runs in CI so a broken paper source fails the build like broken code does.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

TODO_MARKER = r"\textsc{todo}"


def expand(path: Path, root: Path, seen: Set[Path] | None = None) -> Tuple[str, List[str]]:
    """Recursively inline every ``\\input``; return the text and any problems."""
    seen = seen if seen is not None else set()
    resolved = path.resolve()
    problems: List[str] = []
    if resolved in seen:
        return "", [f"circular \\input chain at {path}"]
    seen.add(resolved)
    if not path.exists():
        return "", [f"missing \\input target: {path}"]

    text = path.read_text(encoding="utf-8")
    out = [text]
    for target in re.findall(r"\\input\{([^}]+)\}", text):
        child = root / (target if target.endswith(".tex") else target + ".tex")
        child_text, child_problems = expand(child, root, seen)
        out.append(child_text)
        problems.extend(child_problems)
    return "\n".join(out), problems


def check(root: Path, main: str = "main.tex", allow_todo: bool = False) -> int:
    body, problems = expand(root / main, root)

    for env in sorted(set(re.findall(r"\\begin\{([A-Za-z]+\*?)\}", body))):
        opens = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", body))
        closes = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", body))
        if opens != closes:
            problems.append(f"unbalanced environment {env!r}: {opens} begin, {closes} end")

    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\(?:ref|autoref|Cref|cref)\{([^}]+)\}", body))
    for missing in sorted(refs - labels):
        problems.append(f"undefined reference: \\ref{{{missing}}}")

    cited: Set[str] = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", body):
        cited.update(key.strip() for key in group.split(","))
    bib = root / "references.bib"
    if bib.exists():
        keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
        for missing in sorted(cited - keys):
            problems.append(f"undefined citation: \\cite{{{missing}}}")
        for unused in sorted(keys - cited):
            problems.append(f"note: bibliography entry never cited: {unused}")

    # The check that matters most here.
    todo_tables: Dict[str, int] = {}
    for table in sorted((root / "tables").glob("*.tex")):
        count = table.read_text(encoding="utf-8").count(TODO_MARKER)
        if count:
            todo_tables[table.name] = count
    inputted = {t.split("/")[-1] + ".tex" for t in re.findall(r"\\input\{(tables/[^}]+)\}", body)}
    for name, count in todo_tables.items():
        if name in inputted:
            level = "note" if allow_todo else "ERROR"
            problems.append(
                f"{level}: {name} is \\input by the paper and contains {count} TODO cell(s) -- "
                "those are runs that have not happened"
            )

    errors = [p for p in problems if not p.startswith("note:")]
    for problem in problems:
        print(("  " if problem.startswith("note:") else "  ! ") + problem)
    if not problems:
        print("  no problems found")

    words = len(re.findall(r"\b\w+\b", re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)))
    print(f"\n{len(list((root / 'sections').glob('*.tex')))} sections, ~{words:,} words, "
          f"{len(inputted)} generated tables inlined, {len(labels)} labels, {len(cited)} citations")
    return 1 if errors else 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structural checks on the paper source.")
    parser.add_argument("--paper", default="paper")
    parser.add_argument("--main", default="main.tex")
    parser.add_argument(
        "--allow-todo",
        action="store_true",
        help="Demote TODO-cell findings to notes. For drafts in progress; never for a submission.",
    )
    args = parser.parse_args(argv)
    return check(Path(args.paper), args.main, args.allow_todo)


if __name__ == "__main__":
    raise SystemExit(main())
