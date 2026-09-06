"""Import every module in the package, and fail if any of them cannot be.

This replaces a hand-written list of module names in the CI workflow. That
list silently went stale across a refactor -- five of the twelve names in it
(`datasets`, `memory`, `model`, `schema`, `eval_cli`) had been renamed or
folded into subpackages, so the job failed on the rename rather than on any
real breakage, and it stayed red.

Walking the package instead means the check cannot go out of date, and it
covers every module rather than the twelve someone thought to list. It is
also a genuine smoke test: an import error here means a syntax error, a
circular import, or a missing dependency in requirements.txt.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

# Running this as `python scripts/check_imports.py` puts scripts/ on sys.path,
# not the repository root, so the package it is meant to check is not
# importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flowconx  # noqa: E402 - must follow the sys.path adjustment above


def main() -> int:
    modules = sorted(m.name for m in pkgutil.walk_packages(flowconx.__path__, "flowconx."))
    if not modules:
        print("error: no modules discovered under flowconx/ -- is the package installed?")
        return 1

    failures: list[tuple[str, str]] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - any import failure is a build failure
            failures.append((name, traceback.format_exc()))

    for name, tb in failures:
        print(f"--- {name} ---\n{tb}", file=sys.stderr)

    if failures:
        print(f"{len(failures)} of {len(modules)} modules failed to import", file=sys.stderr)
        return 1

    print(f"all {len(modules)} modules import cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
