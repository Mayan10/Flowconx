#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowconx.synthetic import write_synthetic_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic FlowCon-X data.")
    parser.add_argument("--output", type=str, default="data/synthetic_flows.csv")
    parser.add_argument("--flows-per-app", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-xr", action="store_true", help="Exclude XR classes.")
    args = parser.parse_args()
    path = write_synthetic_csv(args.output, flows_per_app=args.flows_per_app, include_xr=not args.no_xr, seed=args.seed)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
