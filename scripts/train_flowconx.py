#!/usr/bin/env python3
"""Wrapper for python -m flowconx.train."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowconx.train import main


if __name__ == "__main__":
    main()
