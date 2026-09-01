"""Evaluation modes: closed-set, open-set, few-shot, drift, robustness, cost.

Each mode is a module with a single ``evaluate_*`` entry point that takes the
run's config and returns a JSON-serialisable block. ``flowconx/run.py``
switches them on from ``config.eval``; none of them trains anything, and all
of them treat the encoder as frozen.
"""
