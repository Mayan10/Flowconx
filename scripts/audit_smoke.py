#!/usr/bin/env python3
"""End-to-end audit smoke run on a generated table.

The real training CSV is 87 MB and is not committed, so CI cannot audit it.
This generates a small table with the full canonical schema -- including the
provenance columns -- runs the whole audit pipeline over it, and asserts that
every artifact the paper depends on actually gets written.

Run locally with:

    python scripts/audit_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowconx.audit.run_audit import main as run_audit  # noqa: E402
from flowconx.features import infer_condition  # noqa: E402
from flowconx.data.schema import CANONICAL_COLUMNS, make_flow_id  # noqa: E402

SERVICES = ["streaming", "gaming", "browsing", "conferencing"]
ORIGINS = ["source_a", "source_b", "source_c"]
CAPTURES_PER_SERVICE = 6
FLOWS_PER_CAPTURE = 40


def synthetic_table(seed: int = 20240501) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    index = 0
    for service_id, service in enumerate(SERVICES):
        for capture in range(CAPTURES_PER_SERVICE):
            origin = ORIGINS[capture % len(ORIGINS)]
            capture_id = f"{origin}/{service}/capture_{capture}.pcap"
            base_len = 180.0 + 260.0 * service_id + rng.normal(0.0, 30.0)
            base_iat = 4.0 + 8.0 * service_id + rng.normal(0.0, 1.0)
            for _ in range(FLOWS_PER_CAPTURE):
                n = int(rng.integers(16, 48))
                lengths = np.clip(rng.normal(base_len, 60.0, n), 40, 1500)
                iats = np.clip(rng.normal(base_iat, 3.0, n), 0.0, None)
                dirs = rng.choice([1, -1], size=n)
                mean_iat, std_iat = float(iats.mean()), float(iats.std())
                duration = float(iats.sum())
                rows.append(
                    {
                        "flow_id": make_flow_id(origin, capture_id, index),
                        "origin": origin,
                        "capture_id": capture_id,
                        "flow_start_ts": 1_700_000_000.0 + index * 30.0,
                        "server_ip": f"203.0.113.{service_id * 8 + capture}",
                        # Several clients per capture, so client-disjoint
                        # splitting is a genuinely different partition from
                        # session-disjoint and gets covered by the smoke run.
                        "client_ip": f"198.51.100.{(index % 11) + service_id * 11}",
                        # Two apps per service, cycling across captures, so an
                        # app appears in several captures and a session-disjoint
                        # split can place some of its flows in test. Without
                        # this, holding out an app leaves nothing to reject and
                        # the open-set mode correctly but unhelpfully skips.
                        "app": f"{service}_app{capture % 2}",
                        "service": service,
                        "condition": infer_condition(mean_iat, std_iat, 0.0),
                        "packet_lengths": ";".join(f"{v:.2f}" for v in lengths),
                        "iat_values": ";".join(f"{v:.4f}" for v in iats),
                        "directions": ";".join(str(int(v)) for v in dirs),
                        "rtt_ms": mean_iat,
                        "jitter_ms": std_iat,
                        "loss_rate": 0.0,
                        "total packets": n,
                        "total fwd packets": int((dirs > 0).sum()),
                        "total backward packets": int((dirs <= 0).sum()),
                        "packet length mean": float(lengths.mean()),
                        "packet length std": float(lengths.std()),
                        "flow iat mean": mean_iat,
                        "flow iat std": std_iat,
                        "flow duration": duration,
                        "flow bytes/s": float(lengths.sum() / max(duration / 1000.0, 0.001)),
                        "flow packets/s": float(n / max(duration / 1000.0, 0.001)),
                        "protocol": 17,
                    }
                )
                index += 1
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        csv_path = root / "audit_smoke.csv"
        synthetic_table().to_csv(csv_path, index=False)

        exit_code = run_audit(
            [
                "--csv",
                str(csv_path),
                "--splits-dir",
                str(root / "splits"),
                "--results-dir",
                str(root / "results"),
                "--no-bootstrap",
                "--seed",
                "0",
            ]
        )
        if exit_code != 0:
            print("audit returned a non-zero exit code", file=sys.stderr)
            return exit_code

        results = root / "results" / "audit_smoke"
        summary_path = results / "audit_summary.json"
        assert summary_path.exists(), "audit wrote no summary"
        summary = json.loads(summary_path.read_text())

        # Every strict protocol must be reachable on a table that carries the
        # full schema. If one is not, the schema or the splitter regressed.
        for protocol in ("random_flow", "session_disjoint", "temporal", "server_disjoint", "origin_disjoint"):
            entry = summary["splits"].get(protocol, {})
            assert entry.get("status") == "ok", f"{protocol} unavailable on a schema-complete table: {entry}"
            assert (results / protocol / "baselines.json").exists(), f"{protocol} wrote no baselines"
            assert (results / protocol / "leakage.json").exists(), f"{protocol} wrote no leakage report"

        # Session-disjoint must actually keep capture sessions apart.
        session_checks = {c["check"]: c for c in summary["splits"]["session_disjoint"]["leakage"]["checks"]}
        assert session_checks["capture_id_disjoint"]["status"] == "pass", session_checks["capture_id_disjoint"]

        # ...and random_flow must not, otherwise the contrast is vacuous.
        random_checks = {c["check"]: c for c in summary["splits"]["random_flow"]["leakage"]["checks"]}
        assert random_checks["capture_id_disjoint"]["status"] == "FAIL", random_checks["capture_id_disjoint"]

        assert (results / "audit_summary.md").exists(), "audit wrote no markdown summary"
        print("\naudit smoke run OK: every split protocol reachable, artifacts written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
