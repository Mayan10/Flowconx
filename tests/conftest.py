from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = REPO_ROOT / "data" / "processed" / "flowconx_final_labeled_train.csv"

# CI does not carry the 87 MB training CSV, so the leakage tests run against a
# small synthetic table with the same schema. When the real CSV is present
# locally the same tests run against it too, via the `real_frame` fixture.
SYNTHETIC_ROWS = 600


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """A schema-complete table with genuine capture sessions and timestamps.

    Each capture session contributes several correlated flows, which is what
    makes a random split leak and a session-disjoint split not leak. That
    contrast is the property the tests assert on.
    """
    import numpy as np

    from flowconx.features import infer_condition
    from flowconx.data.schema import make_flow_id

    rng = np.random.default_rng(1234)
    services = ["streaming", "gaming", "browsing", "conferencing"]
    # Three source datasets, mirroring the real artifact which merges 5G
    # Traffic, CESNET-QUIC22 and MAWI into one table.
    origins = ["source_a", "source_b", "source_c"]
    rows = []
    index = 0
    for service_id, service in enumerate(services):
        for capture in range(5):
            origin = origins[capture % len(origins)]
            capture_id = f"{origin}/{service}/capture_{capture}.pcap"
            # One template per capture; flows within it are near-copies, the
            # way consecutive windows of one session actually are.
            base_len = 200.0 + 250.0 * service_id
            base_iat = 5.0 + 9.0 * service_id
            for flow in range(SYNTHETIC_ROWS // (len(services) * 5)):
                n = 24
                lengths = np.clip(rng.normal(base_len, 20.0, n), 40, 1500)
                iats = np.clip(rng.normal(base_iat, 1.5, n), 0.0, None)
                dirs = rng.choice([1, -1], size=n)
                mean_iat, std_iat = float(iats.mean()), float(iats.std())
                rows.append(
                    {
                        "flow_id": make_flow_id(origin, capture_id, index),
                        "origin": origin,
                        "capture_id": capture_id,
                        "flow_start_ts": 1_700_000_000.0 + index * 60.0,
                        "server_ip": f"203.0.113.{service_id * 10 + capture}",
                        # Several clients per capture, so client-disjoint
                        # splitting has groups to work with and is a genuinely
                        # different partition from session-disjoint.
                        "client_ip": f"198.51.100.{(index % 7) + service_id * 7}",
                        # Two observation points, as a paired-capture corpus
                        # has. Exercises the vantage-disjoint protocol, whose
                        # two-group case is easy to get wrong.
                        "vantage": "gateway" if capture % 2 else "workstation",
                        "app": f"{service}_app{capture}",
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
                        "flow duration": float(iats.sum()),
                        "flow bytes/s": float(lengths.sum() / max(iats.sum() / 1000.0, 0.001)),
                        "flow packets/s": float(n / max(iats.sum() / 1000.0, 0.001)),
                        "protocol": 17,
                    }
                )
                index += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def real_frame():
    if not REAL_CSV.exists():
        pytest.skip(f"{REAL_CSV} is not present (it is not committed); skipping real-data leakage checks.")
    return pd.read_csv(REAL_CSV)
