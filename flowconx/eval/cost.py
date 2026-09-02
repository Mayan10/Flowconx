"""Deployment cost, measured rather than asserted.

The previous latency number timed ``model.encode`` on random tensors at batch
size 1, on whichever device was auto-selected, excluding all feature
extraction (AUDIT.md 5, M3). It is not a deployment latency.

This measures:

* end-to-end latency -- parsing the packet series out of the stored record,
  building features, forward pass, prototype lookup -- reported as p50/p95/p99
  rather than a mean, because a mean hides the tail that decides whether a
  classifier can sit inline;
* the forward pass alone, so the two can be compared and the feature-extraction
  share is visible;
* throughput in flows per second, batched;
* model size on disk, parameter count, and peak resident memory.

The same function runs for every baseline, so the cost table is comparable.
"""

from __future__ import annotations

import io
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def percentiles(samples: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "p99_ms": float("nan"), "mean_ms": float("nan")}
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "n_samples": int(values.size),
    }


def _synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_forward_latency(model, packet_seq, flow_features, packet_mask, device, runs: int, warmup: int = 20):
    import torch

    model.eval()
    samples: List[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(packet_seq, flow_features, packet_mask, grl_scale=0.0)
        _synchronize(device)
        for _ in range(runs):
            start = time.perf_counter()
            model(packet_seq, flow_features, packet_mask, grl_scale=0.0)
            _synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def measure_end_to_end_latency(model, frame, device, budget: int, runs: int) -> List[float]:
    """Stored record in, decision out. Includes parsing and feature building."""
    import torch

    from ..features.packet import build_flow_features, build_packet_features, parse_float_series

    lengths_col = frame["packet_lengths"].tolist()
    iats_col = frame["iat_values"].tolist()
    dirs_col = frame["directions"].tolist()
    records = frame.to_dict("records")
    n = min(runs, len(records))
    samples: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            start = time.perf_counter()
            lengths = parse_float_series(lengths_col[i])
            iats = parse_float_series(iats_col[i])
            directions = parse_float_series(dirs_col[i])
            packet_features = build_packet_features(lengths, iats, directions, budget)
            if packet_features is None:
                continue
            packet_seq = np.zeros((1, budget, packet_features.shape[1]), dtype=np.float32)
            packet_mask = np.ones((1, budget), dtype=bool)
            packet_seq[0, : packet_features.shape[0]] = packet_features
            packet_mask[0, : packet_features.shape[0]] = False
            flow_features = build_flow_features(records[i], packet_features.shape[0])[None, :]
            outputs = model(
                torch.from_numpy(packet_seq).to(device),
                torch.from_numpy(flow_features).to(device),
                torch.from_numpy(packet_mask).to(device),
                grl_scale=0.0,
            )
            _ = outputs["z_flow"].cpu().numpy()
            _synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def measure_throughput(model, split, device, batch_size: int = 256) -> Dict[str, float]:

    from ..train.loop import extract_embeddings

    n = len(split)
    if n == 0:
        return {"flows_per_second": float("nan"), "n_flows": 0}
    _synchronize(device)
    start = time.perf_counter()
    extract_embeddings(model, split, device, batch_size=batch_size)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "flows_per_second": float(n / elapsed) if elapsed > 0 else float("nan"),
        "batch_size": int(batch_size),
        "n_flows": int(n),
        "seconds": round(elapsed, 4),
    }


def model_size_bytes(model) -> int:
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def measure_cost(
    model,
    split,
    device,
    config,
    runs: int = 200,
    cpu_too: bool = False,
) -> Dict[str, object]:
    """The full cost block written into every ``metrics.json``."""
    import torch

    budget = split.features.observed_packets
    packet_dim = split.features.packet_seq.shape[-1]
    flow_dim = split.features.flow_features.shape[-1]

    def make_inputs(batch: int, target_device):
        return (
            torch.zeros(batch, budget, packet_dim, device=target_device),
            torch.zeros(batch, flow_dim, device=target_device),
            torch.zeros(batch, budget, dtype=torch.bool, device=target_device),
        )

    out: Dict[str, object] = {
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "model_size_bytes": model_size_bytes(model),
        "observed_packets": int(budget),
        "device": str(device),
        "forward_batch1": percentiles(measure_forward_latency(model, *make_inputs(1, device), device, runs)),
        "forward_batch64": percentiles(
            [t / 64.0 for t in measure_forward_latency(model, *make_inputs(64, device), device, max(runs // 4, 10))]
        ),
        "end_to_end_batch1": percentiles(
            measure_end_to_end_latency(model, split.frame, device, budget, min(runs, 200))
        ),
        "throughput_batched": measure_throughput(model, split, device),
    }
    out["feature_extraction_share"] = _share(out["end_to_end_batch1"], out["forward_batch1"])

    # The CPU leg is off by default because it doubles the measurement time on
    # every run in a sweep. The cost table needs it once, not 120 times:
    #   python -m flowconx.run --config <c> --set eval.cost=true   (GPU)
    # then re-measure on CPU for the final table.
    if cpu_too and device.type != "cpu":
        cpu = torch.device("cpu")
        model_cpu = model.to(cpu)
        out["cpu"] = {
            "forward_batch1": percentiles(measure_forward_latency(model_cpu, *make_inputs(1, cpu), cpu, runs)),
            "end_to_end_batch1": percentiles(
                measure_end_to_end_latency(model_cpu, split.frame, cpu, budget, min(runs, 100))
            ),
        }
        model.to(device)
    _ = config
    return out


def _share(end_to_end: Dict[str, float], forward: Dict[str, float]) -> Optional[float]:
    total, fwd = end_to_end.get("p50_ms"), forward.get("p50_ms")
    if total is None or fwd is None or not np.isfinite(total) or total <= 0:
        return None
    return float(max(0.0, (total - fwd) / total))


def measure_callable_latency(fn: Callable[[], object], runs: int = 200, warmup: int = 20) -> Dict[str, float]:
    """Latency of an arbitrary predictor, for the baseline cost table."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return percentiles(samples)
