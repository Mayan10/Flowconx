#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import glob
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowconx.config import infer_service, normalize_label
from flowconx.features import infer_condition


CANONICAL_COLUMNS = [
    "app",
    "service",
    "condition",
    "packet_lengths",
    "iat_values",
    "directions",
    "rtt_ms",
    "jitter_ms",
    "loss_rate",
    "total packets",
    "total fwd packets",
    "total backward packets",
    "packet length mean",
    "packet length std",
    "flow iat mean",
    "flow iat std",
    "flow duration",
    "flow bytes/s",
    "flow packets/s",
    "protocol",
]


DATASET_PRESETS = {
    "vr_broad": {
        "mode": "packet",
        "time_col": "Time",
        "length_cols": ["TCP Segment Len", "UDP length"],
        "direction_col": "Link",
        "protocol_col": "Protocol",
        "app_from_filename": True,
        "service": "xr_interactive",
    },
    "vr_ar_cg": {
        "mode": "aggregate",
        "flow_id_col": "ID",
        "length_col": "PS",
        "iat_col": "IPI",
        "protocol_col": "Protocol",
        "service": "xr_interactive",
        "app_from_filename": True,
    },
    "cesnet_quic22": {
        "mode": "cesnet_quic",
        "app_col": "APP",
        "service_col": "CATEGORY",
        "protocol_value": 17,
    },
    "ciciot2023": {
        "mode": "aggregate",
        "label_col": "label",
        "service": "iot_security",
    },
    "cicflowmeter": {
        "mode": "aggregate",
        "label_col": "Label",
    },
    "generic_packet": {
        "mode": "packet",
    },
    "generic_aggregate": {
        "mode": "aggregate",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real traffic CSV files into FlowCon-X format.")
    parser.add_argument("--input", required=True, help="Input CSV file, directory, or glob pattern.")
    parser.add_argument("--output", required=True, help="Output canonical CSV path.")
    parser.add_argument("--dataset", default="generic_aggregate", choices=sorted(DATASET_PRESETS), help="Dataset preset.")
    parser.add_argument("--mode", choices=["packet", "aggregate", "cesnet_quic"], default=None)
    parser.add_argument("--app", default=None, help="Constant app label to apply to all rows.")
    parser.add_argument("--service", default=None, help="Constant service label to apply to all rows.")
    parser.add_argument("--condition", default=None, help="Constant condition label to apply to all rows.")
    parser.add_argument("--app-col", default=None)
    parser.add_argument("--service-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--flow-id-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--length-col", default=None)
    parser.add_argument("--iat-col", default=None)
    parser.add_argument("--direction-col", default=None)
    parser.add_argument("--protocol-col", default=None)
    parser.add_argument("--rtt-col", default=None)
    parser.add_argument("--jitter-col", default=None)
    parser.add_argument("--loss-col", default=None)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-rows-per-file", type=int, default=None)
    return parser.parse_args()


def resolve_inputs(input_path: str, limit: Optional[int] = None) -> List[Path]:
    path = Path(input_path)
    if any(ch in input_path for ch in "*?[]"):
        files = [Path(item) for item in glob.glob(input_path, recursive=True)]
    elif path.is_dir():
        files = sorted(list(path.rglob("*.csv")) + list(path.rglob("*.csv.gz")) + list(path.rglob("*.parquet")))
    else:
        files = [path]
    files = [item for item in files if item.exists() and item.is_file()]
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No CSV files found for {input_path}")
    return files


def read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    df = pd.read_csv(path, nrows=nrows)
    if len(df.columns) <= 2:
        sample = " ".join(str(col) for col in df.columns)
        if len(df) > 0:
            sample += " " + " ".join(str(value) for value in df.iloc[0].tolist())
        if "\t" in sample:
            df = pd.read_csv(path, sep="\t", header=None, nrows=nrows)
            if df.shape[1] == 4:
                df.columns = ["Time", "Source", "Destination", "Length"]
            elif df.shape[1] == 5:
                df.columns = ["Time", "Source", "Destination", "Protocol", "Length"]
    return df


def clean_file_label(path: Path) -> str:
    stem = path.stem.lower()
    for token in ["trace", "myfile", "ds3", "ex", "csv", "pcap", "features"]:
        stem = stem.replace(token, " ")
    stem = stem.replace("_", " ").replace("-", " ")
    parts = [part for part in stem.split() if not part.isdigit()]
    label = "_".join(parts[:3]) if parts else path.stem
    return normalize_label(label)


def clean_path_app(path: Path) -> str:
    parts = [part for part in path.parts if part not in {"data", "raw", "processed", "interim"}]
    known = [
        "YouTube_Live", "YouTube", "Netflix", "Amazon_Prime", "PrimeVideo",
        "Zoom", "MS_Teams", "Google_Meet", "Roblox", "Zepeto",
        "Battleground", "Teamfight_Tactics", "GeForce_Now", "KT_GameBox",
        "Fortnite", "Forza Horizon 5", "Mortal Kombat 11",
        "beat_saber", "rec_room", "half-life", "vr_chat", "google_earth",
    ]
    joined = "/".join(parts).lower()
    for name in known:
        if name.lower() in joined:
            return normalize_label(name)
    return clean_file_label(path)


def get_col(df: pd.DataFrame, preferred: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    if preferred and preferred.strip().lower() in lowered:
        return lowered[preferred.strip().lower()]
    for candidate in candidates:
        if candidate.strip().lower() in lowered:
            return lowered[candidate.strip().lower()]
    return None


def safe_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=np.float64)


def safe_time(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(1, len(series) // 2):
        values = numeric.ffill().fillna(0.0).to_numpy(dtype=np.float64)
        return values - float(np.nanmin(values))
    cleaned = series.astype(str).str.replace(r"[^\x00-\x7F]+", "", regex=True).str.strip()
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.arange(len(series), dtype=np.float64)
    seconds = (parsed - parsed.min()).dt.total_seconds()
    return seconds.ffill().fillna(0.0).to_numpy(dtype=np.float64)


def series_text(values: Sequence[float], precision: int = 4) -> str:
    return ";".join(f"{float(value):.{precision}f}" for value in values)


def protocol_to_number(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"17", "udp", "quic", "rtp", "dns"} or text.startswith("quic"):
        return 17
    if text in {"6", "tcp", "tls", "ssl", "http", "https"} or text.startswith("tls") or text.startswith("tcp") or text.startswith("http"):
        return 6
    try:
        return int(float(text))
    except ValueError:
        return 0


def direction_from_values(values: Sequence[object]) -> np.ndarray:
    dirs = []
    for value in values:
        text = str(value).strip().lower()
        if "up" in text or text in {"1", "+1", "fwd", "forward", "src"}:
            dirs.append(1)
        elif "down" in text or text in {"-1", "bwd", "backward", "dst"}:
            dirs.append(-1)
        else:
            try:
                number = float(text)
                dirs.append(1 if number >= 0 else -1)
            except ValueError:
                dirs.append(1)
    return np.asarray(dirs, dtype=np.int64)


def direction_from_addresses(df: pd.DataFrame) -> Optional[np.ndarray]:
    src_col = get_col(df, None, ["Source", "src", "SrcIP", "ip.src"])
    dst_col = get_col(df, None, ["Destination", "dst", "DstIP", "ip.dst"])
    if src_col is None or dst_col is None:
        return None

    def is_private(value: object) -> bool:
        text = str(value)
        return (
            text.startswith("10.")
            or text.startswith("192.168.")
            or text.startswith("172.16.")
            or text.startswith("172.17.")
            or text.startswith("172.18.")
            or text.startswith("172.19.")
            or text.startswith("172.2")
            or text.startswith("172.30.")
            or text.startswith("172.31.")
        )

    dirs = []
    for src, dst in zip(df[src_col].tolist(), df[dst_col].tolist()):
        if is_private(src) and not is_private(dst):
            dirs.append(1)
        elif is_private(dst) and not is_private(src):
            dirs.append(-1)
        else:
            dirs.append(1)
    return np.asarray(dirs, dtype=np.int64)


def app_for_file(df: pd.DataFrame, path: Path, args: argparse.Namespace, preset: Dict[str, object]) -> str:
    app_col = args.app_col or preset.get("app_col")
    label_col = args.label_col or preset.get("label_col")
    col = get_col(df, app_col, ["app", "application", "traffic_type", "service", "label"])
    if args.app:
        return normalize_label(args.app)
    if col:
        value = df[col].dropna().astype(str)
        if len(value):
            return normalize_label(value.iloc[0])
    if label_col:
        col = get_col(df, str(label_col), [])
        if col:
            value = df[col].dropna().astype(str)
            if len(value):
                return normalize_label(value.iloc[0])
    return clean_path_app(path)


def service_for_app(app: str, df: pd.DataFrame, args: argparse.Namespace, preset: Dict[str, object]) -> str:
    if args.service:
        return normalize_label(args.service)
    if preset.get("service"):
        return normalize_label(preset["service"])
    service_col = args.service_col or preset.get("service_col")
    col = get_col(df, service_col, ["service", "category", "service_category"])
    if col:
        value = df[col].dropna().astype(str)
        if len(value):
            return normalize_label(value.iloc[0])
    return infer_service(app)


def condition_for_values(rtt: float, jitter: float, loss: float, args: argparse.Namespace) -> str:
    if args.condition:
        return normalize_label(args.condition)
    return infer_condition(rtt, jitter, loss)


def make_row(app: str, service: str, condition: str, lengths: np.ndarray, iats: np.ndarray, dirs: np.ndarray, protocol: int, rtt: float, jitter: float, loss: float) -> Dict[str, object]:
    lengths = np.asarray(lengths, dtype=np.float64)
    iats = np.asarray(iats, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.int64)
    valid = np.isfinite(lengths) & np.isfinite(iats)
    lengths = np.clip(lengths[valid], 0, 65535)
    iats = np.clip(iats[valid], 0, 1_000_000)
    dirs = dirs[valid] if len(dirs) == len(valid) else np.ones(len(lengths), dtype=np.int64)
    if len(lengths) == 0:
        lengths = np.asarray([0.0])
        iats = np.asarray([0.0])
        dirs = np.asarray([1])
    duration = float(np.sum(iats))
    fwd = int(np.sum(dirs > 0))
    bwd = int(np.sum(dirs <= 0))
    return {
        "app": normalize_label(app),
        "service": normalize_label(service),
        "condition": normalize_label(condition),
        "packet_lengths": series_text(lengths, 2),
        "iat_values": series_text(iats, 4),
        "directions": ";".join(str(int(value)) for value in dirs),
        "rtt_ms": float(rtt),
        "jitter_ms": float(jitter),
        "loss_rate": float(loss),
        "total packets": int(len(lengths)),
        "total fwd packets": fwd,
        "total backward packets": bwd,
        "packet length mean": float(np.mean(lengths)),
        "packet length std": float(np.std(lengths)),
        "flow iat mean": float(np.mean(iats)),
        "flow iat std": float(np.std(iats)),
        "flow duration": duration,
        "flow bytes/s": float(np.sum(lengths) / max(duration / 1000.0, 0.001)),
        "flow packets/s": float(len(lengths) / max(duration / 1000.0, 0.001)),
        "protocol": int(protocol),
    }


def prepare_packet_file(path: Path, df: pd.DataFrame, args: argparse.Namespace, preset: Dict[str, object]) -> List[Dict[str, object]]:
    time_col = get_col(df, args.time_col or preset.get("time_col"), ["time", "timestamp", "frame.time_relative", "time_epoch"])
    length_col = get_col(df, args.length_col or preset.get("length_col"), ["length", "packet length", "packet_length", "len", "frame.len"])
    length_cols = preset.get("length_cols", [])
    direction_col = get_col(df, args.direction_col or preset.get("direction_col"), ["direction", "directions", "dir", "link"])
    protocol_col = get_col(df, args.protocol_col or preset.get("protocol_col"), ["protocol", "proto"])
    flow_id_col = get_col(df, args.flow_id_col or preset.get("flow_id_col"), ["flow_id", "id", "flow"])
    rtt_col = get_col(df, args.rtt_col, ["rtt", "rtt_ms", "latency", "latency_ms"])
    jitter_col = get_col(df, args.jitter_col, ["jitter", "jitter_ms"])
    loss_col = get_col(df, args.loss_col, ["loss", "loss_rate", "packet_loss"])

    if time_col is None:
        raise ValueError(f"{path} needs a time column for packet mode.")
    if length_col is None and not length_cols:
        raise ValueError(f"{path} needs a packet length column for packet mode.")

    app = app_for_file(df, path, args, preset)
    service = service_for_app(app, df, args, preset)
    times = safe_time(df[time_col])
    if length_col:
        lengths = safe_numeric(df[length_col])
    else:
        length_arrays = []
        for candidate in length_cols:
            col = get_col(df, str(candidate), [])
            if col:
                length_arrays.append(safe_numeric(df[col]))
        lengths = np.sum(np.vstack(length_arrays), axis=0)
    if direction_col:
        dirs = direction_from_values(df[direction_col].tolist())
    else:
        dirs = direction_from_addresses(df)
        if dirs is None:
            dirs = np.ones(len(df), dtype=np.int64)
    protocols = safe_numeric(df[protocol_col].map(protocol_to_number)) if protocol_col else np.full(len(df), int(preset.get("protocol_value", 0)))
    rtts = safe_numeric(df[rtt_col]) if rtt_col else np.zeros(len(df))
    jitters = safe_numeric(df[jitter_col]) if jitter_col else np.zeros(len(df))
    losses = safe_numeric(df[loss_col]) if loss_col else np.zeros(len(df))

    rows: List[Dict[str, object]] = []
    if flow_id_col:
        group_keys = df[flow_id_col].astype(str).to_numpy()
    else:
        start = float(np.nanmin(times))
        group_keys = np.floor((times - start) / max(args.window_seconds, 0.1)).astype(int)

    for key in pd.unique(group_keys):
        mask = group_keys == key
        order = np.argsort(times[mask])
        group_times = times[mask][order]
        group_lengths = lengths[mask][order]
        group_dirs = dirs[mask][order]
        if len(group_times) > 1:
            iats = np.diff(group_times, prepend=group_times[0]) * 1000.0
        else:
            iats = np.asarray([0.0])
        protocol = int(pd.Series(protocols[mask]).mode().iloc[0]) if np.any(mask) else int(preset.get("protocol_value", 0))
        rtt = float(np.nanmean(rtts[mask])) if np.any(rtts[mask]) else float(np.nanmean(iats))
        jitter = float(np.nanmean(jitters[mask])) if np.any(jitters[mask]) else float(np.nanstd(iats))
        loss = float(np.nanmean(losses[mask])) if np.any(losses[mask]) else 0.0
        condition = condition_for_values(rtt, jitter, loss, args)
        rows.append(make_row(app, service, condition, group_lengths, iats, group_dirs, protocol, rtt, jitter, loss))
    return rows


def prepare_aggregate_file(path: Path, df: pd.DataFrame, args: argparse.Namespace, preset: Dict[str, object]) -> List[Dict[str, object]]:
    app = app_for_file(df, path, args, preset)
    service = service_for_app(app, df, args, preset)
    rows: List[Dict[str, object]] = []
    length_col = get_col(df, args.length_col or preset.get("length_col"), ["packet length mean", "packet_length", "length", "ps", "avg pkt size", "average packet size", "avg", "tot size"])
    iat_col = get_col(df, args.iat_col or preset.get("iat_col"), ["flow iat mean", "iat", "ipi", "inter_packet_time", "artt"])
    protocol_col = get_col(df, args.protocol_col or preset.get("protocol_col"), ["protocol", "proto", "protocol type"])
    rtt_col = get_col(df, args.rtt_col, ["rtt", "rtt_ms", "latency", "latency_ms", "flow iat mean", "artt", "iat"])
    jitter_col = get_col(df, args.jitter_col, ["jitter", "jitter_ms", "flow iat std", "std"])
    loss_col = get_col(df, args.loss_col, ["loss", "loss_rate", "packet_loss"])
    service_col = get_col(df, args.service_col or preset.get("service_col"), ["service", "category", "service_category"])
    app_col = get_col(df, args.app_col or preset.get("app_col"), ["app", "application", "label", "traffic_type"])

    for _, item in df.iterrows():
        item_app = normalize_label(item[app_col]) if app_col else app
        item_service = normalize_label(item[service_col]) if service_col else (service if service != "unknown" else infer_service(item_app))
        rtt = float(pd.to_numeric(pd.Series([item[rtt_col]]), errors="coerce").fillna(0.0).iloc[0]) if rtt_col else 0.0
        jitter = float(pd.to_numeric(pd.Series([item[jitter_col]]), errors="coerce").fillna(0.0).iloc[0]) if jitter_col else 0.0
        loss = float(pd.to_numeric(pd.Series([item[loss_col]]), errors="coerce").fillna(0.0).iloc[0]) if loss_col else 0.0
        condition = condition_for_values(rtt, jitter, loss, args)
        length = float(pd.to_numeric(pd.Series([item[length_col]]), errors="coerce").fillna(0.0).iloc[0]) if length_col else 0.0
        iat = float(pd.to_numeric(pd.Series([item[iat_col]]), errors="coerce").fillna(rtt).iloc[0]) if iat_col else rtt
        protocol = protocol_to_number(item[protocol_col]) if protocol_col else int(preset.get("protocol_value", 0))
        rows.append(make_row(item_app, item_service, condition, np.asarray([length]), np.asarray([iat]), np.asarray([1]), protocol, rtt, jitter, loss))
    return rows


def parse_ppi(value: object) -> Optional[Dict[str, np.ndarray]]:
    text = str(value)
    if not text or text == "nan":
        return None
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        return None
    iats = np.asarray(parsed[0], dtype=np.float64)
    dirs = np.asarray(parsed[1], dtype=np.int64)
    lengths = np.asarray(parsed[2], dtype=np.float64)
    return {"iats": iats, "dirs": dirs, "lengths": lengths}


def prepare_cesnet_quic_file(path: Path, df: pd.DataFrame, args: argparse.Namespace, preset: Dict[str, object]) -> List[Dict[str, object]]:
    ppi_col = get_col(df, None, ["PPI", "ppi"])
    app_col = get_col(df, args.app_col or preset.get("app_col"), ["APP", "app"])
    service_col = get_col(df, args.service_col or preset.get("service_col"), ["CATEGORY", "category"])
    rows: List[Dict[str, object]] = []
    if ppi_col is None:
        return prepare_aggregate_file(path, df, args, preset)
    for _, item in df.iterrows():
        parsed = parse_ppi(item[ppi_col])
        if parsed is None:
            continue
        app = normalize_label(item[app_col]) if app_col else clean_file_label(path)
        service = normalize_label(item[service_col]) if service_col else infer_service(app)
        rtt = float(np.mean(parsed["iats"])) if len(parsed["iats"]) else 0.0
        jitter = float(np.std(parsed["iats"])) if len(parsed["iats"]) else 0.0
        condition = condition_for_values(rtt, jitter, 0.0, args)
        rows.append(make_row(app, service, condition, parsed["lengths"], parsed["iats"], parsed["dirs"], 17, rtt, jitter, 0.0))
    return rows


def main() -> None:
    args = parse_args()
    preset = dict(DATASET_PRESETS[args.dataset])
    mode = args.mode or str(preset["mode"])
    files = resolve_inputs(args.input, args.limit_files)
    all_rows: List[Dict[str, object]] = []
    for path in files:
        df = read_table(path, nrows=args.limit_rows_per_file)
        if mode == "packet":
            rows = prepare_packet_file(path, df, args, preset)
        elif mode == "cesnet_quic":
            rows = prepare_cesnet_quic_file(path, df, args, preset)
        else:
            rows = prepare_aggregate_file(path, df, args, preset)
        all_rows.extend(rows)
        print(f"{path}: wrote {len(rows)} canonical flows")
    out = pd.DataFrame(all_rows)
    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = 0
    out = out[CANONICAL_COLUMNS]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved {len(out)} flows to {output}")


if __name__ == "__main__":
    main()
