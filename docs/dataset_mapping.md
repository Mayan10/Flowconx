# Dataset Mapping Guide

This guide explains how to map public datasets into FlowCon-X.

## Samsung 5G Traffic Datasets

Recommended use:

- Primary training and closed-set service classification.
- App labels from the dataset should map to streaming, gaming, conferencing, browsing, and metaverse-like services.

Command pattern:

```bash
python scripts/train_flowconx.py \
  --csv data/5g_flows.csv \
  --label-col app \
  --app-col app \
  --service-col service \
  --output-dir outputs/5g_run
```

If no service column exists, create one with these mappings:

```text
YouTube, Netflix, Prime Video -> streaming
Zoom, Teams, Google Meet -> conferencing
Roblox, cloud gaming, mobile games -> gaming
Web sessions -> browsing
```

## GAViST5G

Recommended use:

- Strong Samsung KPI alignment.
- Direct examples: YouTube, Netflix, Prime Video, League of Legends, Teamfight Tactics, Valorant.

Mapping:

```text
YouTube, Netflix, Prime Video -> streaming
League of Legends, Teamfight Tactics, Valorant -> gaming
```

## CESNET-QUIC22

Recommended use:

- Encrypted QUIC validation.
- Proves the model works without payload inspection.

Useful columns:

```text
packet sizes, packet directions, inter-packet times, SNI, QUIC version, user agent
```

Mapping options:

- If service labels are available through SNI or metadata, map domains to services.
- If only application categories are available, train service-level prototypes.
- Keep a separate encrypted subset metric.

## MAWI

Recommended use:

- Real-world robustness and domain shift.
- Do not rely on MAWI as the main labeled app classification source unless labels are enriched.

Mapping options:

- Use it for unsupervised embedding visualization.
- Use weak service labels from ports, domain enrichment, or flow heuristics.
- Report as robustness rather than primary KPI if labels are weak.

## CICIoT2023

Recommended use:

- IoT and security robustness.
- Not the primary app-service dataset.

Mapping:

```text
Benign -> iot_security
DDoS, DoS, Recon, Web, Brute Force, Spoofing, Mirai -> iot_security
```

Optional finer labels:

```text
benign_iot, ddos, dos, recon, web_attack, brute_force, spoofing, mirai
```

Use this to show the encoder can be extended to security analytics, but keep Samsung's core KPI on app and service traffic.

## XR Datasets

Recommended use:

- Holdout evaluation.
- Prototype compass demo.
- CIST plus XR frame pulse feature validation.

Datasets:

```text
VR Traffic Dataset
VR-AR-CG Network Telemetry
Cloud Gaming and Cloud VR Traces
NANCY O-RAN VR Dataset
Discern-XR
```

Mapping:

```text
Cloud VR, VR video, AR session, MR session, Metaverse -> xr_interactive
Cloud gaming can be mapped to gaming or xr_interactive depending on the trace objective
```

For zero-shot reporting:

1. Train without XR labels.
2. Build streaming, gaming, conferencing, and bulk prototypes.
3. Feed XR traces.
4. Report nearest prototype and semantic accuracy.

