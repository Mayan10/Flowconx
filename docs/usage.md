# FlowCon-X Usage Guide

## 1. Real Data Policy

FlowCon-X now requires a real CSV for training and evaluation by default. Synthetic data is only a smoke-test option and must be requested explicitly with `--synthetic` or generated through `scripts/make_synthetic_dataset.py`.

For your Samsung submission, train on real public or manually captured data only:

- Samsung 5G Traffic Datasets.
- GAViST5G.
- CESNET-QUIC22.
- MAWI.
- VR, AR, or cloud VR traces.
- Manual packet captures from YouTube, Netflix, gaming, conferencing, downloads, and VR if available.

Use `--augment-count 0` for strict real-data-only training. Counterfactual network augmentation is available, but it is off by default.

## 2. Prepare A Real CSV

For dataset-specific commands, read `docs/preprocessing_guide.md`.

Your CSV should have at least one label column. These are recommended:

```text
app, service, condition, packet_lengths, iat_values, directions, rtt_ms, jitter_ms, loss_rate
```

If true packet sequences are not available, the loader can use aggregate CICFlowMeter-style columns:

```text
Label, Total Fwd Packets, Total Backward Packets, Packet Length Mean,
Packet Length Std, Flow IAT Mean, Flow IAT Std, Flow Duration,
Flow Bytes/s, Flow Packets/s, Protocol
```

Example:

```bash
python scripts/train_flowconx.py \
  --csv data/your_real_flows.csv \
  --label-col Label \
  --epochs 20 \
  --augment-count 0 \
  --output-dir outputs/real_run
```

If labels are app names, FlowCon-X maps them to service classes automatically. If your dataset already has service labels, pass `--service-col service`.

## 3. Optional Smoke Test Data

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_flows.csv --flows-per-app 80
```

The synthetic data includes:

- Streaming: YouTube, Netflix, Prime Video.
- Gaming: Valorant, League of Legends, Roblox.
- Conferencing: Zoom, Teams, Google Meet.
- Bulk transfer: file download, FTP.
- Browsing: web browsing.
- XR interactive: Cloud VR, VR video, AR session.
- Network conditions: good, moderate, degraded, bad.

## 4. Run The Baseline On Real Data

```bash
python scripts/run_baseline.py \
  --csv data/your_real_flows.csv \
  --app-col app \
  --service-col service \
  --output outputs/baseline_metrics.json
```

This baseline uses handcrafted packet and network statistics. It is useful for the ablation table.

## 5. Train The Neural Encoder On Real Data

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then train:

```bash
python scripts/train_flowconx.py \
  --csv data/your_real_flows.csv \
  --app-col app \
  --service-col service \
  --epochs 20 \
  --batch-size 64 \
  --augment-count 0 \
  --output-dir outputs/flowconx_real
```

For a faster CPU test:

```bash
python scripts/train_flowconx.py \
  --csv data/your_real_flows.csv \
  --app-col app \
  --service-col service \
  --epochs 3 \
  --batch-size 32 \
  --limit 400 \
  --output-dir outputs/quick_test
```

## 6. Evaluate On Real Data

```bash
python scripts/evaluate_flowconx.py \
  --checkpoint outputs/flowconx_real/flowconx_checkpoint.pt \
  --csv data/your_real_flows.csv \
  --app-col app \
  --service-col service \
  --output outputs/eval_metrics.json
```

## 7. Prototype Compass Demo

```bash
python scripts/run_prototype_compass.py \
  --checkpoint outputs/flowconx_real/flowconx_checkpoint.pt \
  --calibration-csv data/your_real_flows.csv \
  --stream-csv data/your_real_flows.csv \
  --app-col app \
  --service-col service \
  --limit 10
```

Example output:

```text
flow=1 app=cloud_vr service=xr_interactive compass=xr_interactive:0.821, streaming:0.702, gaming:0.351, bulk_transfer:0.114
```

Use this in the demo to show how a new flow moves relative to semantic prototypes.

## 8. KPI Reporting Template

Use this table in the deck:

| KPI | Target | FlowCon-X result |
|---|---:|---:|
| Service intra cosine | > 0.7 | from metrics.json |
| Service inter cosine | < 0.3 | from metrics.json |
| Classification accuracy | >= 90 percent | best of k-NN and SVM |
| Unseen traffic generalization | >= 85 percent | prototype accuracy |
| Leave-one-app generalization | >= 85 percent | leave_one_app_accuracy |
| Real-time latency | < 100 ms | p95 latency |
| CIST score | > 0.85 | cist_score |

## 9. Manual Capture Workflow

For the strongest demo, collect your own traffic under multiple network conditions.

1. Run one app at a time.
2. Capture packets with tcpdump or Wireshark.
3. Apply good, moderate, degraded, and bad network conditions with a Linux VM or router.
4. Extract flows with CICFlowMeter or tshark.
5. Label each capture with app, service, and condition.
6. Train and evaluate with the commands above.

Recommended capture set:

- YouTube, Netflix, Prime Video.
- Valorant or another game.
- Zoom or Teams.
- File download.
- Browser session.
- VR or cloud VR trace if available.
