# FlowCon-X

Context-aware flow embeddings for adaptive AI based network traffic classification.

FlowCon-X is built for Samsung's problem statement:

- Combine flow features and network characteristics such as RTT and jitter.
- Train embeddings with contrastive learning.
- Classify encrypted and unencrypted traffic in real time.
- Generalize to emerging traffic types such as XR.

## What Makes This Different

Most solutions will train one encoder and show one accuracy number. FlowCon-X is designed around three signals:

1. Service identity should stay stable across network conditions.
2. Network condition should be represented separately.
3. New traffic should be assigned through semantic prototypes and a trust-gated memory bank.

The key extra metric is CIST, the Context Invariance Stress Test. It checks whether the app embedding stays close to itself when RTT, jitter, and loss are changed.

## Project Layout

```text
flowconx/
  config.py        Taxonomy and model dimensions
  features.py      CSV feature extraction and network augmentation
  datasets.py      FlowRecord and PyTorch dataset adapters
  model.py         FlowCon-X encoder
  losses.py        SupCon, prototype, disentanglement, adversarial losses
  memory.py        Contrastive memory and prototype bank
  train.py         Training CLI
  eval_cli.py      KPI evaluation CLI
  baselines.py     NumPy handcrafted baseline
  synthetic.py     Synthetic dataset generator
scripts/
  make_synthetic_dataset.py
  run_baseline.py
  train_flowconx.py
  evaluate_flowconx.py
  run_prototype_compass.py
docs/
  usage.md
  dataset_mapping.md
  preprocessing_guide.md
```

## Install

Use a fresh Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch installation needs a platform-specific wheel, install it from the official selector first:

```bash
pip install torch
pip install -r requirements.txt
```

## Quick Smoke Test Without PyTorch

The NumPy baseline only needs numpy and pandas. This is only for checking the pipeline. It is not the recommended training path.

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_flows.csv --flows-per-app 40
python scripts/run_baseline.py --csv data/synthetic_flows.csv --app-col app --service-col service
```

This creates a first baseline for the ablation table.

## Train FlowCon-X

Training requires a real CSV by default. Use public datasets such as Samsung 5G Traffic Datasets, GAViST5G, CESNET-QUIC22, MAWI, VR or AR traces, or your manual captures.

```bash
python scripts/train_flowconx.py \
  --csv data/real_flows.csv \
  --app-col app \
  --service-col service \
  --epochs 12 \
  --batch-size 64 \
  --augment-count 0 \
  --output-dir outputs/flowconx_real
```

The training script will fail if `--csv` is omitted. Synthetic data is available only through the explicit `--synthetic` flag for smoke tests.

## Evaluate A Checkpoint

```bash
python scripts/evaluate_flowconx.py \
  --checkpoint outputs/flowconx_real/flowconx_checkpoint.pt \
  --csv data/real_flows.csv \
  --app-col app \
  --service-col service \
  --output outputs/eval_metrics.json
```

## Prototype Compass Demo

After training, stream CSV rows and print nearest service prototypes:

```bash
python scripts/run_prototype_compass.py \
  --checkpoint outputs/flowconx_real/flowconx_checkpoint.pt \
  --calibration-csv data/real_flows.csv \
  --stream-csv data/real_flows.csv \
  --app-col app \
  --service-col service \
  --limit 10
```

## KPI Output

The training script writes:

- `flowconx_checkpoint.pt`
- `metrics.json`
- `history.json`

Use these fields in the Samsung result table:

- `service_similarity.intra`
- `service_similarity.inter`
- `classification.knn_accuracy`
- `classification.svm_accuracy`
- `prototype_generalization.prototype_accuracy`
- `leave_one_app_out.leave_one_app_accuracy`
- `cist_score`
- `latency.mean_ms`, `latency.p95_ms`, `latency.p99_ms`

## Real Dataset Plan

Use these datasets in this order:

1. Samsung 5G Traffic Datasets for primary service classification.
2. GAViST5G for YouTube, Netflix, Prime Video, League of Legends, Teamfight Tactics, and Valorant.
3. CESNET-QUIC22 for encrypted QUIC evaluation.
4. VR, AR, and cloud VR datasets for XR holdout.
5. MAWI for real-world robustness.
6. CICIoT2023 for IoT and security robustness.

Details are in `flowconx_realtime_research_plan.md` and `docs/dataset_mapping.md`.

For exact preprocessing commands and where to place each downloaded dataset, use `docs/preprocessing_guide.md`.
