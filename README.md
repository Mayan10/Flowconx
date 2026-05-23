# FlowCon-X

FlowCon-X is a context-aware packet flow encoder for adaptive AI based network traffic classification. It was built for the Samsung problem statement on flow embeddings that work without deep packet inspection, stay useful for encrypted traffic, and remain fast enough for real-time classification.

The final model uses real public data only. Synthetic data was removed from the final code path.

## What This Builds

The system learns a compact embedding for each network flow. Similar services are pulled together, while different services are pushed apart.

The encoder combines:

- Packet-level behavior: packet sizes, inter-arrival times, directions, packet counts, byte rates, protocol.
- Network context: RTT proxy, jitter proxy, packet timing variation, loss placeholder.
- Service supervision: streaming, gaming, conferencing, bulk transfer, browsing, XR interactive.
- Encrypted traffic signals: CESNET QUIC packet metadata and app/category labels.
- Robustness traffic: MAWI backbone background flows kept separate as weak-label unknown traffic.

The model is a CRNN plus Transformer flow encoder with a network-condition encoder and context fusion layer. Training uses supervised contrastive learning, prototype alignment, pairwise margin separation, and a fused-flow service loss so the actual classifier embedding is shaped for the KPI.

## Final Results

Final model folder:

```text
outputs/flowconx_final_labeled_kpi_pass/
```

Final KPI report:

```text
outputs/flowconx_final_labeled_kpi_pass/kpi_report.md
```

Final checkpoint:

```text
outputs/flowconx_final_labeled_kpi_pass/flowconx_checkpoint.pt
```

Final KPI values:

| KPI | Target | Final value | Status |
| --- | --- | --- | --- |
| Embedding intra cosine | > 0.70 | 0.8415 | PASS |
| Embedding inter cosine | < 0.30 | 0.2715 | PASS |
| k-NN classification accuracy | >= 90% | 90.09% | PASS |
| SVM classification accuracy | >= 90% | 90.44% | PASS |
| Prototype generalization | >= 85% | 90.16% | PASS |
| Leave-one-app generalization | >= 85% | 89.92% | PASS |
| Mean latency per flow | < 100 ms | 13.65 ms | PASS |

These metrics were computed on a stratified evaluation cap of 20,000 train embeddings and 10,000 test embeddings because the full final dataset contains more than 112,000 labeled rows. The model was still trained on the full labeled training CSV.

## Repository Layout

```text
flowconx/
  config.py       Label maps and model constants
  datasets.py     CSV loading, label handling, train/test split
  features.py     Packet and network feature construction
  model.py        Encoder, context fusion, condition adversary
  losses.py       SupCon, prototype, pairwise margin losses
  memory.py       Contrastive memory and prototype bank
  train.py        Training and KPI evaluation entry point
  eval_cli.py     Standalone evaluation entry point
  evaluate.py     Metrics, classifiers, latency benchmark

scripts/
  prepare_5g_traffic_dataset.py
  prepare_cesnet_quic22_dataset.py
  prepare_mawi_pcap.py
  train_flowconx.py
  evaluate_flowconx.py
  kpi_report.py

data/processed/
  flowconx_final_labeled_train.csv
  flowconx_mawi_robustness_background.csv

outputs/flowconx_final_labeled_kpi_pass/
  flowconx_checkpoint.pt
  history.json
  metrics.json
  kpi_report.md
```

## Data Sources

### 1. 5G Traffic Datasets

Source:

```text
https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets
```

This dataset supplied real packet-export traffic for video streaming, live streaming, video conferencing, cloud gaming, online gaming, and metaverse-style apps. It gave the project unencrypted and 5G-oriented app behavior.

The raw folder used locally was:

```text
data/5G_Traffic_Datasets/
```

The final preprocessing command was:

```bash
python scripts/prepare_5g_traffic_dataset.py \
  --input data/5G_Traffic_Datasets \
  --output data/processed/5g_traffic_flows_problem_statement.csv \
  --chunk-rows 250000
```

What the script does:

- Reads the full Kaggle folder tree in chunks.
- Handles both Wireshark CSV files and tab-separated packet exports.
- Infers app labels from folder and file names.
- Maps apps into semantic services such as streaming, gaming, conferencing, and XR interactive.
- Builds compact flow windows from raw packet rows.
- Stores packet lengths, inter-arrival times, directions, packet statistics, byte rates, and jitter proxies.

The raw 5G folder was removed after the final processed CSVs were created.

### 2. CESNET-QUIC22

Source:

```text
https://zenodo.org/records/7409924
```

CESNET-QUIC22 supplied the encrypted traffic portion of the final training data. It contains QUIC flows with app labels, category labels, packet metadata sequences, SNI, user agents, packet histograms, byte counts, and packet counts.

The local raw folder was:

```text
data/cesnet-quic22/
```

The final preprocessing command was:

```bash
python scripts/prepare_cesnet_quic22_dataset.py \
  --input data/cesnet-quic22 \
  --output data/processed/cesnet_quic22_fullmonth_balanced.csv \
  --rows-per-service 20000 \
  --chunk-rows 200000
```

What the script does:

- Scans all 28 daily `flows-*.csv.gz` files.
- Uses the full month instead of a small early sample.
- Reads only the columns needed for training.
- Converts CESNET categories into FlowCon-X service classes.
- Parses the `PPI` field into packet lengths, directions, and inter-packet times.
- Keeps a deterministic balanced reservoir per service class.

Final CESNET extraction:

```text
Rows scanned: 153,226,273 encrypted QUIC flows
Rows kept:    100,000 labeled flows
Class mix:    20,000 each for browsing, bulk transfer, conferencing, gaming, streaming
```

The raw CESNET folder was removed after extraction.

### 3. MAWI Working Group Traffic Archive

Source:

```text
https://mawi.wide.ad.jp/mawi/
```

The final trace was:

```text
202605171400.pcap
```

MAWI is real backbone traffic, but it does not provide app-level labels for our target services. For that reason it was not treated as a normal supervised class in the final KPI accuracy calculation. It was processed as weak-label `unknown` background traffic for robustness.

The final preprocessing command was:

```bash
python scripts/prepare_mawi_pcap.py \
  --input data/202605171400.pcap \
  --output data/processed/mawi_202605171400_background.csv \
  --max-flows 20000 \
  --window-seconds 1
```

What the script does:

- Reads pcap records directly without requiring `tshark`.
- Extracts IPv4 and IPv6 TCP/UDP packets.
- Groups packets into compact bidirectional flow windows.
- Keeps a deterministic reservoir of background flows.
- Labels the result as `mawi_background` and service `unknown`.

Final MAWI extraction:

```text
Packets scanned: 229,508,996
TCP/UDP packets accepted: 169,471,034
Background flows kept: 20,000
```

The raw pcap was removed after extraction.

## Final Processed Data

The final supervised KPI training file is:

```text
data/processed/flowconx_final_labeled_train.csv
```

It contains labeled 5G and CESNET rows. This is the file used for the final supervised KPI run.

Service distribution:

```text
streaming         32,021
gaming            20,041
conferencing      20,036
bulk_transfer     20,000
browsing          20,000
xr_interactive        23
```

The MAWI robustness file is:

```text
data/processed/flowconx_mawi_robustness_background.csv
```

It is kept separate because MAWI is weakly labeled background traffic.

## Setup

Create an environment outside the repository, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, make sure PyTorch can see MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Training

Final full labeled training run:

```bash
python -u scripts/train_flowconx.py \
  --csv data/processed/flowconx_final_labeled_train.csv \
  --app-col app \
  --service-col service \
  --epochs 8 \
  --batch-size 256 \
  --augment-count 0 \
  --output-dir outputs/flowconx_final_labeled_flow_tuned \
  --device mps \
  --temperature 0.05 \
  --lambda-app 0.05 \
  --lambda-proto 0.05 \
  --lambda-pair 5.0 \
  --lambda-flow-service 1.0 \
  --lambda-flow-pair 4.0 \
  --pair-negative-margin 0.08 \
  --pair-positive-target 0.75 \
  --memory-per-class 512 \
  --eval-max-train 20000 \
  --eval-max-test 10000
```

Final KPI polish run:

```bash
python -u scripts/train_flowconx.py \
  --csv data/processed/flowconx_final_labeled_train.csv \
  --app-col app \
  --service-col service \
  --epochs 4 \
  --batch-size 256 \
  --augment-count 0 \
  --output-dir outputs/flowconx_final_labeled_kpi_pass \
  --device mps \
  --temperature 0.05 \
  --lambda-app 0.02 \
  --lambda-proto 0.05 \
  --lambda-pair 5.0 \
  --lambda-flow-service 1.5 \
  --lambda-flow-pair 3.0 \
  --pair-negative-margin 0.08 \
  --pair-positive-target 0.75 \
  --memory-per-class 512 \
  --eval-max-train 20000 \
  --eval-max-test 10000 \
  --resume-checkpoint outputs/flowconx_final_labeled_flow_tuned/flowconx_checkpoint.pt
```

Notes:

- `--augment-count 0` keeps the final training real-data-only.
- `--lambda-pair` shapes the application/service embedding space.
- `--lambda-flow-service` and `--lambda-flow-pair` shape the fused embedding used by k-NN and SVM.
- `--eval-max-train` and `--eval-max-test` keep large-dataset evaluation practical while preserving stratified coverage.

## Evaluation

Standalone evaluation:

```bash
python scripts/evaluate_flowconx.py \
  --checkpoint outputs/flowconx_final_labeled_kpi_pass/flowconx_checkpoint.pt \
  --csv data/processed/flowconx_final_labeled_train.csv \
  --app-col app \
  --service-col service \
  --output outputs/flowconx_final_labeled_kpi_pass/eval.json \
  --device mps
```

KPI report:

```bash
python scripts/kpi_report.py \
  --metrics outputs/flowconx_final_labeled_kpi_pass/metrics.json \
  --output outputs/flowconx_final_labeled_kpi_pass/kpi_report.md
```

## Why MAWI Is Separate

MAWI is valuable because it is real backbone traffic, but it does not say which flows are YouTube, Netflix, Zoom, gaming, file transfer, and so on. Treating it as a normal supervised class hurts the Samsung KPI because the KPI asks for known traffic-type classification. The final model therefore uses MAWI as robustness background and uses CESNET plus 5G for supervised KPI reporting.

## Current Clean Workspace

The final workspace intentionally keeps only:

- Core model and training code.
- Dataset extraction scripts that were actually used.
- Final labeled training CSV.
- MAWI background CSV.
- Final checkpoint and KPI report.
- Project metadata and requirements.

Old smoke-test outputs, raw parent datasets, synthetic utilities, baseline demos, virtual environments, caches, and macOS metadata were removed.
