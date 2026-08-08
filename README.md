# FlowCon-X

[![CI](https://github.com/Mayan10/Flowconx/actions/workflows/ci.yml/badge.svg)](https://github.com/Mayan10/Flowconx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-yellow)](#limitations)

FlowCon-X is a context-aware neural encoder for network flow representation learning. It turns packet timing, packet size, directionality, protocol metadata, and network condition signals into compact embeddings that can be used for traffic classification without reading packet payloads.

## Quick Start

```bash
git clone https://github.com/Mayan10/Flowconx.git
cd Flowconx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/evaluate_flowconx.py \
  --checkpoint outputs/flowconx_final_labeled_kpi_pass/flowconx_checkpoint.pt \
  --csv data/processed/flowconx_final_labeled_train.csv \
  --app-col app \
  --service-col service \
  --output /tmp/flowconx_eval.json
```

That reruns evaluation against the checkpoint already included in this repository. Results at a glance:

| Metric | Value |
| --- | ---: |
| Training data | 112,000+ labeled flows, 6 service classes, 2 public datasets |
| SVM / k-NN service accuracy | 90.44% / 90.09% |
| Mean inference latency | 13.65 ms per flow |
| KPI targets cleared | 7 / 7 |

See [Results](#results) for the full metrics table and [Usage](#usage) for a minimal inference example.

## Contents

- [Quick Start](#quick-start)
- [What Makes It Different](#what-makes-it-different)
- [Architecture](#architecture)
- [Training Objective](#training-objective)
- [Memory And Prototypes](#memory-and-prototypes)
- [Data Pipeline](#data-pipeline)
- [Final Processed Files](#final-processed-files)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Design Tradeoffs](#design-tradeoffs)
- [Limitations](#limitations)
- [Why This Shape Works](#why-this-shape-works)
- [License](#license)

The main idea is simple, but the implementation is fairly careful:

- `z_app` should describe what the flow behaves like at the application or service level.
- `z_net` should describe the network condition around the flow.
- `z_flow` should combine both views into the embedding used by downstream classifiers.

That split matters. A video stream on a clean link and the same video stream on a congested link should not become two unrelated application identities. At the same time, network context is still useful because latency, jitter, directionality, and burst structure change how a real flow looks. FlowCon-X keeps those two ideas separate until the fusion stage, then trains the fused embedding directly for classification geometry.

The final training path uses real public traffic data only. No synthetic rows are used in the final training commands.

## What Makes It Different

A quick related-work scan shows many strong neighbors, but not the same architecture as this repository.

- [ET-BERT](https://arxiv.org/abs/2202.06335) learns contextual datagram or burst representations for encrypted traffic with Transformer pretraining. It focuses on byte-like tokenization and large-scale pretraining.
- [CLE-TFE](https://arxiv.org/abs/2402.07501) uses supervised contrastive learning for encrypted traffic and temporal fusion across packet-level and flow-level tasks.
- [MIETT](https://arxiv.org/abs/2412.15306) treats packets as instances inside a flow bag and uses two-level attention for encrypted traffic classification.
- [PacketCLIP](https://arxiv.org/abs/2503.03747) and [FlowCLIP](https://arxiv.org/abs/2606.17746) use CLIP-style alignment between traffic and text or domain supervision.
- [FlowletFormer](https://arxiv.org/abs/2508.19924) builds traffic-aware pretraining around behavioral units and protocol semantics.

FlowCon-X sits in the same research family, but its design is different in four important ways:

1. It has a separate application behavior encoder and network condition encoder.
2. It uses cross-attention from packet behavior tokens into network-condition tokens instead of just concatenating summary features.
3. It uses adversarial condition removal on `z_app`, so the application embedding is pushed away from link-condition shortcuts.
4. It trains the final deployed embedding, `z_flow`, with service-level contrastive and pairwise margin losses, so the downstream k-NN, SVM, and prototype classifiers see the geometry that was actually optimized.

In plain terms, this is not just "Transformer for traffic classification." It is a three-view flow representation system: application identity, network condition, and context-aware fused behavior.

## Architecture

The model lives in [flowconx/model.py](flowconx/model.py).

FlowCon-X produces three normalized embeddings:

| Embedding | Dimension | Purpose |
| --- | ---: | --- |
| `z_app` | 256 | Application and service behavior, trained to be less sensitive to network condition changes |
| `z_net` | 128 | Network context such as RTT proxy, jitter proxy, throughput, loss, and queueing hints |
| `z_flow` | 256 | Final context-aware flow embedding used by classifiers and reports |

The model has four main blocks.

### 1. Application Identity Encoder

Input:

```text
packet_seq: [batch, 128, 16]
```

Each flow is represented as up to 128 packet tokens. Each token has 16 side-channel features:

| Index | Feature |
| ---: | --- |
| 0 | Normalized packet length |
| 1 | Log-scaled inter-arrival time |
| 2 | Direction, forward as `1`, reverse as `-1` |
| 3 | Log-scaled packets per second |
| 4 | Log-scaled bytes per second |
| 5 | Forward packet ratio |
| 6 | Backward packet ratio |
| 7 | TCP indicator |
| 8 | UDP indicator |
| 9 | QUIC-like indicator |
| 10 | Normalized SYN count |
| 11 | Normalized ACK count |
| 12 | Normalized RST count |
| 13 | Log-scaled flow duration |
| 14 | Burst position bucket |
| 15 | Packet position inside the trimmed sequence |

The encoder stack is:

1. Linear projection from 16 features to 192 hidden units.
2. LayerNorm and SiLU activation.
3. Three gated temporal convolution blocks.
4. One Transformer encoder layer with 6 attention heads.
5. Attention pooling over packet tokens.
6. MLP projection to a normalized 256-dimensional `z_app`.

The temporal convolution blocks are depthwise separable convolutions with a learned residual gate. They pick up local timing and burst patterns before self-attention models longer packet relationships.

### 2. Network Condition Encoder

Input:

```text
network_series: [batch, 24, 8]
```

The network context stream has 24 time steps and 8 features:

| Index | Feature |
| ---: | --- |
| 0 | Log-scaled RTT proxy |
| 1 | Log-scaled jitter proxy |
| 2 | Loss rate |
| 3 | Log-scaled retransmission count |
| 4 | Log-scaled throughput |
| 5 | Uplink ratio |
| 6 | Log-scaled queue delay proxy |
| 7 | Condition hint |

The condition encoder uses:

1. Linear projection from 8 features to 128 hidden units.
2. Two-layer bidirectional GRU.
3. Attention pooling.
4. Linear projection to normalized 128-dimensional `z_net`.

This stream lets the model reason about changing network conditions without forcing those conditions to become the application identity.

### 3. Context Fusion

The fusion block turns application tokens and condition tokens into `z_flow`.

The mechanism is cross-attention:

```text
queries = projected packet behavior tokens
keys    = projected network condition tokens
values  = projected network condition tokens
```

The result is pooled and concatenated with `z_app`, then passed through an MLP to produce normalized `z_flow`.

This is the central architectural choice. The model does not simply append RTT and jitter to a flat vector. Packet behavior gets to attend to network context, so the same packet pattern can be interpreted differently under clean, moderate, degraded, or noisy conditions.

### 4. Condition Adversary

The condition adversary is attached to `z_app` through a gradient reversal layer.

During forward pass, it tries to predict the network condition from `z_app`. During backward pass, the gradient is reversed, so the application encoder is penalized when `z_app` contains condition-specific shortcuts.

That gives the model two complementary pressures:

- `z_app` should stay stable when network conditions change.
- `z_flow` should use network context when it helps classification.

## Training Objective

The loss implementation lives in [flowconx/losses.py](flowconx/losses.py).

Training combines several objectives:

| Loss | Applied to | Purpose |
| --- | --- | --- |
| Service supervised contrastive loss | `z_app` | Pull flows from the same service together and push different services apart |
| Service supervised contrastive loss | `z_flow` | Shape the deployed classifier embedding directly |
| App supervised contrastive loss | `z_flow` | Preserve application-level structure inside service groups |
| Prototype alignment | `z_app` | Keep service centroids stable and usable for nearest-prototype classification |
| Pairwise margin loss | `z_app` and `z_flow` | Enforce high same-service cosine similarity and low cross-service cosine similarity |
| Cross-covariance disentanglement | `z_app`, `z_net` | Reduce leakage between application behavior and condition behavior |
| Condition adversarial loss | `z_app` | Discourage condition shortcuts in the application embedding |

The important late-stage detail is that the final classifier uses `z_flow`, so `z_flow` must be trained directly. Earlier experiments shaped `z_app` well but left the fused classifier space slightly under-optimized. Adding `--lambda-flow-service` and `--lambda-flow-pair` fixed that by aligning training with inference.

## Memory And Prototypes

The training loop uses two lightweight stateful structures from [flowconx/memory.py](flowconx/memory.py):

- `EmbeddingMemoryBank` stores recent class-balanced embeddings so supervised contrastive learning sees more positives and negatives than the current mini-batch.
- `PrototypeBank` keeps moving service prototypes that are useful for stability checks and prototype-style evaluation.

The memory bank is not an external database and it is not needed at inference time. It is a training-time structure that makes contrastive learning more stable on imbalanced traffic classes.

## Data Pipeline

All dataset converters write the same canonical CSV schema:

```text
app
service
condition
packet_lengths
iat_values
directions
rtt_ms
jitter_ms
loss_rate
total packets
total fwd packets
total backward packets
packet length mean
packet length std
flow iat mean
flow iat std
flow duration
flow bytes/s
flow packets/s
protocol
```

The core feature builder in [flowconx/features.py](flowconx/features.py) turns each row into the packet sequence and network condition series consumed by the model.

### 5G Traffic Datasets

Source:

[Kaggle 5G Traffic Datasets](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets)

Used for real 5G-oriented application behavior, including video streaming, live streaming, video conferencing, gaming, and metaverse-style application folders.

Converter:

[scripts/prepare_5g_traffic_dataset.py](scripts/prepare_5g_traffic_dataset.py)

Command:

```bash
python scripts/prepare_5g_traffic_dataset.py \
  --input data/5G_Traffic_Datasets \
  --output data/processed/5g_traffic_flows_problem_statement.csv \
  --chunk-rows 250000
```

What it does:

- Streams CSV files in chunks instead of loading the whole dataset into memory.
- Handles Wireshark-style CSV files and tab-separated packet exports.
- Infers app labels from folder and file names.
- Maps app labels into service labels such as streaming, gaming, conferencing, and XR interactive.
- Builds compact flow windows with packet lengths, inter-arrival times, directions, byte rates, packet rates, and jitter proxies.

### CESNET-QUIC22

Source:

[CESNET-QUIC22 on Zenodo](https://zenodo.org/records/7409924)

Used for encrypted QUIC flows with app labels, category labels, packet metadata sequences, packet counts, byte counts, SNI-derived metadata, and category annotations.

Converter:

[scripts/prepare_cesnet_quic22_dataset.py](scripts/prepare_cesnet_quic22_dataset.py)

Command:

```bash
python scripts/prepare_cesnet_quic22_dataset.py \
  --input data/cesnet-quic22 \
  --output data/processed/cesnet_quic22_fullmonth_balanced.csv \
  --rows-per-service 20000 \
  --chunk-rows 200000
```

What it does:

- Scans the full month of daily `flows-*.csv.gz` files.
- Reads compressed files directly, without expanding them first.
- Parses CESNET `PPI` packet metadata into packet lengths, inter-packet times, and directions.
- Converts CESNET category labels into FlowCon-X service labels.
- Keeps a deterministic balanced reservoir per service class.

Final extraction used for this repository:

```text
Rows scanned: 153,226,273 encrypted QUIC flows
Rows kept:    100,000 labeled flows
Class mix:    20,000 each for browsing, bulk transfer, conferencing, gaming, and streaming
```

### MAWI Working Group Traffic Archive

Source:

[MAWI Working Group Traffic Archive](https://mawi.wide.ad.jp/mawi/)

Used as real backbone robustness traffic. MAWI does not provide app-level labels for this task, so it is kept as weak-label background traffic instead of being mixed into the supervised service accuracy report.

Converter:

[scripts/prepare_mawi_pcap.py](scripts/prepare_mawi_pcap.py)

Command:

```bash
python scripts/prepare_mawi_pcap.py \
  --input data/202605171400.pcap \
  --output data/processed/mawi_202605171400_background.csv \
  --max-flows 20000 \
  --window-seconds 1
```

What it does:

- Reads pcap records directly in Python.
- Does not require `tshark`.
- Extracts IPv4 and IPv6 TCP/UDP packets.
- Groups packets into bidirectional flow windows.
- Keeps a deterministic reservoir of background flows.
- Labels the result as app `mawi_background` and service `unknown`.

Final extraction:

```text
Packets scanned:        229,508,996
TCP/UDP packets accepted: 169,471,034
Background flows kept:       20,000
```

## Final Processed Files

The repository keeps the final processed files under:

```text
data/processed/
```

Main supervised training file:

```text
data/processed/flowconx_final_labeled_train.csv
```

This file contains the labeled 5G and CESNET rows used by the final training run.

Service distribution:

```text
streaming         32,021
gaming            20,041
conferencing      20,036
bulk_transfer     20,000
browsing          20,000
xr_interactive        23
```

MAWI robustness file:

```text
data/processed/flowconx_mawi_robustness_background.csv
```

This file is kept separate because it is real traffic but weakly labeled as background.

## Repository Layout

```text
flowconx/
  config.py       Label maps, dimensions, service mappings
  datasets.py     CSV loading, feature construction, train/test split
  eval_cli.py     Evaluation command implementation
  evaluate.py     Similarity, classifier, generalization, and latency metrics
  features.py     Packet sequence and network series construction
  losses.py       SupCon, prototype, pairwise, disentanglement, adversarial losses
  memory.py       Embedding memory bank and prototype bank
  model.py        FlowCon-X encoder architecture
  train.py        Training loop and final report generation

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

outputs/
  flowconx_final_labeled_kpi_pass/
    flowconx_checkpoint.pt
    history.json
    metrics.json
    eval.json
    kpi_report.md
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, verify that PyTorch can use MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

The CI badge above covers `ruff` linting plus a compile and import check on Python 3.10 and 3.11. There is no automated test suite yet.

## Usage

Load the shipped checkpoint and encode a flow directly in Python. `packet_sequence_from_row` and `network_series_from_row` build the same tensors used during training, from a row that follows the [canonical CSV schema](#data-pipeline):

```python
import torch

from flowconx.model import FlowConX
from flowconx.features import network_series_from_row, packet_sequence_from_row

checkpoint = torch.load(
    "outputs/flowconx_final_labeled_kpi_pass/flowconx_checkpoint.pt",
    map_location="cpu",
)
label_maps = checkpoint["label_maps"]

model = FlowConX(n_conditions=len(label_maps["condition"]))
model.load_state_dict(checkpoint["model"])
model.eval()

row = {
    "packet_lengths": "1200;1200;64;1400;90",
    "iat_values": "0;12;3;45;2",
    "directions": "1;1;-1;1;-1",
    "rtt_ms": 28.0,
    "jitter_ms": 4.0,
    "loss_rate": 0.0,
    "protocol": 17,
}

packet_seq = torch.tensor(packet_sequence_from_row(row)).unsqueeze(0)
network_series = torch.tensor(network_series_from_row(row)).unsqueeze(0)

with torch.no_grad():
    embedding = model.encode(packet_seq, network_series)

print(embedding.shape)  # torch.Size([1, 256])
```

`embedding` is the normalized 256-dimensional `z_flow` vector used for the k-NN, SVM, and prototype classifiers reported in [Results](#results).

## Training

The final run was trained in two stages. The first stage trains the full labeled dataset. The second stage resumes from that checkpoint and tightens the fused embedding geometry.

Stage 1:

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

Stage 2:

```bash
python -u scripts/train_flowconx.py \
  --csv data/processed/flowconx_final_labeled_train.csv \
  --app-col app \
  --service-col service \
  --epochs 4 \
  --batch-size 256 \
  --augment-count 0 \
  --output-dir outputs/flowconx_final_stage2 \
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

- `--augment-count 0` keeps the final run real-data-only.
- `--temperature 0.05` sharpens supervised contrastive separation.
- `--lambda-flow-service` applies service contrastive learning to `z_flow`.
- `--lambda-flow-pair` applies explicit cosine margin shaping to `z_flow`.
- `--eval-max-train` and `--eval-max-test` keep k-NN and SVM evaluation practical while preserving stratified coverage.

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

## Results

Current checkpoint:

```text
outputs/flowconx_final_labeled_kpi_pass/flowconx_checkpoint.pt
```

Current metrics file:

```text
outputs/flowconx_final_labeled_kpi_pass/metrics.json
```

Full KPI pass/fail table:

```text
outputs/flowconx_final_labeled_kpi_pass/kpi_report.md
```

The final metrics saved in the repository are:

| Metric | Value |
| --- | ---: |
| Service intra-class cosine similarity | 0.8415 |
| Service inter-class cosine similarity | 0.2715 |
| k-NN service accuracy | 90.09% |
| k-NN macro F1 | 0.8842 |
| Linear SVM service accuracy | 90.44% |
| Linear SVM macro F1 | 0.8094 |
| Prototype accuracy | 90.16% |
| Prototype macro F1 | 0.8254 |
| Leave-one-app-out accuracy | 89.92% |
| Leave-one-app-out macro F1 | 0.7972 |
| Context invariance score | 0.6409 |
| Mean latency per flow | 13.65 ms |
| p50 latency per flow | 14.09 ms |
| p95 latency per flow | 17.01 ms |
| p99 latency per flow | 19.44 ms |

How to read these metrics:

- Intra-class cosine similarity measures how close flows from the same service are in embedding space.
- Inter-class cosine similarity measures how separated different services are.
- k-NN accuracy checks whether the embedding is directly useful without a learned classifier head.
- Linear SVM accuracy checks whether the embedding is linearly separable at service level.
- Prototype accuracy checks whether service centroids are strong enough for nearest-prototype classification.
- Leave-one-app-out accuracy holds out each app and classifies it using prototypes from the other apps.
- Context invariance score compares `z_app` before and after network-condition perturbation.
- Latency is single-flow inference time measured with model synchronization on the selected device.

The reported classifier metrics used stratified caps of 20,000 train embeddings and 10,000 test embeddings for the heavier evaluation steps. The model itself was trained on the full labeled training CSV.

## Design Tradeoffs

FlowCon-X intentionally avoids payload bytes. That makes it more suitable for encrypted traffic and privacy-preserving traffic analysis, but it also means the model must rely on side-channel behavior such as packet timing, length, direction, and rates.

The architecture is compact enough for real-time inference. The expensive part is not the forward pass, but large-scale evaluation with k-NN or SVM over many embeddings. That is why the training script supports stratified evaluation caps.

MAWI is included as a robustness artifact, not as a normal supervised class. It is useful real backbone traffic, but mixing weak labels into supervised accuracy would make the metric less honest.

XR coverage is present but small in the final labeled file. The architecture is ready for richer XR data, and the label map already includes `xr_interactive`, but more real XR captures would make that class much stronger.

## Limitations

- The repository does not ship the raw datasets because they are large.
- The final supervised training file depends on labels provided or inferred from public datasets.
- MAWI background traffic is weakly labeled as `unknown`.
- The memory bank is training-time only. Online continual updating would need a deployment policy around trust, drift, and rollback.
- The model has been validated as a research prototype, not as a production packet processing appliance.

## Why This Shape Works

The useful trick is not one module by itself. It is the agreement between architecture, data, and objective:

1. Real packet-flow data is converted into a common schema.
2. Packet behavior and network condition are encoded separately.
3. `z_app` is trained to keep service identity stable.
4. `z_net` captures the condition context.
5. Cross-attention builds `z_flow` from both views.
6. The final classifier embedding is trained with the same geometry that the evaluation uses.

That is why the project can classify encrypted traffic from metadata only while still producing interpretable embedding metrics.

## License

This project is licensed under the [MIT License](LICENSE).
