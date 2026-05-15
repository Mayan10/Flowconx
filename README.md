# FlowCon-X

Context-aware flow embeddings for adaptive AI based network traffic classification.

FlowCon-X is a research prototype for the Samsung ennovateX problem statement, "Context-Aware Flow Embeddings for Adaptive AI based Network Traffic Classification." The goal is to learn packet-flow embeddings that stay useful when traffic is encrypted, network conditions change, and new traffic types appear.

The core idea is simple: a flow has two stories happening at the same time.

1. What the application is doing.
2. What the network is doing to it.

FlowCon-X tries to separate those two signals. A YouTube flow under bad Wi-Fi should still sit near other streaming flows, not drift into gaming or bulk transfer just because RTT and jitter changed.

## Current Status

This repository contains:

- A PyTorch encoder for packet-flow embeddings.
- Flow and packet feature preprocessing utilities.
- Supervised contrastive learning losses.
- A prototype memory bank for adaptive classification.
- A NumPy baseline for quick sanity checks.
- Training, evaluation, and prototype compass scripts.

Training is real-data-only by default. Synthetic data exists only as an explicit smoke-test utility. The training script fails if no real CSV is provided, unless `--synthetic` is passed on purpose.

## What Makes FlowCon-X Different

Most traffic classifiers optimize only closed-set accuracy. That can look good in a notebook and fail quickly in a real network. FlowCon-X is built around the geometry of the embedding space.

FlowCon-X learns:

- `z_app`: an application and service identity embedding.
- `z_net`: a network condition embedding.
- `z_flow`: a fused embedding used for downstream classification.

The model is trained so that:

- Similar service traffic is close together.
- Different service traffic is far apart.
- Application identity is less sensitive to RTT, jitter, and loss.
- New flows can be matched against semantic service prototypes.

The extra evaluation metric is CIST, the Context Invariance Stress Test. It checks whether the app embedding remains stable when the same flow is evaluated under changed network conditions.

## Samsung KPI Mapping

| KPI | Samsung target | FlowCon-X output |
|---|---:|---|
| Embedding similarity | Intra-class cosine > 0.7, inter-class cosine < 0.3 | `service_similarity` |
| Classification accuracy | At least 90 percent | k-NN and SVM on frozen embeddings |
| Generalization | At least 85 percent on unseen traffic types | prototype and leave-one-app evaluation |
| Real-time performance | < 100 ms per flow | p50, p95, p99 latency |

Samsung's examples treat YouTube and Netflix as similar because both are streaming. FlowCon-X therefore tracks both app labels and service labels.

Example service mapping:

```text
youtube, netflix, prime_video        -> streaming
valorant, roblox, cloud_gaming       -> gaming
zoom, teams, google_meet             -> conferencing
ftp, file_download                   -> bulk_transfer
web_browsing                         -> browsing
cloud_vr, vr_video, ar_session       -> xr_interactive
benign, ddos, mirai, spoofing        -> iot_security
```

## Repository Layout

```text
flowconx/
  config.py        Label taxonomy and model dimensions
  features.py      Flow feature construction and network condition handling
  datasets.py      FlowRecord loading and PyTorch dataset wrappers
  model.py         FlowCon-X neural encoder
  losses.py        SupCon, prototype, disentanglement, adversarial losses
  memory.py        Contrastive memory and prototype bank
  train.py         Training entry point
  eval_cli.py      Evaluation entry point
  baselines.py     NumPy baseline
  compass.py       Prototype compass demo
  synthetic.py     Smoke-test data generator only

scripts/
  prepare_flow_csv.py         Convert real datasets to canonical FlowCon-X CSV
  train_flowconx.py           Train wrapper
  evaluate_flowconx.py        Eval wrapper
  run_baseline.py             NumPy baseline wrapper
  run_prototype_compass.py    Prototype compass wrapper
  make_synthetic_dataset.py   Smoke-test data generator only
```

## Data Policy

Do not train the final model on synthetic data.

Use real public datasets and manual captures. Keep synthetic generation only for checking that the pipeline is wired correctly.

Recommended folder layout:

```text
data/
  raw/
    samsung5g/
    gavist5g/
    cesnet_quic22/
    mawi/
    vr_broad/
    vr_ar_cg/
    nancy_vr/
    ciciot2023/
    manual_capture/
  interim/
    tshark_csv/
    cicflowmeter/
  processed/
    samsung5g_flows.csv
    gavist5g_flows.csv
    cesnet_quic22_flows.csv
    vr_broad_flows.csv
    vr_ar_cg_flows.csv
    ciciot2023_flows.csv
    all_real_flows.csv
```

Training should use files from `data/processed/`, not from `data/raw/`.

## Datasets To Download

Download in this order:

1. 5G Traffic Datasets
   - https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets

2. GAViST5G
   - https://www.kaggle.com/datasets/ahassanein/aggregated-gaming-and-video-streaming-traffic-for-5g

3. CESNET-QUIC22
   - https://zenodo.org/records/7409924

4. VR Traffic Dataset
   - https://figshare.com/articles/dataset/VR_Traffic_Dataset_on_Broad_Range_of_End-user_Activities/22191160

5. VR-AR-CG Network Telemetry
   - https://github.com/dcomp-leris/VR-AR-CG-network-telemetry

6. NANCY VR Video Streaming and iPerf3 Dataset
   - https://zenodo.org/records/13863832

7. MAWI Dataset
   - https://mawi.wide.ad.jp/mawi/

8. CICIoT2023
   - https://www.unb.ca/cic/datasets/iotdataset-2023.html

For CICIoT2023, start with `MERGED_CSV.zip` and `README.pdf`. Do not start with PCAP unless you specifically need packet-level IoT traces.

## Canonical Training CSV

All datasets should be converted into this format:

```text
app, service, condition, packet_lengths, iat_values, directions,
rtt_ms, jitter_ms, loss_rate, total packets, total fwd packets,
total backward packets, packet length mean, packet length std,
flow iat mean, flow iat std, flow duration, flow bytes/s,
flow packets/s, protocol
```

Important columns:

- `app`: exact app or dataset label, such as `youtube`, `netflix`, `valorant`, `zoom`, `cloud_vr`.
- `service`: semantic class, such as `streaming`, `gaming`, `conferencing`, `xr_interactive`.
- `packet_lengths`: semicolon-separated packet sizes.
- `iat_values`: semicolon-separated inter-arrival times in milliseconds.
- `directions`: semicolon-separated packet directions, where `1` means uplink and `-1` means downlink.
- `rtt_ms`, `jitter_ms`, `loss_rate`: network context values.

## Installation

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch needs a platform-specific install command, install PyTorch first from the official selector, then install the rest:

```bash
pip install torch
pip install -r requirements.txt
```

## Preprocessing Real Datasets

The main converter is:

```bash
python scripts/prepare_flow_csv.py --help
```

It supports these presets:

```text
generic_packet
generic_aggregate
cesnet_quic22
vr_broad
vr_ar_cg
ciciot2023
cicflowmeter
```

### 5G Traffic Datasets

Place files here:

```text
data/raw/samsung5g/
```

Convert packet or time-series CSV files:

```bash
python scripts/prepare_flow_csv.py \
  --dataset generic_packet \
  --input "data/raw/samsung5g/**/*.csv" \
  --output data/processed/samsung5g_flows.csv \
  --time-col Time \
  --length-col Length \
  --protocol-col Protocol \
  --window-seconds 10
```

If app labels are stored in a column, add:

```bash
--app-col app
```

If each folder is one app, process per app:

```bash
python scripts/prepare_flow_csv.py \
  --dataset generic_packet \
  --input "data/raw/samsung5g/netflix/**/*.csv" \
  --output data/processed/samsung5g_netflix.csv \
  --app netflix \
  --service streaming \
  --time-col Time \
  --length-col Length \
  --protocol-col Protocol \
  --window-seconds 10
```

### GAViST5G

Place files here:

```text
data/raw/gavist5g/
```

Convert:

```bash
python scripts/prepare_flow_csv.py \
  --dataset generic_packet \
  --input "data/raw/gavist5g/**/*.csv" \
  --output data/processed/gavist5g_flows.csv \
  --time-col Time \
  --length-col Length \
  --protocol-col Protocol \
  --rtt-col RTT \
  --app-col app \
  --window-seconds 10
```

If the app column is named differently, replace `--app-col app` with the actual column name.

### CESNET-QUIC22

Place files here:

```text
data/raw/cesnet_quic22/
```

Convert:

```bash
python scripts/prepare_flow_csv.py \
  --dataset cesnet_quic22 \
  --input "data/raw/cesnet_quic22/**/*" \
  --output data/processed/cesnet_quic22_flows.csv
```

Start with a smaller sample if the files are large:

```bash
python scripts/prepare_flow_csv.py \
  --dataset cesnet_quic22 \
  --input "data/raw/cesnet_quic22/**/*" \
  --output data/processed/cesnet_quic22_sample_flows.csv \
  --limit-files 2 \
  --limit-rows-per-file 200000
```

### VR Traffic Dataset

Place files here:

```text
data/raw/vr_broad/
```

Convert:

```bash
python scripts/prepare_flow_csv.py \
  --dataset vr_broad \
  --input "data/raw/vr_broad/*.csv" \
  --output data/processed/vr_broad_flows.csv \
  --window-seconds 10
```

If filenames are unclear, process per app:

```bash
python scripts/prepare_flow_csv.py \
  --dataset vr_broad \
  --input "data/raw/vr_broad/beat_saber*.csv" \
  --output data/processed/vr_broad_beat_saber.csv \
  --app beat_saber \
  --service xr_interactive \
  --window-seconds 10
```

### VR-AR-CG Network Telemetry

Place files here:

```text
data/raw/vr_ar_cg/
```

Convert AR traces:

```bash
python scripts/prepare_flow_csv.py \
  --dataset vr_ar_cg \
  --input "data/raw/vr_ar_cg/AR dataset/**/*.csv" \
  --output data/processed/vr_ar_cg_ar_flows.csv \
  --service xr_interactive
```

Convert cloud gaming traces:

```bash
python scripts/prepare_flow_csv.py \
  --dataset vr_ar_cg \
  --input "data/raw/vr_ar_cg/CG dataset/**/*Features/*.csv" \
  --output data/processed/vr_ar_cg_cloud_gaming_flows.csv \
  --service gaming
```

### NANCY VR Dataset

Place PCAPs here:

```text
data/raw/nancy_vr/
```

Extract packet CSV with `tshark`:

```bash
mkdir -p data/interim/nancy_vr_tshark
tshark -r data/raw/nancy_vr/your_capture.pcap \
  -T fields \
  -e frame.time_relative \
  -e ip.src \
  -e ip.dst \
  -e _ws.col.Protocol \
  -e frame.len \
  -E header=y \
  -E separator=, \
  -E quote=d \
  > data/interim/nancy_vr_tshark/your_capture.csv
```

Convert:

```bash
python scripts/prepare_flow_csv.py \
  --dataset generic_packet \
  --input "data/interim/nancy_vr_tshark/*.csv" \
  --output data/processed/nancy_vr_flows.csv \
  --time-col frame.time_relative \
  --length-col frame.len \
  --protocol-col _ws.col.Protocol \
  --app vr_video \
  --service xr_interactive \
  --window-seconds 10
```

### MAWI

Place PCAPs here:

```text
data/raw/mawi/
```

Extract packet CSV:

```bash
mkdir -p data/interim/mawi_tshark
tshark -r data/raw/mawi/sample.pcap \
  -T fields \
  -e frame.time_relative \
  -e ip.src \
  -e ip.dst \
  -e _ws.col.Protocol \
  -e frame.len \
  -E header=y \
  -E separator=, \
  -E quote=d \
  > data/interim/mawi_tshark/sample.csv
```

Convert:

```bash
python scripts/prepare_flow_csv.py \
  --dataset generic_packet \
  --input "data/interim/mawi_tshark/*.csv" \
  --output data/processed/mawi_flows.csv \
  --time-col frame.time_relative \
  --length-col frame.len \
  --protocol-col _ws.col.Protocol \
  --app mawi_background \
  --service unknown \
  --window-seconds 10
```

Use MAWI for robustness checks, not the main KPI table, unless you add reliable labels.

### CICIoT2023

Place files here:

```text
data/raw/ciciot2023/
```

Recommended download:

```text
MERGED_CSV.zip
README.pdf
```

Unzip into:

```text
data/raw/ciciot2023/MERGED_CSV/
```

Convert:

```bash
python scripts/prepare_flow_csv.py \
  --dataset ciciot2023 \
  --input "data/raw/ciciot2023/MERGED_CSV/**/*.csv" \
  --output data/processed/ciciot2023_flows.csv
```

Start with a subset of the 63 merged CSVs. CICIoT2023 is useful for IoT and security robustness, but it is not the primary Samsung app-classification dataset.

### Manual Captures

Place PCAPs here:

```text
data/raw/manual_capture/
```

Suggested captures:

```text
youtube_good.pcap
youtube_bad.pcap
netflix_good.pcap
valorant_good.pcap
zoom_good.pcap
file_download_good.pcap
```

Extract:

```bash
mkdir -p data/interim/manual_tshark
tshark -r data/raw/manual_capture/youtube_good.pcap \
  -T fields \
  -e frame.time_relative \
  -e ip.src \
  -e ip.dst \
  -e _ws.col.Protocol \
  -e frame.len \
  -E header=y \
  -E separator=, \
  -E quote=d \
  > data/interim/manual_tshark/youtube_good.csv
```

Convert:

```bash
python scripts/prepare_flow_csv.py \
  --dataset generic_packet \
  --input data/interim/manual_tshark/youtube_good.csv \
  --output data/processed/manual_youtube_good.csv \
  --time-col frame.time_relative \
  --length-col frame.len \
  --protocol-col _ws.col.Protocol \
  --app youtube \
  --service streaming \
  --condition good \
  --window-seconds 10
```

## Combine Processed Files

Combine only the datasets you want in a given experiment.

```bash
python -c "import pandas as pd; files=['data/processed/samsung5g_flows.csv','data/processed/gavist5g_flows.csv','data/processed/vr_broad_flows.csv']; pd.concat([pd.read_csv(f) for f in files]).to_csv('data/processed/all_real_flows.csv', index=False)"
```

Recommended split:

- Main KPI training: 5G Traffic Datasets, GAViST5G, manual captures.
- XR holdout: VR Traffic Dataset, VR-AR-CG, NANCY VR.
- Encrypted validation: CESNET-QUIC22.
- Robustness only: MAWI and CICIoT2023.

## Validate A Processed CSV

Check labels and shape:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/processed/all_real_flows.csv'); print(df.shape); print(df[['app','service','condition']].value_counts().head(20))"
```

Run the baseline:

```bash
python scripts/run_baseline.py \
  --csv data/processed/all_real_flows.csv \
  --app-col app \
  --service-col service \
  --output outputs/baseline_real_metrics.json
```

## Train

Strict real-data-only training:

```bash
python scripts/train_flowconx.py \
  --csv data/processed/all_real_flows.csv \
  --app-col app \
  --service-col service \
  --epochs 20 \
  --batch-size 64 \
  --augment-count 0 \
  --output-dir outputs/flowconx_real
```

The important flag is:

```text
--augment-count 0
```

That keeps training real-data-only. Counterfactual augmentation can be useful later, but it is off by default.

## Evaluate

```bash
python scripts/evaluate_flowconx.py \
  --checkpoint outputs/flowconx_real/flowconx_checkpoint.pt \
  --csv data/processed/all_real_flows.csv \
  --app-col app \
  --service-col service \
  --output outputs/flowconx_real_eval.json
```

The metrics file contains:

```text
service_similarity.intra
service_similarity.inter
classification.knn_accuracy
classification.svm_accuracy
prototype_generalization.prototype_accuracy
leave_one_app_out.leave_one_app_accuracy
cist_score
latency.mean_ms
latency.p95_ms
latency.p99_ms
```

## Prototype Compass

The prototype compass is a simple demo for showing where a new flow lands in service space.

```bash
python scripts/run_prototype_compass.py \
  --checkpoint outputs/flowconx_real/flowconx_checkpoint.pt \
  --calibration-csv data/processed/all_real_flows.csv \
  --stream-csv data/processed/vr_broad_flows.csv \
  --app-col app \
  --service-col service \
  --limit 20
```

Example output:

```text
flow=1 app=cloud_vr service=xr_interactive compass=xr_interactive:0.821, streaming:0.702, gaming:0.351, bulk_transfer:0.114
```

## Smoke Test

Synthetic data is only for checking that scripts run.

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_flows.csv --flows-per-app 40
python scripts/run_baseline.py --csv data/synthetic_flows.csv --app-col app --service-col service
```

Do not use this result in the final Samsung KPI table.

## Notes And Limitations

- The model expects processed CSV files, not raw downloads.
- Raw PCAP datasets need `tshark` or CICFlowMeter before training.
- CICIoT2023 is not the main app-classification dataset. Use it for IoT and security robustness.
- MAWI is useful for domain shift, but labels are weak unless enriched.
- Final numbers should be reported on real datasets only.

