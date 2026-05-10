# Real Dataset Preprocessing Guide

This guide explains where to put downloaded data, how to convert each dataset into FlowCon-X format, and how to verify it before training.

FlowCon-X training expects one canonical CSV:

```text
data/processed/<dataset_name>_flows.csv
```

Each row is one flow or one fixed time-window segment. The required columns are:

```text
app, service, condition, packet_lengths, iat_values, directions,
rtt_ms, jitter_ms, loss_rate, total packets, total fwd packets,
total backward packets, packet length mean, packet length std,
flow iat mean, flow iat std, flow duration, flow bytes/s,
flow packets/s, protocol
```

The most important columns are:

- `app`: exact app or dataset label, for example `youtube`, `netflix`, `valorant`, `zoom`, `cloud_vr`.
- `service`: semantic class, for example `streaming`, `gaming`, `conferencing`, `bulk_transfer`, `browsing`, `xr_interactive`, `iot_security`.
- `packet_lengths`: semicolon-separated packet sizes for a flow segment.
- `iat_values`: semicolon-separated inter-arrival times in milliseconds.
- `directions`: semicolon-separated directions, use `1` for uplink or client-to-server and `-1` for downlink or server-to-client.
- `rtt_ms`, `jitter_ms`, `loss_rate`: network context. Use measured values if present. If unavailable, the converter estimates RTT and jitter from IATs.

## 1. Folder Layout

Place downloaded and processed data like this:

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
    cicflowmeter/
    tshark_csv/
  processed/
    samsung5g_flows.csv
    gavist5g_flows.csv
    cesnet_quic22_flows.csv
    mawi_flows.csv
    vr_broad_flows.csv
    vr_ar_cg_flows.csv
    nancy_vr_flows.csv
    ciciot2023_flows.csv
    manual_capture_flows.csv
    all_real_flows.csv
```

Do not train from `data/raw`. Always train from `data/processed`.

## 2. Inspect A Dataset First

After downloading a dataset, inspect its columns:

```bash
python -c "import pandas as pd; import sys; df=pd.read_csv(sys.argv[1], nrows=5); print(df.columns.tolist()); print(df.head())" data/raw/some_file.csv
```

For parquet:

```bash
python -c "import pandas as pd; import sys; df=pd.read_parquet(sys.argv[1]); print(df.columns.tolist()); print(df.head())" data/raw/some_file.parquet
```

Use the printed column names in the commands below.

## 3. The Converter

The main converter is:

```bash
python scripts/prepare_flow_csv.py --help
```

It supports three broad modes:

- `packet`: input rows are packets or per-second packet-like records.
- `aggregate`: input rows are already flows or aggregate records.
- `cesnet_quic`: input has CESNET-style `PPI`, `APP`, and `CATEGORY` fields.

The converter also has dataset presets:

```text
generic_packet
generic_aggregate
cesnet_quic22
vr_broad
vr_ar_cg
ciciot2023
cicflowmeter
```

## 4. Samsung 5G Traffic Datasets

Role:

- Primary training data for app and service classification.
- Good for streaming, conferencing, metaverse-like, gaming, and mobile traffic.

Expected raw format:

- CSV time series from PCAPdroid style capture.
- The dataset page describes timestamp-mapped traffic with packet header information and source and destination addresses.
- Apps include Netflix, Amazon Prime, YouTube Live, Zoom, MS Teams, Google Meet, Zepeto, Roblox, cloud gaming, and mobile gaming.

Place files:

```text
data/raw/samsung5g/
```

If the files are packet-level or per-second time series:

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

If app labels are in a column:

```bash
python scripts/prepare_flow_csv.py \
  --dataset generic_packet \
  --input "data/raw/samsung5g/**/*.csv" \
  --output data/processed/samsung5g_flows.csv \
  --time-col Time \
  --length-col Length \
  --protocol-col Protocol \
  --app-col app \
  --window-seconds 10
```

If each file or folder is one app, process one app at a time:

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

Repeat for other apps, then combine:

```bash
python -c "import pandas as pd, glob; pd.concat([pd.read_csv(f) for f in glob.glob('data/processed/samsung5g_*.csv')]).to_csv('data/processed/samsung5g_flows.csv', index=False)"
```

## 5. GAViST5G

Role:

- Strongest Samsung KPI alignment for YouTube, Netflix, Prime Video, League of Legends, Teamfight Tactics, and Valorant.
- Includes application labels, timestamps, packet length, protocol, source and destination IPs, geolocation, and RTT in the Kaggle description.

Place files:

```text
data/raw/gavist5g/
```

If using packet-level CSV:

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

If the app label column is named `content_provider`:

```bash
python scripts/prepare_flow_csv.py \
  --dataset generic_packet \
  --input "data/raw/gavist5g/**/*.csv" \
  --output data/processed/gavist5g_flows.csv \
  --time-col Time \
  --length-col Length \
  --protocol-col Protocol \
  --rtt-col RTT \
  --app-col content_provider \
  --window-seconds 10
```

If the downloaded Kaggle version is already aggregated at one-second granularity, keep the same command. Each 10-second window becomes one FlowCon-X training example.

## 6. CESNET-QUIC22

Role:

- Encrypted QUIC validation.
- Use this to prove the model works from packet sizes, directions, inter-packet times, and metadata without DPI.

Expected fields:

- `PPI`: per-packet information, usually inter-packet times, directions, and packet sizes.
- `APP`: web service label.
- `CATEGORY`: service category.
- Flow statistics such as bytes, packets, histograms, duration, and end reason may also exist.

Place files:

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

If the raw files are too large, start with a row cap:

```bash
python scripts/prepare_flow_csv.py \
  --dataset cesnet_quic22 \
  --input "data/raw/cesnet_quic22/**/*" \
  --output data/processed/cesnet_quic22_sample_flows.csv \
  --limit-files 2 \
  --limit-rows-per-file 200000
```

## 7. MAWI

Role:

- Real-world robustness and domain shift.
- MAWI is usually raw packet capture, so you need a flow extraction step first.

Place PCAPs:

```text
data/raw/mawi/
```

Extract packet CSV with tshark:

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

Convert to FlowCon-X:

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

MAWI labels are weak unless you enrich the flows. Use MAWI mainly for robustness, not the primary KPI.

## 8. VR Traffic Dataset On Broad Range Of End-User Activities

Role:

- Real XR holdout data.
- Great for proving XR readiness.

Expected fields:

- `Time`
- `Protocol`
- `TCP Segment Len`
- `UDP length`
- `Link`

The source paper describes 9 CSV traces across Half-Life: Alyx, Beat Saber, Google Earth, Rec Room, and VR Chat.

Place files:

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

If file names are not clean app labels, run per app:

```bash
python scripts/prepare_flow_csv.py \
  --dataset vr_broad \
  --input "data/raw/vr_broad/beat_saber*.csv" \
  --output data/processed/vr_broad_beat_saber.csv \
  --app beat_saber \
  --service xr_interactive \
  --window-seconds 10
```

## 9. VR-AR-CG Network Telemetry

Role:

- AR, VR, and cloud gaming validation.
- Useful because it has extracted feature CSVs, not only PCAP.

Expected AR or cloud gaming feature fields:

- `ID`
- `SrcIP`
- `DstIP`
- `IPVersion`
- `Protocol`
- `PS`
- `IPI`
- `FlowSizeBytes`
- `FlowSizePackets`
- `FS`
- `FS(PKT)`
- `NumFrames`
- `IFI`

Place files:

```text
data/raw/vr_ar_cg/
```

For AR feature CSVs:

```bash
python scripts/prepare_flow_csv.py \
  --dataset vr_ar_cg \
  --input "data/raw/vr_ar_cg/AR dataset/**/*.csv" \
  --output data/processed/vr_ar_cg_ar_flows.csv \
  --service xr_interactive
```

For cloud gaming feature CSVs:

```bash
python scripts/prepare_flow_csv.py \
  --dataset vr_ar_cg \
  --input "data/raw/vr_ar_cg/CG dataset/**/*Features/*.csv" \
  --output data/processed/vr_ar_cg_cloud_gaming_flows.csv \
  --service gaming
```

If you want cloud gaming to test XR-like interactive media, set:

```text
--service xr_interactive
```

## 10. NANCY VR Video Streaming And iPerf3

Role:

- VR video plus 5G and O-RAN context.
- Useful for XR validation and network-condition evaluation.

Expected raw format:

- PCAP files.
- xApp performance metric files.

Place files:

```text
data/raw/nancy_vr/
```

Extract packet CSV from PCAP:

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

If xApp metrics include RTT, queue delay, or throughput, merge them into the packet CSV before conversion or add them later to the canonical CSV.

## 11. CICIoT2023

Role:

- IoT and security robustness.
- Not the primary app/service dataset for Samsung's YouTube versus gaming KPI.

Place files:

```text
data/raw/ciciot2023/
```

Convert:

```bash
python scripts/prepare_flow_csv.py \
  --dataset ciciot2023 \
  --input "data/raw/ciciot2023/**/*.csv" \
  --output data/processed/ciciot2023_flows.csv
```

This maps all labels to service `iot_security`. Keep the original attack label in `app`.

## 12. Manual Packet Capture

Role:

- Best hackathon demo data.
- Lets you show same app under good and bad network conditions.

Place PCAPs:

```text
data/raw/manual_capture/
  youtube_good.pcap
  youtube_bad.pcap
  netflix_good.pcap
  valorant_good.pcap
  zoom_good.pcap
```

Extract packet CSV:

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

Convert one file:

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

Repeat for all captures, then combine:

```bash
python -c "import pandas as pd, glob; pd.concat([pd.read_csv(f) for f in glob.glob('data/processed/manual_*.csv')]).to_csv('data/processed/manual_capture_flows.csv', index=False)"
```

## 13. Combine Real Datasets

After processing multiple datasets:

```bash
python -c "import pandas as pd; files=['data/processed/samsung5g_flows.csv','data/processed/gavist5g_flows.csv','data/processed/vr_broad_flows.csv','data/processed/vr_ar_cg_ar_flows.csv']; pd.concat([pd.read_csv(f) for f in files]).to_csv('data/processed/all_real_flows.csv', index=False)"
```

Do not mix weakly labeled datasets into the main KPI run unless the labels are clean. Recommended:

- Main KPI training: Samsung 5G plus GAViST5G plus manual capture.
- XR holdout: VR Broad plus VR-AR-CG plus NANCY VR.
- Encrypted validation: CESNET-QUIC22.
- Robustness only: MAWI and CICIoT2023.

## 14. Validate Processed CSV

Check columns:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/processed/all_real_flows.csv'); print(df.columns.tolist()); print(df[['app','service','condition']].value_counts().head(20)); print(df.shape)"
```

Run the baseline:

```bash
python scripts/run_baseline.py \
  --csv data/processed/all_real_flows.csv \
  --app-col app \
  --service-col service \
  --output outputs/baseline_real_metrics.json
```

If the baseline reports impossible results, for example 100 percent accuracy with tiny data, check for leakage:

- Do not include filename, scenario ID, or app ID as numeric features.
- Make sure train and test split by capture/session for final reporting.
- Keep XR holdout separate for generalization testing.

## 15. Train On Processed Real Data

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

Evaluate:

```bash
python scripts/evaluate_flowconx.py \
  --checkpoint outputs/flowconx_real/flowconx_checkpoint.pt \
  --csv data/processed/all_real_flows.csv \
  --app-col app \
  --service-col service \
  --output outputs/flowconx_real_eval.json
```

Prototype compass:

```bash
python scripts/run_prototype_compass.py \
  --checkpoint outputs/flowconx_real/flowconx_checkpoint.pt \
  --calibration-csv data/processed/all_real_flows.csv \
  --stream-csv data/processed/vr_broad_flows.csv \
  --app-col app \
  --service-col service \
  --limit 20
```

