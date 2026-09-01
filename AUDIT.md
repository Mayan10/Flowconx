# FlowCon-X Phase 0 Audit

**Commit audited:** `090b4b9` (`Add Quick Start and Usage sections, drop unused dependencies`)
**Date:** 2026-08-28
**Auditor's brief:** find every way the reported 90.09% k-NN accuracy could be a
property of how the dataset was assembled rather than of the architecture.

This document describes the pipeline **as it was at `090b4b9`**. Fixes landed in
Phase 0 are marked *(fixed)* inline and listed in `CHANGELOG.md`.

---

## 1. Data pipeline

### 1.1 Raw inputs

| Source | Preparer | What a raw record is |
| --- | --- | --- |
| Kaggle *5G Traffic Datasets* | `scripts/prepare_5g_traffic_dataset.py` | Wireshark CSV export per capture: `Time, Source, Destination, Protocol, Length` |
| CESNET-QUIC22 | `scripts/prepare_cesnet_quic22_dataset.py` | `flows-*.csv.gz`, one row per bidirectional flow, with a `PPI` field holding per-packet (IAT, direction, length) triples |
| MAWI | `scripts/prepare_mawi_pcap.py` | Raw `.pcap`, parsed byte-by-byte in pure Python |

No raw data is committed. `data/processed/` holds the two derived CSVs:
`flowconx_final_labeled_train.csv` (112,121 rows, 87 MB) and
`flowconx_mawi_robustness_background.csv` (20,000 rows).

### 1.2 What a "flow" is — and it is not the same thing in each source

This is the single most important structural fact about the dataset.

- **CESNET-QUIC22 rows are flows.** One row is one bidirectional conversation,
  and the `PPI` field gives the first ~30 packets of it.
- **5G Traffic rows are not flows.** `prepare_5g_traffic_dataset.py` buckets
  packets into fixed **10-second wall-clock windows** over the *entire capture
  file* (`window_id = (t - t0) // window_seconds`) and aggregates every
  conversation in that window into a single row. A 5G Traffic row is therefore a
  time slice of a whole capture, not a conversation.
- **MAWI rows are 5-tuple flows within 10-second windows**, reservoir-sampled to
  20,000. They carry `service = unknown` and are not part of the labelled set.

Consequences: consecutive rows from one 5G Traffic capture are adjacent slices of
the same continuous session and are strongly autocorrelated; and the *meaning* of
one row differs by source, so a single classifier is being trained across two
different units of observation.

### 1.3 Truncation and padding

- Preparers keep at most `--max-packets` (default 128) per row.
- CESNET `PPI` supplies **at most ~30 packets**; 5G Traffic windows supply up to
  128. **Sequence length is therefore a source-dataset marker.**
- `flowconx/features.py::pad_or_trim` zero-pads to `MAX_PACKETS = 128`.
- The padding mask in `FlowDataset.__getitem__` is
  `np.isclose(packet_seq.sum(axis=1), 0.0)`. A real packet whose 16 feature
  channels happen to sum to zero would be masked out as padding. In practice
  channel 15 (positional index) makes this near-impossible except at index 0,
  but it is a structural rather than a guaranteed property.

### 1.4 Features

**Packet features (`PKT_FEAT_DIM = 16`).** Only channels 0, 1, 2 and 14–15 vary
per packet (length, log-IAT, direction, burst index, position). **Channels 3–13
are flow-level scalars broadcast identically across all 128 positions** —
packets/s, bytes/s, forward ratio, backward ratio, TCP/UDP/QUIC flags,
SYN/ACK/RST counts, duration. Eleven of sixteen channels carry a constant. The
transformer therefore spends most of its input width re-reading a NetFlow record.

**Network features (`NET_FEAT_DIM = 8`, 24 timesteps).** Every channel is a
constant derived from the row's scalars plus Gaussian drift
(`series += drift * (0.5 + t)`). There is no measured time series here: the
"network condition encoder" reads a constant vector with noise on it.
Channel 7 is `condition_to_index(infer_condition(rtt, jitter, loss)) / 3` —
**the declared nuisance label itself, quantised, handed in as an input.**

### 1.5 Label source

| Source | How `app` is set | How `service` is set |
| --- | --- | --- |
| 5G Traffic | Substring match of the **capture file path** against `KNOWN_APPS`, else the parent directory name | `infer_service(app)` — a hand-written dict plus a substring fallback |
| CESNET-QUIC22 | The dataset's `APP` column | The dataset's `CATEGORY` column, mapped through `CATEGORY_TO_SERVICE` |

So the label is a **capture-folder name** for one source and a **publisher's app
catalogue category** for the other. These are two different labelling regimes
fused into one 6-class label space, and the paper must say so.

Within the committed CSV, `app → service` is a **function** (no app carries two
service labels), which means app-level supervision is service supervision at
finer granularity, not independent information.

### 1.6 Hardcoded paths and magic constants

| Location | Constant | Note |
| --- | --- | --- |
| `config.py` | `MAX_PACKETS=128`, `NET_TIMESTEPS=24`, `PKT_FEAT_DIM=16`, `NET_FEAT_DIM=8`, `APP_EMB_DIM=256`, `NET_EMB_DIM=128`, `FLOW_EMB_DIM=256` | No config file; changing any requires editing source |
| `features.py` | Normalisation divisors `1500`, `10`, `20`, `16`, `64`, burst quantiser `8.0` | Undocumented, unvalidated against the data range |
| `features.py` | `infer_condition` thresholds `80/180 ms` RTT, `25 ms` jitter, `0.01/0.04` loss | Defines the nuisance label; see §3, L5 |
| `features.py` | `augment_network_condition` choice lists `[0.7,1.0,1.8,3.2]`, `[0,0.1,0.25,0.55]`, `[0,0.005,0.02,0.06]` | The only "robustness" perturbation in the repo |
| `memory.py` | `momentum=0.95`, `high_confidence=0.75` | Prototype update gate |
| `evaluate.py` | `cist_score(max_records=256, seed=99)`, `benchmark_latency(runs=100)` | Headline CIST number rests on 256 rows |
| `train.py` | `PrototypeBank(..., emb_dim=256)` | Hardcoded; silently breaks if `app_emb_dim` changes |
| `datasets.py` | `split_records(test_fraction=0.2, seed=42)` | See §2 |

---

## 2. Split logic as it stood

`flowconx/datasets.py::split_records` performs a **stratified random split over
individual flows**, keyed on `service`, seed 42, `test_fraction=0.2`. There is
**no validation split** — hyperparameters were selected against the test set, or
against nothing.

It is a random split **over flows**, not over sessions, not over capture files,
and not over time. Nothing in the pipeline retains the information that would
make a stricter split possible: the committed CSV has no flow ID, no capture
identifier, no timestamp, no addresses. *(fixed: `flowconx/schema.py` now defines
`flow_id`, `origin`, `capture_id`, `flow_start_ts`, `server_ip`, and all three
preparers emit them. The existing CSV predates this and must be regenerated.)*

`flowconx/eval_cli.py` re-runs the same random split with the same seed on
whatever CSV it is handed, so an "evaluation" on the training CSV re-derives the
same partition — but only as long as the row order and the seed are unchanged.

---

## 3. Leakage and shortcut surfaces

Ordered by how much damage each does to the headline claim.

**L1 — No grouping in the split.** *(fixed: `flowconx/audit/splits.py`)*
Random flow splitting with no group constraint. This alone invalidates
comparison with any published number obtained under a stricter protocol.

**L2 — 5G Traffic windows straddle the split.** Adjacent 10-second slices of one
continuous capture are near-duplicates of each other and land on both sides. The
model can memorise a session and be scored on the next ten seconds of it.

**L3 — `protocol` is a source-dataset marker, and it is a model input.**
In the committed CSV, `protocol = 0` occurs on 11,985 rows, of which 11,975
(99.9%) are `streaming`; `protocol = 17` covers essentially all CESNET rows. The
value 0 means "the preparer could not parse the Protocol column", i.e. it encodes
*which preparer ran*. It reaches the model as packet-feature channels 7–9.

**L4 — Sequence length is a source-dataset marker.** CESNET rows carry ≤30
packets, 5G Traffic rows up to 128. The padding mask makes this directly visible
to the encoder.

**L5 — The nuisance label is a deterministic function of two model inputs.**
Both preparers set `condition = infer_condition(iat_mean, iat_std, 0.0)`, and
they also write `rtt_ms = iat_mean`, `jitter_ms = iat_std`. So `condition` is a
threshold on `flow iat mean` and `flow iat std`, both of which are model inputs,
and it is *additionally* injected as network-series channel 7. The adversarial
condition-removal head is therefore being asked to delete a quantisation of the
same timing statistics the task classifier depends on. The invariance claim as
stated is circular. `tests/test_leakage.py::test_nuisance_condition_is_flagged_as_derivable`
pins this so it cannot regress silently.

**L6 — `loss_rate` is identically 0.0 across the entire corpus.** Two of the five
branches of `infer_condition` are dead code on this data, and the "packet loss"
robustness story has no measured support.

**L7 — The synthetic-fallback RNG is seeded with the label.**
`records_from_dataframe` calls `np.random.default_rng(stable_seed(prefix, row_idx, app, service))`.
When `packet_lengths`/`iat_values`/`directions` are absent, the *entire* packet
sequence is drawn from that generator, so the sequence becomes a function of the
label. The committed CSV does carry the real series, so this does not fire here —
but any dataset without them would produce a fully synthetic, label-determined
input, and nothing in the code warns about it.

**L8 — Default CLI flags collapse `app` onto `service`.**
`--app-col` and `--service-col` default to `None`. With both unset,
`detect_label_column` picks `service`, then `raw_app = row[label_col]`, so
`app == service` for every row. The app-level contrastive term duplicates the
service term, and `leave_one_app_out_generalization` becomes leave-one-*class*-out,
which deletes the class it is scoring. The committed run passed both flags
explicitly, so its numbers are unaffected — but the documented default command is
not the command that produced them.

**L9 — `infer_service` matches substrings.** The fallback loop
`for key in DEFAULT_APP_TO_SERVICE: if key in token` matches the two-character
keys `ar` and `vr` inside unrelated names. `csgo_m**ar**ket` and
`cloudfl**ar**e_cdnjs` both resolve to `xr_interactive`. 89 of ~106 apps in the
CSV disagree with what `infer_service` would return for them today.

**L10 — Duplicate rows.** 284 byte-identical rows and 763 rows sharing an
identical `(packet_lengths, iat_values, directions)` triple. Under a random split
these straddle train and test.

**L11 — `app → service` is a function**, so the app-level and service-level
objectives are not independent signals.

**L12 — `xr_interactive` is a 23-row outlier class.** Median 102,125 packets per
row versus 23–66 for every other class; it is trivially separable, contributes
~4 test rows, and its per-class F1 is noise that moves macro-F1 by ~17 points.
*(fixed: `flowconx/labels.py` implements `drop` / `merge_into_parent` /
`keep_and_report`.)*

---

## 4. Reproducibility findings

**R1 — The committed headline numbers cannot be regenerated by the committed code.**
`outputs/flowconx_final_labeled_kpi_pass/flowconx_checkpoint.pt` records the
arguments it ran with. Two of them — `--flows-per-app 80` and `--synthetic` — **do
not exist in `flowconx/train.py` at this commit**. The run predates the current
CLI.

**R2 — The headline run resumed from an uncommitted checkpoint.**
`"resume_checkpoint": "outputs/flowconx_final_labeled_flow_tuned/flowconx_checkpoint.pt"`.
That directory is not in the repository and is excluded by `.gitignore`. The
recorded `--epochs 4` is four epochs *on top of an unknown prior state*. Every
number in `metrics.json` therefore has an unreproducible ancestor.

**R3 — Seeding is incomplete.** `train.py` calls `torch.manual_seed(args.seed)`
and nothing else. `numpy.random`, Python's `random`, CUDA/MPS generators and
`torch.use_deterministic_algorithms` are all unset. `DataLoader(shuffle=True)`
draws from the global torch generator, but augmentation, the eval subsampler and
`cist_score` each construct their own generators from unrelated seeds.

**R4 — `metrics.json` records no provenance.** No git commit, no config hash, no
library versions, no device model, no wall-clock. There is no way to attribute a
number to a code state.

**R5 — There is no validation split**, so any hyperparameter choice
(`lambda_pair=5.0`, `lambda_flow_service=1.5`, `lambda_flow_pair=3.0`,
`temperature=0.05` — all far from the defaults in the source) was made against
the test set or by hand.

---

## 5. Metric-definition findings

**M1 — The "context invariance" score (0.6409) has no null model.**
`cist_score` is the mean cosine between `z_app` of a flow and `z_app` of the same
flow after `augment_network_condition`, over the first 256 test records. Its
range is [-1, 1]. A **constant** encoder scores **1.0**; a random 256-dimensional
embedding scores ≈0. The metric is therefore *maximised by a degenerate model*
and says nothing on its own. It must be replaced by adversarial probing (Phase
3.2) or defined against those two reference points explicitly.

**M2 — `leave_one_app_out_generalization` does not measure generalization.**
It receives only `test_emb`, builds class prototypes from the test set, and
scores test rows against them. It is a within-test clustering statistic. It also
never re-encodes anything, so "leave one app out" leaves nothing out of training.

**M3 — `benchmark_latency` measures the wrong thing.** It times
`model.encode` on **random tensors**, batch size 1, excluding CSV parsing,
sequence parsing and feature construction, on whichever device `--device auto`
picked (MPS for the committed run). The 13.65 ms mean is a forward pass on an
Apple GPU, not an end-to-end pcap-to-decision latency.

**M4 — `embedding_similarity` is a Python double loop** over up to 10,000 test
rows (~50M iterations per call), which is why the eval caps exist.

**M5 — Every headline number is accuracy on an imbalanced set.** A
majority-class predictor scores 28.6% accuracy. The paper must report macro-F1,
balanced accuracy and per-class F1. *(fixed: `flowconx/metrics.py`.)*

---

## 6. Script dependency graph

```
                 (raw, not committed)
 5G_Traffic_Datasets/         cesnet-quic22/            *.pcap
        |                           |                      |
        v                           v                      v
 prepare_5g_traffic_        prepare_cesnet_quic22_   prepare_mawi_pcap.py
   dataset.py                  dataset.py                  |
        |                           |                      |
        +------------ manual concatenation ----------------+
                       (NO SCRIPT IN THE REPO)
                              |
                              v
        data/processed/flowconx_final_labeled_train.csv
                              |
        +---------------------+---------------------+
        |                                           |
        v                                           v
 flowconx/train.py                        flowconx/eval_cli.py
  (split + train + evaluate,               (re-splits with the same seed,
   all in one process)                      re-evaluates from a checkpoint)
        |
        v
 outputs/<run>/{metrics.json, history.json, flowconx_checkpoint.pt}
        |
        v
 scripts/kpi_report.py  ->  kpi_report.md
```

**The concatenation step has no script.** `flowconx_final_labeled_train.csv`
interleaves 5G Traffic and CESNET rows in a shuffled order that no committed code
produces, and the ordering destroys the last recoverable trace of capture
structure (the file has 104,789 `app` runs across 112,121 rows — it is fully
shuffled). This is the reason session-disjoint evaluation is impossible on the
committed artifact rather than merely inconvenient.

---

## 7. What Phase 0 changed

| Added | Purpose |
| --- | --- |
| `flowconx/schema.py` | One canonical Flow schema, with provenance columns separated from model inputs |
| `flowconx/metrics.py` | Macro-F1, balanced accuracy, per-class F1, bootstrap CIs, confusion matrices, top confusions |
| `flowconx/labels.py` | `drop` / `merge_into_parent` / `keep_and_report` rare-class policy, recorded in every result |
| `flowconx/audit/splits.py` | Six split protocols, committed manifests with SHA256, hard failure when a protocol is unavailable |
| `flowconx/audit/tabular.py` | Ten trivial and classical feature families (CUMUL, AppScanner, k-FP, …) |
| `flowconx/audit/baselines.py` | Baselines fitted on the identical manifest splits, with every deviation from the original papers recorded |
| `flowconx/audit/leakage.py` | Split-partition, cross-split duplicate/near-duplicate, identifier-as-input and nuisance-derivability probes |
| `flowconx/audit/run_audit.py` | One command producing `results/audit/**` and `splits/**` |
| `tests/test_leakage.py`, `tests/test_metrics.py` | 56 tests, run in CI |
| `scripts/audit_smoke.py` | End-to-end audit run in CI on a schema-complete generated table |
| Provenance emission in all three preparers | Makes session-disjoint, temporal and server-disjoint splits possible on regenerated data |

Results: see `results/audit/flowconx_final_labeled_train/audit_summary.md` and
§8 of `paper/RESULTS.md`.

---

## 8. Audit results

Source: `results/audit/flowconx_final_labeled_train/audit_summary.md`, generated by

```bash
python -m flowconx.audit.run_audit \
  --csv data/processed/flowconx_final_labeled_train.csv \
  --seed 42 --rare-class-mode drop
```

112,098 rows after dropping `xr_interactive`; 5 classes; splits 78,469 / 11,210 /
22,419 (train/val/test) under `random_flow`.

### 8.1 Trivial baselines — macro-F1

| Baseline | Feature count | `random_flow` | `app_disjoint` |
| --- | ---: | ---: | ---: |
| Majority class | 0 | 0.0889 | 0.0003 |
| Protocol number only, depth-4 tree | 1 | 0.1729 | 0.0645 |
| Nuisance `condition` only, depth-4 tree | 1 | 0.2064 | 0.1593 |
| CUMUL (SVM-RBF, ≤8k rows) | 104 | 0.4975 | 0.1797 |
| Five NetFlow statistics, random forest | 5 | **0.6414** | 0.1247 |
| All flow-level scalars, XGBoost | 14 | 0.7111 | 0.1435 |
| k-fingerprinting, random forest | 52 | 0.7739 | 0.2554 |
| First 10 signed packet sizes, XGBoost | 10 | 0.8028 | 0.2516 |
| Packet-size histogram, XGBoost | 24 | 0.8250 | 0.1694 |
| First 20 signed packet sizes, XGBoost | 20 | 0.8641 | **0.3109** |
| **AppScanner (2016), random forest** | 72 | **0.8690** | 0.2018 |
| *FlowCon-X, recorded* | — | *0.8842* † | not run |

† `outputs/flowconx_final_labeled_kpi_pass/metrics.json`, `classification.knn_macro_f1`.
**Not comparable and not reproducible.** It was computed on a different
partition (a stratified subsample of 20,000 train / 10,000 test embeddings,
`--eval-max-train` / `--eval-max-test`), under the six-class label space that
still contained `xr_interactive`, by a CLI that no longer exists, resumed from an
uncommitted checkpoint (§4, R1–R2). It is printed here only to show the scale of
the gap that has to be re-measured under the Phase 1 harness.

### 8.2 What this says

**A 2016 statistical baseline is within ~1.5 macro-F1 points of the recorded
FlowCon-X number, under the same split protocol.** AppScanner is eighteen
summary statistics of the packet-size series fed to a random forest. It has no
learned representation, no fusion, no adversarial head, and no contrastive
objective. On the field's usual random-flow protocol it reaches 0.8690 against
FlowCon-X's recorded 0.8842.

Applying the standing rule from the brief: **the dataset is the story, not the
model.** The evaluation protocol has to change before anything is written up.

Three further readings:

1. **Five NetFlow numbers get macro-F1 0.6414.** Packet count, byte count,
   duration, mean IAT and mean packet size — no packet content whatsoever —
   recover two thirds of the task. Most of what looks like encrypted-traffic
   fingerprinting here is flow-volume discrimination.

2. **Every baseline collapses under `app_disjoint`.** AppScanner falls from
   0.8690 to 0.2018, a drop of 0.667 macro-F1; the best surviving baseline
   manages 0.3109. On a random split these methods look solved; on unseen
   applications they are barely above the floor. That contrast — measured, on
   committed splits, with the same code — is itself a measurement contribution
   and is the strongest single result Phase 0 produced.

3. **The two protocols are not equally leaky, and the probes show it.** Under
   `random_flow`, 53 test rows are byte-identical to a training row and **21.3%
   of sampled test rows have a training row at cosine ≥ 0.999** in standardised
   AppScanner space (max observed cosine 1.0). Under `app_disjoint` the same
   probe reports 2.1%. The random split is measurably recalling near-copies.

### 8.3 Leakage verdicts

| Check | `random_flow` | `app_disjoint` |
| --- | --- | --- |
| Row-index partition | pass | pass |
| `flow_id` disjoint | vacuous ‡ | vacuous ‡ |
| `capture_id` / `five_tuple` / `client_ip` / `server_ip` / `origin` disjoint | **unavailable** | **unavailable** |
| No exact duplicates across splits | **FAIL** (53 rows, 0.24% of test) | pass |
| No near duplicates across splits (cos ≥ 0.999) | **FAIL** (21.3% of sampled test) | **FAIL** (2.1%) |
| Identifiers excluded from declared inputs | pass | pass |
| Nuisance label not derivable from inputs | **FAIL** (agreement **1.000**) | **FAIL** (agreement **1.000**) |

‡ The committed CSV has no `flow_id`, so the audit synthesizes one from row
content. It is unique by construction, which makes the check vacuous; the
exact-duplicate probe is the load-bearing one. The report says so explicitly
rather than banking a pass.

**Five of the seven identity probes could not run at all** because the committed
CSV retains no provenance. That is the finding, not a limitation of the tool: a
reviewer cannot verify that flows from one capture session stayed on one side,
because the artifact does not record which session a flow came from.

**`condition` is reconstructed from `flow iat mean` and `flow iat std` with
agreement 1.000** — not 0.99, exactly 1.0 on all 112,098 rows. The declared
nuisance variable is a threshold function of two model inputs. Any claim that the
adversarial head removes "network condition" while preserving task information
is, as currently constructed, a claim about removing a quantisation of the flow's
own timing statistics.

### 8.4 What has to change before the paper

1. **Regenerate the dataset with provenance** (the preparers now emit it) and
   move the headline table to `session_disjoint` and `temporal`. Keep
   `random_flow` as a clearly-labelled contrast column — the inflation it
   produces is a result worth reporting.
2. **Report the trivial baselines in the paper regardless of outcome.** A paper
   that shows AppScanner at 0.8690 and explains why its own protocol is harder is
   far more credible than one that omits it.
3. **Re-measure FlowCon-X on the committed manifest splits** under the Phase 1
   harness, with 5+ seeds, before any comparison is stated.
4. **Restate or drop the invariance claim.** Either define `condition` from
   something that is not a function of the model's inputs, or reframe the
   adversarial head as removing a specific known confound and probe it properly
   (Phase 3.2).
