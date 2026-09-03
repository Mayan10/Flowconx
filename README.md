# FlowCon-X

**Encrypted traffic classification under realistic protocols.**

A flow-level metric-trained embedding for application and service
classification from packet metadata alone — no deep packet inspection, no
payload bytes, no SNI at inference — together with the audit machinery needed
to tell whether such a result means anything.

This repository is the artifact behind a paper in preparation. It is built so
that a reviewer can download it, run it, and try to break the claims.

---

## The short version

We started by trying to break our own result, and succeeded.

On 5G Traffic, the same model and the same code score **macro-F1 0.992 under
the split protocol the encrypted-traffic literature standardly uses, and 0.574
under a session-disjoint one.** Nothing changes but the partition. Written up,
0.992 across six service classes would read as a strong contribution; the audit
says precisely why it is not, because under that protocol the **server IP alone
scores 0.999** and the **capture-session ID alone scores 0.983**.

On CESNET-QUIC22 the same experiment finds nothing — random, session-disjoint
and temporal splitting agree to within one seed's noise — but **server-disjoint
splitting costs 0.19 macro-F1**. The axis that carries the label differs by
corpus, and controlling the wrong one protects nothing.

That is not a result about a model. It is a result about a protocol, and it is
the reason this repository leads with `AUDIT.md` rather than with a headline
number.

Three things follow, and they shape everything here:

1. **Every reported number sits next to a trivial baseline.** Destination port
   alone, SNI alone, server IP alone, capture-session ID alone. If one of them
   approaches the model, the benchmark is not measuring traffic analysis, and
   the audit says so before the paper does.
2. **The random flow split is a contrast column, never a headline.** Headline
   numbers use session-disjoint, temporal and server-disjoint protocols.
3. **Nothing is reported that a script did not produce.** Table cells with no
   run behind them read `TODO`. `scripts/make_paper_assets.py` counts them.
4. **Every model number here is a lower bound.** 13 of 16 runs stopped with
   validation accuracy still rising — the 15-epoch budget is a compute
   decision, not a modelling one. See `paper/THREATS.md` §8.

---

## Claims

| # | Claim | Evidence | Status |
| --- | --- | --- | --- |
| C1 | The split axis that matters is corpus-specific; controlling the wrong one protects nothing | `paper/tables/split_contrast.tex` | supported (5G: capture axis; CESNET: server axis, −0.19 macro-F1) |
| C2 | Competitive under strict protocols — **not** state of the art | `paper/tables/main_comparison_*.tex` | beats all deep baselines; loses to XGBoost on both corpora (0.783 vs 0.790; 0.547 vs 0.849) |
| C3 | Identifier shortcuts do not explain the task | main comparison, shortcut tier | **not supported** — SNI alone scores 0.967 on CESNET, above every model |
| C4 | Metric-trained embedding rejects unknown applications better than softmax | `results/flowconx_open_set/` | supported on FPR@95TPR (0.34 vs 0.61); AUROC gap within seed spread |
| C5 | New applications enroll from a handful of flows, no retraining | `paper/figures/enrollment_curve.pdf` | **not supported on 5G** — curve flat, failed its pre-registered rule |
| — | *Reframing:* the model is an application fingerprinter, not a service classifier | `results/flowconx_app_task/` | internal comparison holds (0.701 vs 0.547); baselines on the app task not yet run |
| C6 | Degrades gracefully over time; cheaply restored by re-enrollment | `drift` block | **withdrawn** — 4-week corpus cannot exhibit drift |
| C7 | Survives padding defences at a stated overhead | `paper/figures/robustness_overhead.pdf` | supported (only 2.28× -overhead defences bite) |
| C8 | Decides from few packets, fast enough to sit inline | `paper/tables/cost.tex` | partially supported (p50 6.1 ms end-to-end, 7.8k flows/s) |
| C9 | Adversarial removal reduces nuisance leakage without destroying the representation | `probes` block | **not supported** — leak equals the raw-feature control |

Full statements, the test behind each, and the claims we explicitly do **not**
make are in [`paper/CLAIMS.md`](paper/CLAIMS.md). Limitations are in
[`paper/THREATS.md`](paper/THREATS.md), including the ethics statement.

**We do not claim to beat ET-BERT.** We could not run it: it tokenises raw
payload bytes, which QUIC/TLS does not expose and our records do not retain.
[`flowconx/baselines/WHY_NOT_RUN.md`](flowconx/baselines/WHY_NOT_RUN.md) gives
the reason for each of the six pre-trained models in that position.

---

## Quick start

```bash
git clone <repository-url> && cd flowconx
make setup                    # install the package and dev dependencies
make test                     # lint, unit tests, leakage tests (~10 s, no data needed)

./scripts/download_data.sh    # licences, DOIs, checksums; ~24 GB
make data                     # build both datasets from the archives (~30 min)
make audit                    # shortcut and leakage audit (~15 min)
make repro-small              # reduced end-to-end pipeline (<30 min on one GPU)
```

`make test` needs no data and is the fastest way to confirm the install.

Docker, if you prefer:

```bash
docker build -t flowconx .    # the leakage suite runs during the build
docker run --rm -v "$PWD/data:/work/data" -v "$PWD/results:/work/results" \
  flowconx make repro-small
```

### Hardware and time

| Step | Time | Hardware | Disk |
| --- | ---: | --- | ---: |
| `make test` | 10 s | any | — |
| `make data` | ~30 min | 8 cores; no GPU | 24 GB raw (read-only), 420 MB output |
| `make audit` | ~15 min | 8 cores; no GPU | 5 MB |
| One training run | ~10 min | 1 GPU (or MPS) | 2 MB |
| `make repro-small` | ~30 min | 1 GPU | 10 MB |
| `make repro-full` | ~20 h | 1 GPU | 200 MB |

**The archives are never expanded.** They are 3.2 GB and 21 GB; every loader
streams members out of the zip in place. 24 GB of free disk is enough to run
everything.

---

## Reproducing each table and figure

Each row is self-contained: run the command, get the asset.

| Asset | Command |
| --- | --- |
| Table 1 — split protocol contrast | `make data && make audit && python scripts/run_all_experiments.py --stages headline split_contrast && make paper` |
| Table 2 — main comparison | `make audit && python -m flowconx.baselines.run_baselines --config configs/cesnet_main.yaml --seed 0 && make paper` |
| Table 3 — ablations | `python scripts/run_all_experiments.py --stages ablations --seeds 0 1 2 3 4 && python -m flowconx.analysis.significance && make paper` |
| Table 4 — deployment cost | `python -m flowconx.run --config configs/cesnet_main.yaml --seed 0 && make paper` |
| Fig. 1 — early classification | `python -m flowconx.run --config configs/cesnet_main.yaml --seed 0 --set eval.early_classification=true` |
| Fig. 2 — enrollment curve | `python -m flowconx.run --config configs/fiveg_open_set.yaml --seed 0` |
| Fig. 3 — robustness vs overhead | `python -m flowconx.run --config configs/fiveg_main.yaml --seed 0 --set eval.robustness=true` |
| Fig. 4 — adversarial trade-off | `python scripts/run_all_experiments.py --stages ablations --seeds 0 1 2` (the `adv_weight_*` family) |
| Audit tables (`AUDIT.md` §8) | `make audit` |

Every command writes JSON into `results/`; `make paper` renders `results/`
into `paper/tables/*.tex` and `paper/figures/*.pdf`. **Tables are never edited
by hand.**

---

## Repository layout

```
flowconx/
  data/          loaders (one module per source), canonical schema, zip streaming
  features/      deterministic feature extraction, unit tested
  models/        encoders, fusion, heads, memory and prototype banks
  losses/        contrastive, margin, adversarial — each independently toggleable
  train/         the training loop
  eval/          closed-set, open-set, few-shot, drift, robustness, early, cost, probes
  audit/         split protocols, trivial baselines, leakage probes
  baselines/     deep non-pretrained baselines, and WHY_NOT_RUN.md
  analysis/      significance testing and seed aggregation
  run.py         the single entry point
configs/         every experiment is a YAML file; ablations/ holds 35 of them
scripts/         download_data.sh, run_all_experiments.py, make_paper_assets.py, anonymize_repo.sh
splits/          committed split manifests: flow-ID lists plus SHA256 per side
results/         metrics.json per run, committed (small JSON only)
paper/           auto-generated tables and figures, plus CLAIMS/THREATS/VENUE/RESULTS
tests/           97 tests, including the leakage suite that gates CI
```

### One entry point

```bash
python -m flowconx.run --config configs/cesnet_main.yaml --seed 0
python -m flowconx.run --config configs/cesnet_main.yaml --seed 0 --set model.fusion=concat
python -m flowconx.run --config configs/cesnet_main.yaml --dry-run   # resolve and print
```

Nothing important is reachable only from a notebook. Every ablation is a
config override, so an ablation table row is a file rather than a code branch.
A run refuses to overwrite an existing `metrics.json` unless `--overwrite` is
given, so a sweep cannot silently replace a number a table already cites.

### Determinism

Each run seeds Python, NumPy, Torch and CUDA, requests deterministic
algorithms, and records into `metrics.json`: the seed, the config hash, the git
commit **and whether the tree was dirty**, library versions, the device, and
wall-clock time. `tests/test_determinism.py` asserts that the same config and
seed give identical metrics — and that different seeds do not, since otherwise
every error bar in the paper would be fictitious.

---

## Datasets

Both are public. Neither is redistributed here; `scripts/download_data.sh`
carries the licences, DOIs and checksums.

| | 5G Traffic | CESNET-QUIC22 |
| --- | --- | --- |
| Source | Kaggle, CC BY 4.0 | Zenodo `10.5281/zenodo.7409648`, CC BY 4.0 |
| Raw | 75 packet captures, 44 GB | 28 daily flow files, 4 weeks |
| Read | 330 M packet rows | 153 M flow rows |
| Flows kept | 164,348 | 201,600 |
| Classes | 6 service categories | 18 service categories |
| Applications | 15 | 104 |
| Timeline | May–Oct 2022 | 2022-10-31 → 2022-11-27 |
| Provenance | capture, server IP, port, timestamp | capture day, client/server IP, port, SNI, AS, timestamp |

**They are kept separate and never concatenated.** An earlier version merged
them, and the merge itself became the shortcut: `protocol == 0` was 99.9% one
class, because that value recorded which preparer had run rather than anything
about the traffic (`AUDIT.md` §3, L3).

A row is one **conversation segment**: packets between one client and one
server over one transport, bounded by an idle and an active timeout. The
provenance columns — addresses, ports, SNI, capture ID — are retained
deliberately, so the audit can measure how much each explains on its own, and
are excluded from `MODEL_INPUT_COLUMNS`. `tests/test_leakage.py` asserts the
exclusion mechanically rather than trusting it.

---

## What the audit checks

`make audit` runs, for every split protocol the data supports:

- **Identifier probes** — port only, SNI only, server IP and /24 prefix only,
  server AS only, capture ID only, transport protocol only.
- **Classical baselines** — 5 flow statistics (RF), packet-size histogram
  (XGBoost), CUMUL, AppScanner, k-fingerprinting, all on the identical splits.
- **Leakage probes** — split partition, per-identifier disjointness, exact and
  near-duplicate rows across splits, identifiers-as-inputs, and whether the
  declared nuisance variable is derivable from the model's own inputs.

A probe whose precondition is missing reports `unavailable` with the reason.
It never quietly passes, and it never reports a degenerate number as if it
were a finding: a dataset with no SNI column reports `n/a` for the SNI probe,
not "SNI does not help".

`AUDIT.md` is the full Phase 0 audit of the previous pipeline, including the
defects this version fixes.

---

## Citing the data

```bibtex
@article{luxemburk2023cesnet,
  title   = {{CESNET-QUIC22}: A large one-month {QUIC} network traffic dataset
             from backbone lines},
  author  = {Luxemburk, Jan and Hyn{\v{c}}ica, Karel and {\v{S}}ediv{\'y}, Jakub},
  journal = {Data in Brief},
  year    = {2023},
  doi     = {10.1016/j.dib.2023.108888}
}
```

The 5G Traffic Datasets are distributed on Kaggle under CC BY 4.0; see
`scripts/download_data.sh` for the attribution the licence requires.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
