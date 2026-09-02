# Changelog

Every change that could move a reported number gets an entry here, dated, with
what changed and why. Results generated before a change are marked stale rather
than deleted.

## [Unreleased]

### 2026-09-02 — Phase 1-5: rebuilt data pipeline, harness, and evaluation

**Data.** Both corpora rebuilt from the raw archives, streamed from the zips
without expanding them. 5G Traffic: 164,348 conversation segments from 330M
packet rows across all 75 captures. CESNET-QUIC22: 201,600 flows sampled from
153M rows across all 28 days and four weeks. Both carry full provenance, so
five of six split protocols are available where two were before.

**The two datasets are no longer merged.** The merge was itself a shortcut:
`protocol == 0` was 99.9% one class because it recorded which preparer had
run. `protocol` is now provenance rather than a model input.

**Impact on results: total.** No number from the previous corpus carries over.
`AUDIT.md` §9 is the audit of the new one.

**Changes that alter what the paper can claim** (per standing rule 3):

1. *The nuisance variable changed.* It was `condition`, a threshold on two of
   the model's own input features, reconstructed with agreement 1.000. It is
   now drawn from provenance the model never sees (capture session, week,
   server AS). The invariance claim now refers to a different quantity and the
   0.6409 CIST score does not carry over. Recorded in `paper/CLAIMS.md` C9 and
   `paper/THREATS.md` §6.
2. *Labels come from each dataset's own taxonomy.* 5G Traffic uses its
   directory structure; CESNET uses its `CATEGORY` field. The hand-written
   mapping with its substring fallback is gone, along with the mislabels it
   produced (`csgo_market` and `cloudflare_cdnjs` both matched "ar" and were
   labelled `xr_interactive`).
3. *A row is a conversation segment, not a 10-second window over a whole
   capture.* The unit of observation is now the same across both datasets.

**Defects found and fixed while building** (each changed results, not only
speed):

- `byte_progress` and `time_progress` normalised by the observation window's
  total, so the value at packet 1 depended on packets not yet seen. Harmless
  for full-flow scoring, but it inflated the early-classification curve, which
  is one of the paper's claims. Both channels are causal now; a test asserts
  that truncating a sequence equals observing fewer packets.
- The memory bank's `sample()` broke at `max_total` while looping over
  classes, so the contrastive loss saw only whichever 4 of 18 classes came
  first in dictionary order. First-epoch validation macro-F1 moved 0.164 →
  0.513 when fixed.
- The `defaults:` chain was resolved one level deep, so every ablation
  silently fell back to dataclass defaults for its grandparent's keys and
  would have run on a different dataset and split than its own reference row.
- The audit reported a degenerate number for a probe whose column was absent
  or constant, which reads as "this identifier does not help" when it means
  "this dataset has no such field". It now reports `unavailable` with the
  reason.

**Harness.** Single entry point (`python -m flowconx.run`), 45 YAML configs
including 35 ablations sharing one documented reduced budget, full determinism
and provenance recording, Makefile, Dockerfile, 100 tests.

**Evaluation.** Open-set rejection with four scoring rules on one embedding,
few-shot enrollment curves, temporal drift with prototype re-enrollment,
eleven condition and countermeasure perturbations reported against the
overhead each imposes, early-classification curves, end-to-end latency
percentiles, and adversarial probing to replace the CIST score.

### 2026-08-28 — Phase 0: audit, split protocols, leakage tests

Standing-rule compliance: no number in this repository is hand-entered. Every
value in `results/` is written by `python -m flowconx.audit.run_audit`.

**Added**

- `flowconx/schema.py` — the canonical Flow schema, with provenance columns
  (`flow_id`, `origin`, `capture_id`, `flow_start_ts`, `server_ip`) declared
  separately from `MODEL_INPUT_COLUMNS`. Nothing in the provenance group may
  reach a feature extractor.
- `flowconx/metrics.py` — macro-F1, balanced accuracy, per-class F1, per-class
  support, confusion matrices, top-confusion lists and percentile bootstrap 95%
  CIs. Reason: a majority-class predictor scores 28.6% accuracy on this dataset,
  so accuracy alone is not informative (AUDIT.md §5, M5).
- `flowconx/labels.py` — explicit rare-class policy with three modes
  (`drop`, `merge_into_parent`, `keep_and_report`) and a recorded decision in
  every result. Default `drop` at `min_class_count=100`, which removes
  `xr_interactive` (23 rows) from the headline table. Reason: AUDIT.md §3, L12.
- `flowconx/audit/splits.py` — six split protocols (`random_flow`,
  `session_disjoint`, `temporal`, `server_disjoint`, `app_disjoint`,
  `origin_disjoint`), committed manifests carrying per-side flow-ID lists and
  SHA256 checksums, and a hard `SplitUnavailable` when a protocol's grouping
  column is missing. Reason: the previous `split_records` was a stratified
  random split over flows with no grouping and no validation side (AUDIT.md §2).
- `flowconx/audit/tabular.py` — ten trivial and classical feature families,
  including CUMUL (Panchenko et al., NDSS 2016), AppScanner (Taylor et al.,
  EuroS&P 2016) and k-fingerprinting (Hayes and Danezis, USENIX Security 2016).
- `flowconx/audit/baselines.py` — those families fitted on the identical
  manifest-defined splits, with every deviation from the originating paper
  recorded in the result JSON rather than in a footnote nobody reads.
- `flowconx/audit/leakage.py` — split-partition, per-identifier disjointness,
  exact-duplicate, standardised-cosine near-duplicate, identifier-as-input and
  nuisance-derivability probes. Probes whose precondition is missing return
  `unavailable`, never a silent pass.
- `flowconx/audit/run_audit.py` — single entry point writing
  `results/audit/<dataset>/<protocol>/{baselines,leakage}.json`, an
  `audit_summary.json` carrying git commit and library versions, and a generated
  `audit_summary.md`.
- `tests/test_leakage.py`, `tests/test_metrics.py` — 56 tests, run in CI.
- `scripts/audit_smoke.py` — end-to-end audit over a generated schema-complete
  table; asserts every strict protocol is reachable, that `session_disjoint`
  keeps capture sessions apart, and that `random_flow` demonstrably does not.

**Changed**

- `scripts/prepare_5g_traffic_dataset.py`, `scripts/prepare_cesnet_quic22_dataset.py`,
  `scripts/prepare_mawi_pcap.py` now import `CANONICAL_COLUMNS` from
  `flowconx.schema` and emit the provenance columns. 5G Traffic groups on the
  source capture file; CESNET on the source day-file and `TIME_FIRST`; MAWI on
  the pcap name plus a low-port server heuristic.
  **Impact: no change to any model number.** These columns are not model inputs.
  They make session-disjoint, temporal and server-disjoint evaluation possible
  on regenerated data. The committed CSV predates them and must be regenerated
  before the headline table can move off `random_flow`.
- `.github/workflows/ci.yml` — added a `tests` job running the leakage suite and
  an `audit-smoke` job running the full audit pipeline end to end.

**Result**

First audit run, `results/audit/flowconx_final_labeled_train/` at commit
`090b4b9`: AppScanner (Taylor et al., 2016) reaches macro-F1 **0.8690** under
`random_flow`, against FlowCon-X's recorded **0.8842**; five NetFlow scalars
reach **0.6414**. Under `app_disjoint` every baseline collapses (AppScanner
0.8690 -> **0.2018**). Four of six split protocols are unavailable on the
committed CSV. Full reading in `AUDIT.md` §8. **No model change was made in
response to these numbers.**

**Known stale**

- `outputs/flowconx_final_labeled_kpi_pass/metrics.json` is **not reproducible**
  at any commit in this repository. It was produced by a CLI carrying
  `--flows-per-app` and `--synthetic`, neither of which exists in
  `flowconx/train.py`, and it resumed from
  `outputs/flowconx_final_labeled_flow_tuned/flowconx_checkpoint.pt`, which is
  not committed (AUDIT.md §4, R1–R2). Its numbers are retained for the record
  and must not appear in the paper until regenerated under the Phase 1 harness.
