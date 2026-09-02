# CLAIMS

Every claim we intend to make in the abstract and introduction, each mapped to
the table, figure and significance test that supports it.

**A claim with no mapped result is deleted before submission.** This file is
the checklist that enforces that, and it is updated as results land, not
afterwards.

Status legend: **supported** (result exists and the test passed) ·
**pending** (the run is planned but has not produced a number) ·
**withdrawn** (the result contradicted it).

---

## C1 — The split axis that matters is corpus-specific, and controlling the wrong one protects nothing

> Encrypted-traffic results are inflated by splitting on the wrong axis, and
> which axis matters depends on how the corpus was collected. On single-session
> captures it is the capture; on backbone traffic it is the server. Controlling
> session-disjointness on backbone traffic, or server-disjointness on
> single-session captures, provides no protection at all.

*(Superseded the original wording, "random flow splitting inflates results".
Our own second dataset contradicts that as a general statement: on
CESNET-QUIC22 random, session-disjoint and temporal splitting are
indistinguishable.)*

- **Evidence:** `paper/tables/split_contrast.tex`, from
  `results/audit/*/audit_summary.json` and `results/flowconx_*/`
- **Test:** paired comparison across seeds, `results/significance.json`
- **Status:** **supported on 5G Traffic, not supported on CESNET-QUIC22.**
  On 5G, server IP alone reaches 0.999 under a random split and capture ID
  alone 0.983, and AppScanner falls 0.996 → 0.640 under session-disjoint.
  On CESNET, all three protocols agree to within one seed standard deviation
  for both the model (0.781 / 0.786 / 0.781) and the baselines
  (0.768 / 0.772 / 0.762).
- **Full evidence.** CESNET-QUIC22, three seeds, k-NN, versus session-disjoint:
  random −0.002 (d = 0.23), temporal −0.001 (d = 0.12), **server-disjoint
  −0.191 (d = 19.3)**. 5G Traffic: capture ID alone predicts the label at
  macro-F1 0.983 and AppScanner falls 0.996 → 0.640 from random to
  session-disjoint. The audit's identifier probes are the diagnostic that tells
  you which axis to control: `server_ip_only` scores 0.772 under a random split
  and 0.026 under server-disjoint.
- **Why this version is better than the original.** It is more actionable — it
  tells a reader which grouping to check on their own data rather than issuing
  a blanket warning — and it is the version our evidence actually supports.
- **Open:** three seeds cannot produce a p-value. The server-disjoint contrast
  is the one comparison worth six seeds, about an hour of compute, purely so
  the test reports a number rather than `undetermined`.

## C2 — FlowCon-X is competitive under strict protocols, not state of the art

> We do not claim to beat pre-trained byte-level transformers on curated
> benchmarks; those are saturated above 98% and a fractional win is not a
> contribution.

- **Evidence:** `paper/tables/main_comparison_*.tex`
- **Test:** Wilcoxon across seeds against every baseline we ran, Holm-Bonferroni
  corrected within the `against_baselines` family
- **Status:** **supported, and smaller than hoped.** CESNET-QUIC22,
  session-disjoint, 3 seeds: macro-F1 0.7889 ± 0.0046 (linear head) against
  AppScanner at 0.7718 and XGBoost on the first 20 packet sizes at 0.7897 —
  the latter is *within one seed standard deviation*. The closed-set gain is
  about 1.5 points over a 2016 baseline and roughly nil over a well-tuned
  gradient-boosted tree.
- **Not supported on 5G Traffic.** Session-disjoint, seed 0: FlowCon-X reaches
  0.5953 (k-NN) against 0.8494 for XGBoost on thirteen flow scalars and 0.6399
  for AppScanner. A 0.25 macro-F1 loss, driven by two classes collapsing where
  the split is accidentally app-disjoint (`metaverse` F1 = 0.008, its test set
  being almost entirely an application absent from training).
- **Consequence for the paper:** the closed-set table cannot be the
  contribution. It establishes competitiveness on one dataset and a clear loss
  on the other, and the introduction must not imply otherwise. C4–C8 carry the
  argument or there isn't one.
- **Constraint:** ET-BERT, YaTC, CLE-TFE, MIETT, FlowletFormer and PacketCLIP
  were **not run** — all six tokenise raw payload bytes, which our QUIC/TLS
  records do not retain. See `flowconx/baselines/WHY_NOT_RUN.md`. The claim is
  worded to be true given that gap.

## C3 — Identifier shortcuts do not explain the task

> Destination port, SNI, server address and server AS, each used alone, fall
> far short of the behavioural models, so the benchmark is measuring traffic
> analysis rather than name resolution.

- **Evidence:** `paper/tables/main_comparison_*.tex`, "Identifier shortcuts" tier
- **Test:** none needed; the comparison is against the majority-class floor
- **Status:** **pending**
- **Risk if it fails:** if SNI alone approaches the model, no result in the
  paper means anything and we have to change datasets. This is checked first.

## C4 — A metric-trained deployed embedding rejects unknown applications better than a softmax classifier

> Held-out applications are rejected by distance to the nearest prototype with
> higher AUROC and lower FPR@95TPR than maximum-softmax-probability,
> energy-based or Mahalanobis rejection on the same embedding.

- **Evidence:** `results/flowconx_open_set/`, OSCR curves
- **Test:** paired across seeds
- **Status:** **pending**
- **Note:** this is a claim ET-BERT-class models cannot easily make, since a
  fine-tuned softmax head has one logit per training class and no "none of
  these". Stated as a structural argument, not as a measured win over them.

## C5 — New applications enroll from a handful of labelled flows, without retraining

> With the encoder frozen, a new application reaches usable accuracy from
> k = 5–25 labelled flows and a prototype update, at a cost of k forward passes
> against a fine-tuning run for every baseline.

- **Evidence:** `paper/figures/enrollment_curve.pdf`, `paper/tables/cost.tex`
- **Test:** enrollment curves with spread over repeated draws
- **Status:** **pending, and now load-bearing.**
- **Both halves are required.** The accuracy curve alone is not the claim; the
  cost asymmetry has to be measured in seconds and GPU-hours, or the claim is
  rhetorical.
- **Decision rule, written down before the result arrives.** The 5G loss shows
  the embedding does not transfer zero-shot to an unseen application, which is
  consistent with a design that expects enrollment. **If enrollment at
  k = 5–25 labelled flows does not close that 0.25 macro-F1 gap, the
  architecture has no case on this dataset and the paper says so** rather than
  reframing around whatever else happens to look good.

## C6 — Performance degrades gracefully over time and is cheaply restored

> Trained on the earliest week of CESNET-QUIC22 and evaluated week by week,
> accuracy declines; refreshing prototypes from k labelled flows recovers most
> of the loss without touching the encoder.

- **Evidence:** `results/flowconx_temporal/cesnet_quic22/temporal/seed*/metrics.json`, `drift` block
- **Status:** **withdrawn — untestable on this data.** Over the six held-out
  days, macro-F1 *rises* from 0.7678 to 0.7910 (total change +0.0232). A
  four-week corpus with a six-day held-out tail cannot exhibit drift, and the
  re-enrollment remedy is equally untestable because there is nothing to
  restore. Reporting a flat curve as drift resistance would be read as a claim
  about deployment lifetime, which this evidence cannot support. Reinstate
  only with a corpus spanning months.

## C7 — The model survives padding and quantisation defences better than the alternatives, at a stated overhead

> Accuracy degradation is reported against the byte and packet overhead each
> countermeasure imposes. A defence that halves accuracy by tripling bandwidth
> is not the same result as one that does it for free.

- **Evidence:** `paper/figures/robustness_overhead.pdf`
- **Status:** **supported (5G, seed 0).** Network conditions barely register:
  25 ms jitter costs 0.025 macro-F1 and 5% packet loss 0.011. Among
  countermeasures, only those that destroy packet-size information work, and
  they cost 2.28x bandwidth: constant-rate padding −0.344, MTU padding −0.298.
  The cheap defences do essentially nothing — random padding −0.0004 at 1.15x
  overhead, dummy injection −0.010 at 1.17x, 128-byte quantisation −0.011 at
  1.08x. Stated as a defender's result, that is the useful direction: a
  defender deploying the cheap options is paying for nothing.

## C8 — Decisions are made from few packets, fast enough to sit inline

> Accuracy at packets 1, 2, 3, 5, 10 and 20, and end-to-end p50/p95/p99 latency
> including feature construction.

- **Evidence:** `paper/figures/early_classification.pdf`, `paper/tables/cost.tex`
- **Status:** **pending**
- **Correction to the previous draft:** the earlier "sub-15 ms" figure was a
  forward pass on random tensors at batch size 1 on an Apple GPU, excluding all
  feature extraction (`AUDIT.md` §5, M3). It must not be reused.

## C9 — Adversarial removal reduces measurable nuisance leakage without destroying the representation

> An MLP probe recovers capture session / week from `z_flow` at close to the
> majority-class floor, while downstream task macro-F1 is retained.

- **Evidence:** `paper/figures/adversarial_tradeoff.pdf`, `probes` block
- **Status:** **pending**
- **Scope changed since the previous draft.** The nuisance variable used to be
  `condition`, which the audit reconstructed from two model input features with
  agreement 1.000 — removing it was circular (`AUDIT.md` §3, L5). It is now
  drawn from provenance the model never sees. **The claim now refers to a
  different quantity than the earlier draft's "context invariance", and the
  0.6409 CIST score must not be carried over.**

---

## Claims we are NOT making

Recorded so they do not creep back in during writing:

- Not: state of the art on ISCX-VPN, USTC-TFC, or any saturated benchmark.
- Not: beating ET-BERT. We did not run it and say so.
- Not: "context invariance" as previously defined. The metric was maximised by
  a constant encoder.
- Not: any claim about packet loss robustness derived from the v1 corpus, where
  `loss_rate` was identically zero on all 112,121 rows.
