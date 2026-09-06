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
- **CERTIFIED at eight seeds.** CESNET, five protocols, Wilcoxon against
  session-disjoint with Holm-Bonferroni across the family:

  | Protocol | Macro-F1 | d | p |
  | --- | --- | --- | --- |
  | Session-disjoint (ref.) | 0.7798 ± 0.0061 | — | — |
  | Random flow | 0.7805 ± 0.0057 | −0.12 | 0.742 |
  | Temporal | 0.7811 ± 0.0051 | −0.31 | 0.547 |
  | Client-disjoint | 0.7791 ± 0.0055 | 0.13 | 0.945 |
  | **Server-disjoint** | **0.5881 ± 0.0068** | **17.03** | **0.0078 ✓** |

  Four axes null, one decisive, and the positive result **survives the
  correction** — the first comparison in this project to do so. Six seeds could
  not have certified it at any effect size; eight clears the 0.0125 threshold
  by one step.

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
- **Not supported on 5G Traffic, and the reason is not this architecture.**
  Session-disjoint: FlowCon-X 0.574, DeepPacket-style CNN 0.575, FS-Net 0.585,
  bi-LSTM 0.614, flow-statistics MLP 0.568 — every neural model in a 0.05 band,
  with the plain bi-LSTM ahead of ours. XGBoost on thirteen flow scalars
  reaches 0.849 on the same split. The MLP result is the sharpest: 0.568 from
  the ten flow scalars alone, against 0.849 for a tree on the same kind of
  input. A 0.28 gap attributable purely to model class.
- **So the 5G finding is about neural traffic classification, not about us.**
  Four independent architectures fail by the same margin. That is a fairer and
  more interesting thing to report than a single model underperforming.
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

- **Evidence:** `results/flowconx_open_set/fiveg_traffic/session_disjoint/seed0/metrics.json`
- **Test:** paired across seeds (2 more seeds queued)
- **Status:** **supported, narrowed.** 5G Traffic, three held-out applications
  spanning three service classes, 7,509 unknown against 25,350 known test rows.
  All four rules score the same frozen embedding:

  Three seeds:

  | Rule | AUROC | FPR@95TPR |
  | --- | --- | --- |
  | prototype cosine | 0.9189 ± 0.0327 | 0.3410 ± 0.1074 |
  | softmax MSP | 0.8372 ± 0.0853 | 0.6102 ± 0.1360 |
  | energy | 0.8699 ± 0.0806 | 0.3747 ± 0.1377 |
  | Mahalanobis | **0.9271 ± 0.0304** | **0.3162 ± 0.1329** |

  **Report FPR@95TPR, not AUROC.** The AUROC gap (0.082) is within the pooled
  seed spread; the FPR@95TPR gap (0.27) is about twice it. The claim is that
  **at the same true-positive rate, prototype rejection accepts roughly half as
  many unknown applications as softmax thresholding.**
- **Narrowing, because a reviewer will find it:** Mahalanobis on the *same*
  embedding is marginally better. The defensible claim is therefore *"a
  metric-trained embedding supports distance-based rejection, and
  distance-based rejection beats softmax thresholding"* — not "our rejector is
  best". The comparison among distance rules is about scoring functions, not
  architectures.
- **Note:** the structural argument still holds — a fine-tuned softmax head has
  one logit per training class and no "none of these" — but it is an argument,
  not a measurement against ET-BERT, which we did not run.

## C5 — New applications enroll from a handful of labelled flows, without retraining

> With the encoder frozen, a new application reaches usable accuracy from
> k = 5–25 labelled flows and a prototype update, at a cost of k forward passes
> against a fine-tuning run for every baseline.

- **Evidence:** `paper/figures/enrollment_curve.pdf`, `paper/tables/cost.tex`
- **Test:** enrollment curves with spread over repeated draws
- **Status:** **MIS-SPECIFIED, not false. Withdrawn in its original form;
  supported in a narrower one.**
- **The claim was about the wrong setting.** Within a corpus the enrollment
  curve is flat (+0.003 from k=1 to k=100) because the prototypes are already
  right — there is nothing for extra examples to fix. Across corpora, where
  they are genuinely wrong, **five labelled flows per class recover 0.676 of a
  0.246 zero-shot deficit on CESNET → 5G, which is 91% of the value reached at
  a hundred**, with the encoder frozen.
- **Defensible form:** *prototype re-enrollment cheaply adapts a frozen encoder
  to a new network*, not *to a new class on a network it already knows*.
- Caveats that must travel with it: three shared classes, one seed, an explicit
  and arguable taxonomy mapping, and no baseline run in this setting.
- **The original test and its verdict stand below**, because the rule was
  pre-registered and it failed:
- **Result (three seeds).** The enrollment curve is flat. Service macro-F1
  moves from 0.5020 ± 0.0697 at k = 1 to 0.5054 ± 0.0698 at k = 100 — +0.003,
  against a seed spread of ± 0.07. Application enrollment moves +0.005. At
  k = 25 the service figure is 0.5056, against 0.8494 for XGBoost on thirteen
  flow scalars. The 0.25 gap is not closed and not narrowed.
- **The rule, as recorded in advance:** *"If enrollment at k = 5–25 labelled
  flows does not close that 0.25 macro-F1 gap, the architecture has no case on
  this dataset and the paper says so."* It does not. **C5 is withdrawn on 5G
  Traffic.**
- **What the flat curve means.** Not a mechanical failure: the prototype is
  already converged at k = 1, which says the embedding places same-class flows
  tightly. It is *where* it places them. Cheap enrollment of a weak classifier
  is not a contribution.
- **The diagnosis, from the same run.** Identical embedding, identical k,
  identical procedure: **0.7572 ± 0.0182 identifying the application at k = 1,
  0.5020 ± 0.0697 for the service category.** The application figure also has
  a quarter of the variance, which is what one expects when a model is scored
  on what it actually represents. The model is an application fingerprinter
  scored against a service taxonomy it does not encode. On CESNET that
  mismatch is small because service and application correlate; on 5G Traffic,
  where `metaverse` spans Zepeto and Roblox, it is fatal.
- **The reframing was tested and the internal comparison holds.** A full
  supervised run on the application label, identical split and budget, reaches
  **0.701 macro-F1 and 0.761 balanced accuracy** over the eight applications it
  could have learned, against **0.547 ± 0.046 and 0.643** for the service
  taxonomy. Three applications exceed 0.89. Two independent routes — frozen
  prototype enrollment and a full supervised run — agree.
- **It does not yet rescue anything.** Nobody has run the baselines on the
  application task. XGBoost reaches 0.849 on the *service* taxonomy; its score
  on the *application* one is unmeasured. The plausible reading is that
  application identity is where sequence structure should beat volume
  statistics, but this project has had four hypotheses of that shape fail, and
  the comparison is queued rather than assumed.
- **Still unmeasured:** the cost asymmetry in seconds and GPU-hours. Without an
  accuracy result to pair it with, it is not worth measuring yet.

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

- **Evidence:** `probes` block, three seeds, `flowconx_main` on both datasets
- **Status:** **NOT SUPPORTED.** The adversarial head removes nothing.

  | Dataset | Nuisance leak from `z_flow` | From raw features |
  | --- | ---: | ---: |
  | CESNET (week) | +0.0006 ± 0.0006 | −0.0049 ± 0.0041 |
  | 5G (capture session) | +0.2935 ± 0.0205 | +0.2790 ± 0.0854 |

  On CESNET there was no nuisance information in the input to begin with, so
  the near-zero leak from the embedding is vacuous. On 5G, where capture
  identity *is* decodable from the input, the embedding leaks it at the same
  rate after adversarial removal at λ = 0.15.
- **Settled: it is the mechanism, not the weight.** Swept λ ∈ {0.01, 0.1, 0.5,
  1.0, 2.0} on 5G, three seeds each. Leakage is flat — +0.319, +0.325, +0.306,
  +0.319, +0.290 — against a raw-feature control of +0.279. Every value is at
  or above the control, including at thirteen times the default weight. Task
  macro-F1 does not move either.
- **The component should be deleted, not retuned.**
- **What did work:** the probing protocol itself. The CIST score it replaced
  would have reported 0.6409 and said nothing, being maximised by a constant
  encoder.
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
