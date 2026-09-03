# THREATS TO VALIDITY

Written as we go, not assembled at the end. Everything here is intended to
appear in the paper in substantially this form.

---

## 1. Labelling method and its assumptions

**5G Traffic.** Labels come from the dataset's own directory taxonomy:
`5G_Traffic_Datasets/<Service>/<App>/`. Every flow in a capture inherits the
capture's application. This assumes the capture contains only that
application's traffic. It largely does — captures are single-app sessions on a
dedicated device — but background OS traffic, DNS, and CDN fetches shared with
other applications are labelled as the foreground app. We do not filter them,
because a filter would need a ground truth we do not have, and filtering by
heuristic would be a hidden preprocessing choice.

**CESNET-QUIC22.** Labels are the dataset's own `APP` and `CATEGORY` fields,
derived by the publishers from SNI. This is the standard method and it is also
a limitation: **the label is a function of a field we exclude from the model's
inputs.** The task is therefore "predict what SNI would have said, without
seeing SNI". That is the right task for a world with Encrypted Client Hello,
but it means the ceiling is set by how well SNI itself categorises traffic, and
a flow the publishers could not resolve is absent rather than labelled unknown.

We quantify the shortcut directly: the SNI-only probe in
`results/audit/*/baselines.json` reports exactly how much of the task the
label's own source explains.

## 2. Encrypted Client Hello removes the labelling mechanism

Both corpora predate widespread ECH deployment. Once ECH is common, SNI-based
labelling is unavailable, which affects this work in two opposite directions:
it strengthens the motivation (traffic analysis without name resolution becomes
necessary) and it weakens future data collection (no cheap ground truth). We
state this rather than assuming the datasets remain reproducible in kind.

## 3. Split protocol, and what remains uncontrolled

We report session-disjoint, temporal and server-disjoint splits and treat the
random flow split as a contrast column. Residual dependence remains:

- **5G Traffic:** conversation segments from one capture share a client host, a
  time window and a network path. `capture_id` grouping removes the worst of it.
  Segments from *different* captures of the same app on the same day may still
  be correlated through the device and the access network.
- **5G session-disjoint is partly app-disjoint, by accident.** Several
  applications have only one or two capture files -- Roblox and Netflix have
  one each -- so grouping on `capture_id` puts an entire application on one
  side of the split. In the seed-42 split, the whole of `roblox` lands in test
  while training sees only `zepeto` for the `metaverse` class, and its per-class
  F1 is 0.008 as a result. **The 5G session-disjoint number is therefore a
  hybrid of session- and application-disjointness and is not comparable to the
  CESNET session-disjoint number**, where 104 applications spread across all 28
  capture days. Both numbers appear in the paper with this stated; presenting
  them in one column without it would be misleading.

  A second consequence: **`roblox` is an undeclared unknown application.** It
  has one capture file, so all 6,000 of its flows land in test while training
  sees only `zepeto` for the metaverse class. Those rows are scored as knowns,
  which is why that class's per-class F1 is 0.008 in the headline run. The
  open-set experiment deliberately does *not* declare it, so that its result
  is about the applications we chose to hold out rather than about an artefact
  of the split.
- **CESNET-QUIC22:** we group on the capture day. Flows from one client on
  consecutive days are not separated. A client-disjoint split is possible
  (`SRC_IP` is retained) and is the obvious next control.
- **Server-disjoint** on 5G Traffic is close to degenerate: several captures
  contact a single server address, so the protocol has few groups to work with.

## 4. Sampling

CESNET rows are a seeded reservoir of 400 flows per class per day, taken over
all 28 days: every row is *considered*, but the retained sample is 202k of
roughly 100M flows. 5G Traffic keeps a reservoir of at most 6,000 conversation
segments per capture, out of 330M packet rows read. Both are recorded in
`results/data/*_manifest.json` with the seed. Class balance is imposed at read
time, which means the reported class priors are ours, not the network's, and
no claim about base rates can be made from these datasets.

## 5. Baselines we could not run

Six named pre-trained transformers are absent because they tokenise raw payload
bytes that QUIC/TLS does not expose and our schema does not retain. This is
documented per model in `flowconx/baselines/WHY_NOT_RUN.md`. **Consequence: the
paper cannot claim state-of-the-art accuracy.** It claims a narrower thing about
behaviour under strict protocols, which is what the evidence supports.

Where our preprocessing differs from a baseline's original, the deviation is
recorded in the model's `SPEC.deviation` and copied into the results JSON —
never left as an unstated difference.

## 6. The nuisance variable changed, and so did what "invariance" means

The previous pipeline's nuisance label was a threshold on two of the model's own
input features and was reconstructed with agreement 1.000 (`AUDIT.md` §3, L5).
It has been replaced with provenance the model never sees. This is a genuine
improvement, but it means **the invariance claim now refers to a different
quantity**, and no number from the previous formulation — including the 0.6409
"CIST" score — carries over. The earlier score was in any case maximised by a
constant encoder.

## 7. Generalisation limits

- Two datasets, two networks (one Korean 5G access network, one Czech academic
  backbone), one protocol family (QUIC/TLS). No claim about other regions,
  access technologies, or non-QUIC traffic.
- 5G Traffic is 15 applications; CESNET is roughly 100. Neither approaches the
  open world.
- Both are 2022. Application traffic patterns change over months and years;
  **the temporal experiment measures a six-day held-out tail of a four-week
  corpus and finds no drift at all** (macro-F1 rises 0.7678 to 0.7910). That is
  a statement about the corpus, not about the method: this data cannot exhibit
  the phenomenon. Claim C6 is withdrawn rather than supported by a flat curve,
  because a reader would take a flat curve as a claim about deployment
  lifetime.

## 8. The model is under-trained, and every number is a lower bound

13 of 16 completed runs ended with `best_epoch == epochs_run`: early stopping
never fired because validation macro-F1 was still rising at the epoch budget.
The CESNET seed-0 trajectory climbs monotonically from 0.457 to 0.779 across
15 epochs with no plateau.

The 15-epoch budget is a compute decision — it makes a 21-run sweep fit in
seven hours on one machine — not a modelling one. Every FlowCon-X figure here
is therefore a lower bound, and the comparison against classical baselines is
unfair in our own disfavour, since those are trained to their own convergence
criteria while the neural model is cut off mid-ascent. The headline
configurations must be re-run to convergence before submission.

## 9. Compute and statistical power

Runs are on a single machine. Where fewer than the stated number of seeds were
completed, `results/aggregate.json` marks the cell and the Wilcoxon test
**refuses to emit a p-value below six seeds** rather than reporting an
underpowered one.

**Six seeds is not enough either, and the reason is arithmetic.** A two-sided
Wilcoxon signed-rank test on *n* paired seeds has a minimum achievable p-value
of 2^−(n−1). At six seeds that floor is 0.03125, so a family of three
comparisons corrected by Holm-Bonferroni — which requires 0.0167 at rank 1 —
cannot certify *any* effect, however large. Our largest measured effect
(server-disjoint versus session-disjoint, Cohen's d = 20.3) hits exactly this
wall. Eight seeds are required, and are running. Any paper reporting corrected
Wilcoxon tests over five seeds is reporting a test that could not have passed.

## 10. Ethical and privacy considerations

*(Mandatory for NDSS, USENIX Security and IMC.)*

**What this work enables.** Application-level classification of encrypted
traffic from packet metadata. The same capability serves network operations
(capacity planning, QoE management, anomaly detection) and censorship or
surveillance. We take the dual-use seriously rather than treating it as
incidental.

**Data.** No data was collected by us. Both corpora are public and published for
research. CESNET-QUIC22 is anonymised at source by its publishers under an
institutional data-handling policy; addresses in it are already transformed. The
5G captures are from the authors' own devices on a commercial network. We
retain client addresses only as split-grouping keys, never as model inputs, and
`tests/test_leakage.py` asserts that mechanically.

**Our own handling.** No raw capture is committed. Committed artifacts are split
manifests (flow identifiers and checksums) and metric JSON. Addresses that
appear in the processed CSVs are not redistributed.

**Countermeasure asymmetry.** Section 7's defence evaluation exists partly so
that the padding and quantisation overheads needed to defeat this class of
classifier are stated publicly, which is the information a defender needs.

**Position.** The realistic-conditions framing cuts against a surveillance use
as much as for it: we show these classifiers degrade sharply under session
disjointness, unseen applications and modest padding. Overstating their
reliability would be the more harmful error.
