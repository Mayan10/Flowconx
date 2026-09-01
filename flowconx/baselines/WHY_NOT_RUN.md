# Baselines we could not run, and precisely why

The paper names ET-BERT, CLE-TFE, MIETT, YaTC, FlowletFormer and
PacketCLIP/FlowCLIP. This file records, for each, whether it was run and if
not what specifically blocked it. **No number for any model listed here
appears anywhere in this repository or in the paper as an "ours-run" result.**
Where a published number is cited, it is marked as cited and attributed to the
primary paper.

The blocker is the same for most of them and it is a property of our data, not
a matter of effort: **we do not retain payload bytes.** Both corpora we use are
QUIC/TLS traffic where the payload is encrypted, and our canonical schema
stores packet sizes, directions and inter-arrival times. Every model below
tokenises raw bytes.

---

## ET-BERT — not run

**Reference.** Lin et al., "ET-BERT: A Contextualized Datagram Representation
with Pre-training Transformers for Encrypted Traffic Classification", WWW 2022.

**Why not.**
1. **Input mismatch, not fixable by adaptation.** ET-BERT's input is the
   BURST-level hex byte string of the packet payload, tokenised as bigrams
   over a 65,536-entry vocabulary. Our records retain no payload. Feeding it
   header-derived features would not be ET-BERT; it would be a differently
   sized transformer with ET-BERT's name on it, which is worse than not
   running it.
2. **Pre-training corpus unavailable to us.** The published checkpoint is
   pre-trained on 30 GB of unlabelled traffic that is not redistributed. A
   from-scratch pre-training run is possible in principle but would not be the
   published model.

**What we do instead.** Cited comparison only, on the standard benchmarks ET-BERT
reports on, clearly marked as cited. See `paper/CLAIMS.md` for the claim this
does and does not support: **we do not claim to beat ET-BERT on accuracy.**

## YaTC — not run

**Reference.** Zhao et al., "Yet Another Traffic Classifier: A Masked
Autoencoder Based Traffic Transformer with Multi-Level Flow Representation",
AAAI 2023.

**Why not.** Input is a 40x40 greyscale image built from the first 5 packets'
header and payload bytes. Payload unavailable, as above. The header half alone
would be a different model.

## CLE-TFE — not run

**Reference.** Zhou et al., contrastive learning with byte-level and
packet-level augmentation for encrypted traffic.

**Why not.** Its augmentations are defined over byte sequences (byte masking,
byte substitution). Without payload the augmentation family is empty and the
contrastive objective degenerates.

## MIETT — not run

**Reference.** Multi-instance encrypted traffic transformer.

**Why not.** At the time of writing we could not locate a public reference
implementation, and the paper does not specify the tokeniser in enough detail
to re-implement it faithfully. Re-implementing from an under-specified
description and reporting the result as MIETT would be worse than reporting
nothing.

## FlowletFormer — not run

**Why not.** Same as MIETT: no public implementation located. Its flowlet
segmentation is described but the pre-training corpus is not released.

## PacketCLIP / FlowCLIP — not run

**Why not.** Requires paired text descriptions of traffic for the
contrastive-language objective. No such pairing exists for either of our
corpora, and constructing one from our labels would make the text a direct
encoding of the label, which is a leak rather than a baseline.

---

## What we run instead, and why it is the right comparison

Running an under-fed re-implementation of a pre-trained byte-level transformer
would produce a number that flatters us for the wrong reason. The honest
alternative is a strong, *fairly trained* set of baselines on the exact inputs
we use:

| Tier | Models | Status |
| --- | --- | --- |
| Trivial / identifier | port-only, SNI-only, server-IP-only, server-ASN-only, capture-id-only, protocol-only | **run** (`flowconx/audit/`) |
| Classical statistical | 5-stat RF, packet-size histogram XGBoost, CUMUL, AppScanner, k-fingerprinting | **run** (`flowconx/audit/`) |
| Deep, non-pretrained | DeepPacket-style CNN, FS-Net, bi-LSTM + attention, flow-statistics MLP | **run** (`flowconx/baselines/deep.py`) |
| Pretrained transformer | ET-BERT, YaTC, CLE-TFE, MIETT, FlowletFormer, PacketCLIP | **not run**, reasons above |

Every deep baseline is trained on the identical manifest-defined splits, with
the same optimiser, batch size, epoch budget, class weighting and
early-stopping rule as FlowCon-X. Deviations from each original are recorded
in the model's `SPEC.deviation` and copied into the results JSON.

## The claim this constrains

Because the pretrained tier is missing, the paper **cannot** claim
state-of-the-art accuracy on encrypted traffic classification. It claims
something narrower and testable: that under session-disjoint and temporal
evaluation, with unknown applications and padding defences present, a
metric-trained flow embedding degrades less and enrolls new classes more
cheaply than the alternatives we could run. `paper/THREATS.md` states this
limitation in the form it will appear in the paper.
