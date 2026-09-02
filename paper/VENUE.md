# VENUE

Target venue, deadline, format and artifact requirements. Updated as results
land; the recommendation is a function of which claims survive.

---

## Recommendation as of 2026-09-02 (updated after the main comparison landed)

**Primary: ACM IMC. The decision rule below has now resolved, and it resolves
to IMC.**

The modelling case is gone. On CESNET, XGBoost over twenty packet sizes beats
FlowCon-X (0.790 vs 0.783) in three seconds against a thousand. On 5G Traffic
FlowCon-X loses by 0.25 to XGBoost on thirteen flow scalars. C5 (enrollment)
and C6 (drift) are withdrawn on their own evidence.

What stands is measurement: the split-protocol contrast (0.992 → 0.574 on 5G,
0.786 → 0.582 on CESNET under the *different* axis that matters there), the
identifier probes that explain both, and the defence-overhead table. Plus one
modelling result, C4, that rejection by distance halves the false-accept rate
of softmax thresholding at matched TPR.

**That is an IMC paper with a modelling section, not a modelling paper.**

**Backup: PETS.**

This is a change from the earlier NDSS/USENIX framing, and the reason is what
Phase 0 measured. The strongest result in hand is a *measurement* result: a
2016 statistical baseline reaches macro-F1 0.869 under the split protocol the
literature standardly uses, and 0.202 under a strict one, on identical data
with identical code. That is an IMC paper. It is a weaker security paper,
because it does not describe an attack or a defence — it describes what the
field has been measuring.

The decision rule, stated now so it is not rationalised later:

| If | Then |
| --- | --- |
| The split-protocol gap (C1) replicates on the regenerated corpora and is large | **IMC.** Lead with the measurement, position FlowCon-X as the design that survives. |
| C1 is modest but open-set (C4) and enrollment (C5) are strong | **PETS.** The deployed-embedding story is a privacy/traffic-analysis contribution and PETS reviews that framing well. |
| Both are modest | **Do not submit to a top venue this cycle.** A workshop paper on the audit methodology is honest; a stretched full paper is not. |

## Deadlines

| Venue | Cycle | Abstract | Full paper | Notes |
| --- | --- | --- | --- | --- |
| ACM IMC | annual, ~May | ~mid-May | ~late May | Single track. Strong measurement-methodology fit. |
| PETS | 4 rolling deadlines (Feb / May / Aug / Nov) | — | rolling | Rolling deadlines are a real advantage: a miss costs one quarter, not one year. |
| NDSS | summer | ~April | ~April | Two cycles. Needs a security framing we do not currently have. |
| USENIX Security | 3 cycles | — | rolling | Very strong artifact culture. |
| WWW | ~October | ~Oct | ~Oct | Where ET-BERT appeared. Wrong fit for a negative-result-heavy paper. |

**Verify every date against the official call before planning around it.**
The table records the usual pattern, not a commitment.

## Format and length

| Venue | Format | Length |
| --- | --- | --- |
| IMC | `acmart`, `sigconf` | 14 pages excl. references |
| PETS | `popets` LaTeX class | no hard limit; ~20 pages typical |
| NDSS | NDSS template | 13 pages excl. references and appendices |
| USENIX Security | USENIX template | 13 pages excl. references and appendix |

## Anonymity

| Venue | Review | Artifact link |
| --- | --- | --- |
| IMC | Double-blind | Anonymous repository required at submission |
| PETS | Double-blind | Anonymous repository expected |
| NDSS | Double-blind | Anonymous artifact encouraged |
| USENIX Security | Double-blind | Anonymous artifact |

All four require an anonymous artifact link. `scripts/anonymize_repo.sh`
produces one; it strips the git history entirely, because commit metadata
carries names, emails and remote URLs that a reviewer can trivially read.
**Run it and read the diff. It is a first pass, not a guarantee.**

## Artifact evaluation

IMC, NDSS and USENIX Security all run artifact evaluation. What each badge
needs, and where we stand:

| Requirement | Status |
| --- | --- |
| Artifact available at a permanent URL (Zenodo DOI) | **todo** — mint at submission |
| Builds and runs from documentation alone | **done** — `Dockerfile`, `make setup` |
| Reduced pipeline in reasonable time | **done** — `make repro-small` |
| Full pipeline reproducible | **partial** — `make repro-full` exists; not yet run end to end |
| Every table and figure traceable to a command | **done** — `README.md` mapping table |
| Fixed seeds, recorded environment | **done** — `flowconx/determinism.py` |
| Raw data obtainable by a third party | **done** — both corpora are public; `scripts/download_data.sh` carries DOIs, licences and checksums |
| No proprietary dependency | **done** |

The reduced pipeline mattering more than the full one is the usual reviewer
reality: an evaluator will run `make repro-small`, not a 100-run sweep. It has
to work on a clean machine with no GPU.

## Ethics review

IMC, NDSS and USENIX all require an ethics statement for network measurement
work. `paper/THREATS.md` §9 is written to be lifted into the paper. Key points:
no data collected by us, both corpora public and anonymised at source, no raw
captures redistributed, and an explicit position on the dual-use question
rather than a disclaimer.
