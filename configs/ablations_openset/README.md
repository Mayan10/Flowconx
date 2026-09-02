# Open-set ablations

The closed-set ablation family (`configs/ablations/`) measures macro-F1, which
is the right target for a modelling contribution. That contribution did not
survive: XGBoost on twenty packet sizes beats the model on CESNET-QUIC22, and
XGBoost on thirteen flow scalars beats it by 0.25 on 5G Traffic.

What survived is **C4** — rejection by distance to the nearest prototype
accepts roughly half as many unknown applications as softmax thresholding at
matched true-positive rate. That claim rests on the deployed embedding being
metric-trained, and **nothing in the closed-set ablation family tests it.**

This family does. Same ablations, scored on AUROC and FPR@95TPR against
held-out applications.

**The reference row is `configs/fiveg_open_set.yaml`** (experiment name
`flowconx_open_set`), already run at three seeds — the settings are identical,
so a separate `full.yaml` here would be the same experiment under another name
and the sweep runner would correctly refuse to run it twice.

Four members, restricted on purpose. Running all 35 closed-set ablations
against the open-set metric would cost a day for rows that cannot bear on a
rejection claim (temperature sweeps, input-budget sweeps).

| Config | Question |
| --- | --- |
| `no_flow_metric` | **Decisive.** No metric objective on the deployed embedding at all — the field-standard `z_app`-only training. If prototype rejection still beats softmax here, the metric objective is not what produces the advantage and C4's mechanism is misattributed. |
| `no_pair_margin` | Margin removed, contrastive kept. Which of the two objectives carries the behaviour? |
| `no_flow_supcon` | Contrastive removed, margin kept. The complement. |
| `no_adversarial` | Removing capture identity could plausibly help or hurt rejection of unseen applications. We have no prediction, which is why it is worth running. |

Run with:

```bash
python scripts/run_all_experiments.py --stages ablations_openset --seeds 0 1 2
python -m flowconx.analysis.significance
```
