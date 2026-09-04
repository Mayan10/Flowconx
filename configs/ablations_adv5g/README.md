# Adversarial weight sweep on 5G Traffic

The closed-set adversarial sweep (`configs/ablations/adv_weight_*.yaml`)
inherits `cesnet_main.yaml`, so it ran on CESNET-QUIC22 — **where the nuisance
is not present.** Week is undecodable from the raw features there (−0.005
above chance), so the sweep could only ever show a flat line, and it did:
task macro-F1 moved 0.002 across λ ∈ [0, 1] and the leak stayed at zero
throughout.

That is not a diagnosis of C9. It is a measurement taken where the phenomenon
does not occur.

On 5G Traffic the capture session **is** decodable — +0.279 ± 0.085 above
chance from raw flow features — and at λ = 0.15 the embedding leaks it at
+0.294 ± 0.021, i.e. removal achieves nothing. This family asks the question
that actually matters: **is there any λ at which it works?**

λ ∈ {0, 0.01, 0.1, 0.5, 1.0, 2.0}, three seeds, probes on.

If leakage stays flat across two orders of magnitude in λ, the failure is in
the mechanism rather than the weight, and the component should be removed
rather than retuned.
