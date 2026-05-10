# FlowCon-X Research And Build Plan

Research date: 2026-05-01

## 1. Samsung Problem Statement, Restated Precisely

Samsung asks for a packet flow encoder that generates context-aware embeddings from:

- Flow-level features such as 5-tuple, packet rate, packet size, inter-arrival time, direction, flow duration, and burst behavior.
- Network characteristics such as RTT, jitter, loss, retransmission, queue delay, and changing link condition.

The embedding must support adaptive AI based traffic classification in dynamic environments, including:

- Encrypted and unencrypted traffic.
- Changing network conditions.
- Emerging traffic types such as XR.
- Real-time use, below 100 ms per flow.

The core KPI is semantic geometry:

- Similar traffic should be nearby. Example: YouTube and Netflix.
- Dissimilar traffic should be far apart. Example: video streaming and gaming.
- New traffic should land near the right service category, even if the exact app was unseen.

## 2. Definitive KPIs

These are the KPIs we will optimize and report.

| KPI | Samsung target | FlowCon-X evaluation |
|---|---:|---|
| Embedding similarity | Intra-class cosine similarity > 0.7 and inter-class < 0.3 | Report app-level and service-level cosine similarity with confidence intervals |
| Classification accuracy | At least 90 percent | k-NN and SVM over frozen embeddings, report best and both individual results |
| Generalization | At least 85 percent on unseen traffic types | Leave-one-app-out and leave-one-dataset-out semantic prototype classification |
| Real-time performance | < 100 ms per flow | CPU p50, p95, p99 latency with batch size 1 |

Important framing: Samsung's example treats YouTube and Netflix as "intra-class" because both are video streaming. We should therefore report two levels:

- App class: YouTube, Netflix, Valorant, Zoom, and so on.
- Service class: streaming, gaming, conferencing, bulk transfer, browsing, IoT, XR interactive media.

The main KPI should be service-class geometry because that is what enables unseen XR routing.

## 3. Dataset Strategy

Training policy:

- Final training must use real public datasets or manually captured packet flows.
- Synthetic traffic is not part of the training plan.
- Counterfactual network perturbation is optional and is off by default in code.
- CIST may still use counterfactual perturbation for evaluation because it is a stress test, not a training dataset.

### A. Samsung Suggested Datasets

Use all of these, but with clear roles.

| Dataset | Role |
|---|---|
| 5G Traffic Datasets | Primary app and service classification dataset |
| CESNET-QUIC22 | Encrypted QUIC validation and packet timing validation |
| MAWI | Real-world robustness and domain shift validation |
| Manual packet capture | Controlled good and bad network conditions for the disentanglement demo |
| PacketCLIP | Baseline inspiration for semantic interpretability, not a drop-in solution |

Samsung 5G Traffic Datasets:

- https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets

CESNET-QUIC22:

- https://zenodo.org/records/7409924
- https://zenodo.org/records/7963302

MAWI:

- https://mawi.wide.ad.jp/mawi/

PacketCLIP:

- https://arxiv.org/abs/2503.03747

### B. Datasets That Make Us Stand Out

Most teams will use the obvious Samsung list. We should add XR and cloud gaming traces so our "emerging traffic" story is real, not only simulated.

| Dataset | Why it matters |
|---|---|
| GAViST5G | Direct overlap with YouTube, Netflix, Prime Video, League of Legends, Teamfight Tactics, Valorant |
| VR Traffic Dataset | Real Meta Quest 2 VR user activity, useful for XR holdout |
| VR-AR-CG Network Telemetry | AR, VR, and cloud gaming features, including packet and frame timing |
| Cloud Gaming and Cloud VR Traces | Includes normal and disturbed network conditions |
| NANCY VR Video Streaming on O-RAN 5G | VR PCAPs plus O-RAN xApp metrics |
| Discern-XR | Recent online classifier paper for Metaverse traffic |
| CICIoT2023 | IoT attack and benign robustness, not primary app classification |

Links:

- GAViST5G: https://www.kaggle.com/datasets/ahassanein/aggregated-gaming-and-video-streaming-traffic-for-5g
- VR Traffic Dataset paper: https://www.mdpi.com/2306-5729/8/8/132
- VR Traffic Dataset files: https://doi.org/10.6084/m9.figshare.22191160
- VR-AR-CG telemetry: https://github.com/dcomp-leris/VR-AR-CG-network-telemetry
- Cloud gaming and Cloud VR traces: https://cloud-gaming-traces.lhs.loria.fr/
- NANCY O-RAN VR dataset: https://zenodo.org/records/13863832
- Discern-XR: https://arxiv.org/abs/2411.05184
- CICIoT2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html

### C. Correct Use Of CICIoT2023

CICIoT2023 is useful, but not perfect for this problem. It contains benign IoT plus 33 attack types across attack families such as DDoS, DoS, Recon, Web, Brute Force, Spoofing, and Mirai.

Use it for:

- IoT robustness.
- Encrypted or opaque traffic style feature testing.
- Security side story: the same embedding engine can support traffic management and security.

Do not use it as the primary proof for YouTube versus Netflix versus gaming, because its labels do not match Samsung's service examples.

## 4. The Unique Angle

The obvious AI answer is: "Use an LSTM or Transformer with contrastive loss." That is not enough to win.

FlowCon-X should be positioned as:

> A context-invariant flow embedding engine that separates service identity from network condition, updates semantic prototypes online, and proves XR readiness with real XR traces plus counterfactual network stress tests.

The differentiators:

1. Dual geometry, not one embedding plot
   - z_app should cluster by service.
   - z_net should move with RTT, jitter, and loss.
   - z_flow should preserve enough context for routing and classification.

2. Counterfactual network stress testing
   - Take the same flow and generate controlled variants at low, medium, and bad network conditions.
   - The service embedding should remain stable while the network embedding moves.
   - This directly proves context awareness.

3. XR frame pulse features
   - Extract burst and frame-like timing patterns from packet sequences.
   - XR often appears as repeated downlink frame bursts plus uplink pose or control traffic.
   - This is more original than generic packet size and IAT features.

4. Trust-gated prototype memory
   - Update prototypes only when confidence is high.
   - Buffer uncertain flows.
   - Create provisional "unknown emerging service" clusters when novelty persists.
   - This gives the "adaptive" part a concrete mechanism.

5. CIST score
   - CIST means Context Invariance Stress Test.
   - It measures how much app embedding changes when network condition changes.
   - Judges can understand this quickly: "A good model keeps YouTube near YouTube even under bad Wi-Fi."

Suggested CIST metric:

```text
CIST = mean cosine(z_app(original), z_app(counterfactual_condition))
Target: > 0.85 for same service under changed network condition
```

6. Prototype compass demo
   - For a new XR flow, show the nearest service prototypes.
   - Example output: XR interactive media 0.82, streaming 0.74, gaming 0.41, bulk 0.09.
   - This is more memorable than only showing a confusion matrix.

## 5. Architecture

### Inputs

Each flow becomes three aligned views.

1. Packet sequence view
   - Packet length
   - Inter-arrival time
   - Direction
   - Protocol hints
   - TCP flag counts where available
   - Burst index
   - Packet rate and byte rate context

2. Network condition view
   - RTT or RTT proxy
   - Jitter
   - Loss or retransmission proxy
   - Throughput
   - Uplink/downlink ratio
   - Queue delay or delay proxy

3. Semantic label view
   - App label where available.
   - Service label for Samsung KPI geometry.
   - Network condition bin for adversarial disentanglement.

### Encoder

FlowCon-X has three learned components.

1. Application identity encoder
   - Temporal convolution for local packet rhythm.
   - Transformer or attention block for longer packet relationships.
   - Output: z_app.

2. Network condition encoder
   - GRU or LSTM over network condition windows.
   - Output sequence plus pooled z_net.

3. Context fusion head
   - z_app packet tokens attend over network condition tokens.
   - Output: z_flow for downstream classification.

### Loss

Use a multi-term objective:

```text
L = L_supcon_service
  + 0.5 * L_supcon_app
  + lambda_proto * L_prototype
  + lambda_dis * L_cross_cov_disentangle
  + lambda_adv * L_network_condition_adversary
```

The key idea:

- z_app should predict service and app.
- z_app should not predict network condition.
- z_net should capture network condition.
- z_flow should be useful for classification and routing.

### Memory Bank

Use three memory systems.

1. Contrastive memory
   - Class-balanced queue of previous embeddings.
   - Increases positive and negative pairs without huge batch sizes.

2. Prototype bank
   - One prototype per service class.
   - Updated by exponential moving average from high-confidence embeddings.

3. Novelty buffer
   - Holds low-confidence but recurring flows.
   - Promotes a cluster to "unknown emerging type" after repeated evidence.

Online update rule:

```text
For each flow:
  z = encoder(flow)
  c, sim = nearest_prototype(z)

  if sim >= high_confidence_threshold:
      update prototype c with EMA
      add z to class memory
  else if novelty persists for K nearby flows:
      create provisional unknown prototype
  else:
      keep in uncertainty buffer
```

## 6. Evaluation Plan

### KPI 1: Embedding Similarity

Report:

- Service intra-class cosine.
- Service inter-class cosine.
- App intra-class cosine.
- CIST score under synthetic network condition changes.

Pass condition:

- Service intra > 0.7.
- Service inter < 0.3.
- CIST > 0.85 for same-service counterfactual pairs.

### KPI 2: Classification Accuracy

Train frozen-embedding classifiers:

- k-NN with cosine metric.
- SVM with RBF kernel.
- Optional logistic regression for speed.

Report:

- Accuracy.
- Macro F1.
- Confusion matrix.
- Encrypted subset accuracy for CESNET-QUIC22 or TLS data.

Pass condition:

- Best classifier >= 90 percent.

### KPI 3: Generalization

Use three tests:

1. Leave-one-app-out
   - Example: train with YouTube and Prime Video, hold out Netflix.
   - Correct result: Netflix maps to streaming.

2. Leave-XR-out
   - Train without XR.
   - Test on VR, AR, or cloud VR traces.
   - Correct result: XR maps to XR interactive media or interactive streaming.

3. Leave-one-dataset-out
   - Train on 5G/GAViST5G.
   - Test on CESNET or MAWI service groups where labels can be mapped.

Pass condition:

- Semantic prototype accuracy >= 85 percent.

### KPI 4: Real-Time Performance

Benchmark:

- CPU batch size 1.
- p50, p95, p99 latency.
- Mean latency.

Pass condition:

- p95 and mean below 100 ms per flow.

## 7. Demo Design

Build a demo that tells the story in one minute.

1. Live flow card
   - Shows predicted service, confidence, and nearest prototypes.

2. Embedding map
   - Color by service.
   - Optional second view color by network condition.

3. Impairment slider
   - RTT: 10 ms to 250 ms.
   - Jitter: 1 ms to 80 ms.
   - Loss: 0 percent to 5 percent.
   - Shows z_app stability and z_net movement.

4. XR reveal
   - Feed held-out XR flow.
   - Show it lands near XR interactive media or streaming plus interactive services.

## 8. Immediate Build Plan

1. Create dataset adapters for generic CSV, Samsung 5G CSV, GAViST5G CSV, CICFlowMeter CSV, CESNET-QUIC22, MAWI, and XR datasets.
2. Implement packet sequence construction from either true packet columns or aggregate flow columns.
3. Implement FlowCon-X encoder, SupCon losses, cross-covariance disentanglement, adversarial condition loss, and prototype memory.
4. Implement training CLI.
5. Implement evaluation CLI for all KPIs.
6. Keep synthetic generation only as an explicit smoke-test utility, not as the training path.
7. Implement guide and dataset mapping documentation.

## 9. Practical Winning Strategy

The first version should not chase every paper baseline. It should produce a crisp, defensible result:

- Baseline: packet statistics plus Random Forest or SVM.
- Baseline 2: single encoder with SupCon.
- FlowCon-X: dual branch plus context invariance plus memory bank.

Show an ablation table:

| Model | Intra sim | Inter sim | Accuracy | Generalization | p95 latency |
|---|---:|---:|---:|---:|---:|
| Stats baseline | TBD | TBD | TBD | TBD | TBD |
| SupCon encoder | TBD | TBD | TBD | TBD | TBD |
| FlowCon-X without memory | TBD | TBD | TBD | TBD | TBD |
| FlowCon-X full | TBD | TBD | TBD | TBD | TBD |

This gives the judges a reason to believe each innovation matters.
