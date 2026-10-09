# DUAL-X Subsea Fiber Neural Routing — research prototype

Inspired by the user's earlier fiber-within-layer topology concept. This is a software routing analogy, not literal optical networking or a validated novel architecture.

## Graph design
- Nodes: product text embedding, user intent, behavioral sequence, commission/risk context.
- Fiber paths: independent representation channels between nodes, each with capacity, latency, reliability and learned signal.
- Routing: softmax((signal + ln(reliability) + ln(capacity) - ln(1 + latency_ms))/temperature).
- Reliability=0 or capacity=0 disables the path. If all paths fail, return NO_AVAILABLE_PATH, never fabricate predictions.
- Fusion: z = sum_k alpha_k h_k. All channel embeddings must have equal dimensions.

## Training plan
1. Train per-channel encoders on permitted datasets with provenance.
2. Freeze encoders; train small routing head on chronological ranking labels.
3. Fine-tune routing + last encoder layers using calibrated real Thai-market data.
4. Optimize ranking quality and latency with capacity and resilience penalties.
5. Evaluate failure injection: remove channels, corrupt features, increase latency.
6. Ablations: concat baseline, uniform fusion, learned soft routing, reliability-gated routing.

## Evidence requirements
NDCG@10, Recall@20, Brier, calibration error, approved EPC/1000, p95 inference latency, CPU/GPU memory and performance under channel outages. Use identical splits, seeds, hardware and budget.

## Limits
The current Python module implements deterministic routing and fusion only. No model weights are trained. Latency is a metadata feature, not measured network transit time. No claim of superior accuracy until paired evaluation.
