# Nano Fiber-MoE Agent (experimental)

## Architecture
- Token embedding -> repeated fiber-within-layer blocks -> tied vocabulary head.
- Three heterogeneous expert families: GELU, SiLU and tanh, each with low-rank down/up projection.
- Token-level top-k router; sparse dispatch computes selected experts only.
- Zero-initialized up projections preserve identity at initialization.
- Explicit available-fiber mask supports outage experiments.

## Critical distinctions
This is a **small causal-language-model component prototype**, NOT a pretrained production LLM. It currently has no attention or temporal state, so cannot model context adequately. Add causal attention / recurrent state before language pretraining. Distinct activations are not the same as distinct pretrained model families. Compression of separate teacher models requires supervised distillation into expert fibers and empirical verification. There are no claims of SOTA or guaranteed accuracy.

## Distillation plan
1. Choose licensed small specialist teachers (Thai language, product retrieval, structured analytics).
2. Collect permissioned teacher outputs with provenance and confidence.
3. Match shared hidden dimensions with learned projections; use per-task distillation (KL logits + task loss + routing balance).
4. Add causal sequence backbone, tool-use policy, and refusal/permission checks.
5. Compare equal-parameter dense, standard MoE, uniform fibers, heterogeneous fibers, and ablated fibers.
6. Report held-out Thai product QA, ranking NDCG, calibration, approved EPC, tokens/s, VRAM, p95 latency and power on RTX 3090.

Install optional torch separately; tests skip if torch is absent. This branch contains code but no trained checkpoint.
