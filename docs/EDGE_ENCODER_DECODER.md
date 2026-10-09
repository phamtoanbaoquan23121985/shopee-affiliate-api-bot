# Nano Fiber Graph: boundary encoder and terminal decoder

## User topology
Input tokens -> **CausalStem encoder** -> parallel specialized fiber trunks -> sparse gated microtube connections at chosen depths -> terminal fusion -> **TerminalDecoder** -> vocabulary logits.

Both boundary modules are newly composed research prototypes, not proven novel components:
- Encoder: token embedding + left-padded causal depthwise temporal convolution + learned sigmoid gate + normalization.
- Decoder: terminal normalization + gated nonlinear residual refinement + tied embedding output head.
- Model: NanoFiberCodecLM, differentiable end-to-end next-token cross-entropy.
- No future-token leakage in the encoder by construction; a dedicated test checks prefix invariance.
- No full attention yet; finite receptive field limits long-context language modeling. No trained weights or accuracy claims.
- Distinct expert species require separate teacher distillation; activation variants alone are not separate pretrained models.

## Research ablations
Compare identity encoder, convolutional encoder, attention-based encoder; linear decoder vs gated decoder; no bridges vs sparse bridges; matched parameters/FLOPs and identical train/validation splits. Measure perplexity, Thai domain tasks, throughput, memory, and prefix-invariance.
