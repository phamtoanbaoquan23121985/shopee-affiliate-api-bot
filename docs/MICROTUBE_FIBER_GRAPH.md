# Nano Fiber Graph: selective microtube connections

Independent fiber trunks retain their own parameters and layer stacks. Small directed, low-rank pipes connect **only** at chosen checkpoints where measured transfer is beneficial; terminal fusion remains the final aggregator.

- Nodes: (fiber, depth) hidden states.
- Edges: (depth, source, destination); each microtube has rank-r down/up and sigmoid gate.
- No implicit all-to-all exchange. Edges at each checkpoint read the same snapshot.
- Zero-initialized bridge output preserves initial isolation; gradients can learn to open useful bridges.
- Disabled edges support fault injection and ablations.
- Candidate edges must be chosen on TRAIN/VALIDATION only, never test; score by paired marginal gain in downstream loss minus FLOPs/latency/interference penalties.
- Avoid data leakage from profitability labels to product representation when making predictions.
- Compare no bridges, all bridges, random sparse, and evidence-selected sparse at equal parameter/FLOP budgets.

Prototype caveat: propose_edges accepts a supplied synergy matrix; it does not yet measure synergy. Current lanes are tokenwise low-rank residual blocks without causal attention. This is not a trained LLM or a proven accuracy improvement. Dense evaluation of all lanes remains expensive; add conditional lane execution in later work.
