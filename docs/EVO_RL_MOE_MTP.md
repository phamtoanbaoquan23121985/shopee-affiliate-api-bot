# EVO-RL-MoE-MTP research integration

Code tasks are a first-class task family for the Omni Fiber architecture.
- **MoE**: existing specialized fibers and gated cross-fiber microtubes.
- **MTP**: independent future-token heads predicting offsets 1..K from causal states, with masked sequence boundaries.
- **RL**: PPO-style clipped objective for policy updates; must be driven by observed rewards from sandboxed external tests, never self-reported success.
- **EVO**: deterministic architecture-neighbor proposals; only accept candidates after paired, held-out evaluations and budget checks.

Proposed code agent loop: task -> context/retrieval -> generate patch -> isolated lint/typecheck/tests/security checks -> evidence receipt -> reward -> offline policy update -> re-evaluate. Never execute untrusted generated code without sandbox isolation, resource/network restrictions, and approval gates.

This module implements losses, candidate generation and reward scoring only. It does NOT run PPO training, execute code, implement a verifier sandbox, integrate an MTP head into the Omni backbone, or demonstrate benchmark improvement. Next: expose causal hidden states from OmniFiberLM, train supervised code completion, connect verified evaluation harness and add matched-budget ablations.
