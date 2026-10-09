# Overlapping independent fibers, terminal fusion

User design correction: fibers remain separate throughout their depth; they only merge at the terminal endpoint. This differs from mixing experts at every layer.

Input x -> independent Fiber A depth L
        -> independent Fiber B depth L
        -> independent Fiber C depth L
        -> terminal gated fusion -> output

Implementation: dual_x/terminal_fiber.py.
Every lane has independent parameters and identical input; no lane-to-lane information exchange before terminal fusion. Terminal weights are computed from shared input and optionally masked for failed lanes.

Caveats:
- Current lanes are low-rank tokenwise residual transforms, not pretrained LLMs.
- Output LayerNorm means zero-initialized experts do not imply an identity output.
- All lanes execute; this is NOT sparse compute. Future terminal top-k selection must occur before lane execution to save FLOPs.
- Add separate causal attention/state within each lane for a real autoregressive language model.
- For heterogeneous teacher families, distill each teacher into its lane separately, then train the terminal fusion with frozen lanes before joint fine-tuning.
- Compare against equal-FLOP early-fusion, mid-fusion, independent-lane late-fusion and standard MoE. Measure language perplexity, task accuracy, interference, memory, latency and failure robustness.
