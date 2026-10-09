# Phase 2: Revenue Intelligence, Dual-Profit, Operations

Implemented:
- Observed approved commission EPC per 1,000 clicks, net of reported ad spend.
- Repeat-buyer ratio measured only on eligible cohorts.
- Minimum sample gates (100 clicks, 30 eligible buyers, 30 days), configurable.
- Evidence-gated ranking: incomplete cohorts cannot outrank eligible cohorts.
- Campaign lifecycle draft -> approved -> published; cancellation of unpublished campaigns.
- Unit tests for economics, gating and workflow.

Important:
- Cohort figures must be imported from genuine authorized reports; none are connected yet.
- A repeat-buyer ratio is NOT causal incremental lifetime value.
- Reported ad spend does not include every operating cost; 'net_1000' is contribution after ad spend only.
- Publishing status is set only after external confirmation; this code does not publish posts.
- Do not attribute cross-platform conversions without permitted tracking and consent.
- Run python -m unittest discover -s tests -v.
