# DUAL-X Commerce: evidence-first upgrade

## Observed legacy issues
- `main.py` eagerly imports social bots, schedules every job, and loops indefinitely without graceful shutdown.
- `auth.py` performs interactive Twitter OAuth at import time and prints access credentials. Do not import it in production.
- `requirements.txt` is unpinned and contains multiple unrelated social integrations.

## Delivered in this branch
- Offline, deterministic product commission and expected contribution model.
- Missing conversion data is explicitly marked INSUFFICIENT_DATA, not guessed.
- Input validation and standard-library unit tests.

## Next engineering gates
1. Audit every source file for leaked credentials and revoke any discovered secrets.
2. Split legacy scheduler and connectors into isolated optional modules; remove import-time side effects.
3. Build permissioned Shopee TH and Lazada TH adapters with official API access checks, retry/backoff, rate limiting, provenance and redacted logs.
4. Store click attribution, approved/pending/reversed commissions separately. Never label estimates as settled revenue.
5. Add PostgreSQL migrations, encrypted secrets, idempotent jobs, signed webhooks where supported, observability and CI.
6. Build responsive storefront first, React Native merchant dashboard second.
7. Validate links, disclosure requirements, promotion-channel rules, API terms and data privacy before publishing.
8. Measure real EPC, approved commission, net margin and retention against baseline before enabling automated promotion.

Run: `python -m unittest discover -s tests -v`
