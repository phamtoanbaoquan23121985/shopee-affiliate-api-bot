# Initial security review (partial, evidence-limited)

Inspected: main.py, auth.py, requirements.txt. Full repository audit is NOT complete.

- HIGH: auth.py prints access token and access token secret. Remove print statements, revoke exposed credentials if logs exist.
- HIGH: auth.py performs interactive Twitter OAuth on import. Keep out of production startup.
- MEDIUM: main.py wildcard-imports all bot modules and runs an infinite scheduling loop; isolation and graceful shutdown required.
- MEDIUM: requirements.txt has unpinned dependencies. Introduce lockfiles, vulnerability scanning and provenance verification.
- UNKNOWN: nested bot/database modules have not been audited.
- No claim of credential leak is made without log/history evidence.

New dual_x package never reads platform secrets and does not call undocumented APIs.
Authorized CSV feeds only until platform-specific API permissions and contract are verified.
