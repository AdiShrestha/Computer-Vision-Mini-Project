# Claude-review verification epoch

Date: 2026-10-01. Starting revision: `7f0a7d8084c3df362b917643a1d7bdb4b5ef3478`. No observations, trained research weights or scientific performance estimates were produced. Read [the assessment](../../research/claude_review_check.md) and root plan section 16.

The final complete checks pass **58 core engineering tests and 308 factory offline tests** on the installed macOS arm64 Python 3.12.8 environment. [Final verification](run_03/verification.json) records exact argv, exits, counts, raw-log hashes, dependency snapshot, supplied-review hashes and covered source bytes. Covered membership and bytes were stable throughout execution. No clean environment installation, MPS domain execution or supervised Sentinel-GL domain run occurred.

| Evidence | Actual meaning |
|---|---|
| `baseline_probes.json`, `probe_before.py` | Replayed constructed defects from the recorded migrated baseline; isolated untrained legacy architecture counts at a verified source hash; no legacy observations/checkpoints admitted |
| `historical_inventory_check.json` | All 147 prior inventory rows match Git at their recorded commit, with no unexplained tracked omissions; not a check of an unavailable Claude archive or provider truth |
| `run_01/` | Intermediate stage: 53 core and 308 factory checks passed; superseded code/checker format, not current evidence |
| `run_02/` | Core: 3 failed / 55 passed because a new test called a keyword-only trainer positionally; factory: 308 passed. Failure retained |
| `focused_test_attempt.txt` | Actual subsequent failure output: keyword correction still omitted four required trainer options; 3 failed / 12 passed. This stores the complete returned tool output; it is not a supervisor receipt |
| `run_03/` | Corrected final stage: 58 core / 308 factory checks passed. This is the current evidence referenced by readiness |
| `current_inventory.json`, `inspect_current.py` | Every nonexcluded current file byte-read/hashed, UTF-8 lines indexed and Python ASTs parsed. Roles distinguish active core/factory, constructed fixtures, historical context and untrusted supplied reports. This is coverage, not expert semantic validation of every line |
| `consistency_check.json` | Actual output of the readiness/recorded-evidence checker on the final stage; no scientific truth or execution authenticity guarantee |

Two separate read-only agents investigated and reviewed the credential-variable fix under the security fix skill. Both traced typed/legacy caller paths and used dummy values; no real credentials were exposed. This bounded independent check does not close the full architect factory-maintenance review or implement filesystem/network isolation.

Run `PYTHONDONTWRITEBYTECODE=1 python3 tools/verify.py --check-recorded` from the repository root. Fresh checks require a new directory: `PYTHONDONTWRITEBYTECODE=1 python3 tools/verify.py --run --output-dir docs/audit/<new-epoch>/<new-attempt>`. Preserve previous attempts and byte manifests. The current schema intentionally reports research blocked with no scientific claims; changing scientific readiness requires reviewed evidence and a prospective schema/contract update.

The pasted Claude reports are preserved separately with provenance. Their large differential experiment, exact Linux environment and claimed replication are not treated as locally executed evidence because raw inputs/logs were not supplied. Research remains **RESEARCH_NOT_READY**.
