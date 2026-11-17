# Use of AI tools

PLOS ONE asks authors to disclose AI tools used to generate or revise content or analysis code. This file records what the repository itself documents. The author must confirm it before submission (see `submission/plos_one/OWNER_CHECKLIST.md`).

| Tool | Where it was used (as recorded in the project) |
|---|---|
| OpenAI Codex (agent runtime) | Earlier pipeline stages, the earlier ICT-CEEL manuscript builds and their verification scripts (October 2026). |
| Google Antigravity (agent) | Architecture and contract reports for earlier work packages (October 2026). |
| Anthropic Claude (Claude Opus 5.5, Claude Code) | External code reviews; the 2026-11-17 submission-readiness review; the corrections and re-analysis in protocol amendment PLOS-001 (trainer and scoring fixes, re-centred acquisition, inventory zones, positive control, bootstrap, reference predictors), the figures, the result packet, the verifier and drafting of the PLOS ONE manuscript text. |

How output was checked:

* Every number in the manuscript is generated from `docs/results/sr1_plos_packet.json` by `tools/build_plos_manuscript.py`; `tools/verify_plos_package.py` recomputes the packet from the retained prediction rows and compares.
* Regression tests encode the defects found in review and were run against the defective code before the fix (`docs/protocol/PRE_FIX_REGRESSION_RUN.txt`).
* Predictions are replayed from the saved checkpoints by an independent implementation (`tools/replay_sr1_plos.py`).
* References were checked against Crossref, DataCite or arXiv records (`tools/verify_references.py`).
* No AI tool generated observations, labels or result values; all values come from executed code on retained Sentinel-2 data. No independent human scientific or statistical review has taken place.
