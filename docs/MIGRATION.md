# Migrating v2.6 projects

V3 does not silently bless v2.6 artifacts. Keep the old Factory under `factory/legacy/v2_6_0/` for historical reproduction. Create a new plan that names the real source/data, recompute every prediction with `factory/engine/metrics.py`, repair labels and group splits, rerun all preregistered seeds with a real stopping policy, and derive failure analysis from prediction IDs. A v2.6 `contract_report.md`, SHA-256, or PASS line is not a v3 receipt.

The supplied 2.6 test project is intentionally not migrated as evidence. Its independent forensic report is `docs/v26_forensic_results.json`; the read-only tool is `docs/audit_tools/audit_v26_project.py`. It records 50 two-epoch runs, 12 rows per run, 43 AUROC disagreements, 26 AUPRC disagreements, six binary-only probability files, and four taxonomy IDs absent from the prediction corpus. The v3 metric oracle corrects the original rank direction and distinguishes average precision from threshold precision. The project’s random negative construction and estimated/hardcoded hardware telemetry remain findings requiring a genuine rerun.

## Sentinel-GL repository transition, 1 October 2026

The paragraph above describes a supplied historical factory example from another project; its numbers are not Sentinel-GL results. The Sentinel-GL audit and rebuild are specified in root plan.md.

Legacy remote main: `aaf063f80905f5f2049fad3d6440e915263d181e`. Annotated tag `legacy/sentinel-gl-2026-09-30` preserves it. Annotated tag `legacy/sentinel-gl-working-tree-2026-09-30` preserves tracked and non-ignored uncommitted legacy work at `836601e4c25147ba96dc2fcd179b1a00247f7809`. Both are unvalidated forensic snapshots, not research certifications.

Local `.migration/` contains the complete working-tree tar (excluding outer Git internals), history bundle, binary patch, status, path list and backup manifest. The old folder is retained. The archive SHA-256 is `1cd3c304635b2a834d9331bedea75ae20fad4b60d41b23e6c8b342e663f8f725`; bundle SHA-256 is `7c5e68d090c481743ebf7dd971ceabdf5d46d306eb4b0a0c0b7a484c913596f8`. Local backups are excluded from Git and active research.

Published and verified on 1 October 2026. Rebuild commit `ac167e537fd38d2d633d1dbb64ef3dc8efa70f29` replaced current main contents in an ordinary descendant commit, retaining history and both tags. GitHub main and both peeled tag targets were read back using git ls-remote. A fresh shallow checkout from GitHub passed all 17 core engineering tests in the same installed environment. This demonstrates the published tree works under that environment; clean installation and full observational research remain pending. A subsequent documentation commit records this verification. No force-push was used; the old local folder is retained.
