# Audit evidence, not research results

The audit inventoried all 2,048 legacy files excluding Git internals, read/hashed their bytes, inspected numeric arrays without pickle, and indexed all 148 source Python files. Selected pipeline behavior received semantic review and constructed counterexamples. Binary hashing and automated full-text indexing do not amount to complete scientific validation of every artifact. No unsafe checkpoint/object deserialization was used.

See ../../plan.md for 31 findings, their evidence, inference limits and remediation gates. legacy_inventory.json and legacy_source_index.md preserve all-file coverage. legacy_evidence_excerpts.md and legacy_generation_commit.diff preserve direct evidence of fabricated inputs/results and flawed operators. legacy_counterexamples.json contains engineering examples only.

core_tests.txt records 17 new numerical-core tests. factory_tests.txt records 276 supplied-factory offline tests; an initial sandbox socket restriction required a permitted rerun. legacy_test_collection.txt records 296 collected legacy tests, not a full test execution. environment.json and source/requirements.lock record the installed environment, not a clean-install certificate.

inventory.py accepts ROOT OUTPUT; extract_legacy_evidence.py accepts LEGACY_ROOT OUTPUT_DIRECTORY. Restore legacy tags separately to use them. No active pipeline may parse these forensic objects as measurement, fitted-state or metric inputs. A checksum identifies bytes and does not prove provider authenticity.
