# Reproduction status and preparation

A complete clean-environment reproduction has not been verified. source/requirements.lock records an installed dependency snapshot; matching versions alone does not ensure identical numerical results, a portable installation or a fully bound environment.

Current derived features, fitted models, predictions and figures are excluded from scientific use. Replaying those bytes cannot validate the original observations or restore a contaminated holdout. Earlier statements that all artifacts are independently reproduced or bit-identical across platforms are withdrawn.

Before a scientific release, construct a clean platform-specific environment outside frozen source, verify dependency artifacts and transitive versions, record actual runtime/device identity, and run the genuine pipeline on admitted inputs. Preserve failures. Check a fresh-process replay against a tolerance selected before final evaluation; disclose CPU/MPS differences and fallback explicitly.

Measure representative synchronized workload timings and platform-specific memory scopes separately. Constructed microbenchmarks do not measure the final study or sustained thermal behavior. Unsupported energy/thermal/total-unified-memory fields remain unavailable.

For a focused numerical contract check from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest source/tests/test_integrity.py source/tests/test_numerical_contracts.py -q -p no:cacheprovider
```

Run broader tests that generate report/release artifacts only in a disposable inventoried copy until output isolation is repaired. These checks provide engineering evidence, not current observational performance or submission readiness. Final installation, data retrieval, method configuration and figure-generation instructions will follow verified execution and source permissions.
