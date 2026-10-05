# Sentinel-GL

Sentinel-GL is a verifiable remote-sensing research pipeline for glacial lake observation and anomaly monitoring. The project is focused on honest, reproducible methodology for evaluating satellite observation capabilities across optical and radar sensors.

## Architecture and Numerical Core

The system's numerical core (in `source/sentinel_gl/`) implements:
- Explicit missingness handling and observation masks
- Masked-target training objectives
- Training-only transformations and estimators (preventing temporal and cross-lake leakage)
- Fitted scoring functions and robust standardization
- Empirical alarm calibration and operational exposure accounting
- Honest undefined/unestimable metrics for incomplete or single-class evaluation windows

## Getting Started

### Prerequisites

- Python 3.12+
- Dependencies specified in `source/requirements.lock`

### Installation and Testing

To install the environment and execute the verified core engineering test suite:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest source/tests -q -p no:cacheprovider
```

All core tests verify numerical invariants, boundary condition handling, temporal ordering, and data integrity contracts.

## Research Context and Integrity

The Sentinel-GL research effort prioritizes reproducible measurement and transparent reporting of both positive and null findings. Legacy data, simulated scores, and unverified predictions are quarantined in historical archives; only independently verified provider observations and documented physical features are admitted into active research.

For detailed documentation on the migration and forensic baseline, see:
- [Forensic Audit Summary](docs/FORENSIC_AUDIT.md)
- [Migration and Provenance](docs/MIGRATION.md)
- [References and Literature](docs/REFERENCES.md)
- [Release Scope and Assurance Limits](docs/RELEASE_SCOPE.md)

## License

This repository is governed by the license terms in [LICENSE](LICENSE).
