# Sentinel-GL research rebuild

Sentinel-GL is being rebuilt as a verifiable remote-sensing research system. The legacy audit found generated observations, simulated evaluation scores, invalid statistics, fabricated plots, and temporal leakage. Legacy data, checkpoints and numerical claims are quarantined in recoverable Git history; they are not active research inputs.

Read [plan.md](plan.md) for the comprehensive audit, 31 findings, mathematical specification, experiment design, Software Factory v3.3 integration requirements, staged implementation gates, M3 resource plan and architect/implementor handoffs. It is the starting point for agents working only in this folder.

The corrected [numerical core](source/README.md) implements explicit missingness, masked-target loss, training-only transforms, fitted scoring, empirical calibration and honest undefined metrics. **This is not yet a complete research system or a submission-ready GLOF predictor.** There are no empirical performance claims or migrated trained weights.

From this folder:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 factory/run_self_tests.py
```

The migration baseline passes 17 core engineering tests and 276 factory offline tests. Constructed test tensors are fixtures, never observations. See [audit evidence](docs/audit/README.md) for verification scope and limitations.

[Software Factory v3.3](factory/README.md) remains supplied policy. Its binary profile/runtime and receipt fields need the reviewed domain/maintenance work in the plan before Sentinel-GL research can be certified. [Readiness](project/READINESS.json) records the blockers; the existing research-plan JSON is an unfilled template, not a frozen experiment. Historical factory documents concerning other projects are background only.

[Migration and recovery](docs/MIGRATION.md) records the legacy tags and local backup. Use this new folder for further work. No code may fall back to the old folder, audit excerpts or archives for missing measurements/results. The existing proprietary [license](LICENSE) is retained; a future reproducible public research release needs the owner's explicit license decision and provider attribution/redistribution checks.
