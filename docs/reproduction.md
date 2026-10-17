# Reproducing the 2023 analysis

The public source contains the source-bound version-2 panel, full quadrant masking, fit-only standardization, asynchronous transformer, source-excluded latent scoring, fused scores, sensor-stratified baselines, native SAR calibration and bounded optical COG reader. Numerical baseline and checkpoint-selection definitions were extracted unchanged from the implementation used for the experiment. Import paths and local configuration loading were separated for this distribution. The original empirical artifacts were not recomputed with this packaging refactor.

Use Python 3.12.8 with `PYTHONPATH=source`. The dependency snapshot records the local installed environment, rather than guaranteeing cross-platform installation. Run the offline fixture suite before attempting reproduction. Acquisition requires separate network access and applicable provider credentials. Never put credentials in tracked files or command arguments.

The numerical protocol is `configs/domain_2023_protocol.json`. The fit interval ends exclusively at `2023-10-02T18:30:00Z`; an acquisition must end before that instant to qualify. Fitting uses parity 0 of 16-pixel checkerboard blocks. Parity 1 is same-scene development monitoring. Four folds hide all optical channels in each target quadrant. Missing channels are never interpolated. Independent threshold-calibration and event-label support are unavailable; the recorded annual alarm, lead-time and classification metrics remain unestimated.

After preparing genuine ROI arrays and receipts as described in `docs/data_preparation.md`, calculate the exact manifest digest and run into a fresh output directory:

```sh
shasum -a 256 data/derived/expansion_2023/observations.v2.json
PYTHONPATH=source python3 -B source/runners/run_2023_diagnostics_v2.py \
  --observations data/derived/expansion_2023/observations.v2.json \
  --expected-ready-sha256 YOUR_VERIFIED_MANIFEST_SHA256 \
  --output data/derived/expansion_2023/reproduction_001
```

The command performs five-seed training and inference. Its manifest hash must describe the supplied bytes; a rebuilt dataset need not match the original archive hashes. Compare actual source IDs, timestamps, QA, ROI geometry and configuration before interpreting differences. GPU/CPU and library changes can affect optimization. No claim of bit-identical retraining is made.

Existing recorded outputs in `results/2023/` are copied verbatim and include fit transforms, per-seed epoch histories, model configuration, the 950-row score ledger and analysis summary. `original_output_hashes.json` verifies those copies. Checkpoints, source rasters and attention tensors are not included, so independent checkpoint/attention replay requires obtaining those exact artifacts separately. A hash establishes identity, not authenticity or scientific validity.

The original working implementation's full source suite previously reported 450 passed and one skipped. That is a separate retained execution record. Public packaging verification is recorded in `docs/verification.json`; its counts describe this distribution only.
