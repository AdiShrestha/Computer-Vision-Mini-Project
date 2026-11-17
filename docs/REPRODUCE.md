# Reproducing the PLOS ONE analysis

Python 3.12 on macOS arm64 (the reported run) or Linux. All commands run from the repository root.

```sh
python3.12 -m venv .venv && . .venv/bin/activate
make install          # pip install --require-hashes -r requirements.lock
make test             # constructed-input tests; no data needed
```

## Level 1 — from the deposited checkpoints (exact)

Unzip the data deposit (`sentinel_gl_data_v1.zip`) at the repository root, then

```sh
make all-from-deposit   # replay -> packet -> figures -> manuscript -> verify
```

`make replay` re-runs every primary checkpoint with an independent implementation of the cross-mask and Score A and requires a maximum absolute difference of 0 for predictions and at most 1e-9 for Score A. `make packet` recomputes every reported number from the retained rows; `make verify` checks the manuscript against a freshly recomputed packet and the code constants.

## Level 2 — re-train from the corpus

In a copy without `data/derived/sr1_multiyear/attempt_007/plos_001/training` and `scores` (stages never overwrite), with the Hi-MAG archive downloaded to `data/raw/external/hi_mag_zenodo_4275164/` (doi:10.5281/zenodo.4275164):

```sh
make zones train score replay packet figures manuscript verify
```

Training is seeded but floating-point results can differ across CPUs, thread counts and library builds; the reported run used 4 worker processes with 2 torch threads each on an Apple M3 CPU. Re-trained values are expected to differ slightly from the deposited ones; the deposited checkpoints are the reference for Level 1.

## Level 3 — re-acquire the South Lhonak windows

`make acquire` re-reads the 48 Earth Search items (2018–2022) on the re-centred window by HTTP range requests and re-crops the 24 retained 2023 scenes. It needs network access and the attempt_006 provenance files from the deposit, and writes a new immutable attempt directory.

## Pipeline order

| Step | Command | Output |
|---|---|---|
| Acquisition (attempt_006, Oct 2026) | `source/runners/run_sr1_multiyear.py`, then `tools/enrich_sr1_native_metadata_receipts.py` | 141-scene corpus; as-run versions also in git history |
| Re-centring (attempt_007) | `make acquire` | South Lhonak windows on the Hi-MAG centroid |
| Zones | `make zones` | inventory zones, agreement with the spectral rule |
| Training | `make train` | 55 fits + extended-budget refits, run manifest |
| Scoring | `make score` | predictions, reconstruction errors, positive control, replay arrays |
| Replay | `make replay` | replay_report.json |
| Packet | `make packet` | docs/results/sr1_plos_packet.json |
| Figures | `make figures` | docs/figures/plos/Fig1–5 (TIFF + PNG) |
| References | `make references` | docs/manuscript/references_plos.json (network) |
| Manuscript | `make manuscript` | submission/plos_one/ (DOCX, SI tables, ledger) |
| Verification | `make verify` | submission/plos_one/verification_report.json |

The submission DOCX is the only manuscript format produced; no hand-rendered PDF is part of the chain.
