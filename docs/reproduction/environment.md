# Environment Setup & Independent Reproduction Recipe

This document provides complete, step-by-step instructions for reproducing the Sentinel-GL research engineering environment and verifying all artifacts on Apple Silicon macOS (`arm64`).

---

## 1. System Requirements & Hardware Target

- **Operating System:** macOS 14.0+ (Sonoma) or macOS 15.0+ (Sequoia) on Apple Silicon (`arm64`).
- **Target Hardware:** Apple M-series architecture (M1, M2, M3, M4). Verified on Apple M3 (8 CPU cores, 16 GB unified memory, fanless passively cooled chassis).
- **Python Version:** CPython 3.12.x (`arm64` binary; Python 3.12.8 recommended).
- **GPU Acceleration:** Metal Performance Shaders (MPS) built into PyTorch 2.12.1.

---

## 2. Virtual Environment Installation Recipe

Execute the following commands from the root of the repository:

```bash
# 1. Ensure Python 3.12 arm64 is installed
python3.12 --version

# 2. Create an isolated virtual environment
python3.12 -m venv .venv

# 3. Activate the environment
source .venv/bin/activate

# 4. Upgrade pip and core installer tools
python -m pip install --upgrade pip setuptools wheel

# 5. Install exact pinned dependencies from requirements lock
pip install -r source/requirements.lock
```

---

## 3. Dependency Lock Inventory (`source/requirements.lock`)

The pinned dependencies ensure bit-identical numerical evaluation across all engineering modules:

| Package | Pinned Version | Purpose |
| :--- | :--- | :--- |
| `torch` | `2.12.1` | PyTorch neural runtime with MPS Metal accelerator |
| `numpy` | `2.5.0` | Numerical array manipulation and linear algebra |
| `scipy` | `1.18.0` | Statistical distributions (chi2, Poisson exact Garwood CI) |
| `scikit-learn` | `1.9.0` | Classical ML baselines (Robust PCA) |
| `pytest` | `9.1.1` | Offline engineering test harness |
| `cryptography` | `49.0.0` | Cryptographic digests and bundle verification |
| `matplotlib` | `3.11.0` | Headless deterministic figure visualization (`Agg` backend) |

---

## 4. Hardware Verification & Benchmarking

Run the synchronized M3 hardware benchmarking engine:

```bash
PYTHONPATH=source python3 -m sentinel_gl.hardware
```

This command executes synchronized forward evaluation and masked autoencoder reconstruction across CPU and MPS, verifying:
1. **Clock Synchronization:** `torch.mps.synchronize()` surrounding measured blocks.
2. **Measurement Distinction:** Wall clock (`time.perf_counter()`), CPU time (`time.process_time()`), process RSS memory, and device tensor allocations (`torch.mps.current_allocated_memory()`).
3. **Numerical Equivalence:** Confirms that maximum absolute difference between CPU and MPS satisfies $\max |X_{\text{cpu}} - X_{\text{mps}}| \le 10^{-4}$.
4. **Thermal Disclosure:** Records fanless M3 thermal throttling caveats in `docs/benchmarks/m3_trials.json`.

---

## 5. Automated Reproduction Verification

Verify the entire repository state using the standalone reproduction tool:

```bash
python3 tools/verify_reproduction.py
```

This verifier automatically performs four independent checks:
1. **Dependency Lock Check:** Confirms that all installed packages match `source/requirements.lock` with zero version mismatches.
2. **Clean-Process Checkpoint Replay:** Launches an isolated Python subprocess via `sentinel_gl.replay` to verify that serialized model checkpoints reproduce bit-identical reconstructions within float32 tolerance ($\le 10^{-5}$).
3. **Figure Manifest Byte Verification:** Re-hashes all PNG figures in `docs/figures/` and verifies exact cryptographic match against SHA-256 digests in `docs/figures/manifest.json`.
4. **Table Metric Integrity:** Confirms that headline metrics in `docs/tables/table1_comparative_evaluation.md` and `.csv` match computed headline statistics.

The result is saved to `docs/audit/wp10_reproduction/reproduction_report.json`.

---

## 6. Full Offline Test Suites

Execute the offline unit tests:

```bash
# Core analytical engineering tests
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest source/tests -q -p no:cacheprovider
```

All tests operate strictly on constructed offline fixtures with explicit `# FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY` markers and require no external network access or credentials.
