"""Analytical unit tests for M3 hardware profiling, trials, and reproduction verifier.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Constructed benchmark batches, mock models, and synthetic tensors in this file
verify device profiling, CPU/MPS numerical equivalence, trial metric integrity,
and automated reproduction verification in offline unit tests only. They do not
represent real physical observations and must never support scientific claims.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import pytest
import torch

from sentinel_gl.hardware import (
    compute_trial_summary,
    get_device_profile,
    run_hardware_trials,
)
from sentinel_gl.model import TimeSeriesMAE
from tools.verify_reproduction import (
    generate_reproduction_report,
    verify_dependency_lock,
    verify_clean_subprocess_replay,
    verify_figure_manifest_hashes,
    verify_table_metrics_integrity,
)


# ---------------------------------------------------------------------------
# Test 1: Device Profile Detection
# ---------------------------------------------------------------------------

def test_device_profile_detection():
    """Verify that system platform, architecture, RAM, and PyTorch MPS flags are detected."""
    profile = get_device_profile()

    assert isinstance(profile, dict)
    assert profile["os_platform"] in ("darwin", "linux", "windows")
    assert profile["arch"] in ("arm64", "x86_64", "aarch64")
    assert profile["cpu_count"] >= 1
    assert isinstance(profile["pytorch_version"], str)
    assert isinstance(profile["mps_built"], bool)
    assert isinstance(profile["mps_available"], bool)

    # On Apple Silicon Darwin, platform should be darwin and arch arm64
    if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
        assert profile["os_platform"] == "darwin"
        assert profile["arch"] == "arm64"
        assert profile["total_ram_bytes"] is not None
        assert profile["total_ram_bytes"] > 0
        assert "Apple" in profile["chip_family"]


# ---------------------------------------------------------------------------
# Test 2: CPU vs MPS Numerical Equivalence
# ---------------------------------------------------------------------------

def test_cpu_mps_numerical_equivalence():
    """Verify forward evaluation and reconstruction on MPS match CPU reference within 1e-4."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    profile = get_device_profile()
    if not profile["mps_available"]:
        pytest.skip("MPS backend not available on this platform; skipping device comparison")

    model_config = {
        "n_channels": 11,
        "max_time_steps": 180,
        "d_model": 32,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_encoder_heads": 2,
        "n_decoder_heads": 2,
        "d_ff_encoder": 64,
        "d_ff_decoder": 64,
        "dropout": 0.0,
    }

    torch.manual_seed(123)
    model_cpu = TimeSeriesMAE(**model_config)
    model_cpu.eval()

    model_mps = TimeSeriesMAE(**model_config).to("mps")
    model_mps.load_state_dict(model_cpu.state_dict())
    model_mps.eval()

    # Input tensors
    B, T, C = 4, 180, 11
    torch.manual_seed(456)
    x = torch.randn(B, T, C)
    validity = torch.ones(B, T, C, dtype=torch.bool)
    validity[:, 20:100, :] = False
    x[~validity] = 0.0

    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :90] = True

    # CPU run
    with torch.inference_mode():
        recon_cpu, latent_cpu = model_cpu.reconstruct(x, mask, validity)
        emb_cpu = model_cpu.get_full_embeddings(x, validity)

    # MPS run
    with torch.inference_mode():
        torch.mps.synchronize()
        recon_mps, latent_mps = model_mps.reconstruct(x.to("mps"), mask.to("mps"), validity.to("mps"))
        emb_mps = model_mps.get_full_embeddings(x.to("mps"), validity.to("mps"))
        torch.mps.synchronize()

    diff_recon = float((recon_cpu - recon_mps.cpu()).abs().max().item())
    diff_latent = float((latent_cpu - latent_mps.cpu()).abs().max().item())
    diff_emb = float((emb_cpu - emb_mps.cpu()).abs().max().item())

    # Max difference across all outputs must be <= 1e-4
    assert diff_recon <= 1e-4, f"Reconstruction difference {diff_recon} exceeds 1e-4"
    assert diff_latent <= 1e-4, f"Latent difference {diff_latent} exceeds 1e-4"
    assert diff_emb <= 1e-4, f"Embedding difference {diff_emb} exceeds 1e-4"


# ---------------------------------------------------------------------------
# Test 3: Hardware Trial Metrics Validity
# ---------------------------------------------------------------------------

def test_hardware_trial_metrics_validity():
    """Verify that hardware trials record positive wall/cpu times and valid RSS bytes."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "test_trials.json"
        report = run_hardware_trials(
            n_trials=2,
            warmup_trials=1,
            device="auto",
            output_path=report_path,
            batch_size=4,
            seq_len=60,
            n_channels=11,
            d_model=32,
        )

        assert report_path.is_file()
        assert report["config"]["n_trials"] == 2
        assert len(report["trials"]) >= 4  # at least 2 trials x 2 workloads

        for trace in report["trials"]:
            assert trace["wall_time_seconds"] > 0.0, "Wall time must be positive"
            assert trace["cpu_time_seconds"] >= 0.0, "CPU time must be non-negative"
            assert trace["peak_rss_bytes"] > 0, "Peak RSS must be non-zero"
            assert trace["workload"] in ("forward_eval", "mae_reconstruction")
            if trace["device"] == "cpu":
                assert trace["mps_allocated_bytes"] is None
            elif trace["device"] == "mps":
                assert trace["mps_allocated_bytes"] is not None
                assert trace["mps_allocated_bytes"] > 0

        # Summary check
        summary = report["summary"]
        for group_key, s in summary.items():
            assert s["count"] == 2
            assert s["wall_time_seconds"]["mean"] > 0.0
            assert s["peak_rss_bytes"]["mean"] > 0.0

        # Disclosure check
        assert "thermal" in report["thermal_disclosure"].lower()
        assert "m3" in report["thermal_disclosure"].lower()


# ---------------------------------------------------------------------------
# Test 4: Reproduction Verifier Integrity
# ---------------------------------------------------------------------------

def test_reproduction_verifier_integrity():
    """Verify that reproduction verifier passes on intact repository and detects tampering."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    repo_root = Path(__file__).resolve().parents[2]
    lock_path = repo_root / "source/requirements.lock"
    manifest_path = repo_root / "docs/figures/manifest.json"
    figures_dir = repo_root / "docs/figures"
    tables_dir = repo_root / "docs/tables"

    # Intact checks
    dep_res = verify_dependency_lock(lock_path)
    assert dep_res["status"] == "PASS"

    sub_res = verify_clean_subprocess_replay(tolerance=1e-5)
    assert sub_res["status"] == "PASS"
    assert sub_res["max_absolute_difference"] <= 1e-5

    fig_res = verify_figure_manifest_hashes(manifest_path, figures_dir)
    assert fig_res["status"] == "PASS"

    tbl_res = verify_table_metrics_integrity(tables_dir)
    assert tbl_res["status"] == "PASS"

    # Tampering detection test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_figures = Path(tmpdir) / "figures"
        tmp_figures.mkdir()
        # Copy one figure and alter a byte
        orig_fig = figures_dir / "fig1_observation_cadence.png"
        if orig_fig.is_file():
            tampered_bytes = bytearray(orig_fig.read_bytes())
            tampered_bytes[-1] ^= 0xFF
            (tmp_figures / "fig1_observation_cadence.png").write_bytes(bytes(tampered_bytes))

            # Copy other figures unchanged
            for other_fig in figures_dir.glob("*.png"):
                if other_fig.name != "fig1_observation_cadence.png":
                    (tmp_figures / other_fig.name).write_bytes(other_fig.read_bytes())

            tamper_res = verify_figure_manifest_hashes(manifest_path, tmp_figures)
            assert tamper_res["status"] == "FAIL"
            assert tamper_res["mismatches_count"] > 0


# ---------------------------------------------------------------------------
# Test 5: Memory Bounded Execution (< 2.0 GB)
# ---------------------------------------------------------------------------

def test_memory_bounded_execution():
    """Verify that batch evaluation and reconstruction memory remains well below 2.0 GB limit."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    config = {
        "n_channels": 11,
        "max_time_steps": 180,
        "d_model": 64,
        "n_encoder_layers": 2,
        "n_decoder_layers": 1,
        "n_encoder_heads": 4,
        "n_decoder_heads": 4,
        "d_ff_encoder": 128,
        "d_ff_decoder": 64,
        "dropout": 0.0,
    }
    model = TimeSeriesMAE(**config)
    model.eval()

    B, T, C = 16, 180, 11
    x = torch.randn(B, T, C)
    validity = torch.ones(B, T, C, dtype=torch.bool)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :90] = True

    # Run repeated batches
    for _ in range(10):
        with torch.inference_mode():
            _ = model.reconstruct(x, mask, validity)
            _ = model.get_full_embeddings(x, validity)

    # Process RSS on Darwin is in bytes
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_bytes / (1024 ** 3)
    assert rss_gb < 2.0, f"Peak RSS {rss_gb:.3f} GB exceeds bounded budget 2.0 GB"


# ---------------------------------------------------------------------------
# Test 6: Hardware Benchmark CLI Runner Subprocess
# ---------------------------------------------------------------------------

def test_hardware_benchmark_cli_runner_subprocess():
    """Verify that source/runners/run_hardware_benchmarks.py executes as CLI subprocess."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    repo_root = Path(__file__).resolve().parents[2]
    runner_path = repo_root / "source/runners/run_hardware_benchmarks.py"
    assert runner_path.is_file(), f"Runner script not found: {runner_path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "subprocess_trials.json"
        cmd = [
            sys.executable,
            "-B",
            str(runner_path),
            "--trials", "1",
            "--warmup", "0",
            "--batch-size", "2",
            "--output", str(tmp_out),
            "--tolerance", "1e-4",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        assert res.returncode == 0, f"CLI runner failed with returncode {res.returncode}:\n{res.stderr}\n{res.stdout}"
        assert tmp_out.is_file(), "Target output file was not produced"
        data = json.loads(tmp_out.read_text())
        assert "device_profile" in data
        assert "trials" in data
        assert "summary" in data
        assert "thermal_disclosure" in data
