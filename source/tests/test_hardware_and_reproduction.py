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
from sentinel_gl.features import NUM_CHANNELS
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
    from unittest.mock import patch
    with patch('sentinel_gl.hardware.subprocess.check_output',side_effect=OSError('unavailable')), patch('sentinel_gl.hardware.platform.processor',return_value=''):
        profile=get_device_profile()
    assert profile['chip_family'] is None and profile['chip_status']=='NOT_MEASURED'
    assert type(profile['mps_available']) is bool
    assert profile['total_ram_bytes'] is None or profile['total_ram_bytes']>0



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
        "n_channels": NUM_CHANNELS,
        "max_time_steps": 24,
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
    B, T, C = 4, 24, NUM_CHANNELS
    torch.manual_seed(456)
    x = torch.randn(B, T, C)
    validity = torch.ones(B, T, C, dtype=torch.bool)
    validity[:, 16:20, :] = False
    x[~validity] = 0.0

    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :12] = True

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
            seq_len=36,
            n_channels=NUM_CHANNELS,
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
        assert report["origin"] == "fixture"
        assert report["research_admissible"] is False


# ---------------------------------------------------------------------------
# Test 4: Reproduction Verifier Integrity
# ---------------------------------------------------------------------------

def test_reproduction_verifier_integrity(tmp_path):
    result=generate_reproduction_report(tmp_path/'reproduction.json')
    assert result['overall_status']=='BLOCKED'
    assert verify_figure_manifest_hashes(None,None)['status']=='BLOCKED'
    assert verify_table_metrics_integrity(None)['status']=='BLOCKED'
    with pytest.raises(FileExistsError):generate_reproduction_report(tmp_path/'reproduction.json')



# ---------------------------------------------------------------------------
# Test 5: Memory Bounded Execution (< 2.0 GB)
# ---------------------------------------------------------------------------

def test_memory_bounded_execution():
    """Verify that batch evaluation and reconstruction memory remains well below 2.0 GB limit."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    repo_root = Path(__file__).resolve().parents[2]
    script = r'''
import pathlib
import resource
import sys
import torch

sys.path.insert(0, str(pathlib.Path.cwd() / "source"))
from sentinel_gl.model import TimeSeriesMAE

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
for _ in range(10):
    with torch.inference_mode():
        model.reconstruct(x, mask, validity)
        model.get_full_embeddings(x, validity)

# Darwin reports bytes; other supported platforms report KiB.
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_bytes = rss if sys.platform == "darwin" else rss * 1024
rss_gb = rss_bytes / (1024 ** 3)
print(f"peak_rss_bytes={rss_bytes}")
assert rss_gb < 2.0, f"Peak RSS {rss_gb:.3f} GB exceeds bounded budget 2.0 GB"
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, (
        f"Isolated model memory check failed with returncode {result.returncode}:\n"
        f"{result.stderr}\n{result.stdout}"
    )
    assert "peak_rss_bytes=" in result.stdout


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
            "--sequence-length", "12",
            "--channels", str(NUM_CHANNELS),
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
