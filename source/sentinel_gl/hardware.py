"""M3 hardware benchmarking, device profiling, and synchronized execution engine.

Provides accurate hardware profiling on Apple Silicon macOS, synchronized execution
across CPU and Metal Performance Shaders (MPS), numerical equivalence verification
within float32 tolerance (<= 1e-4), and thermal throttling disclosures for fanless M3.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .model import TimeSeriesMAE


def get_device_profile() -> Dict[str, Any]:
    """Detect platform, architecture, CPU count, RAM, chip family, PyTorch version, and MPS status."""
    os_platform = platform.system().lower()
    arch = platform.machine().lower()
    cpu_count = os.cpu_count() or 1

    # Detect total RAM in bytes
    total_ram_bytes: Optional[int] = None
    if os_platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            total_ram_bytes = int(out)
        except Exception:
            pass
    if total_ram_bytes is None:
        try:
            total_ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            total_ram_bytes = None

    # Detect chip family / CPU brand
    chip_family = "Apple Silicon"
    if os_platform == "darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if brand:
                chip_family = brand
        except Exception:
            chip_family = platform.processor() or "Apple Silicon"
    else:
        chip_family = platform.processor() or "Generic CPU"

    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())

    return {
        "os_platform": os_platform,
        "arch": arch,
        "cpu_count": cpu_count,
        "total_ram_bytes": total_ram_bytes,
        "chip_family": chip_family,
        "pytorch_version": torch.__version__,
        "mps_built": mps_built,
        "mps_available": mps_available,
    }


def compute_metric_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics (mean, median, std, min, max) for numeric series."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_trial_summary(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics grouped by device and workload."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for trace in traces:
        key = (trace["device"], trace["workload"])
        groups.setdefault(key, []).append(trace)

    summary: Dict[str, Any] = {}
    for (dev, workload), items in groups.items():
        group_key = f"{dev}_{workload}"
        wall_times = [it["wall_time_seconds"] for it in items if it.get("wall_time_seconds") is not None]
        cpu_times = [it["cpu_time_seconds"] for it in items if it.get("cpu_time_seconds") is not None]
        rss_bytes = [float(it["peak_rss_bytes"]) for it in items if it.get("peak_rss_bytes") is not None]
        mps_bytes = [float(it["mps_allocated_bytes"]) for it in items if it.get("mps_allocated_bytes") is not None]

        group_summary: Dict[str, Any] = {
            "device": dev,
            "workload": workload,
            "count": len(items),
            "wall_time_seconds": compute_metric_stats(wall_times),
            "cpu_time_seconds": compute_metric_stats(cpu_times),
            "peak_rss_bytes": compute_metric_stats(rss_bytes),
        }
        if mps_bytes:
            group_summary["mps_allocated_bytes"] = compute_metric_stats(mps_bytes)
        else:
            group_summary["mps_allocated_bytes"] = None

        summary[group_key] = group_summary

    return summary


def run_hardware_trials(
    n_trials: int = 5,
    warmup_trials: int = 2,
    device: str = "mps",
    output_path: Optional[Union[str, Path]] = "docs/benchmarks/m3_trials.json",
    batch_size: int = 16,
    seq_len: int = 180,
    n_channels: int = 11,
    d_model: int = 64,
) -> Dict[str, Any]:
    """Execute synchronized hardware trials on CPU and MPS with numerical verification.

    Workloads:
      1. forward_eval: Multi-modal (T=180, C=11) batch evaluation.
      2. mae_reconstruction: Masked autoencoder reconstruction over observation entries.

    Returns:
      Structured HardwareReport dictionary.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if warmup_trials < 0:
        raise ValueError("warmup_trials must be non-negative")

    device_profile = get_device_profile()
    mps_ready = device_profile["mps_available"] and device in ("mps", "all", "auto")

    # Devices to benchmark
    target_devices = ["cpu"]
    if mps_ready:
        target_devices.append("mps")

    # Model configuration
    model_config = {
        "n_channels": n_channels,
        "max_time_steps": seq_len,
        "d_model": d_model,
        "n_encoder_layers": 2,
        "n_decoder_layers": 1,
        "n_encoder_heads": 4,
        "n_decoder_heads": 4,
        "d_ff_encoder": 128,
        "d_ff_decoder": 64,
        "dropout": 0.0,
    }

    # Reference CPU model
    torch.manual_seed(42)
    model_cpu = TimeSeriesMAE(**model_config)
    model_cpu.eval()

    # Input tensors
    torch.manual_seed(101)
    x_cpu = torch.randn(batch_size, seq_len, n_channels)
    validity_cpu = torch.ones(batch_size, seq_len, n_channels, dtype=torch.bool)
    # Simulate partially observed intervals while preserving visible and target observations
    if seq_len > 8:
        validity_cpu[:, (seq_len // 4) : (seq_len // 2), :] = False
    x_cpu[~validity_cpu] = 0.0

    mask_cpu = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_cpu[:, ::2] = True

    traces: List[Dict[str, Any]] = []

    for dev_name in target_devices:
        dev = torch.device(dev_name)
        if dev_name == "cpu":
            model = model_cpu
            x = x_cpu
            validity = validity_cpu
            mask = mask_cpu
        else:
            model = TimeSeriesMAE(**model_config).to(dev)
            model.load_state_dict(model_cpu.state_dict())
            model.eval()
            x = x_cpu.to(dev)
            validity = validity_cpu.to(dev)
            mask = mask_cpu.to(dev)

        # 1. Warm-up trials (clock frequency stabilization)
        for _ in range(warmup_trials):
            with torch.inference_mode():
                _ = model.get_full_embeddings(x, validity)
                _ = model.reconstruct(x, mask, validity)
            if dev_name == "mps":
                torch.mps.synchronize()

        # 2. Timed trials
        for trial_idx in range(1, n_trials + 1):
            # Workload 1: Forward evaluation
            if dev_name == "mps":
                torch.mps.synchronize()
            t_wall_start = time.perf_counter()
            t_cpu_start = time.process_time()

            with torch.inference_mode():
                _ = model.get_full_embeddings(x, validity)

            if dev_name == "mps":
                torch.mps.synchronize()
            t_wall_end = time.perf_counter()
            t_cpu_end = time.process_time()

            wall_time = max(t_wall_end - t_wall_start, 1e-9)
            cpu_time = max(t_cpu_end - t_cpu_start, 1e-9)
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mps_alloc = torch.mps.current_allocated_memory() if dev_name == "mps" else None

            traces.append({
                "trial_index": trial_idx,
                "device": dev_name,
                "workload": "forward_eval",
                "wall_time_seconds": round(wall_time, 6),
                "cpu_time_seconds": round(cpu_time, 6),
                "peak_rss_bytes": rss_bytes,
                "mps_allocated_bytes": mps_alloc,
            })

            # Workload 2: MAE reconstruction
            if dev_name == "mps":
                torch.mps.synchronize()
            t_wall_start = time.perf_counter()
            t_cpu_start = time.process_time()

            with torch.inference_mode():
                _ = model.reconstruct(x, mask, validity)

            if dev_name == "mps":
                torch.mps.synchronize()
            t_wall_end = time.perf_counter()
            t_cpu_end = time.process_time()

            wall_time = max(t_wall_end - t_wall_start, 1e-9)
            cpu_time = max(t_cpu_end - t_cpu_start, 1e-9)
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mps_alloc = torch.mps.current_allocated_memory() if dev_name == "mps" else None

            traces.append({
                "trial_index": trial_idx,
                "device": dev_name,
                "workload": "mae_reconstruction",
                "wall_time_seconds": round(wall_time, 6),
                "cpu_time_seconds": round(cpu_time, 6),
                "peak_rss_bytes": rss_bytes,
                "mps_allocated_bytes": mps_alloc,
            })

    # Numerical Equivalence Verification across CPU and MPS
    numerical_eq: Dict[str, Any] = {}
    if "mps" in target_devices:
        model_mps = TimeSeriesMAE(**model_config).to("mps")
        model_mps.load_state_dict(model_cpu.state_dict())
        model_mps.eval()

        with torch.inference_mode():
            recon_cpu, latent_cpu = model_cpu.reconstruct(x_cpu, mask_cpu, validity_cpu)
            emb_cpu = model_cpu.get_full_embeddings(x_cpu, validity_cpu)

            torch.mps.synchronize()
            recon_mps, latent_mps = model_mps.reconstruct(
                x_cpu.to("mps"), mask_cpu.to("mps"), validity_cpu.to("mps")
            )
            emb_mps = model_mps.get_full_embeddings(x_cpu.to("mps"), validity_cpu.to("mps"))
            torch.mps.synchronize()

        diff_recon = float((recon_cpu - recon_mps.cpu()).abs().max().item())
        diff_latent = float((latent_cpu - latent_mps.cpu()).abs().max().item())
        diff_emb = float((emb_cpu - emb_mps.cpu()).abs().max().item())
        max_abs_diff = max(diff_recon, diff_latent, diff_emb)
        tolerance = 1e-4
        eq_status = "PASS" if max_abs_diff <= tolerance else "FAIL"

        numerical_eq = {
            "status": eq_status,
            "max_absolute_difference": max_abs_diff,
            "diff_reconstruction": diff_recon,
            "diff_latent": diff_latent,
            "diff_embedding": diff_emb,
            "tolerance": tolerance,
        }
    else:
        numerical_eq = {
            "status": "CPU_ONLY",
            "max_absolute_difference": 0.0,
            "tolerance": 1e-4,
            "note": "MPS backend unavailable; evaluated on CPU reference only",
        }

    # Summary statistics
    summary = compute_trial_summary(traces)

    # Fanless M3 Thermal Disclosure
    thermal_disclosure = (
        "Apple Silicon M3 MacBook Air is a fanless, passively cooled architecture. Sustained CPU "
        "and GPU execution leads to passive thermal throttling and clock downscaling over extended "
        "run durations. Benchmark figures reflect burst execution under stable thermal conditions. "
        "Production deployments and long-running training loops must account for thermal headroom, "
        "bounded batch memory allocations (B <= 16), and explicit memory reclamation."
    )

    report = {
        "report_version": 1,
        "device_profile": device_profile,
        "config": {
            "n_trials": n_trials,
            "warmup_trials": warmup_trials,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "n_channels": n_channels,
            "d_model": d_model,
        },
        "trials": traces,
        "summary": summary,
        "numerical_equivalence": numerical_eq,
        "thermal_disclosure": thermal_disclosure,
    }

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main() -> int:
    """CLI entrypoint for running hardware benchmark trials."""
    parser = argparse.ArgumentParser(description="Sentinel-GL M3 Hardware Benchmarking Engine.")
    parser.add_argument("--trials", type=int, default=5, help="Number of timed trials per workload")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu", "auto"], help="Target device")
    parser.add_argument("--output", type=str, default="docs/benchmarks/m3_trials.json", help="Output path for JSON report")
    args = parser.parse_args()

    print("=" * 60)
    print("Sentinel-GL Hardware Benchmarking & M3 Resource Trials")
    print("=" * 60)

    report = run_hardware_trials(
        n_trials=args.trials,
        warmup_trials=args.warmup,
        device=args.device,
        output_path=args.output,
    )

    prof = report["device_profile"]
    print(f"Platform: {prof['os_platform']} ({prof['arch']}) | Chip: {prof['chip_family']} ({prof['cpu_count']} cores)")
    ram_gb = f"{prof['total_ram_bytes'] / (1024**3):.1f} GB" if prof['total_ram_bytes'] else "Unknown"
    print(f"Total RAM: {ram_gb} | PyTorch: {prof['pytorch_version']} | MPS: {prof['mps_available']}")
    print("-" * 60)

    print("Summary Results (Mean Wall Time / Mean CPU Time):")
    for group_key, s in report["summary"].items():
        w_mean = s["wall_time_seconds"]["mean"] * 1000.0
        c_mean = s["cpu_time_seconds"]["mean"] * 1000.0
        rss_mb = s["peak_rss_bytes"]["mean"] / (1024 * 1024)
        print(f"  {group_key:25s}: wall={w_mean:6.2f} ms | cpu={c_mean:6.2f} ms | rss={rss_mb:6.1f} MB")

    neq = report["numerical_equivalence"]
    print("-" * 60)
    print(f"Numerical Equivalence (CPU vs MPS): {neq['status']} (max diff: {neq['max_absolute_difference']:.2e}, tol: {neq['tolerance']:.1e})")
    print(f"Benchmark artifact written to: {args.output}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
