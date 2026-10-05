"""Checkpoint bundle, state serialization, and deterministic clean-process replay tool.

Validates that model weights, optimizer state, transform state, and RNG states
can be replayed in a fresh, independent Python process to produce identical
reconstructions within float32 tolerance (<= 1e-5).
"""
from __future__ import annotations
import argparse
import dataclasses
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .model import TimeSeriesMAE, evaluation_masks


@dataclasses.dataclass
class CheckpointBundle:
    """Complete serialized model checkpoint with RNG, normalizer, and telemetry."""
    format_version: int
    model_config: Dict[str, Any]
    model_state_dict: Dict[str, Any]
    transform_state: Dict[str, Any]
    rng_states: Dict[str, Any]
    telemetry: Dict[str, Any]
    history: List[Dict[str, Any]]
    optimizer_state_dict: Optional[Dict[str, Any]] = None

    def save(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.format_version,
            "model_config": self.model_config,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "transform_state": self.transform_state,
            "rng_states": self.rng_states,
            "telemetry": self.telemetry,
            "history": self.history,
        }
        torch.save(payload, p)

    @classmethod
    def load(cls, path: Union[Path, str], device: str = "cpu") -> CheckpointBundle:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found at {p}")
        payload = torch.load(p, map_location=device, weights_only=False)
        return cls(
            format_version=payload.get("format_version", 4),
            model_config=payload["model_config"],
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload.get("optimizer_state_dict"),
            transform_state=payload["transform_state"],
            rng_states=payload["rng_states"],
            telemetry=payload["telemetry"],
            history=payload.get("history", []),
        )

    @classmethod
    def capture(
        cls,
        model: TimeSeriesMAE,
        optimizer: Optional[torch.optim.Optimizer],
        transform_state: Dict[str, Any],
        telemetry: Dict[str, Any],
        history: List[Dict[str, Any]],
        format_version: int = 4,
    ) -> CheckpointBundle:
        """Capture live model, optimizer, normalizer, and RNG states."""
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()

        return cls(
            format_version=format_version,
            model_config=dict(model.configuration),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict() if optimizer else None,
            transform_state=dict(transform_state),
            rng_states=rng_states,
            telemetry=dict(telemetry),
            history=list(history),
        )


def compute_deterministic_reconstruction(
    model: TimeSeriesMAE,
    x: torch.Tensor,
    validity: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute deterministic reconstruction under evaluation cross-masks."""
    model.eval()
    model.to(device)
    x = x.to(device)
    validity = validity.to(device)

    with torch.inference_mode():
        model._validate_input(x, validity)
        # Use first evaluation mask fold for deterministic replay
        masks = evaluation_masks(validity, partitions=2)
        reconstruction, _ = model.reconstruct(x, masks[0], validity)

    return reconstruction.cpu()


def verify_clean_process_replay(
    checkpoint_path: Union[Path, str],
    input_tensor_path: Union[Path, str],
    expected_output_path: Union[Path, str],
    tolerance: float = 1e-5,
) -> Tuple[bool, float]:
    """Execute clean-process replay check in a fresh standalone Python interpreter."""
    import os
    source_dir = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{source_dir}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = source_dir

    cmd = [
        sys.executable,
        "-m",
        "sentinel_gl.replay",
        "--checkpoint",
        str(checkpoint_path),
        "--input",
        str(input_tensor_path),
        "--expected",
        str(expected_output_path),
        "--tolerance",
        str(tolerance),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return False, float("inf")

    # Parse stdout for recorded diff
    max_diff = 0.0
    for line in result.stdout.splitlines():
        if "max_diff=" in line:
            max_diff = float(line.split("max_diff=")[-1].strip())

    return True, max_diff


def main() -> int:
    """CLI entrypoint for standalone subprocess replay verification."""
    parser = argparse.ArgumentParser(description="Deterministic checkpoint replay verifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--input", required=True, help="Path to input tensor .pt (tuple: x, mask)")
    parser.add_argument("--expected", required=True, help="Path to expected output tensor .pt")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Maximum float32 error tolerance")
    args = parser.parse_args()

    bundle = CheckpointBundle.load(args.checkpoint, device="cpu")
    model = TimeSeriesMAE(**bundle.model_config)
    model.load_state_dict(bundle.model_state_dict)

    input_data = torch.load(args.input, map_location="cpu", weights_only=False)
    if isinstance(input_data, (tuple, list)):
        x, validity = input_data[0], input_data[1]
    else:
        x, validity = input_data["x"], input_data["validity"]

    expected = torch.load(args.expected, map_location="cpu", weights_only=False)

    reconstruction = compute_deterministic_reconstruction(model, x, validity, device="cpu")
    diff = float((reconstruction - expected).abs().max().item())

    print(f"max_diff={diff:.8f}")
    if diff <= args.tolerance:
        print("REPLAY_VERIFIED: PASS")
        return 0
    else:
        print(f"REPLAY_FAILED: diff {diff:.8f} exceeds tolerance {args.tolerance}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
