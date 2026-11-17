#!/usr/bin/env python3
"""Run the explicitly authorized quarantined WP06 local diagnostic."""
import argparse
import json
from pathlib import Path

from sentinel_gl.wp06_diagnostic_training import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(args.root, args.output, device_name=args.device)
    print(json.dumps({"status": result["status"], "parameter_count": result["architecture"]["parameter_count"],
        "pilot_epochs": result["pilot"]["epochs_executed"],
        "seed_epochs": [x["epochs_executed"] for x in result["five_seed_runs"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
