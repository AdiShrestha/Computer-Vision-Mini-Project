#!/usr/bin/env python3
"""Run quarantined WP07/WP08 diagnostic baselines and inference operators."""
import argparse
import json
from pathlib import Path

from sentinel_gl.wp07_wp08_diagnostic import run


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=run(args.root,args.output)
    print(json.dumps({"status":result["status"],"research_admissible":result["research_admissible"],
        "fit_rows":result["row_summary"]["fit_rows"],"monitor_rows":result["row_summary"]["monitor_rows"],
        "methods":sorted(result["method_states"]),"calibration_status":result["calibration"]["status"]},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
