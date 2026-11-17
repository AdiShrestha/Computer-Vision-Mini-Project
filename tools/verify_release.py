#!/usr/bin/env python3
"""Research release preflight: unsupported or missing evidence blocks release."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from sentinel_gl.integrity import file_hash, loads_strict
from sentinel_gl.admission import QUARANTINED


def generate_release_report(output_path=None, *, root=ROOT):
    root = Path(root)
    excluded = []
    for folder in ('data', 'docs/figures', 'docs/tables', 'docs/insar'):
        for p in sorted((root / folder).rglob('*')):
            if p.is_file() and not p.is_symlink() and file_hash(p) in QUARANTINED:
                excluded.append({'path': str(p.relative_to(root)), 'sha256': file_hash(p)})
    report = {
        'report_version': 2, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'overall_status': 'BLOCKED', 'scientific_claims': [],
        'gates': {
            'measurement_admission': {'status': 'BLOCKED', 'reason': 'NO_REVIEWED_OBSERVATIONAL_DECODER', 'excluded_current_bytes': excluded},
            'task_and_design': {'status': 'BLOCKED', 'reason': 'NO_REVIEWED_EXECUTABLE_RESEARCH_CONTRACT'},
            'reproduction': {'status': 'NOT_VERIFIED', 'reason': 'NO_ADMITTED_RESEARCH_RUN_OR_CLEAN_INSTALL'},
            'publication_and_license': {'status': 'NOT_VERIFIED', 'reason': 'FULL_HISTORY_REVIEW_AND_OWNER_AUTHORIZATION_REQUIRED'},
        },
        'scope': 'Research release remains disabled pending evidence-bound adapter review. Document keywords and desired performance cannot pass it.',
    }
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    report = generate_release_report(args.output, root=args.root)
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
