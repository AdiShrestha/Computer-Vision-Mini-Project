#!/usr/bin/env python3
"""Research execution is blocked until source and task adapters are reviewed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentinel_gl.admission import AdmissionError, blocked

def main(argv=None):
    try:
        blocked("research execution")
    except AdmissionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
