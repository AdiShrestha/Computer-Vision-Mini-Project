#!/usr/bin/env python3
"""Insert the Zenodo DOIs into the release documents (run once after reserving them).

Usage: make dois DATA_DOI=10.5281/zenodo.NNNN CODE_DOI=10.5281/zenodo.MMMM
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "CITATION.cff", "docs/DATA_AVAILABILITY.md", "submission/plos_one/SUBMISSION_FORM_ANSWERS.md"]


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not all(re.fullmatch(r"10\.\d{4,9}/\S+", x) for x in argv):
        raise SystemExit("usage: set_dois.py DATA_DOI CODE_DOI (each like 10.5281/zenodo.1234567)")
    data, code = argv
    for name in FILES:
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text()
        new = text.replace("DATA_DOI_PENDING", data).replace("CODE_DOI_PENDING", code)
        if new != text:
            path.write_text(new)
            print(f"updated {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
