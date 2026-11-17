#!/usr/bin/env python3
"""Discover and stage genuine provider products for a bounded development pilot."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentinel_gl.ingestion.development import main

if __name__ == '__main__':
    raise SystemExit(main())
