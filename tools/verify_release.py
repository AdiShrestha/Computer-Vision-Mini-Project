#!/usr/bin/env python3
"""Standalone Release Integrity & Verification Auditor for Sentinel-GL.

Executes comprehensive release gates:
  1. Reproduction verification: runs verify_reproduction checks (dependency lock,
     clean subprocess checkpoint replay, figure manifest digests, table metrics).
  2. Release documentation completeness: validates docs/MAINTENANCE.md,
     docs/LICENSE_NOTICE.md, docs/CITATIONS.bib, and docs/manuscript/manuscript.md.
  3. Secret & credential leak audit: scans repository files for unauthorized private
     keys, provider tokens, and credential markers.
  4. Manuscript & scientific contract verification: confirms integration of InSAR
     monsoon decorrelation, regional cohort expansion, and small-N power bounds.

Emits structured audit artifact:
  docs/audit/wp12_release/release_verification.json
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

# Ensure repository root and tools are discoverable
REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
SOURCE_DIR = REPO_ROOT / "source"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

try:
    from tools.verify_reproduction import generate_reproduction_report
except ImportError:
    from verify_reproduction import generate_reproduction_report


def verify_reproduction_subsystem() -> Dict[str, Any]:
    """Execute reproduction subsystem checks."""
    report_path = REPO_ROOT / "docs/audit/wp10_reproduction/reproduction_report.json"
    rep_result = generate_reproduction_report(output_path=report_path)
    status = "PASS" if rep_result.get("overall_status") == "PASS" else "FAIL"
    return {
        "status": status,
        "details": rep_result.get("checks", {}),
        "report_path": str(report_path.relative_to(REPO_ROOT)),
    }


def verify_document_completeness() -> Dict[str, Any]:
    """Verify presence and required content in release documents."""
    required_docs = {
        "docs/MAINTENANCE.md": [
            "Sensor & Provider Lifecycle Management",
            "Sentinel-1C",
            "Sentinel-2C",
            "ERA5",
            "Safe Credential Rotation",
            "Model Retraining",
            "Apple Silicon",
            "CPU Reference",
        ],
        "docs/LICENSE_NOTICE.md": [
            "Copernicus Sentinel",
            "ECMWF",
            "Randolph Glacier Inventory",
            "Apache License",
        ],
        "docs/CITATIONS.bib": [
            "@article{brun2017spatially",
            "@article{veh2018detecting",
            "@article{veh2019hazard",
            "@article{nie2018inventory",
            "@article{taylor2023glacial",
            "@article{zheng2021complex",
            "doi",
        ],
        "docs/manuscript/manuscript.md": [
            "Sentinel-GL",
            "InSAR",
            "decorrelation",
            "0.25",
            "FAIL_TO_REJECT",
            "statistical power",
            "N \\ge 30",
            "non-operational",
        ],
    }

    results: Dict[str, Any] = {}
    all_ok = True

    for rel_path, required_terms in required_docs.items():
        doc_path = REPO_ROOT / rel_path
        if not doc_path.is_file():
            results[rel_path] = {
                "status": "FAIL",
                "error": "FILE_NOT_FOUND",
                "missing_terms": required_terms,
            }
            all_ok = False
            continue

        content = doc_path.read_text(encoding="utf-8")
        size_bytes = len(content.encode("utf-8"))
        missing = [t for t in required_terms if t not in content]

        doc_status = "PASS" if not missing and size_bytes > 500 else "FAIL"
        results[rel_path] = {
            "status": doc_status,
            "size_bytes": size_bytes,
            "missing_terms": missing,
        }
        if doc_status != "PASS":
            all_ok = False

    return {
        "status": "PASS" if all_ok else "FAIL",
        "documents": results,
    }


def verify_credential_and_secret_hygiene() -> Dict[str, Any]:
    """Audit project files to guarantee zero uncommitted secrets or private keys."""
    # Forbidden patterns that must never appear in public files
    forbidden_patterns = [
        (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "PRIVATE_KEY_BLOCK"),
        (re.compile(r"(?i)cdsapi_key\s*=\s*['\"][0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}['\"]"), "CDSAPI_RAW_KEY"),
        (re.compile(r"(?i)aws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{40}['\"]"), "AWS_SECRET_KEY"),
        (re.compile(r"(?i)password\s*=\s*['\"][A-Za-z0-9!@#$%^&*]{8,}['\"]"), "PLAINTEXT_PASSWORD"),
    ]

    scanned_dirs = ["source", "docs", "tools", "data"]
    findings: List[Dict[str, str]] = []
    files_scanned = 0

    for s_dir in scanned_dirs:
        dir_path = REPO_ROOT / s_dir
        if not dir_path.is_dir():
            continue
        for p in dir_path.rglob("*"):
            if not p.is_file():
                continue
            # Skip binary figure PNGs and cache files
            if p.suffix in (".png", ".pt", ".pyc", ".lock", ".DS_Store"):
                continue
            if "__pycache__" in str(p) or ".pytest_cache" in str(p):
                continue

            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                files_scanned += 1
                for pattern, label in forbidden_patterns:
                    if pattern.search(text):
                        findings.append({
                            "file": str(p.relative_to(REPO_ROOT)),
                            "violation": label,
                        })
            except Exception as e:
                findings.append({
                    "file": str(p.relative_to(REPO_ROOT)),
                    "violation": f"READ_ERROR: {str(e)}",
                })

    status = "PASS" if not findings else "FAIL"
    return {
        "status": status,
        "files_scanned": files_scanned,
        "findings_count": len(findings),
        "findings": findings,
    }


def verify_cohort_and_registry_invariants() -> Dict[str, Any]:
    """Verify that lake and event registries satisfy expanded cohort invariants."""
    lake_reg = REPO_ROOT / "data/lake_registry_expanded.csv"
    event_reg = REPO_ROOT / "data/event_registry_expanded.csv"

    if not lake_reg.is_file() or not event_reg.is_file():
        return {
            "status": "FAIL",
            "error": "Registries missing in data/",
        }

    try:
        from sentinel_gl.cohort import (
            audit_cohort_integrity,
            get_expanded_event_registry,
            get_expanded_lake_registry,
        )
        lakes = get_expanded_lake_registry()
        events = get_expanded_event_registry()
        audit = audit_cohort_integrity(lakes, events)
        return {
            "status": audit["status"],
            "lake_count": len(lakes),
            "event_count": len(events),
            "split_leakage_detected": audit["split_leakage_detected"],
            "errors": audit.get("errors", []),
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
        }


def generate_release_report(
    output_path: Path = REPO_ROOT / "docs/audit/wp12_release/release_verification.json",
) -> Dict[str, Any]:
    """Execute all release verification gates and emit structured report."""
    rep_gate = verify_reproduction_subsystem()
    doc_gate = verify_document_completeness()
    sec_gate = verify_credential_and_secret_hygiene()
    coh_gate = verify_cohort_and_registry_invariants()

    all_passed = (
        rep_gate["status"] == "PASS"
        and doc_gate["status"] == "PASS"
        and sec_gate["status"] == "PASS"
        and coh_gate["status"] == "PASS"
    )

    report = {
        "report_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if all_passed else "FAIL",
        "release_contract": "WP12-RELEASE-MAINTENANCE-01",
        "gates": {
            "reproduction_subsystem": rep_gate,
            "document_completeness": doc_gate,
            "credential_and_secret_hygiene": sec_gate,
            "cohort_and_registry_invariants": coh_gate,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> int:
    """CLI entrypoint for standalone release verification."""
    parser = argparse.ArgumentParser(description="Sentinel-GL Release Integrity Verifier.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "docs/audit/wp12_release/release_verification.json"),
        help="Path for emitted release verification report JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose summary output",
    )
    args = parser.parse_args()

    report = generate_release_report(Path(args.output))

    if not args.quiet:
        print("=" * 70)
        print("Sentinel-GL Comprehensive Release Integrity Verifier")
        print("=" * 70)
        gates = report["gates"]
        print(f"1. Reproduction Subsystem Gate:   {gates['reproduction_subsystem']['status']}")
        print(f"2. Document Completeness Gate:    {gates['document_completeness']['status']}")
        print(f"3. Secret & Credential Hygiene:   {gates['credential_and_secret_hygiene']['status']} ({gates['credential_and_secret_hygiene']['files_scanned']} files scanned)")
        print(f"4. Cohort & Registry Invariants:  {gates['cohort_and_registry_invariants']['status']} ({gates['cohort_and_registry_invariants'].get('lake_count', 0)} lakes, {gates['cohort_and_registry_invariants'].get('event_count', 0)} events)")
        print("-" * 70)
        print(f"OVERALL RELEASE VERIFICATION:     {report['overall_status']}")
        print(f"Report written to:                {args.output}")
        print("=" * 70)

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
