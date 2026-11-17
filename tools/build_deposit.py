#!/usr/bin/env python3
"""Assemble the data deposit and the code archive for Zenodo (or a similar repository).

Writes into release/ (git-ignored):

* sentinel_gl_data_v1.zip: every file the corpus manifest, the 2023 parent
  records and the plos_001 analysis reference, plus MANIFEST.sha256, README.md
  and LICENSE-DATA.txt;
* sentinel_gl_code_v1.zip: ``git archive`` of HEAD;
* zenodo_data_metadata.json and zenodo_code_metadata.json: upload metadata.

Uploading is an owner action (it needs the owner's account); this tool never
contacts a repository.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release"
ATT7 = ROOT / "data/derived/sr1_multiyear/attempt_007"
NOTICE = "Contains modified Copernicus Sentinel data [2018–2023]."


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def referenced_files() -> set[Path]:
    files: set[Path] = set()
    manifest_path = ATT7 / "observations.sr1.json"
    files.add(manifest_path)
    files.add(ROOT / "data/derived/sr1_multiyear/attempt_006/observations.sr1.json")
    files.add(ROOT / "data/derived/expansion_2023_recentered/observations.v2.json")
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["observations"]:
        for key in ("values_file", "receipt_file", "item_file"):
            if row.get(key):
                files.add(ROOT / row[key])
        if row.get("provider") == "Earth Search C1 L2A" and not row.get("item_file"):
            files.add((ROOT / row["receipt_file"]).parent / "item.json")
        for meta in row.get("metadata_receipts", {}).values():
            files.add(ROOT / meta["path"])
        parent = row.get("parent_manifest_observation")
        if parent:
            files.add(ROOT / parent["values_file"])
            files.add(ROOT / parent["receipt_file"])
        grid = row.get("grid", {}).get("grid_path")
        if grid:
            files.add(ROOT / grid)
    for folder in (ATT7 / "plos_001", ROOT / "data/raw/sr1_multiyear/attempt_007"):
        files.update(p for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    for ledger in ("attempt_006", "attempt_007"):
        p = ROOT / "data/raw/sr1_multiyear" / ledger / "transfer_ledger.json"
        if p.exists():
            files.add(p)
    missing = [p for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} referenced files are missing, e.g. {missing[:3]}")
    return files


def main() -> int:
    REL.mkdir(exist_ok=True)
    files = sorted(referenced_files())
    lines = [f"{sha(p)}  {p.relative_to(ROOT).as_posix()}" for p in files]
    readme = f"""# Sentinel-GL data deposit (v1)

Inputs and outputs of the PLOS ONE analysis (protocol amendment SR1_AMENDMENT_PLOS_001).
Unzip at the root of the Sentinel-GL code release (same version); paths are repository-relative.
`MANIFEST.sha256` lists every file with its SHA-256 (`shasum -a 256 -c MANIFEST.sha256`).

Contents: 141 Sentinel-2 Collection-1 L2A 160 x 160 ROI arrays for South Lhonak and Khangchung Tsho
(2018-2023) with STAC items, HTTP byte-range receipts, native SAFE metadata and transfer ledgers; the
retained 2023 parent arrays and receipts; inventory zones; training run manifest and all checkpoints;
predictions, reconstruction errors, positive-control rows and replay arrays.

The Hi-MAG inventory (doi:10.5281/zenodo.4275164, CC BY 4.0) is not redistributed; download it to
data/raw/external/hi_mag_zenodo_4275164/ to recompute zones.

Licence: CC BY 4.0. {NOTICE}
"""
    licence = ("Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
               "https://creativecommons.org/licenses/by/4.0/legalcode\n\n" + NOTICE + "\n"
               "Lake zones are adapted from the Hi-MAG database (Chen et al. 2020, doi:10.5281/zenodo.4275164, CC BY 4.0).\n")
    data_zip = REL / "sentinel_gl_data_v1.zip"
    with zipfile.ZipFile(data_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("MANIFEST.sha256", "\n".join(lines) + "\n")
        z.writestr("README.md", readme)
        z.writestr("LICENSE-DATA.txt", licence)
    code_zip = REL / "sentinel_gl_code_v1.zip"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=sentinel-gl/", "-o", str(code_zip), "HEAD"],
                   cwd=ROOT, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    creators = [{"name": "Shrestha, Aditya",
                 "affiliation": "Department of Computer Science and Engineering, School of Engineering, Kathmandu University"}]
    (REL / "zenodo_data_metadata.json").write_text(json.dumps({"metadata": {
        "upload_type": "dataset", "title": "Sentinel-GL data deposit: Sentinel-2 ROI time series, zones, checkpoints and scores for South Lhonak and Khangchung Tsho (2018-2023)",
        "creators": creators, "license": "cc-by-4.0", "version": "1.0.0",
        "description": readme.replace("\n", "<br>"), "keywords": ["glacial lakes", "Sentinel-2", "GLOF", "South Lhonak"]}}, indent=2) + "\n")
    (REL / "zenodo_code_metadata.json").write_text(json.dumps({"metadata": {
        "upload_type": "software", "title": "Sentinel-GL analysis code v1.0.0", "creators": creators,
        "license": "mit", "version": "1.0.0", "description": f"Code release for the PLOS ONE submission; git commit {head}."}}, indent=2) + "\n")
    summary = {"data_zip": data_zip.name, "data_zip_bytes": data_zip.stat().st_size, "data_zip_sha256": sha(data_zip),
               "data_files": len(files), "code_zip": code_zip.name, "code_zip_sha256": sha(code_zip), "code_commit": head}
    (REL / "deposit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
