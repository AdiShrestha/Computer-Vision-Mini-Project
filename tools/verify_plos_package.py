#!/usr/bin/env python3
"""Verify the PLOS ONE submission package against data, not against expected strings.

1. Recompute the result packet from the retained analysis rows into a temporary
   file and require it to equal docs/results/sr1_plos_packet.json (apart from
   the generation timestamp).
2. Re-resolve every entry of the manuscript's number ledger against the
   recomputed packet (numbers), the code (protocol constants) or a boolean
   expression (qualitative claims) and require identical text / truth value.
3. Scan the manuscript body for numeric tokens that no ledger entry explains.
4. Check PLOS format rules: abstract <= 300 words, short title <= 70 characters,
   line numbering, double spacing, native equations, numbered references cited
   in order and all cited, every figure cited, TIFF figures <= 2,250 px wide at
   300 dpi matching their manifest hashes, no unresolved placeholders.

Writes submission/plos_one/verification_report.json; exit status 1 on failure.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from docx import Document
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "source"))
import build_sr1_plos_packet as packet_builder  # noqa: E402
from plos_ledger import const_value, evaluate, fmt, resolve  # noqa: E402

SUB = ROOT / "submission/plos_one"
PACKET = ROOT / "docs/results/sr1_plos_packet.json"
MANUSCRIPT = SUB / "Manuscript.docx"
LEDGER = SUB / "number_ledger.json"
FIGS = ROOT / "docs/figures/plos"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strip(packet: dict) -> dict:
    return {k: v for k, v in packet.items() if k != "generated_utc"}


def paragraph_texts(doc) -> list[str]:
    out = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                out.append(cell.text)
    return out


def main() -> int:
    failures, checks = [], {}
    # 1. Packet recomputation.
    with tempfile.TemporaryDirectory() as tmp:
        packet_builder.PACKET = Path(tmp) / "packet.json"
        packet_builder.main()
        recomputed = json.loads(packet_builder.PACKET.read_text())
    stored = json.loads(PACKET.read_text())
    same = json.dumps(strip(recomputed), sort_keys=True) == json.dumps(strip(stored), sort_keys=True)
    checks["packet_recomputed_equal"] = same
    if not same:
        failures.append("stored result packet differs from a fresh recomputation")
    # 2. Ledger.
    ledger = json.loads(LEDGER.read_text())
    if ledger["packet_sha256"] != sha(PACKET):
        failures.append("ledger was built from a different packet")
    texts_used, n_num, n_const, n_claim = set(), 0, 0, 0
    for entry in ledger["entries"]:
        if entry["kind"] == "packet":
            text = fmt(resolve(recomputed, entry["path"]) * entry["scale"], entry["spec"])
            n_num += 1
        elif entry["kind"] == "const":
            text = fmt(const_value(entry["module"], entry["attr"]) * entry["scale"], entry["spec"])
            n_const += 1
        elif entry["kind"] == "fixed":
            texts_used.add(entry["text"])
            continue
        else:
            n_claim += 1
            if bool(evaluate(recomputed, entry["expression"])) != entry["value"]:
                failures.append(f"claim no longer holds: {entry['description']}")
            continue
        if text != entry["text"]:
            failures.append(f"ledger mismatch {entry.get('path') or entry.get('attr')}: {entry['text']} vs {text}")
        texts_used.add(text)
    checks["ledger_entries"] = {"numbers": n_num, "constants": n_const, "claims": n_claim}
    # 3. Manuscript text.
    doc = Document(MANUSCRIPT)
    paras = paragraph_texts(doc)
    full = "\n".join(paras)
    missing = sorted(t for t in texts_used if t not in full)
    if missing:
        failures.append(f"ledger values absent from manuscript: {missing[:10]}")
    body_end = next(i for i, p in enumerate(doc.paragraphs) if p.text.strip() == "References")
    body = "\n".join(p.text for p in doc.paragraphs[:body_end]) + "\n" + "\n".join(
        c.text for t in doc.tables for r in t.rows for c in r.cells)
    allowed = set(texts_used) | set(ledger.get("fixed_tokens", []))
    unexplained, contexts = [], {}
    scrub = re.sub(r"\[[\d,–\-– ]+\]", " ", body)                       # citations
    scrub = re.sub(r"\b(Fig|Table|S)\s?\d+\b", " ", scrub)                    # figure/table/SI labels
    scrub = re.sub(r"\(\d+\)", " ", scrub)                                     # equation numbers
    scrub = re.sub(r"\b(1[89]|20)\d{2}(-\d{2}(-\d{2})?)?\b", " ", scrub)      # years and ISO dates
    scrub = re.sub(r"\b\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December)\b", " ", scrub)
    scrub = re.sub(r"\b\d+\.\d+\.\d+\b", " ", scrub)                              # software versions
    scrub = scrub.replace("Element 84", " ")                                         # company name
    scrub = re.sub(r"\b(Sentinel|B|E|Level|Collection|EPSG|UTM|S2|R|C|Q|attempt_|PLOS|CC BY|Opus)[ :-]?\d+(\.\d+)?\b", " ", scrub)
    scrub = re.sub(r"[A-Za-z_]+\d+[A-Za-z_\d]*", " ", scrub)                  # identifiers such as GL088195E27912N
    for token in re.findall(r"[−\-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[−\-+]?\d+)?%?", scrub):
        if token in allowed or token.lstrip("+−-") in allowed:
            continue
        unexplained.append(token)
        k = scrub.find(token)
        contexts.setdefault(token, scrub[max(0, k - 50):k + 30].replace("\n", " "))
    checks["unexplained_numeric_tokens"] = {t: contexts[t] for t in sorted(set(unexplained))}
    if unexplained:
        failures.append(f"numbers without a ledger source: {sorted(set(unexplained))[:25]}")
    if re.search(r"\[(OWNER|TODO|TBD|XX|CONFIRM)", full) or "??" in full:
        failures.append("unresolved placeholder in manuscript")
    # 4. Format rules.
    abstract = next(paras[i + 1] for i, p in enumerate(paras) if p.strip() == "Abstract")
    checks["abstract_words"] = len(abstract.split())
    if checks["abstract_words"] > 300:
        failures.append("abstract exceeds 300 words")
    short = next((p for p in paras if p.startswith("Short title:")), "")
    checks["short_title_chars"] = len(short.replace("Short title:", "").strip())
    if not 0 < checks["short_title_chars"] <= 70:
        failures.append("short title missing or longer than 70 characters")
    xml = doc.element.xml
    checks["line_numbering"] = "lnNumType" in xml
    checks["native_equations"] = xml.count(f"<m:oMath>") + xml.count("<m:oMath ")
    if not checks["line_numbering"]:
        failures.append("continuous line numbering missing")
    if checks["native_equations"] < 6:
        failures.append("fewer than six native (OMML) equations")
    spacing = {p.paragraph_format.line_spacing for p in doc.paragraphs[:body_end] if p.text.strip()}
    checks["body_line_spacing"] = sorted(str(s) for s in spacing)
    if spacing - {2.0}:
        failures.append(f"body is not double spaced: {spacing}")
    refs = [p.text for p in doc.paragraphs[body_end + 1:] if re.match(r"^\d+\. ", p.text)]
    groups = re.findall(r"\[([\d,–\- ]+)\]", "\n".join(p.text for p in doc.paragraphs[:body_end]))
    groups = [g for g in groups if not re.fullmatch(r"(19|20)\d{2}\s*[–-]\s*(19|20)\d{2}", g.strip())]  # data-year notices
    cited = [int(x) for grp in groups for x in _expand(grp)]
    first_order = list(dict.fromkeys(cited))
    checks["references"] = len(refs)
    if sorted(set(cited)) != list(range(1, len(refs) + 1)):
        failures.append("not every reference is cited, or a citation has no reference")
    if first_order != sorted(first_order):
        failures.append("references are not numbered in order of first citation")
    figs = json.loads((FIGS / "figure_manifest.json").read_text())["figures"]
    for name, meta in figs.items():
        tif = ROOT / meta["tif"]
        if sha(tif) != meta["tif_sha256"]:
            failures.append(f"{name} differs from its manifest")
        with Image.open(tif) as im:
            dpi = tuple(round(float(x)) for x in im.info.get("dpi", (0, 0)))
            if im.width > 2250 or dpi != (300, 300):
                failures.append(f"{name}: {im.width}px at {dpi} dpi")
        if not re.search(rf"\bFig {name[3:]}\b", body):
            failures.append(f"{name} is never cited in the text")
        if not (SUB / f"{name}.tif").exists() or sha(SUB / f"{name}.tif") != meta["tif_sha256"]:
            failures.append(f"{name}.tif missing from the submission folder or stale")
    checks["figures"] = len(figs)
    report = {"status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks,
              "manuscript_sha256": sha(MANUSCRIPT), "packet_sha256": sha(PACKET), "ledger_sha256": sha(LEDGER)}
    (SUB / "verification_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "failures": failures[:12], "checks": checks}, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


def _expand(group: str) -> list[int]:
    out = []
    for part in group.split(","):
        part = part.strip()
        m = re.match(r"(\d+)\s*[–\-]\s*(\d+)", part)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
