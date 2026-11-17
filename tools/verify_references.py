#!/usr/bin/env python3
"""Check every manuscript reference against its primary record and format it (Vancouver).

DOIs are resolved through the Crossref REST API (DataCite for Zenodo DOIs) and arXiv identifiers through
the arXiv API; web documents are checked for an HTTP 200 response. Raw
responses are kept under data/raw/external/reference_checks/ (not released);
the verified registry docs/manuscript/references_plos.json records the
source, retrieval time, response SHA-256 and the formatted entry. A
reference whose record cannot be retrieved, or whose retrieved title does not
contain the expected title words, stops the build.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external/reference_checks"
OUT = ROOT / "docs/manuscript/references_plos.json"
UA = {"User-Agent": "sentinel-gl-reference-check/1.0 (research software; no contact supplied)"}

# key, kind, identifier, expected title words (lower case), optional manual fields
REFERENCES = [
    ("shugar2020", "doi", "10.1038/s41558-020-0855-4", "rapid worldwide growth of glacial lakes"),
    ("zheng2021", "doi", "10.1038/s41558-021-01028-3", "increasing risk of glacial lake outburst floods"),
    ("taylor2023", "doi", "10.1038/s41467-023-36033-x", "glacial lake outburst floods threaten millions"),
    ("sattar2025", "doi", "10.1126/science.ads2659", "sikkim flood of october 2023"),
    ("veh2018", "doi", "10.1016/j.rse.2017.12.025", "detecting himalayan glacial lake outburst floods"),
    ("yu2024", "doi", "10.3390/rs16132307", "south lhonak"),
    ("chen2021", "doi", "10.5194/essd-13-741-2021", "annual 30 m dataset for glacial lakes"),
    ("chen2020data", "datacite", "10.5281/zenodo.4275164", "annual 30-meter dataset for glacial lakes"),
    ("drusch2012", "doi", "10.1016/j.rse.2011.11.026", "sentinel-2"),
    ("mcfeeters1996", "doi", "10.1080/01431169608948714", "normalized difference water index"),
    ("xu2006", "doi", "10.1080/01431160600589179", "modification of normalised difference water index"),
    ("vaswani2017", "arxiv", "1706.03762", "attention is all you need"),
    ("he2022", "doi", "10.1109/CVPR52688.2022.01553", "masked autoencoders are scalable vision learners"),
    ("zerveas2021", "doi", "10.1145/3447548.3467401", "transformer-based framework for multivariate time series"),
    ("cong2022", "arxiv", "2207.08051", "satmae"),
    ("tseng2023", "arxiv", "2304.14065", ""),
    ("szwarcman2024", "arxiv", "2412.02732", "prithvi-eo-2.0"),
    ("loshchilov2019", "arxiv", "1711.05101", "decoupled weight decay regularization"),
    ("verbesselt2010", "doi", "10.1016/j.rse.2009.08.014", "detecting trend and seasonal changes"),
    ("zhu2014", "doi", "10.1016/j.rse.2014.01.011", "continuous change detection and classification"),
    ("kennedy2010", "doi", "10.1016/j.rse.2010.07.008", "landtrendr"),
    ("liu2008", "doi", "10.1109/ICDM.2008.17", "isolation forest"),
    ("scholkopf2001", "doi", "10.1162/089976601750264965", "estimating the support of a high-dimensional distribution"),
    ("page1954", "doi", "10.1093/biomet/41.1-2.100", "continuous inspection schemes"),
    ("candes2011", "doi", "10.1145/1970392.1970395", "robust principal component analysis"),
    ("wu2021", "doi", "10.1109/TKDE.2021.3112126", "current time series anomaly detection benchmarks are flawed"),
    ("tatbul2018", "arxiv", "1803.03639", "precision and recall for time series"),
    ("kunsch1989", "doi", "10.1214/aos/1176347265", "jackknife and the bootstrap for general stationary observations"),
    ("paszke2019", "arxiv", "1912.01703", "pytorch"),
    ("sarfraz2024", "web", "https://proceedings.mlr.press/v235/sarfraz24a.html", "quo vadis"),
    ("pedregosa2011", "web", "https://jmlr.org/papers/v12/pedregosa11a.html", "scikit-learn"),
    ("earthsearch", "web", "https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a", "sentinel-2"),
    ("copernicus_notice", "web", "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice", ""),
]

# Fields Crossref/arXiv do not supply for web documents; titles are checked against the page.
WEB_ENTRIES = {
    "sarfraz2024": "Sarfraz MS, Chen MY, Layer L, Peng K, Koulakis M. Position: Quo vadis, unsupervised time series anomaly detection? In: Proceedings of the 41st International Conference on Machine Learning. PMLR; 2024. p. 43461–43476. Available from: https://proceedings.mlr.press/v235/sarfraz24a.html",
    "pedregosa2011": "Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825–2830. Available from: https://jmlr.org/papers/v12/pedregosa11a.html",
    "earthsearch": "Element 84. Earth Search STAC API: Sentinel-2 Collection 1 Level-2A (sentinel-2-c1-l2a) [Internet]. Registry of Open Data on AWS. Available from: https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a",
    "copernicus_notice": "European Commission. Legal notice on the use of Copernicus Sentinel data and service information [Internet]. Available from: https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice",
}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9.\- ]+", " ", re.sub(r"\s+", " ", text))


def get(url: str, accept: str | None = None) -> tuple[int, bytes]:
    headers = dict(UA)
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read()


def vancouver_authors(names: list[tuple[str, str]]) -> str:
    out = []
    for family, given in names[:6]:
        initials = "".join(part[0] for part in re.split(r"[\s\-.]+", given) if part and part[0].isalpha()).upper()
        out.append(f"{family} {initials}".strip())
    text = ", ".join(out)
    return text + (", et al." if len(names) > 6 else ".")


def from_crossref(msg: dict) -> tuple[str, str]:
    names = [(a.get("family", a.get("name", "")), a.get("given", "")) for a in msg.get("author", [])]
    names = [(f.title() if f.isupper() else f, g) for f, g in names]
    # Spelling fixes where the Crossref record differs from the printed article (checked against the article PDFs' bylines).
    names = [({"McFEETERS": "McFeeters", "Kunsch": "Künsch"}.get(f, f), g) for f, g in names]
    if not names and msg.get("publisher"):
        names = [(msg["publisher"], "")]
    title = html.unescape(re.sub(r"<[^>]+>", "", (msg.get("title") or [""])[0])).strip().rstrip(".")
    if title.isupper():
        title = title.capitalize()
    container = html.unescape((msg.get("short-container-title") or msg.get("container-title") or [""])[0])
    parts = msg.get("published-print") or msg.get("published-online") or msg.get("issued") or msg.get("published")
    year = parts["date-parts"][0][0]
    vol, issue, page = msg.get("volume"), msg.get("issue"), msg.get("page") or msg.get("article-number")
    if msg.get("type") == "dataset":
        entry = f"{vancouver_authors(names)} {title} [Dataset]. {msg.get('publisher', '')}; {year}. doi:{msg['DOI']}"
    elif msg.get("type") in ("proceedings-article", "book-chapter"):
        entry = f"{vancouver_authors(names)} {title}. In: {container}. {year}." + (f" p. {page}." if page else "") + f" doi:{msg['DOI']}"
    else:
        entry = f"{vancouver_authors(names)} {title}. {container}. {year}"
        entry += (f";{vol}" if vol else "") + (f"({issue})" if issue else "") + (f":{page}" if page else "")
        entry += f". doi:{msg['DOI']}"
        entry = entry.replace(":1-1.", " (early access).")
    return title, re.sub(r"\s+", " ", entry.replace("..", ".").replace("?.", "?"))


def from_arxiv(xml: bytes, ident: str) -> tuple[str, str]:
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    entry = ET.fromstring(xml).find("a:entry", ns)
    if entry is None:
        raise ValueError("arXiv returned no entry")
    title = re.sub(r"\s+", " ", entry.find("a:title", ns).text).strip()
    names = []
    for a in entry.findall("a:author", ns):
        full = a.find("a:name", ns).text.strip()
        parts = full.split()
        names.append((parts[-1], " ".join(parts[:-1])))
    year = entry.find("a:published", ns).text[:4]
    journal = entry.find("x:journal_ref", ns)
    ref = f"{vancouver_authors(names)} {title}. arXiv:{ident} [Preprint]. {year}."
    if journal is not None and journal.text:
        ref += f" Published as: {re.sub(r'\s+', ' ', journal.text.strip())}."
    return title, ref


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    registry = {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "style": "Vancouver (PLOS)",
                "references": {}}
    failures = []
    for key, kind, ident, expect in REFERENCES:
        try:
            if kind == "doi":
                url = "https://api.crossref.org/works/" + urllib.parse.quote(ident)
                status, body = get(url)
                title, entry = from_crossref(json.loads(body)["message"])
            elif kind == "datacite":
                url = "https://api.datacite.org/dois/" + urllib.parse.quote(ident)
                status, body = get(url)
                a = json.loads(body)["data"]["attributes"]
                title = a["titles"][0]["title"]
                names = [(c.get("familyName", c["name"]), c.get("givenName", "")) for c in a["creators"]]
                entry = f"{vancouver_authors(names)} {title} [Dataset]. {a['publisher']}; {a['publicationYear']}. doi:{ident}"
            elif kind == "arxiv":
                url = f"https://export.arxiv.org/api/query?id_list={ident}"
                status, body = get(url)
                title, entry = from_arxiv(body, ident)
                time.sleep(3)  # arXiv API etiquette
            else:
                url = ident
                status, body = get(url)
                text = body.decode("utf-8", "ignore")
                m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
                title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ("(non-HTML document)" if body else "")
                entry = WEB_ENTRIES[key]
            if status != 200:
                raise ValueError(f"HTTP {status}")
            if expect and norm(expect) not in norm(title) and not (kind == "web" and norm(expect) in norm(body.decode("utf-8", "ignore")[:200000])):
                raise ValueError(f"retrieved title does not match: {title!r}")
            raw = RAW / f"{key}.{'json' if kind == 'doi' else 'xml' if kind == 'arxiv' else 'html'}"
            raw.write_bytes(body)
            registry["references"][key] = {"kind": kind, "identifier": ident, "source_url": url,
                                           "retrieved_title": title, "response_sha256": hashlib.sha256(body).hexdigest(),
                                           "status": "PRIMARY_RECORD_RETRIEVED_TITLE_MATCHED", "vancouver": entry}
            print(f"OK   {key}: {title[:90]}")
        except Exception as exc:  # recorded, never silently replaced
            failures.append({"key": key, "identifier": ident, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {key}: {exc}")
    registry["failures"] = failures
    OUT.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
