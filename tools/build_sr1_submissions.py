#!/usr/bin/env python3
"""Build all ICT-CEEL 2027 submission formats (Word DOCX and LaTeX PDF) from the sealed SR-1 result packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

from manuscript_sr1_model import make_sr1_models, resolve

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "abstract": "ICT_CEEL_2027_Abstract.docx",
    "extended": "ICT_CEEL_2027_Extended_Abstract.docx",
    "fullpaper": "ICT_CEEL_2027_Fullpaper.docx",
}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def math_node(tag: str, *children, **attrs):
    node = OxmlElement("m:" + tag)
    for key, value in attrs.items():
        node.set(qn("m:" + key), str(value))
    for child in children:
        node.append(child)
    return node


def mr(text: str):
    t = math_node("t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    return math_node("r", t)


def ms(base: str, sub: str):
    return math_node("sSub", math_node("e", mr(base)), math_node("sub", mr(sub)))


def frac(top: list, bottom: list):
    return math_node("f", math_node("num", *top), math_node("den", *bottom))


def square(children: list):
    return math_node("sSup", math_node("e", *children), math_node("sup", mr("2")))


def sigma(sub: Any, children: list):
    subnodes = [mr(sub)] if isinstance(sub, str) else sub
    return math_node(
        "nary",
        math_node("naryPr", math_node("chr", val="∑"), math_node("limLoc", val="subSup"), math_node("supHide", val="1")),
        math_node("sub", *subnodes),
        math_node("sup"),
        math_node("e", *children),
    )


def norm2(children: list):
    return math_node("sSub", math_node("e", mr("‖"), *children, mr("‖")), math_node("sub", mr("2")))


def omml(number: int):
    if number == 1:
        nodes = []
        for name, band in [("N", "08"), ("M", "11")]:
            if nodes:
                nodes.append(mr(",     "))
            nodes += [
                ms("I", name),
                mr(" = "),
                frac([ms("ρ", "03"), mr(" − "), ms("ρ", band)], [mr("max(δ, "), ms("ρ", "03"), mr(" + "), ms("ρ", band), mr(")")]),
            ]
    elif number == 2:
        nodes = [
            ms("x̄", "tqc"),
            mr(" = "),
            frac([mr("1")], [ms("n", "tqc")]),
            sigma([mr("p∈"), ms("Ω", "tqc")], [frac([ms("x", "tpc"), mr(" − "), ms("med", "c")], [ms("MAD", "c")])]),
        ]
    elif number == 3:
        nodes = [
            ms("A", "t"),
            mr(" = "),
            frac(
                [sigma("q,c", [ms("m", "tqc"), square([mr("("), ms("x̂", "tqc"), mr(" − "), ms("x̄", "tqc"), mr(")")])])],
                [sigma("q,c", [ms("m", "tqc")])],
            ),
        ]
    elif number == 4:
        first = [
            ms("B", "t"),
            mr(" = "),
            frac([mr("1")], [mr("|"), ms("Q", "t"), mr("|")]),
            sigma([mr("q∈"), ms("Q", "t")], [frac([mr("1")], [mr("k")]), sigma([mr("r∈"), ms("N", "k"), mr("(q)")], [mr("d(q,r)")])]),
        ]
        second = [
            mr("d(q,r) = "),
            norm2([mr("P("), ms("h", "tq"), mr(") − P("), ms("h", "rq"), mr(")")]),
        ]
        nodes = [
            math_node("eqArr", math_node("eqArrPr", math_node("baseJc", val="center")), math_node("e", *first), math_node("e", *second))
        ]
    else:
        nodes = [
            ms("C", "t"),
            mr(" = α "),
            ms("z", "fit"),
            mr("(ln "),
            ms("A", "t"),
            mr(") + (1 − α) "),
            ms("z", "fit"),
            mr("("),
            ms("B", "t"),
            mr("),     α = "),
            frac([mr("1")], [mr("2")]),
        ]
    return math_node("oMath", *nodes)


def base_document(reference: Path) -> Document:
    doc = Document(reference)
    body = doc._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    for key, rel in list(doc.part.package.rels.items()):
        if rel.reltype.endswith("/metadata/thumbnail"):
            del doc.part.package.rels[key]
    for key, rel in list(doc.part.rels.items()):
        if rel.reltype == RT.IMAGE:
            doc.part.drop_rel(key)
    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.top_margin = section.bottom_margin = Mm(25)
        section.left_margin = section.right_margin = Mm(20)
        section.different_first_page_header_footer = False
        section.footer_distance = Mm(12)
    for name in ["Normal", "Title", "Heading 1", "Heading 2"]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.size = Pt(12 if name == "Title" else 10.5)
    for style_name in ["Title", "Normal"]:
        for border in doc.styles[style_name].element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    normal = doc.styles["Normal"].paragraph_format
    normal.space_before = Pt(0)
    normal.space_after = Pt(0)
    normal.line_spacing = 1.10
    settings = doc.settings.element
    existing = settings.find(qn("w:updateFields"))
    if existing is None:
        existing = OxmlElement("w:updateFields")
        settings.append(existing)
    existing.set(qn("w:val"), "true")
    return doc


def paragraph(doc: Document, text: str, *, bold: bool = False, center: bool = False, size: float = 10.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    r.font.name = "Times New Roman"
    return p


def add_footer(doc: Document):
    for section in doc.sections:
        footer = section.footer
        footer.is_linked_to_previous = False
        for p in list(footer.paragraphs)[1:]:
            p._element.getparent().remove(p._element)
        p = footer.paragraphs[0]
        p.clear()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for kind, text in [("begin", None), (None, " PAGE "), ("separate", None), (None, "1"), ("end", None)]:
            r = p.add_run()
            r.font.size = Pt(9)
            if kind:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), kind)
            else:
                el = OxmlElement("w:instrText" if text == " PAGE " else "w:t")
                el.text = text
                if text == " PAGE ":
                    el.set(qn("xml:space"), "preserve")
            r._r.append(el)


def build_docx(model: Dict[str, Any], reference: Path, target: Path, figures: Path, binding_sha: str):
    doc = base_document(reference)
    title = paragraph(doc, model["title"], bold=True, center=True, size=12)
    title.style = doc.styles["Title"]
    title.paragraph_format.space_after = Pt(9)
    p = paragraph(doc, "Aditya Shrestha¹*", bold=True, center=True)
    p.paragraph_format.space_after = Pt(10.5)
    p = paragraph(
        doc,
        "¹ Department of Computer Science and Engineering (DoCSE), School of Engineering, Kathmandu University, Dhulikhel, Nepal",
        center=True,
        size=10,
    )
    p.paragraph_format.space_after = Pt(10.5)
    paragraph(doc, "* Correspondence: adityashrestha39@gmail.com; institutional email: as03233622@student.ku.edu.np", center=True, size=9)
    if target.name != NAMES["abstract"]:
        paragraph(doc, "(Received:                    Accepted: to be filled by the Editorial Board)", size=9)
    paragraph(doc, "Abstract", bold=True, center=True)
    paragraph(doc, model["abstract"])
    paragraph(doc, "Keywords: " + model["keywords"], size=10)
    for block in model["blocks"]:
        kind = block["kind"]
        if kind == "heading":
            p = paragraph(doc, block["text"], bold=True)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
        elif kind == "paragraph":
            p = paragraph(doc, resolve(block["text"], model["references"]))
            p.paragraph_format.first_line_indent = Mm(6.3)
        elif kind == "equation":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.tab_stops.add_tab_stop(Mm(80), WD_TAB_ALIGNMENT.CENTER)
            p.paragraph_format.tab_stops.add_tab_stop(Mm(170), WD_TAB_ALIGNMENT.RIGHT)
            p.add_run("\t")
            p._p.append(omml(block["number"]))
            p.add_run("\t(" + str(block["number"]) + ")")
        elif kind == "figure":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(6)
            p.add_run().add_picture(str(figures / block["file"]), width=Mm(160))
            p = paragraph(doc, resolve(block["caption"], model["references"]), size=9.5)
            p.paragraph_format.space_after = Pt(6)
        elif kind == "table":
            p = paragraph(doc, block["caption"], bold=True, size=9.5)
            p.paragraph_format.keep_with_next = True
            tab = doc.add_table(rows=1, cols=len(block["headers"]))
            tab.autofit = False
            widths = [52] + [(170 - 52) / (len(block["headers"]) - 1)] * (len(block["headers"]) - 1)
            for cell, text, w in zip(tab.rows[0].cells, block["headers"], widths):
                cell.text = text
                cell.width = Mm(w)
            for row in block["rows"]:
                for cell, text, w in zip(tab.add_row().cells, row, widths):
                    cell.text = text
                    cell.width = Mm(w)
            for i, row in enumerate(tab.rows):
                for cell in row.cells:
                    props = cell._tc.get_or_add_tcPr()
                    borders = OxmlElement("w:tcBorders")
                    for side in ["top", "bottom", "left", "right"]:
                        border = OxmlElement("w:" + side)
                        border.set(qn("w:val"), "single")
                        border.set(qn("w:sz"), "4")
                        border.set(qn("w:color"), "D9D9D9")
                        borders.append(border)
                    props.append(borders)
                    if i == 0:
                        shade = OxmlElement("w:shd")
                        shade.set(qn("w:fill"), "EDEDED")
                        props.append(shade)
                    for p in cell.paragraphs:
                        p.paragraph_format.space_before = Pt(3)
                        p.paragraph_format.space_after = Pt(3)
                        for r in p.runs:
                            r.font.size = Pt(9.5)
                            r.bold = i == 0
                if i == 0:
                    repeat = OxmlElement("w:tblHeader")
                    row._tr.get_or_add_trPr().append(repeat)
    if model["references"]:
        p = paragraph(doc, "References", bold=True)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(8)
        for ref in model["references"].values():
            p = paragraph(doc, ref["reference"], size=9.5)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Mm(6)
            p.paragraph_format.first_line_indent = Mm(-6)
            p.paragraph_format.space_after = Pt(3)
    if target.name != NAMES["abstract"]:
        add_footer(doc)
    doc.core_properties.title = model["title"]
    doc.core_properties.author = "Aditya Shrestha"
    doc.core_properties.comments = "Scientific result packet SHA-256: " + binding_sha
    doc.save(target)
    print(f"Saved {target.name}")


def tex(text: str) -> str:
    specials = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    converted = "".join(specials.get(c, c) for c in text)
    for a, b in [
        ("−", "-"),
        ("–", "--"),
        ("—", "---"),
        ("×", r"$\times$"),
        ("²", r"$^2$"),
        ("³", r"$^3$"),
        ("⁴", r"$^4$"),
        ("⁵", r"$^5$"),
        ("⁻", r"$^{-}$"),
        ("°", r"$^\circ$"),
        ("≤", r"$\leq$"),
        ("≥", r"$\geq$"),
        ("±", r"$\pm$"),
        ("δ", r"$\delta$"),
        ("ρ", r"$\rho$"),
        ("μ", r"$\mu$"),
        ("σ", r"$\sigma$"),
        ("Δ", r"$\Delta$"),
        ("κ", r"$\kappa$"),
        ("α", r"$\alpha$"),
        ("’", "'"),
        ("‘", "'"),
        ("“", "``"),
        ("”", "''"),
    ]:
        converted = converted.replace(a, b)
    return converted


def reference_tex(text: str) -> str:
    parts = re.split(r"(https?://\S+)", text)
    return "".join(r"\url{" + p + "}" if p.startswith("http") else tex(p) for p in parts)


def build_tex(model: Dict[str, Any], figures: Path, output: Path) -> str:
    prefix = (
        r"""\documentclass[10pt,twocolumn]{article}
\usepackage[a4paper,top=25mm,bottom=25mm,left=20mm,right=20mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,graphicx,amsmath,amssymb,booktabs}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hyphens]{url}
\usepackage[hidelinks,breaklinks]{hyperref}
\renewcommand{\rmdefault}{ptm}
\setlength{\columnsep}{6mm}
\setlength{\emergencystretch}{2em}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{0.95}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\hypersetup{pdfauthor={Aditya Shrestha}}
\title{\fontsize{12}{14}\selectfont\bfseries """
        + tex(model["title"])
        + r"""}
\author{\normalsize Aditya Shrestha$^{1*}$\\
\small $^1$Department of Computer Science and Engineering (DoCSE), School of Engineering,\\
\small Kathmandu University, Dhulikhel, Nepal\\
\small *Correspondence: \nolinkurl{adityashrestha39@gmail.com}\\
\small Institutional email: \nolinkurl{as03233622@student.ku.edu.np}\\
\small (Received: \hspace{1cm} Accepted: to be filled by the Editorial Board)}
\date{}
\begin{document}
\fontsize{10.5}{12.0}\selectfont
\raggedbottom
\maketitle
\begin{abstract}
"""
        + tex(model["abstract"])
        + r"""
\end{abstract}
\par\noindent\textbf{Keywords:} """
        + tex(model["keywords"])
        + "\n\n"
    )
    lines = [prefix]
    for b in model["blocks"]:
        kind = b["kind"]
        if kind == "heading":
            command = r"\subsection*" if re.match(r"^\d+\.\d+\.", b["text"]) else r"\section*"
            lines.append(command + "{" + tex(b["text"]) + "}")
        elif kind == "paragraph":
            lines.append(tex(resolve(b["text"], model["references"])) + "\n")
        elif kind == "equation":
            lines.append(r"\begin{equation}" + b["latex"] + r"\tag{" + str(b["number"]) + r"}\end{equation}")
        elif kind == "figure":
            path = Path(os.path.relpath((figures / b["file"]).with_suffix(".pdf"), output)).as_posix()
            caption = re.sub(r"^Figure \d+\. ", "", resolve(b["caption"], model["references"]))
            lines.append(
                r"\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{"
                + path
                + r"}\caption{"
                + tex(caption)
                + r"}\end{figure*}"
            )
        elif kind == "table":
            caption = re.sub(r"^Table \d+\. ", "", b["caption"])
            columns = "l" + "r" * (len(b["headers"]) - 1)
            lines.append(r"\begin{table*}[t]\centering\small\caption{" + tex(caption) + r"}\begin{tabular}{" + columns + r"}\toprule")
            lines.append(" & ".join(tex(v) for v in b["headers"]) + r"\\\midrule")
            lines.extend(" & ".join(tex(v) for v in row) + r"\\" for row in b["rows"])
            lines.append(r"\bottomrule\end{tabular}\end{table*}")
    lines.append(
        r"\clearpage\section*{References}\begingroup\fontsize{9.5}{10.5}\selectfont\begin{list}{}{\setlength{\leftmargin}{1em}\setlength{\itemindent}{-1em}\setlength{\itemsep}{3pt}}\raggedright"
    )
    for ref in model["references"].values():
        lines.append(r"\item " + reference_tex(ref["reference"]))
    lines.append(r"\end{list}\endgroup\end{document}")
    return "\n".join(lines) + "\n"


def main():
    bindings_path = ROOT / "docs/tables/sr1/result_bindings.json"
    figures_dir = ROOT / "docs/figures/sr1"
    output_dir = ROOT / "docs/manuscript"
    reference_dir = ROOT / "docs/manuscript"
    registry_path = ROOT / "docs/manuscript/citation_registry_sr1.json"

    b = json.loads(bindings_path.read_text())
    registry = json.loads(registry_path.read_text())
    b_sha = sha(bindings_path)

    print("Generating SR-1 semantic manuscript models...")
    models = make_sr1_models(b, registry)

    # 1. Build Word Documents
    for key, name in NAMES.items():
        build_docx(models[key], reference_dir / name, output_dir / name, figures_dir, b_sha)

    # 2. Build TeX Documents
    print("Writing LaTeX manuscripts...")
    tex_source = build_tex(models["fullpaper"], figures_dir, output_dir)
    (output_dir / "sentinel_gl_ictceel2027.tex").write_text(tex_source)
    (output_dir / "sentinel_gl_submission.tex").write_text(tex_source)
    print("Saved sentinel_gl_ictceel2027.tex and sentinel_gl_submission.tex")

    # 3. Compile LaTeX PDF
    compile_dir = ROOT / "scratch/tex_compile"
    if compile_dir.exists():
        shutil.rmtree(compile_dir)
    compile_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling LaTeX PDF with pdflatex (2 passes)...")
    for run in range(2):
        p = subprocess.run(
            ["/Library/TeX/texbin/pdflatex", "-halt-on-error", "-interaction=nonstopmode", f"-output-directory={compile_dir}", "sentinel_gl_ictceel2027.tex"],
            cwd=output_dir,
            capture_output=True,
            text=True,
        )
        (compile_dir / f"pass_{run+1}.stdout.log").write_text(p.stdout)
        (compile_dir / f"pass_{run+1}.stderr.log").write_text(p.stderr)
        if p.returncode != 0:
            print("LaTeX compilation failed! Error log:")
            print(p.stdout[-2000:])
            raise RuntimeError("pdflatex compilation failed")

    # Copy output PDFs
    shutil.copy2(compile_dir / "sentinel_gl_ictceel2027.pdf", output_dir / "sentinel_gl_ictceel2027.pdf")
    shutil.copy2(compile_dir / "sentinel_gl_ictceel2027.pdf", output_dir / "sentinel_gl_submission.pdf")
    print("Successfully built sentinel_gl_ictceel2027.pdf and sentinel_gl_submission.pdf!")

    # 4. Save Manuscript Model
    model_path = ROOT / "docs/tables/sr1/manuscript_model.json"
    model_path.write_text(json.dumps({"bindings_sha256": b_sha, "models": models}, indent=2, ensure_ascii=False) + "\n")
    print("Saved docs/tables/sr1/manuscript_model.json")

    # 5. Compute and update SUBMISSION_SHA256SUMS
    submission_files = [
        "sentinel_gl_ictceel2027.pdf",
        "sentinel_gl_submission.pdf",
        "sentinel_gl_ictceel2027.tex",
        "sentinel_gl_submission.tex",
        "ICT_CEEL_2027_Abstract.docx",
        "ICT_CEEL_2027_Extended_Abstract.docx",
        "ICT_CEEL_2027_Fullpaper.docx",
    ]
    checksum_lines = []
    for fn in submission_files:
        fpath = output_dir / fn
        if fpath.exists():
            checksum_lines.append(f"{sha(fpath)}  {fn}\n")
    (output_dir / "SUBMISSION_SHA256SUMS").write_text("".join(checksum_lines))
    print("Updated docs/manuscript/SUBMISSION_SHA256SUMS")

    print(json.dumps({
        "status": "SUCCESS_SR1_SUBMISSIONS_BUILT",
        "abstract_words": models["abstract"]["abstract_word_count"],
        "fullpaper_references": len(models["fullpaper"]["references"]),
        "outputs": [str(output_dir / n) for n in submission_files],
    }, indent=2))


if __name__ == "__main__":
    main()
