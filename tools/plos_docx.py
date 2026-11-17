"""DOCX helpers for a PLOS ONE manuscript: double spacing, line numbers, OMML equations."""
from __future__ import annotations

import re
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'


# ------------------------------------------------------------------ OMML builders
def r(text: str, italic: bool = True) -> str:
    sty = "" if italic else '<m:rPr><m:sty m:val="p"/></m:rPr>'
    return f'<m:r>{sty}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def t(text: str) -> str:  # upright text
    return r(text, italic=False)


def f(num: str, den: str) -> str:
    return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def sub(base: str, s: str) -> str:
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{s}</m:sub></m:sSub>"


def sup(base: str, s: str) -> str:
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{s}</m:sup></m:sSup>"


def subsup(base: str, s: str, p: str) -> str:
    return f"<m:sSubSup><m:e>{base}</m:e><m:sub>{s}</m:sub><m:sup>{p}</m:sup></m:sSubSup>"


def nary(symbol: str, lower: str, body: str) -> str:
    return (f'<m:nary><m:naryPr><m:chr m:val="{symbol}"/><m:limLoc m:val="undOvr"/><m:supHide m:val="1"/></m:naryPr>'
            f"<m:sub>{lower}</m:sub><m:sup/><m:e>{body}</m:e></m:nary>")


def d(body: str, beg: str = "(", end: str = ")") -> str:
    return f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr><m:e>{body}</m:e></m:d>'


def bar(body: str) -> str:
    return f'<m:acc><m:accPr><m:chr m:val="̄"/></m:accPr><m:e>{body}</m:e></m:acc>'


def hat(body: str) -> str:
    return f'<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>{body}</m:e></m:acc>'


def tilde(body: str) -> str:
    return f'<m:acc><m:accPr><m:chr m:val="̃"/></m:accPr><m:e>{body}</m:e></m:acc>'


# ------------------------------------------------------------------ document
class Manuscript:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Inches(8.5), Inches(11)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Inches(1))
        normal = self.doc.styles["Normal"]
        normal.font.name = "Times New Roman"
        normal.font.size = Pt(12)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        pf = normal.paragraph_format
        pf.line_spacing = 2.0
        pf.space_after = Pt(0)
        for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12), ("Title", 16)):
            st = self.doc.styles[name]
            st.font.name, st.font.size, st.font.bold = "Times New Roman", Pt(size), True
            st.font.color.rgb = None
            st.paragraph_format.line_spacing = 2.0
            st.paragraph_format.space_before = Pt(6)
            st.paragraph_format.space_after = Pt(0)
            rpr = st.element.get_or_add_rPr()
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts")
                rpr.append(rfonts)
            for key in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rfonts.set(qn(key), "Times New Roman")
            for theme in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rfonts.get(qn(theme)) is not None:
                    del rfonts.attrib[qn(theme)]
        sect = sec._sectPr
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        ln.set(qn("w:distance"), "360")
        sect.insert(0, ln)
        footer = sec.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run()
        for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if kind:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), kind)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)

    def heading(self, text: str, level: int = 1):
        p = self.doc.add_heading(text, level=level)
        p.paragraph_format.line_spacing = 2.0
        return p

    def para(self, text: str, *, bold_prefix: str | None = None, align=None, italic=False):
        p = self.doc.add_paragraph()
        p.paragraph_format.line_spacing = 2.0
        if align:
            p.alignment = align
        if bold_prefix:
            p.add_run(bold_prefix).bold = True
        self._rich(p, text, italic=italic)
        return p

    def _rich(self, p, text: str, italic=False):
        # *italic* and ^superscript^ markup
        for chunk in re.split(r"(\*[^*]+\*|\^[^^]+\^)", text):
            if not chunk:
                continue
            if chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
                run = p.add_run(chunk[1:-1])
                run.italic = True
            elif chunk.startswith("^") and chunk.endswith("^") and len(chunk) > 2:
                run = p.add_run(chunk[1:-1])
                run.font.superscript = True
            else:
                run = p.add_run(chunk)
                run.italic = italic

    def equation(self, omml: str, number: int):
        p = self.doc.add_paragraph()
        p.paragraph_format.line_spacing = 2.0
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p._p.append(parse_xml(f"<m:oMath {M}>{omml}</m:oMath>"))
        p.add_run(f"\t({number})")
        return p

    def table(self, caption_bold: str, caption_rest: str, headers: list[str], rows: list[list[str]], notes: str | None = None):
        cap = self.para(caption_rest, bold_prefix=caption_bold + " ")
        tbl = self.doc.add_table(rows=1, cols=len(headers))
        tbl.style = "Table Grid"
        for i, h in enumerate(headers):
            cell = tbl.rows[0].cells[i]
            cell.text = ""
            cell.paragraphs[0].add_run(h).bold = True
        for row in rows:
            cells = tbl.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)
        for row in tbl.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    for run in p.runs:
                        run.font.size = Pt(9)
        if notes:
            self.para(notes)
        return cap

    def save(self, path):
        self.doc.save(path)
