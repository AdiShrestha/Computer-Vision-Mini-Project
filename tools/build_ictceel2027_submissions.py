#!/usr/bin/env python3
"""Generate polished, template-compliant ICT-CEEL 2027 .docx files (Abstract,
Extended Abstract, Full Paper) and <=8-page A4 10.5pt Author-Year PDF."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from docx import Document
from docx.shared import Mm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = ROOT / "docs/manuscript"
FIG_DIR = ROOT / "docs/figures/2023"
BINDINGS = json.loads((ROOT / "docs/tables/2023/result_bindings.json").read_text(encoding="utf-8"))
TRAIN = json.loads((ROOT / "results/2023/training/training_report.v2.json").read_text(encoding="utf-8"))

TITLE = "Sentinel-GL: Dependency-Safe Multisensor Masked Autoencoding and Retrospective Anomaly Diagnostics for Himalayan Glacial Lakes"
KEYWORDS = "Glacial lakes, Sentinel-1, Sentinel-2, masked autoencoders, anomaly detection"

ABSTRACT_TEXT = (
    "Reliable satellite anomaly analysis for glacial lakes requires separating measurement support, "
    "deterministic feature dependencies, and developmental model fitting. This study investigates what "
    "source-supported reconstruction and latent anomaly trajectories emerge over two Himalayan glacial lake "
    "regions—South Lhonak and Khangchung Tsho—during the 2023 cascade season when derived-index leakage, "
    "fitting support, and sensor missingness are strictly controlled. We assemble 48 Sentinel-2 L2A optical "
    "observations spanning January–November 2023 across 20-m EPSG:32645 query grids alongside one shared "
    "September 2023 Sentinel-1 VV/VH SAR acquisition. A 1,062,919-parameter temporal masked autoencoder (T-MAE) "
    "is trained across five seeds using four-fold quadrant cross-masking, optical band-index dependency closure, "
    "and parity-0 pre-breach Welford normalization, and is compared against Isolation Forest, One-Class SVM, "
    "sensor-stratified Robust PCA, and CUSUM. Across all 188 genuinely observed query quadrants per seed, "
    "South Lhonak's median masked reconstruction error changes from 0.0477 pre-breach to 0.0442 post-breach, "
    "whereas latent embedding distance rises from 1.0086 to 3.1894 at South Lhonak and 1.3871 to 2.9591 at "
    "Khangchung Tsho, with One-Class SVM and CUSUM also increasing post-event. Optical Robust PCA converges in "
    "413 iterations at rank 3. These findings characterize retrospective surface-anomaly diagnostics and "
    "demonstrate that ROI-wide anomaly scores require shoreline isolation and multi-year exposure before "
    "operational GLOF warning use."
)

ACKNOWLEDGEMENT_TEXT = (
    "This research was conducted independently at the Department of Computer Science and Engineering (DoCSE), "
    "School of Engineering, Kathmandu University, without institutional or external grant funding. Copernicus "
    "Sentinel-1 and Sentinel-2 data were accessed via the Copernicus Data Space Ecosystem and Element 84 Earth "
    "Search. AI tools assisted with coding, verification scripting, and manuscript drafting under the author's "
    "architectural direction; Aditya Shrestha retains sole responsibility for all technical decisions, verification, "
    "and content."
)

REFERENCES_APA = [
    "Candès, E.J., Li, X., Ma, Y. & Wright, J. (2009). Robust principal component analysis? arXiv:0912.3599.",
    "Davison, A.C. & Smith, R.L. (1990). Models for exceedances over high thresholds. Journal of the Royal Statistical Society: Series B, 52(3), 393-425.",
    "Drusch, M., Del Bello, U., Carlier, S. et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. Remote Sensing of Environment, 120, 25-36.",
    "He, K., Chen, X., Xie, S., Li, Y., Dollár, P. & Girshick, R. (2022). Masked autoencoders are scalable vision learners. Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 16000-16009.",
    "Lin, Z., Chen, M. & Ma, Y. (2010). The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices. arXiv:1009.5055.",
    "Liu, F.T., Ting, K.M. & Zhou, Z.H. (2008). Isolation forest. Proc. of 8th IEEE International Conf. on Data Mining, 413-422.",
    "Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. Proc. of International Conf. on Learning Representations.",
    "McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing, 17(7), 1425-1432.",
    "Page, E.S. (1954). Continuous inspection schemes. Biometrika, 41(1-2), 100-115.",
    "Pickands, J. (1975). Statistical inference using extreme order statistics. The Annals of Statistics, 3(1), 119-131.",
    "Sarfraz, M.S., Chen, M.Y., Layer, L., Peng, K. & Koulakis, M. (2024). Position: Quo vadis, unsupervised time series anomaly detection? Proc. of 41st International Conf. on Machine Learning, 235, 43461-43476.",
    "Sattar, A., Cook, K.L., Rai, S.K. et al. (2025). The Sikkim flood of October 2023: Drivers, causes, and impacts of a multihazard cascade. Science, 387, eads2659.",
    "Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J. & Williamson, R.C. (2001). Estimating the support of a high-dimensional distribution. Neural Computation, 13(7), 1443-1471.",
    "Shugar, D.H., Burr, A., Haritashya, U.K. et al. (2020). Rapid worldwide growth of glacial lakes since 1990. Nature Climate Change, 10, 939-945.",
    "Taylor, C., Robinson, T.R., Dunning, S., Carr, J.R. & Westoby, M. (2023). Glacial lake outburst floods threaten millions globally. Nature Communications, 14, 487.",
    "Torres, R., Snoeij, P., Geudtner, D. et al. (2012). GMES Sentinel-1 mission. Remote Sensing of Environment, 120, 9-24.",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "Veh, G., Korup, O., Roessner, S. & Walz, A. (2018). Detecting Himalayan glacial lake outburst floods from Landsat time series. Remote Sensing of Environment, 207, 84-97.",
    "Welford, B.P. (1962). Note on a method for calculating corrected sums of squares and products. Technometrics, 4(3), 419-420.",
    "Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. International Journal of Remote Sensing, 27(14), 3025-3033.",
    "Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A. & Eickhoff, C. (2021). A transformer-based framework for multivariate time series representation learning. Proc. of 27th ACM SIGKDD Conf. on Knowledge Discovery and Data Mining, 2114-2124.",
    "Zheng, G., Allen, S.K., Bao, A. et al. (2021). Increasing risk of glacial lake outburst floods from future Third Pole deglaciation. Nature Climate Change, 11, 411-417.",
]


def setup_a4_margins_and_style(doc: Document) -> None:
    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.top_margin = Mm(25)
        section.bottom_margin = Mm(25)
        section.left_margin = Mm(20)
        section.right_margin = Mm(20)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(0)
    style.paragraph_format.line_spacing = 1.12


def add_header_block(doc: Document, include_corr_box: bool = True) -> None:
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(10.5)
    r_title = p_title.add_run(TITLE)
    r_title.bold = True
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(12)

    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(10.5)
    r_a = p_auth.add_run("Aditya Shrestha")
    r_a.bold = True
    r_a.font.size = Pt(10.5)
    r_sup = p_auth.add_run("1*" if include_corr_box else "1")
    r_sup.font.superscript = True
    r_sup.bold = True

    p_aff = doc.add_paragraph()
    p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_aff.paragraph_format.space_after = Pt(10.5)
    r_s = p_aff.add_run("1")
    r_s.font.superscript = True
    r_s.italic = True
    r_af = p_aff.add_run(
        "Department of Computer Science and Engineering (DoCSE), School of Engineering, "
        "Kathmandu University, Dhulikhel, Nepal"
    )
    r_af.italic = True
    r_af.font.size = Pt(10.5)

    p_abs_h = doc.add_paragraph()
    p_abs_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_abs_h.paragraph_format.space_after = Pt(3)
    r_ah = p_abs_h.add_run("Abstract")
    r_ah.bold = True
    r_ah.font.size = Pt(10.5)

    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_after = Pt(6)
    r_ab = p_abs.add_run(ABSTRACT_TEXT)
    r_ab.font.size = Pt(10.5)

    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.space_after = Pt(10.5)
    r_kw1 = p_kw.add_run("Keywords: ")
    r_kw1.bold = True
    r_kw1.italic = True
    r_kw2 = p_kw.add_run(KEYWORDS)
    r_kw2.font.size = Pt(10.5)

    if include_corr_box:
        corr_lines = [
            "*Corresponding author: Aditya Shrestha",
            "Department of Computer Science and Engineering (DoCSE), School of Engineering, Kathmandu University, Dhulikhel, Nepal",
            "Email: adityashrestha39@gmail.com (Institutional: as03233622@student.ku.edu.np)",
            "(Received:                    Accepted: to be filled by the Editorial Board)",
        ]
        for idx, line in enumerate(corr_lines):
            p_c = doc.add_paragraph()
            p_c.paragraph_format.space_after = Pt(8 if idx == len(corr_lines) - 1 else 1)
            r_c = p_c.add_run(line)
            r_c.font.size = Pt(9.5)
            if idx == len(corr_lines) - 1:
                r_c.italic = True


def add_section_heading(doc: Document, text: str, space_before: float = 12.0) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10.5)


def add_body_para(doc: Document, text: str, indent: bool = True) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Mm(6.3)
    r = p.add_run(text)
    r.font.size = Pt(10.5)


def add_equation_para(doc: Document, formula_text: str, eq_num: int) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(5)
    tabs = p.paragraph_format.tab_stops
    tabs.add_tab_stop(Mm(85), WD_TAB_ALIGNMENT.CENTER)
    tabs.add_tab_stop(Mm(170), WD_TAB_ALIGNMENT.RIGHT)
    r_lead = p.add_run("\t")
    r_eq = p.add_run(formula_text)
    r_eq.italic = True
    r_eq.font.size = Pt(10.5)
    r_num = p.add_run(f"\t({eq_num})")
    r_num.font.size = Pt(10.5)


def add_figure_block(doc: Document, png_name: str, caption: str, width_mm: float = 160.0) -> None:
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(18)
    p_img.paragraph_format.space_after = Pt(4)
    run = p_img.add_run()
    run.add_picture(str(FIG_DIR / png_name), width=Mm(width_mm))
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap.paragraph_format.space_after = Pt(18)
    r_c = p_cap.add_run(caption)
    r_c.font.size = Pt(9.5)


def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge, val in kwargs.items():
        edge_el = tcBorders.find(qn(f"w:{edge}"))
        if edge_el is None:
            edge_el = OxmlElement(f"w:{edge}")
            tcBorders.append(edge_el)
        edge_el.set(qn("w:val"), val.get("val", "single"))
        edge_el.set(qn("w:sz"), str(val.get("sz", 6)))
        edge_el.set(qn("w:space"), "0")
        edge_el.set(qn("w:color"), val.get("color", "000000"))


def add_styled_table(doc: Document, caption: str, headers: list[str], rows: list[list[str]]) -> None:
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(10)
    p_cap.paragraph_format.space_after = Pt(3)
    r_c = p_cap.add_run(caption)
    r_c.bold = True
    r_c.font.size = Pt(10)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        set_cell_border(cell, top={"sz": 8}, bottom={"sz": 6})
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)
    for i, row in enumerate(rows, start=1):
        is_last = i == len(rows)
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = val
            if is_last:
                set_cell_border(cell, bottom={"sz": 8})
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.size = Pt(9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def add_ack_and_references(doc: Document, cited_subset: list[str] | None = None) -> None:
    add_section_heading(doc, "Acknowledgement")
    add_body_para(doc, ACKNOWLEDGEMENT_TEXT, indent=False)
    add_section_heading(doc, "References")
    for ref in (cited_subset or REFERENCES_APA):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Mm(6.3)
        p.paragraph_format.first_line_indent = Mm(-6.3)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(ref)
        r.font.size = Pt(9.5)


def get_table3_rows() -> list[list[str]]:
    stats = {(r["site_id"], r["method"]): r for r in BINDINGS["pre_post_descriptive"]}
    methods = [
        ("tmae_score_a", "T-MAE A (MSE)"),
        ("tmae_score_b", "T-MAE B (Latent)"),
        ("tmae_score_c", "T-MAE C (Fused)"),
        ("isolation_forest", "Isolation Forest"),
        ("psd_rbf_one_class_svm", "One-Class SVM"),
        ("inexact_alm_rpca", "RPCA residual"),
        ("cusum", "CUSUM"),
    ]
    rows = []
    for m_id, label in methods:
        sl = stats[("PILOT-SOUTH-LHONAK", m_id)]
        kt = stats[("PILOT-KHANGCHUNG-TSHO", m_id)]
        rows.append([
            label,
            f"{sl['pre_median']:.4f}",
            f"{sl['post_median']:.4f}",
            f"{kt['pre_median']:.4f}",
            f"{kt['post_median']:.4f}",
        ])
    return rows


def build_abstract_docx() -> Path:
    doc = Document()
    setup_a4_margins_and_style(doc)
    add_header_block(doc, include_corr_box=False)
    subset = [
        r for r in REFERENCES_APA
        if any(k in r for k in ("Sattar", "Shugar", "Veh", "He, K.", "Sarfraz", "Candès"))
    ]
    add_ack_and_references(doc, cited_subset=subset)
    out = MANUSCRIPT_DIR / "ICT_CEEL_2027_Abstract.docx"
    doc.save(out)
    return out


def build_extended_abstract_docx() -> Path:
    doc = Document()
    setup_a4_margins_and_style(doc)
    add_header_block(doc, include_corr_box=True)

    add_section_heading(doc, "1. Introduction")
    add_body_para(
        doc,
        "Rapid glacial lake expansion across High Mountain Asia increases exposure to glacial lake outburst "
        "floods (GLOFs) (Shugar et al., 2020; Zheng et al., 2021; Taylor et al., 2023). The 3–4 October 2023 "
        "South Lhonak moraine breach in Sikkim demonstrated how a cascading hazard requires rigorous physical "
        "and statistical interpretation (Veh et al., 2018; Sattar et al., 2025). When applying self-supervised transformers to "
        "satellite time series, band-index algebraic dependencies, cloud-induced missingness, and spatial "
        "autocorrelation can create silent data leakage (Sarfraz et al., 2024). We evaluate a dependency-safe "
        "multisensor masked autoencoder (Sentinel-GL) over South Lhonak and Khangchung Tsho across January–November 2023.",
        indent=False,
    )

    add_section_heading(doc, "2. Methodology")
    add_body_para(
        doc,
        "Fixed 499 × 499 query grids (20-m pixel spacing in EPSG:32645; 99.6004 km²) are centered at South Lhonak "
        "(27.915°N, 88.204°E) and Khangchung Tsho (27.985°N, 88.752°E). Across 24 equal-duration temporal bins "
        "between 1 January and 15 November 2023, score-blind catalogue selection retains 24 Sentinel-2 L2A scenes "
        "per site (Drusch et al., 2012) plus one shared 1 September 2023 Sentinel-1 IW GRD VV/VH acquisition "
        "(Torres et al., 2012). Optical validity requires Scene Classification Layer (SCL) classes 4, 5, or 6, "
        "and NDWI (McFeeters, 1996) and MNDWI (Xu, 2006) are computed without epsilon clamping.",
        indent=False,
    )
    add_body_para(
        doc,
        "A 16-pixel checkerboard split assigns parity-0 pixels of pre-breach acquisitions (ending strictly before "
        "2 October 2023, 18:30 UTC) to fitting and parity-1 pixels to development monitoring. Streaming float64 "
        "Welford moments normalize each channel using parity-0 support only (Welford, 1962). Each acquisition is "
        "pooled into four spatial quadrant tokens. A 1,062,919-parameter Temporal Masked Autoencoder (Vaswani et al., "
        "2017; He et al., 2022; Zerveas et al., 2021) masks each quadrant in four separate folds while hiding all "
        "five optical channels at the targeted quadrant to prevent algebraic reconstruction of B03 from B08 and NDWI. "
        "Five seeds are optimized with AdamW (Loshchilov and Hutter, 2019) for up to 80 epochs. Reconstruction MSE "
        "(T-MAE A), eight-component PCA five-nearest-neighbor latent distance (T-MAE B), and fit-standardized fused "
        "scores (T-MAE C) are compared with Isolation Forest (Liu et al., 2008), PSD RBF One-Class SVM (Schölkopf "
        "et al., 2001), sensor-stratified ADMM Robust PCA (Candès et al., 2009; Lin et al., 2010), and CUSUM (Page, 1954), while noting that Peaks-Over-Threshold calibration (Pickands, 1975; Davison and Smith, 1990) requires multi-year exposure.",
        indent=True,
    )

    add_section_heading(doc, "3. Results")
    add_body_para(
        doc,
        "All 188 genuinely observed parity-1 quadrants per seed are scored across 49 unique provider products "
        "(950 method-seed-observation rows), with zero checkpoint replay error across all five seeds (selected monitor "
        "MSE 0.101839 to 0.215871). Optical Robust PCA converges in 413 iterations at rank 3 (normalized primal "
        "residual 9.9450 × 10⁻⁶; dual residual 1.4947 × 10⁻⁶).",
        indent=False,
    )
    add_styled_table(
        doc,
        "Table 1. Optical acquisition-level median anomaly scores across pre- and post-breach periods (SL: South Lhonak; KT: Khangchung Tsho).",
        ["Method", "SL pre (n=21)", "SL post (n=3)", "KT pre (n=19)", "KT post (n=4)"],
        get_table3_rows(),
    )
    add_figure_block(
        doc,
        "fig4_anomaly_trajectories.png",
        "Figure 1. Optical acquisition-level parity-1 anomaly score trajectories across 2023 for South Lhonak and Khangchung Tsho. Shaded vertical band marks the 3–4 October 2023 breach.",
        width_mm=160.0,
    )

    add_section_heading(doc, "4. Conclusion")
    add_body_para(
        doc,
        "Sentinel-GL establishes a dependency-safe, byte-auditable multisensor anomaly analysis workflow. While "
        "post-breach surface drainage is visually evident in Sentinel-2 imagery and increases latent distance, "
        "One-Class SVM, Isolation Forest, and CUSUM scores at South Lhonak, ROI-wide reconstruction error does not "
        "spike because surrounding terrain and seasonal snow/cloud support dominate the 99.6 km² grid. Shoreline-isolated "
        "masks, multi-date SAR time series, and multi-year exposure calibration are required before operational warning use.",
        indent=False,
    )
    add_ack_and_references(doc)
    out = MANUSCRIPT_DIR / "ICT_CEEL_2027_Extended_Abstract.docx"
    doc.save(out)
    return out


def build_fullpaper_docx() -> Path:
    doc = Document()
    setup_a4_margins_and_style(doc)
    add_header_block(doc, include_corr_box=True)

    add_section_heading(doc, "1. Introduction")
    add_body_para(
        doc,
        "Glacial lake growth and changing downstream exposure motivate satellite monitoring across High Mountain "
        "Asia (Shugar et al., 2020; Zheng et al., 2021; Taylor et al., 2023). The October 2023 South Lhonak disaster "
        "in Sikkim illustrates a cascading process whose hazard interpretation requires physical evidence beyond an "
        "unusual image score (Sattar et al., 2025). Retrospective satellite detection of lake changes is an established "
        "research direction (Veh et al., 2018); consequently, an anomaly detector's contribution cannot be justified "
        "solely by applying a transformer to a known event.",
        indent=False,
    )
    add_body_para(
        doc,
        "A central methodological difficulty is observational dependence. Optical water indices reuse the same spectral "
        "bands, spatial patches share landscapes and atmospheric conditions, and repeated optimization seeds share the "
        "same source corpus. Treating those dependencies as independent information can produce an apparently successful "
        "model without establishing warning skill. We investigate a bounded retrospective question: what source-supported "
        "reconstruction and fitted anomaly trajectories arise over South Lhonak and Khangchung Tsho during 2023 when "
        "derived-channel leakage, fitting support, sensor missingness, and temporal context are explicitly controlled?",
        indent=True,
    )

    add_section_heading(doc, "2. Literature")
    add_body_para(
        doc,
        "Optical multispectral monitoring with Sentinel-2 (Drusch et al., 2012) relies on spectral ratios such as "
        "NDWI (McFeeters, 1996) and MNDWI (Xu, 2006) to enhance open-water features, while C-band Synthetic Aperture "
        "Radar (SAR) from Sentinel-1 provides cloud-penetrating backscatter observations (Torres et al., 2012). In "
        "multivariate time-series representation learning, transformer encoders (Vaswani et al., 2017; Zerveas et al., "
        "2021) and masked autoencoders (He et al., 2022) reconstruct masked tokens from unmasked context. However, "
        "recent critical reviews of unsupervised time-series anomaly detection demonstrate that complex deep models "
        "frequently fail to outperform simple statistical baselines such as Isolation Forest (Liu et al., 2008), "
        "One-Class SVM (Schölkopf et al., 2001), Robust PCA (Candès et al., 2009; Lin et al., 2010), and CUSUM "
        "(Page, 1954) when evaluation protocols suffer from threshold or feature leakage (Sarfraz et al., 2024). "
        "Extreme-value threshold calibration via Peaks-Over-Threshold (Pickands, 1975; Davison and Smith, 1990) "
        "further requires verified multi-year exposure.",
        indent=False,
    )

    add_section_heading(doc, "3. Methods and Procedures")
    add_section_heading(doc, "3.1 Spatial Design and 2023 Observation Corpus", space_before=6)
    add_body_para(
        doc,
        "We fix query centroids at (27.915°N, 88.204°E) for South Lhonak and (27.985°N, 88.752°E) for Khangchung Tsho. "
        "Each region uses an integer-aligned 499 × 499 grid in EPSG:32645 with 20-m pixels (9.98 km side length; "
        "99.6004 km² area). Within 24 equal-duration bins across 1 January–15 November 2023, score-blind catalogue "
        "selection retains the lowest-cloud Sentinel-2 L2A granule with full projected ROI coverage, yielding 48 optical "
        "observations (24 per site) plus one shared 1 September 2023 ascending Sentinel-1 IW GRD VV/VH product (Fig. 1).",
        indent=False,
    )
    add_figure_block(
        doc,
        "fig1_observation_timeline.png",
        "Figure 1. Source-supported 2023 observation sequence across South Lhonak and Khangchung Tsho. Optical points show the fraction of the fixed ROI assigned SCL 4/5/6; diamonds mark the 1 September SAR acquisition.",
        width_mm=160.0,
    )

    add_section_heading(doc, "3.2 Optical Radiometry and Native SAR Calibration", space_before=6)
    add_body_para(
        doc,
        "Bands B03, B08, B11, and SCL are extracted via HTTP byte-range reads verified against native XML quantification "
        "values (Q = 10000) and additive offsets. B03 and B08 are reduced from 10 m to 20 m by exact 2 × 2 arithmetic "
        "means requiring all four valid source pixels. Optical validity requires SCL classes 4, 5, or 6. NDWI and MNDWI "
        "are computed in float64 before float32 serialization:",
        indent=False,
    )
    add_equation_para(doc, "I_N = (ρ_3 − ρ_8) / (ρ_3 + ρ_8),    I_M = (ρ_3 − ρ_11) / (ρ_3 + ρ_11)", 1)
    add_body_para(
        doc,
        "For Sentinel-1 IW GRD digital numbers d, interpolated sigma-nought LUT A_σ, and range/azimuth thermal "
        "noise terms N_R and N_A, we compute raw linear power and signed noise-subtracted power on bounded native "
        "radar windows before valid-weight bilinear resampling to the target grid (Fig. 2):",
        indent=True,
    )
    add_equation_para(doc, "σ_r⁰ = d² / A_σ²,    NESZ = (N_R N_A) / A_σ²,    σ_s⁰ = (d² − N_R N_A) / A_σ²", 2)
    add_figure_block(
        doc,
        "fig2_source_distributions.png",
        "Figure 2. South Lhonak source diagnostics: (a,b) nearest pre/post optical false-color pairs (26 September vs. 9 October 2023); (c,d) ROI reflectance and index empirical CDFs; (e–h) 1 September SAR VV/VH distributions.",
        width_mm=160.0,
    )

    add_section_heading(doc, "3.3 Spatial Parity Firewall and Dependency-Safe T-MAE", space_before=6)
    add_body_para(
        doc,
        "For raster row r and column c, spatial checkerboard parity is defined as:",
        indent=False,
    )
    add_equation_para(doc, "P(r, c) = (⌊r / 16⌋ + ⌊c / 16⌋) mod 2", 3)
    add_body_para(
        doc,
        "Parity 0 supplies fitting pixels and parity 1 supplies development monitoring, restricted strictly to "
        "acquisitions ending before 2 October 2023, 18:30 UTC (3 October 00:00 NPT). Per-channel float64 Welford "
        "standardization (Welford, 1962) is fitted on parity-0 pre-breach pixels only. Each acquisition forms four "
        "quadrant tokens. In each of four evaluation folds (q = 0, 1, 2, 3), optical dependency closure hides all "
        "five optical channels of the target quadrant so that hidden bands cannot be algebraically recovered via:",
        indent=True,
    )
    add_equation_para(doc, "ρ_3 = ρ_8 (1 + I_N) / (1 − I_N)", 4)
    add_body_para(
        doc,
        "Five seeds are trained using AdamW (Loshchilov and Hutter, 2019) with learning rate 3e-4 and weight decay 1e-4 for up to 80 epochs. Five seeds are trained using AdamW (Loshchilov and Hutter, 2019) with learning rate 3e-4 and weight decay 1e-4 for up to 80 epochs. The T-MAE (1,062,919 parameters; d_model = 128, 4 heads, 4 encoder layers, 2 decoder layers) scores "
        "out-of-context quadrant reconstruction error A(t), eight-component PCA five-neighbor latent distance B(t) "
        "(excluding same-product and same-timestamp neighbors), and pre-breach parity-0 standardized fusion C(t):",
        indent=True,
    )
    add_equation_para(doc, "A(t) = ∑_{q, j ∈ T_q} (ẑ_{qj} − z_{qj})² / ∑_q |T_q|,    C(t) = ½(A − μ_A)/s_A + ½(B − μ_B)/s_B", 5)

    add_section_heading(doc, "4. Results")
    add_body_para(
        doc,
        "Retained optical clear fractions range from 1.27% to 66.46% at South Lhonak (median 41.17%) and 0% to 89.25% "
        "at Khangchung Tsho (median 45.50%) (Table 1). Across five training seeds (80-epoch budget, patience 12), "
        "every seed scores all 188 genuinely observed parity-1 quadrants (49 acquisition-level scores per seed), and "
        "all selected checkpoints replay with 0.0 absolute error (Table 2; Fig. 3).",
        indent=False,
    )
    add_styled_table(
        doc,
        "Table 1. Recorded 2023 observation support by region and sensor.",
        ["Region", "Optical pre-breach", "Optical post-breach", "SAR dates"],
        [["South Lhonak", "21", "3", "1"], ["Khangchung Tsho", "20 (19 usable)", "4", "1"]],
    )
    seed_rows = [
        [str(s["seed"]), str(s["epochs_executed"]), str(s["selected_epoch"]), f"{s['selected_monitor_mse']:.6f}"]
        for s in TRAIN["seeds"]
    ]
    add_styled_table(
        doc,
        "Table 2. Five-seed T-MAE selected checkpoint evidence on parity-1 pre-breach monitor support.",
        ["Seed", "Epochs run", "Selected epoch", "Monitor MSE"],
        seed_rows,
    )
    add_figure_block(
        doc,
        "fig3_optimization.png",
        "Figure 3. Five-seed T-MAE optimization histories on (a) parity-0 training pixels and (b) parity-1 development monitor pixels.",
        width_mm=160.0,
    )
    add_styled_table(
        doc,
        "Table 3. Optical acquisition-level median scores by method and period (SL: South Lhonak; KT: Khangchung Tsho).",
        ["Method", "SL pre", "SL post", "KT pre", "KT post"],
        get_table3_rows(),
    )
    add_body_para(
        doc,
        "As shown in Table 3 and Figure 4, South Lhonak's median reconstruction score (T-MAE A) changes from 0.0477 "
        "pre-breach to 0.0442 post-breach, whereas latent embedding distance (T-MAE B) increases from 1.0086 to 3.1894 "
        "at South Lhonak and from 1.3871 to 2.9591 at Khangchung Tsho (with elevated late-monsoon September medians of "
        "4.4797 and 2.9420). One-Class SVM (−0.1898 to 0.0394), Isolation Forest (0.4277 to 0.4890), and CUSUM "
        "(0.0000 to 0.2171) also increase post-event at South Lhonak. Sensor-stratified optical Robust PCA converges "
        "numerically in 413 iterations at rank 3 (normalized primal residual 9.9450 × 10⁻⁶; dual residual 1.4947 × 10⁻⁶). "
        "Representative seed-0 attention maps are shown in Figure 5.",
        indent=False,
    )
    add_figure_block(
        doc,
        "fig4_anomaly_trajectories.png",
        "Figure 4. Optical acquisition-level parity-1 trajectories across 2023 for all seven diagnostic methods. Shaded band marks the 3–4 October 2023 breach.",
        width_mm=160.0,
    )
    add_figure_block(
        doc,
        "fig5_attention.png",
        "Figure 5. Representative seed-0 encoder and decoder self-attention weights (layer 0, q0 fold) for the first post-breach South Lhonak optical query (9 October 2023).",
        width_mm=155.0,
    )

    add_section_heading(doc, "5. Conclusion")
    add_body_para(
        doc,
        "Sentinel-GL integrates multi-date 2023 Sentinel-2 optical imagery and native-calibrated Sentinel-1 SAR "
        "within a dependency-safe, byte-verified masked autoencoding framework. While physical lake drainage after "
        "the October 2023 South Lhonak breach is clear in optical imagery and elevates latent, SVM, Isolation Forest, "
        "and CUSUM trajectories, ROI-wide reconstruction error does not spike due to surrounding terrain and seasonal "
        "support variations. Future operational warning skill requires shoreline-specific polygon masks, multi-date "
        "terrain-corrected SAR time series, and multi-year false-alarm calibration.",
        indent=False,
    )
    add_ack_and_references(doc)
    out = MANUSCRIPT_DIR / "ICT_CEEL_2027_Fullpaper.docx"
    doc.save(out)
    return out


def build_ictceel2027_pdf() -> Path:
    src = (MANUSCRIPT_DIR / "sentinel_gl_submission.tex").read_text(encoding="utf-8")

    # 1. A4 paper with 25mm top/bottom, 20mm left/right margins per Fullpaper 2027.docx
    src = src.replace(
        r"\usepackage[letterpaper,margin=0.7in]{geometry}",
        r"\usepackage[a4paper,top=25mm,bottom=25mm,left=20mm,right=20mm]{geometry}"
    )

    # 2. Bold Arabic section hierarchy (1, 1.1) and compact vertical spacing so all 22 refs fit on <= 8 pages
    src = src.replace(
        "\\renewcommand{\\thesection}{\\Roman{section}}\n\\renewcommand{\\thesubsection}{\\Alph{subsection}}",
        "\\renewcommand{\\thesection}{\\arabic{section}}\n\\renewcommand{\\thesubsection}{\\thesection.\\arabic{subsection}}\n"
        "\\setlength{\\textfloatsep}{8pt plus 2pt minus 2pt}\n"
        "\\setlength{\\dbltextfloatsep}{8pt plus 2pt minus 2pt}\n"
        "\\setlength{\\intextsep}{6pt plus 2pt minus 2pt}\n"
        "\\captionsetup{skip=3pt}"
    )

    # 3. Title (12pt bold), single asterisk on author (Aditya Shrestha^{1*}), and Corresponding Author footnote box
    src = src.replace(
        r"\title{\large\bfseries Sentinel-GL: Dependency-Safe Multisensor Masked Autoencoding and Retrospective Anomaly Diagnostics for Himalayan Glacial Lakes}",
        r"\title{\vspace{-1.2em}\fontsize{12}{14}\selectfont\bfseries Sentinel-GL: Dependency-Safe Multisensor Masked Autoencoding and Retrospective Anomaly Diagnostics for Himalayan Glacial Lakes}"
    )
    src = src.replace(
        "\\author{\\normalsize Aditya Shrestha\\thanks{Institutional student email: \\href{mailto:as03233622@student.ku.edu.np}{\\nolinkurl{as03233622@student.ku.edu.np}}.}\\\\\n"
        "\\small Department of Computer Science and Engineering (DoCSE), School of Engineering\\\\\n"
        "\\small Kathmandu University, Dhulikhel, Nepal\\\\\n"
        "\\small Correspondence: \\href{mailto:adityashrestha39@gmail.com}{\\nolinkurl{adityashrestha39@gmail.com}}}",
        "\\author{\\fontsize{10.5}{12.5}\\selectfont\\bfseries Aditya Shrestha$^{1}$\\thanks{Corresponding author: Aditya Shrestha, Department of Computer Science and Engineering (DoCSE), School of Engineering, Kathmandu University, Dhulikhel, Nepal. Email: \\href{mailto:adityashrestha39@gmail.com}{\\nolinkurl{adityashrestha39@gmail.com}} (Institutional: \\href{mailto:as03233622@student.ku.edu.np}{\\nolinkurl{as03233622@student.ku.edu.np}}). (Received: \\hspace{1.0cm} Accepted: to be filled by the Editorial Board)}\\\\\n"
        "\\small $^1$Department of Computer Science and Engineering (DoCSE), School of Engineering,\\\\\n"
        "\\small Kathmandu University, Dhulikhel, Nepal}"
    )
    src = src.replace(
        r"\noindent\textbf{Index terms---}Glacial lakes, Sentinel-1, Sentinel-2, masked autoencoders, missing observations, retrospective anomaly analysis.",
        r"\noindent\textbf{\textit{Keywords:}} Glacial lakes, Sentinel-1, Sentinel-2, masked autoencoders, anomaly detection."
    )

    # 4. Add explicit Section 2 (Literature) and Author-Year citations matching Fullpaper 2027.docx
    src = src.replace(
        r"\section{Study Data and Observation Support}",
        "\\section{Literature}\n"
        "Optical multispectral monitoring with Sentinel-2 (Drusch et al., 2012) relies on spectral ratios such as NDWI (McFeeters, 1996) and MNDWI (Xu, 2006) to delineate open-water features, while C-band Synthetic Aperture Radar (SAR) from Sentinel-1 provides cloud-penetrating backscatter observations (Torres et al., 2012). In multivariate time-series representation learning, transformer encoders (Vaswani et al., 2017; Zerveas et al., 2021) and masked autoencoders (He et al., 2022) learn contextual representations by reconstructing masked inputs. However, critical reviews of unsupervised time-series anomaly detection caution that deep architectures require transparent comparisons against classical detectors such as Isolation Forest (Liu et al., 2008), One-Class SVM (Sch\\\"olkopf et al., 2001), Robust PCA (Cand\\`es et al., 2009; Lin et al., 2010), and CUSUM (Page, 1954), as well as extreme-value threshold calibration (Pickands, 1975; Davison and Smith, 1990) free of evaluation leakage (Sarfraz et al., 2024).\n\n"
        "\\section{Methods and Procedures}"
    )
    src = src.replace(
        "\\section{Dependency-Safe Modeling}\n\\subsection{Fitting support and asynchronous tokens}",
        "\\subsection{Fitting support and asynchronous tokens}"
    )

    # 5. Convert bracketed \cite{...} in body text to ICT-CEEL Author-Year format
    cite_replacements = {
        r"~\cite{shugar2020,zheng2021,taylor2023}": " (Shugar et al., 2020; Zheng et al., 2021; Taylor et al., 2023)",
        r"~\cite{sattar2025}": " (Sattar et al., 2025)",
        r"~\cite{veh2018}": " (Veh et al., 2018)",
        r"~\cite{sarfraz2024}": " (Sarfraz et al., 2024)",
        r"~\cite{drusch2012}": " (Drusch et al., 2012)",
        r"~\cite{mcfeeters1996}": " (McFeeters, 1996)",
        r"~\cite{xu2006}": " (Xu, 2006)",
        r"~\cite{torres2012}": " (Torres et al., 2012)",
        r"~\cite{welford1962}": " (Welford, 1962)",
        r"~\cite{vaswani2017}": " (Vaswani et al., 2017)",
        r"~\cite{he2022}": " (He et al., 2022)",
        r"~\cite{zerveas2021}": " (Zerveas et al., 2021)",
        r"~\cite{loshchilov2019}": " (Loshchilov and Hutter, 2019)",
        r"~\cite{liu2008}": " (Liu et al., 2008)",
        r"~\cite{scholkopf2001}": " (Sch\\\"olkopf et al., 2001)",
        r"~\cite{candes2009}": " (Cand\\`es et al., 2009)",
        r"~\cite{lin2010}": " (Lin et al., 2010)",
        r"~\cite{page1954}": " (Page, 1954)",
        r"~\cite{pickands1975,davison1990}": " (Pickands, 1975; Davison and Smith, 1990)",
    }
    for k, v in cite_replacements.items():
        src = src.replace(k, v)

    # 6. Replace Declarations + BibTeX with Acknowledgement + Alphabetical ICT-CEEL Author-Year References
    apa_items = [
        r"\item Cand\`es, E.J., Li, X., Ma, Y. \& Wright, J. (2009). Robust principal component analysis? \textit{arXiv:0912.3599}.",
        r"\item Davison, A.C. \& Smith, R.L. (1990). Models for exceedances over high thresholds. \textit{Journal of the Royal Statistical Society: Series B}, 52(3), 393--425.",
        r"\item Drusch, M., Del Bello, U., Carlier, S. \textit{et al.} (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. \textit{Remote Sensing of Environment}, 120, 25--36.",
        r"\item He, K., Chen, X., Xie, S., Li, Y., Doll\"ar, P. \& Girshick, R. (2022). Masked autoencoders are scalable vision learners. \textit{Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition}, 16000--16009.",
        r"\item Lin, Z., Chen, M. \& Ma, Y. (2010). The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices. \textit{arXiv:1009.5055}.",
        r"\item Liu, F.T., Ting, K.M. \& Zhou, Z.H. (2008). Isolation forest. \textit{Proc. of 8th IEEE International Conf. on Data Mining}, 413--422.",
        r"\item Loshchilov, I. \& Hutter, F. (2019). Decoupled weight decay regularization. \textit{Proc. of International Conf. on Learning Representations}.",
        r"\item McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. \textit{International Journal of Remote Sensing}, 17(7), 1425--1432.",
        r"\item Page, E.S. (1954). Continuous inspection schemes. \textit{Biometrika}, 41(1--2), 100--115.",
        r"\item Pickands, J. (1975). Statistical inference using extreme order statistics. \textit{The Annals of Statistics}, 3(1), 119--131.",
        r"\item Sarfraz, M.S., Chen, M.Y., Layer, L., Peng, K. \& Koulakis, M. (2024). Position: Quo vadis, unsupervised time series anomaly detection? \textit{Proc. of 41st International Conf. on Machine Learning}, 235, 43461--43476.",
        r"\item Sattar, A., Cook, K.L., Rai, S.K. \textit{et al.} (2025). The Sikkim flood of October 2023: Drivers, causes, and impacts of a multihazard cascade. \textit{Science}, 387, eads2659.",
        r"\item Sch\"olkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J. \& Williamson, R.C. (2001). Estimating the support of a high-dimensional distribution. \textit{Neural Computation}, 13(7), 1443--1471.",
        r"\item Shugar, D.H., Burr, A., Haritashya, U.K. \textit{et al.} (2020). Rapid worldwide growth of glacial lakes since 1990. \textit{Nature Climate Change}, 10, 939--945.",
        r"\item Taylor, C., Robinson, T.R., Dunning, S., Carr, J.R. \& Westoby, M. (2023). Glacial lake outburst floods threaten millions globally. \textit{Nature Communications}, 14, 487.",
        r"\item Torres, R., Snoeij, P., Geudtner, D. \textit{et al.} (2012). GMES Sentinel-1 mission. \textit{Remote Sensing of Environment}, 120, 9--24.",
        r"\item Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. \& Polosukhin, I. (2017). Attention is all you need. \textit{Advances in Neural Information Processing Systems}, 30.",
        r"\item Veh, G., Korup, O., Roessner, S. \& Walz, A. (2018). Detecting Himalayan glacial lake outburst floods from Landsat time series. \textit{Remote Sensing of Environment}, 207, 84--97.",
        r"\item Welford, B.P. (1962). Note on a method for calculating corrected sums of squares and products. \textit{Technometrics}, 4(3), 419--420.",
        r"\item Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. \textit{International Journal of Remote Sensing}, 27(14), 3025--3033.",
        r"\item Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A. \& Eickhoff, C. (2021). A transformer-based framework for multivariate time series representation learning. \textit{Proc. of 27th ACM SIGKDD Conf. on Knowledge Discovery and Data Mining}, 2114--2124.",
        r"\item Zheng, G., Allen, S.K., Bao, A. \textit{et al.} (2021). Increasing risk of glacial lake outburst floods from future Third Pole deglaciation. \textit{Nature Climate Change}, 11, 411--417.",
    ]
    ref_block = (
        "\\section*{Acknowledgement}\n"
        "The research was conducted independently without institutional or external grant funding. "
        "The institutional affiliation identifies the author's academic affiliation and does not imply "
        "institutional sponsorship or endorsement. AI tools assisted with coding, literature organization, "
        "verification scripting and manuscript drafting under the author's architectural direction. "
        "Aditya Shrestha retains sole responsibility for technical review, intellectual decisions, source "
        "verification, interpretation and the submitted content. No AI system is an author. No operational "
        "civil-defense use is claimed.\n\n"
        "\\section*{References}\n"
        "{\\small\\begin{list}{}{\\setlength{\\leftmargin}{1.4em}\\setlength{\\itemindent}{-1.4em}"
        "\\setlength{\\itemsep}{1.5pt}\\setlength{\\parsep}{0pt}\\setlength{\\topsep}{2pt}}\n"
        + "\n".join(apa_items)
        + "\n\\end{list}}\n\\end{document}"
    )
    src = src[:src.index(r"\section*{Declarations}")] + ref_block

    # 7. Set 10.5pt font size (with 12pt leading) per ICT-CEEL template so all content fits within 8 pages
    src = src.replace(
        "\\begin{document}\n\\raggedbottom",
        "\\begin{document}\n\\fontsize{10.5}{12.0}\\selectfont\n\\raggedbottom"
    )

    out_tex = MANUSCRIPT_DIR / "sentinel_gl_ictceel2027.tex"
    out_tex.write_text(src, encoding="utf-8")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "sentinel_gl_ictceel2027.tex"],
            cwd=MANUSCRIPT_DIR, check=True, stdout=subprocess.DEVNULL
        )
    return out_tex.with_suffix(".pdf")


if __name__ == "__main__":
    p1 = build_abstract_docx()
    p2 = build_extended_abstract_docx()
    p3 = build_fullpaper_docx()
    p4 = build_ictceel2027_pdf()
    log_text = (MANUSCRIPT_DIR / "sentinel_gl_ictceel2027.log").read_text(encoding="utf-8")
    page_line = next((line for line in log_text.splitlines() if "Output written on sentinel_gl_ictceel2027.pdf" in line), "")
    print(json.dumps({
        "abstract_docx": str(p1.relative_to(ROOT)),
        "extended_abstract_docx": str(p2.relative_to(ROOT)),
        "fullpaper_docx": str(p3.relative_to(ROOT)),
        "ictceel2027_pdf": str(p4.relative_to(ROOT)),
        "pdf_output_summary": page_line,
        "abstract_word_count": len(ABSTRACT_TEXT.split()),
    }, indent=2))
