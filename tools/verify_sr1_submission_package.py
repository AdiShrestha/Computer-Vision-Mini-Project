#!/usr/bin/env python3
"""Verify the corrected, result-bound SR-1 manuscript package."""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from pathlib import Path
from docx import Document

ROOT=Path(__file__).resolve().parents[1]
MAN=ROOT/'docs/manuscript'
PKT=ROOT/'docs/tables/sr1_final/result_bindings.json'
OUT=ROOT/'project/remediation/2026-10-27/third_party_review_001/SUBMISSION_VERIFICATION_013.json'
DOCS=['ICT_CEEL_2027_Abstract.docx','ICT_CEEL_2027_Extended_Abstract.docx','ICT_CEEL_2027_Fullpaper.docx']
PDFS=['sentinel_gl_ictceel2027.pdf','sentinel_gl_submission.pdf']
FIG=ROOT/'docs/figures/sr1_final_006'
VISUAL_QA=ROOT/'project/remediation/2026-10-27/third_party_review_001/MANUSCRIPT_VISUAL_QA_005/visual_qa.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def doc_text(doc):
    parts=[p.text for p in doc.paragraphs]
    parts.extend(c.text for t in doc.tables for row in t.rows for c in row.cells)
    return '\n'.join(parts)
def first_author_and_year(reference):
    m=re.match(r'\s*([^.,]+).*?\((\d{4})\)',reference)
    if not m: raise AssertionError(f'cannot parse reference: {reference}')
    return m.group(1).strip(),m.group(2)
def ensure(condition,message):
    if not condition: raise AssertionError(message)

def main():
    report={'schema_version':1,'status':'UNKNOWN','artifact':'LOCAL_TECHNICAL_MANUSCRIPT_BUILD','checks':{},'errors':[]}
    try:
        packet=json.loads(PKT.read_text())
        for relative,digest in packet['inputs'].items():
            path=ROOT/relative
            ensure(path.is_file() and sha(path)==digest,f'input binding mismatch: {relative}')
        report['checks']['bound_inputs']=len(packet['inputs'])
        names=DOCS+PDFS
        for name in names: ensure((MAN/name).is_file(),f'missing deliverable: {name}')
        for name in DOCS: ensure(packet['outputs'].get(name)==sha(MAN/name),f'packet output hash mismatch: {name}')
        report['checks']['docx_pdf_files']='PASS'
        ensure(sha(MAN/PDFS[0])==sha(MAN/PDFS[1]),'paired PDF outputs differ')
        report['checks']['paired_pdfs_identical']='PASS'
        docs={name:Document(MAN/name) for name in DOCS}
        texts={name:doc_text(doc) for name,doc in docs.items()}
        abstract=texts[DOCS[0]]
        # Count the abstract paragraph separately from the rest of its DOCX.
        absdoc=docs[DOCS[0]]
        abstract_par=next((p.text for p in absdoc.paragraphs if p.text.startswith('Cloud, snow and irregular revisits')),None)
        ensure(abstract_par is not None,'abstract paragraph missing')
        wc=len(abstract_par.split())
        ensure(150<=wc<=200,f'abstract word count outside 150–200: {wc}')
        ensure('0.8498 (SD 0.2371)' in abstract_par and '350 target-bearing' in abstract_par and '80 no-target' in abstract_par,'primary validation result or support counts absent from abstract')
        ensure('Six of eight' in abstract_par and 'seven prior FIT scenes per site' in abstract_par and 'three/four EVAL scenes' in abstract_par,'matched-season result or sample sizes absent')
        ensure('0.121/0.072' in abstract_par and '0.189/0.231' in abstract_par and 'FIT n=7/5' in abstract_par and 'EVAL n=3/4' in abstract_par,'available-scene zone contrasts or their support counts absent')
        ensure('0.410→0.539' in abstract_par and '1.144→1.074' in abstract_par and '1.736→1.932' in abstract_par,'site-specific matched-autumn medians absent from abstract')
        ensure('207,685-parameter' in abstract_par and 'event timing, false-alert burden and warning value remain unestimated' in abstract_par,'model description or inference limits absent from abstract')
        report['checks']['abstract_words']=wc
        full=docs['ICT_CEEL_2027_Fullpaper.docx']; fulltext=texts['ICT_CEEL_2027_Fullpaper.docx']
        ext=docs['ICT_CEEL_2027_Extended_Abstract.docx']; exttext=texts['ICT_CEEL_2027_Extended_Abstract.docx']
        required=['1. Introduction','2. Literature','3. Data and Methods','3.1 Study Regions and 2018–2023 Sentinel-2 Corpus','3.2 Patch Radiometry, Denominator Guard, and Candidate Zones','3.3 Missingness-Aware Factorized T-MAE and 50-Fit Matrix','3.4 Anomaly Scores and Baseline Definitions','4. Results','5. Interpretation and Limitations','6. Conclusion','Acknowledgement','References']
        for label in required: ensure(label in fulltext,f'missing required section label: {label}')
        for i in range(1,6): ensure(f'Figure {i}.' in fulltext,f'numbered figure caption missing: {i}')
        for i in range(1,6): ensure(f'Figure {i}.' in exttext,f'extended abstract numbered figure caption missing: {i}')
        img_count=sum(1 for rel in full.part.rels.values() if rel.reltype.endswith('/image'))
        ensure(img_count==5,f'full-paper embedded figure count is {img_count}, expected 5')
        report['checks']['required_hierarchy_and_figures']='PASS'
        ensure('half of the observed patches are sampled at random for masking' in fulltext and 'validation masks every other observed patch' in fulltext,'training/validation masking procedure is not explicit')
        ensure('different target spaces and scales' in fulltext and 'not directly comparable' in fulltext,'channel-count MSE caveat is missing')
        ensure('2022 VAL falls between FIT periods' in fulltext and 'rather than forward-time prediction' in fulltext and '27.988333° N' in fulltext and '88.815° E' in fulltext,'split or current reference-coordinate disclosure is missing')
        ensure('seasonal transfer' in fulltext.casefold() and 'false-alert burden per event-free lake-year' in fulltext.casefold() and 'incremental warning value' in fulltext.casefold(),'scientific quantities are not defined in plain language')
        ensure('1.399 (9 October)' in fulltext and '0.539 (26 October)' in fulltext and '0.528 (13 November)' in fulltext,'scene-level trajectory interpretation is missing')
        ensure('0.121 at South Lhonak (seven prior FIT scenes)' in fulltext and '0.072 at Khangchung (five)' in fulltext and '0.189 and 0.231' in fulltext,'site-specific autumn zone contrast or available-scene counts are missing')
        equation_tables=[t for t in full.tables if len(t.rows)==1 and len(t.columns)==3 and t.rows[0].cells[2].text.strip() in {'(1)','(2)','(3)','(4)','(5)'}]
        ensure(len(equation_tables)==5,'five numbered borderless equation rows with centered formula cells are required')
        symbols=next((t for t in full.tables if len(t.rows)>=8 and t.rows[0].cells[0].text=='Symbol'),None)
        ensure(symbols is not None and 'x̃ₜ,ₚ,𝚌' in '\n'.join(c.text for row in symbols.rows for c in row.cells) and 'PCAFIT' in '\n'.join(c.text for row in symbols.rows for c in row.cells),'corrected Table 1 symbols are missing')
        for internal in ['source-bound','receipt-bound','frozen v4','research-readiness gate','The present package records']:
            ensure(internal.casefold() not in fulltext.casefold(),f'internal workflow phrase remains in manuscript: {internal}')
        ensure(not re.search(r'\bE[1-5]\b',fulltext),'opaque E1-E5 labels remain in manuscript')
        report['checks']['review_findings_and_plain_language']='PASS'
        # Citation/reference parity in the body of both DOCX artifacts.
        for label,text in [('fullpaper',fulltext),('extended',exttext)]:
            ensure('References' in text,f'{label}: references section missing')
            body=text.split('References',1)[0]
            ref_lines=text.split('References',1)[1].splitlines()
            refs=[line.strip() for line in ref_lines if re.match(r'^[^\n].*\(\d{4}\)\.',line.strip())]
            ensure(len(refs)>=30,f'{label}: incomplete bibliography ({len(refs)} entries)')
            missing=[]
            for ref in refs:
                author,year=first_author_and_year(ref)
                if author not in body or year not in body: missing.append(f'{author} ({year})')
            ensure(not missing,f'{label}: bibliography entries not cited in body: {missing}')
            report['checks'][label+'_citation_parity']={'references':len(refs),'uncited':0}
        from pypdf import PdfReader
        reader=PdfReader(str(MAN/PDFS[0]));pdftext='\n'.join(page.extract_text() or '' for page in reader.pages)
        ref_block=pdftext.split('References',1)[1]
        pdfbody=re.sub(r'\s+',' ',pdftext.split('References',1)[0])
        pdfrefs=[line.strip() for line in ref_block.splitlines() if re.match(r'^[^\n].*\(\d{4}\)\.',line.strip())]
        pdf_missing=[]
        for ref in pdfrefs:
            author,year=first_author_and_year(re.sub(r'\s+',' ',ref))
            if author not in pdfbody or year not in pdfbody: pdf_missing.append(f'{author} ({year})')
        ensure(len(pdfrefs)>=30 and not pdf_missing,f'PDF citation parity failed: {pdf_missing}')
        report['checks']['pdf_citation_parity']={'references':len(pdfrefs),'uncited':0}
        # Table 4 must contain all six site/period cells, plus RPCA rank-2 sensitivity.
        t4=next((t for t in full.tables if len(t.rows)>1 and 'SL FIT/VAL' in ' '.join(c.text for c in t.rows[0].cells)),None)
        ensure(t4 is not None,'site-separated Table 4 not found')
        headers=' '.join(c.text for c in t4.rows[0].cells)
        for token in ['SL FIT/VAL','SL Autumn 18–21','SL EVAL','KT FIT/VAL','KT Autumn 18–21','KT EVAL']:
            ensure(token in headers,f'Table 4 column missing: {token}')
        ensure('Post-hoc rank-2 SVD residual' in '\n'.join(c.text for row in t4.rows for c in row.cells),'rank-2 residual missing from Table 4')
        report['checks']['table4_site_and_autumn_columns']='PASS'
        ensure('Figure 5. Site-separated' in fulltext and 'each autumn point represents one scene after seed aggregation' in fulltext,'Figure 5 caption does not identify scene-level points')
        ensure((FIG/'fig5_zone_and_autumn_contrasts.png').is_file(),'Figure 5 rendered PNG missing')
        from PIL import Image, ImageStat
        im=Image.open(FIG/'fig5_zone_and_autumn_contrasts.png').convert('RGB')
        ensure(sum(ImageStat.Stat(im).var)>1000,'Figure 5 appears blank')
        report['checks']['figure5_scene_level_data_panels']='PASS'
        # Autumn table cells must count unique acquisitions, not seed-scene rows.
        t4text='\n'.join(c.text for row in t4.rows for c in row.cells)
        ensure('n=7 scenes' in t4text and 'n=3 scenes' in t4text and 'n=4 scenes' in t4text,'Table 4 lacks expected unique-scene counts')
        ensure('prior FIT scenes were used for training' in fulltext and 'FIT scenes are in-sample' in abstract_par,'in-sample versus held-out limitation missing')
        report['checks']['table4_scene_unit_and_sample_sizes']='PASS'
        pages=len(reader.pages)
        ensure(pages<=8,f'full-paper PDF exceeds eight pages: {pages}')
        report['checks']['pdf_pages']=pages
        qa=json.loads(VISUAL_QA.read_text())
        ensure(qa.get('status')=='PASS_VISUAL_REVIEW','visual QA record is not passing')
        qc=qa.get('page_counts',{})
        ensure(qc.get('abstract_docx_pages')==1 and qc.get('extended_abstract_docx_pages')==6 and qc.get('fullpaper_docx_pages')==pages and qc.get('fullpaper_pdf_pages')==pages and qc.get('fullpaper_pdf_pair_identical') is True and qc.get('embedded_figures')==5,'visual QA counts do not match final files')
        ensure(len(qa.get('rendered_png_sha256',{}))==1+6+pages+5,'visual QA rendered page/figure inventory is incomplete')
        report['checks']['visual_review']='PASS_VISUAL_REVIEW'
        report['checks']['visual_qa_record']={'path':str(VISUAL_QA.relative_to(ROOT)),'rendered_pngs':1+6+pages+5,'docx_pages':[1,6,pages]}
        report['outputs']={name:{'sha256':sha(MAN/name),'size_bytes':(MAN/name).stat().st_size} for name in names}
        report['outputs']['abstract_word_count']=wc
        report['status']='PASS_LOCAL_MANUSCRIPT_BUILD_RESEARCH_GATES_OPEN'
    except Exception as exc:
        report['status']='FAIL';report['errors'].append(f'{type(exc).__name__}: {exc}')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    ensure_not=OUT.exists()
    if ensure_not: raise SystemExit(f'refusing to overwrite verification record: {OUT}')
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'record':str(OUT),'checks':report['checks'],'errors':report['errors']},indent=2))
    return 0 if report['status'].startswith('PASS') else 1

if __name__=='__main__': sys.exit(main())
