#!/usr/bin/env python3
"""Rebuild the three SR-1 Word drafts and matched two PDF drafts from bound outputs."""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess
from pathlib import Path
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/manuscript'
FIG_DIR=ROOT/'docs/figures/sr1_true'
NAMES={'abstract':'ICT_CEEL_2027_Abstract.docx','extended':'ICT_CEEL_2027_Extended_Abstract.docx','fullpaper':'ICT_CEEL_2027_Fullpaper.docx'}
FIGS=[('fig1_observation_cadence','Figure 1'),('fig2_fit_only_candidate_zones','Figure 2'),
      ('fig3_reflectance_distributions','Figure 3')]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v): return 'not estimable' if v is None else f'{float(v):.4g}'
def role(value):
    from datetime import datetime,timezone
    d=datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone.utc)
    if 2018<=d.year<=2021:return 'FIT'
    if d.year==2022:return 'VAL'
    if d.year==2023 and d.month<=6:return 'FIT'
    if d.year==2023 and d.month<=9:return 'VAL'
    if d>=datetime.fromisoformat('2023-10-04T18:30:00+00:00'):return 'EVAL'
    if d>=datetime.fromisoformat('2023-10-02T18:30:00+00:00'):return 'EVENT_UNCERTAIN'
    return 'OUT'

def add_text(doc, text, *, bold=False, center=False, size=10.5, keep=False):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after=Pt(4);p.paragraph_format.keep_with_next=keep
    r=p.add_run(str(text));r.bold=bold;r.font.name='Times New Roman';r.font.size=Pt(size)
    return p

def add_heading(doc, text, level=1):
    p=doc.add_paragraph();p.style=f'Heading {level}';p.paragraph_format.space_before=Pt(8);p.paragraph_format.space_after=Pt(3);p.paragraph_format.keep_with_next=True
    r=p.add_run(text);r.font.name='Times New Roman';r.font.size=Pt(11 if level==1 else 10.5);r.bold=True

def setup_doc(title, abstract, keywords):
    d=Document();sec=d.sections[0];sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.7);sec.left_margin=Inches(.75);sec.right_margin=Inches(.75)
    normal=d.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10.5);normal.paragraph_format.line_spacing=1.08;normal.paragraph_format.space_after=Pt(3)
    for sn in ('Heading 1','Heading 2'):
        st=d.styles[sn];st.font.name='Times New Roman';st.font.color.rgb=None
    add_text(d,title,bold=True,center=True,size=14)
    add_text(d,'Aditya Shrestha¹*',bold=True,center=True,size=10.5)
    add_text(d,'¹ Department of Computer Science and Engineering (DoCSE), School of Engineering, Kathmandu University, Dhulikhel, Nepal',center=True,size=9)
    add_text(d,'* Correspondence: adityashrestha39@gmail.com; institutional email: as03233622@student.ku.edu.np',center=True,size=8.5)
    add_text(d,'Abstract',bold=True,center=True,size=10.5)
    add_text(d,abstract,size=9.5)
    add_text(d,'Keywords: '+keywords,size=9)
    return d

def add_table(doc, headers, rows, caption):
    add_text(doc,caption,bold=True,size=9.5,keep=True)
    t=doc.add_table(rows=1,cols=len(headers));t.style='Table Grid'
    for cell,value in zip(t.rows[0].cells,headers):cell.text=str(value)
    for row in rows:
        cells=t.add_row().cells
        for cell,value in zip(cells,row):cell.text=str(value)
    for ri,row in enumerate(t.rows):
        row_properties=row._tr.get_or_add_trPr()
        row_properties.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            # Repeat the table's column labels when Word paginates a long table.
            row_properties.append(OxmlElement('w:tblHeader'))
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(1)
                for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(8);r.bold=(ri==0)
    return t

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',default='attempt_006');ap.add_argument('--analysis',default='analysis_001')
    ap.add_argument('--compile-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,default=Path('docs/manuscript'))
    ap.add_argument('--figure-dir',type=Path,default=Path('docs/figures/sr1_true'))
    ap.add_argument('--result-packet',type=Path,default=Path('docs/tables/sr1_true/result_bindings.json'))
    a=ap.parse_args();manifest_path=ROOT/'data/derived/sr1_multiyear'/a.attempt/'observations.sr1.json'
    figure_dir=ROOT/a.figure_dir
    analysis_dir=manifest_path.parent/a.analysis;manifest=json.loads(manifest_path.read_text());report=json.loads((analysis_dir/'analysis_report.json').read_text())
    hist=json.loads((analysis_dir/'training_histories.json').read_text())
    protocol=ROOT/'project/sr1_protocol_true_v4.json';protocol_hash=sha(protocol)
    packet_path=ROOT/a.result_packet
    packet=json.loads(packet_path.read_text())
    if packet['protocol_sha256']!=sha(protocol):raise ValueError('Result packet protocol hash differs from the frozen protocol')
    sites=manifest['sites'];siteyear={}
    for r in manifest['observations']:
        if r.get('status')!='EXPORT_FAILED':siteyear[(r['site_id'],r['acquired_at'][:4])]=siteyear.get((r['site_id'],r['acquired_at'][:4]),0)+1
    unique_by_role={k:len(v) for k,v in report['split'].items() if k.endswith('_ids')}
    seed_rows=[]
    reexec_path=ROOT/'data/derived/sr1_multiyear/attempt_006/reexecution_001/run_manifest.json'
    reexec=json.loads(reexec_path.read_text())
    if reexec.get('status')!='PASS_COMPLETE_VAL' or reexec.get('jobs_completed')!=50 or reexec.get('jobs_failed')!=0 or reexec.get('eval_ids_accessed'):
        raise ValueError('corrective FIT/VAL re-execution is incomplete or accessed EVAL')
    for item in reexec['jobs']:
        if item.get('family')=='primary':
            seed_rows.append([item['seed'],item['epochs_completed'],item['best_epoch'],fmt(item['best_val_loss']),item['optimizer_steps'],('epoch budget reached' if item['stopping_status']=='MAX_EPOCH_BUDGET_REACHED' else 'validation plateau')])
    seed_rows.sort(key=lambda row:row[0])
    if len(seed_rows)!=5: raise ValueError('expected five corrected primary runs')
    summary_path=ROOT/'project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_SENSITIVITY_SUMMARY_001.json'
    corrected_summary=json.loads(summary_path.read_text())
    config_rows=[]
    for cell in corrected_summary['cell_summaries']:
        if cell['family']=='capacity': label=f"Capacity, d={cell['cell'].split('_')[-1]}"
        elif cell['family']=='primary': label='Primary, 5 channels, FIT zones, context 8'
        else: label=f"{cell['channels']} channels, {'FIT' if cell['zones_enabled'] else 'constant'} zones, context {cell['context_scenes_max']}"
        config_rows.append([label,f"{cell['mean_best_val_masked_mse']:.4f} ± {cell['sample_sd']:.4f}",f"{cell['target_bearing_val_blocks_total']} / {cell['no_target_val_blocks_total']}",cell['seed_count']])
    counts_by_site=[]
    for s in sites:
        values=[f'{y}: {siteyear.get((s,str(y)),0)}' for y in range(2018,2024)]
        counts_by_site.append([s.replace('PILOT-','').replace('-',' ').title(),'; '.join(values),'Not used (optical-only SR-1)'])
    est_rows=[[k,'Conditional descriptive' if k=='E4' else 'Not estimable',v['reason']]
              for k,v in report['estimands'].items()]
    title='A Missingness-Aware Sentinel-2 Time-Series Pipeline for South Lhonak and Khangchung Tsho'
    abstract=(f"This study develops a source-traceable Sentinel-2 time-series pipeline for South Lhonak and Khangchung Tsho, using {len([r for r in manifest['observations'] if r.get('status')!='EXPORT_FAILED'])} retained acquisitions from 2018–2023. "
      "A missingness-aware temporal autoencoder was trained on observations assigned to FIT and selected using a separate VAL period. After correcting a causal-attention failure under leading missing observations, a frozen re-execution completed 50 fits across primary, capacity and factorial configurations. An independent validation replay matched all selected checkpoints across 350 target-bearing VAL blocks; 80 blocks without valid targets were recorded as unavailable. For the five-seed primary configuration, mean selected VAL masked MSE was 0.8498 (SD 0.2371). These are optimization diagnostics on one two-lake corpus, not independent site-level replications. The corrected models have not been evaluated on EVAL, and event detection, false-alarm rates, seasonal generalization and operational warning performance are not established. The study is exploratory and limited by two sites, unvalidated candidate zones and incomplete event/control evidence.")
    keywords='glacial lakes; Sentinel-2; anomaly detection; missing data; temporal validation; exploratory analysis'
    out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True)
    # Abstract document
    d=setup_doc(title,abstract,keywords);d.core_properties.title=title;d.core_properties.author='Aditya Shrestha'
    d.core_properties.comments='Protocol SHA-256: '+protocol_hash;d.save(out/NAMES['abstract'])
    # Extended abstract
    d=setup_doc(title,abstract,keywords)
    add_heading(d,'1. Introduction')
    p1=('Glacial-lake monitoring is constrained by sparse field observations, seasonal surface change, cloud cover, and uncertain event labels. Satellite time series can support repeatable measurement, but anomaly scores alone do not establish a lake-outburst warning system. A credible analysis must bind each pixel window to its source product and acquisition time, separate model fitting from temporal validation and evaluation, represent missing channels explicitly, and report failures as well as supported outcomes. These requirements are especially important when apparent changes may reflect weather, illumination, product processing, or incomplete observations rather than lake change.')
    p2=('The research question is whether a compact, location-aware temporal masked autoencoder yields useful descriptive anomalies for the two specified lake regions under a leakage-controlled multi-year protocol. E1 concerns event timing conditional on an independently reviewed event bracket and operational availability; E2 concerns false alarms over independently verified event-free lake-years; E3 concerns seasonally matched out-of-sample ranks under rolling-origin fitting; E4 is a within-scene zone contrast; and E5 is incremental value against fair, availability-matched baselines. The contributions are a receipt-bound multi-year optical corpus, a missingness-aware patch pipeline, a causal temporal model, and a result packet that leaves unsupported estimands null.')
    add_text(d,p1);add_text(d,p2)
    add_heading(d,'2. Related work')
    related=('Self-supervised Earth-observation models use satellite-specific temporal and spectral structure: SatMAE applies masked pretraining to temporal or multispectral imagery, Presto learns from remote-sensing pixel time series, and Prithvi-EO is a large HLS-pretrained geospatial model (Cong et al., 2022; Tseng et al., 2023; Szwarcman et al., 2024). These works establish relevant model families, but do not validate this study’s two-site optical anomaly scores or its event estimands.')
    add_text(d,related)
    related=('Classical satellite time-series change methods provide important comparison families: BFAST models trend and seasonal structure, CCDC processes dense Landsat observations for continuous change detection, and LandTrendr segments spectral trajectories (Verbesselt et al., 2010; Zhu and Woodcock, 2014; Kennedy et al., 2010). They motivate explicit seasonal and change-detection baselines; the current exploratory packet does not claim a fair calibrated comparison against these methods.')
    add_text(d,related)
    related=('Anomaly evaluation also depends on the evaluation protocol. Wu and Keogh document benchmark construction problems, Tatbul et al. formalize range-aware precision and recall, and Sarfraz et al. discuss methodological challenges in unsupervised time-series anomaly detection (Wu and Keogh, 2021; Tatbul et al., 2018; Sarfraz et al., 2024). Here, event labels, exposure, and alarm operating points remain insufficient for those performance estimands, so E1, E2, and E5 are not estimable.')
    add_text(d,related)
    related=('For the target hazard, optical and SAR observations have been used retrospectively to examine South Lhonak’s 2023 outburst (Yu et al., 2024), while regional GLOF hazard work frames lake change within a wider physical risk assessment (Taylor et al., 2023; NRSC/ISRO, 2025). A published early-warning architecture describes ultrasonic water-level and ice monitoring at two other Sikkim lakes (Kumar et al., 2020). A recent unreviewed preprint by Kahn et al. explicitly separates site susceptibility from trigger timing and tests strong simple baselines; it is a current novelty challenge, not an independently reproduced benchmark here (Kahn et al., 2026). This study is optical-only, retrospective, and does not estimate susceptibility, trigger timing, or warning performance.')
    add_text(d,related)
    add_heading(d,'3. Data and methods')
    add_text(d,f"The retained manifest contains {len([r for r in manifest['observations'] if r.get('status')!='EXPORT_FAILED'])} observational rows, including source-manifest-bound 2023 records. For 2018–2022, selection used the lowest provider cloud-cover item within each equal-duration bin among full-ROI-coverage products; unfilled bins remain absent. All newly acquired raster windows are HTTP 206 range reads. Transfer receipts, native metadata, processing baseline, BOA scale and offset, grid definition, and SHA-256 bindings are retained. The 10 GB cumulative cap includes earlier attempts and conservative interrupted-transfer reserves.")
    add_text(d,'Each region uses a nearest-pixel-centered 160 × 160 window in EPSG:32645 at 20 m resolution, summarized into 100 non-overlapping 16 × 16 patches. Clear support uses SCL classes 4, 5, and 6 with at least 25% valid pixels. Patch-mean B03/B08/B11 values are computed in float64. NDWI and MNDWI are calculated from those means; absolute denominators below 0.002 yield NaN and an explicit undefined status. Negative reflectance is preserved. Candidate zones are derived from FIT-only spectral composites and are not independently validated lake or moraine geometries.')
    add_text(d,'FIT comprises 2018–2021 and January–June 2023; VAL comprises 2022 and July–September 2023; the two-day event bracket is excluded. The held-out EVAL period begins 2023-10-04 18:30 UTC. All preprocessing statistics and optimizer inputs use FIT observations only. The primary model has d=64, four heads, three factorized encoder layers and one decoder layer; it was trained with AdamW, cosine learning-rate decay and EMA. The temporal encoder applies causal attention only to each patch’s observed timestamps, preserving missingness without invalid attention rows. The primary, capacity and factorial configurations were fitted on FIT and selected on VAL. A separate replay matched saved checkpoint losses across all target-bearing validation blocks; the EVAL period was not used in this re-execution.')
    add_heading(d,'4. Results')
    add_table(d,['Region','Retained acquisitions by year (UTC)','SAR dates'],counts_by_site,'Table 1. Source observations by region and year.')
    add_table(d,['Seed','Epochs','Selected epoch','VAL masked MSE','Optimizer steps','Stop status'],seed_rows,'Table 2. Corrected primary configuration, trained on FIT and selected on VAL (five optimization seeds).')
    add_table(d,['Configuration','Mean selected VAL MSE (SD)','Target / no-target blocks','Seeds'],config_rows,'Table 3. Corrected capacity and factorial validation diagnostics. Means and standard deviations summarize optimizer seeds on the same fixed corpus; block counts are totals across seeds.')
    add_text(d,'The primary configuration had mean selected validation masked MSE 0.8498 (SD 0.2371) across five optimizer seeds. All three target-bearing blocks were scored for each run. Across the complete 50-fit matrix, the independent replay matched 350 target-bearing block instances. Eighty additional blocks contained no valid reconstruction targets and were retained as unavailable. Validation loss describes reconstruction under this protocol; it does not measure event detection or generalization to other lakes.')
    add_heading(d,'5. Figures')
    for stem,label,text in [
      (FIGS[0][0],'Figure 1','Number of retained Sentinel-2 acquisitions by year and region.'),
      (FIGS[1][0],'Figure 2','FIT-derived spectral candidate zones. These masks have not been validated against independent lake or moraine boundaries.'),
      (FIGS[2][0],'Figure 3','Distributions of valid surface-reflectance observations; negative values are retained.')]:
        add_text(d,f'{label}. {text}')
    add_heading(d,'6. Limitations and conclusion')
    add_text(d,'This is an exploratory retrospective study of two lake regions, and earlier inspection of 2023 outcomes precludes confirmatory interpretation. The corrected training and validation results establish that the implementation can select checkpoints using the declared target-bearing VAL blocks; they do not establish that the anomaly scores detect outbursts. Event timing, false-alarm exposure, seasonal generalization and incremental value over baselines remain unestimated because independent event/control labels, adequate rolling-origin reference years and a common calibrated operating point are not available. The within-scene zone contrast is descriptive because candidate zones lack independent geometry validation. Corrected-model EVAL scoring has not been performed. Further work requires event and control review, validated lake geometry, fair baseline calibration and independent scientific and statistical assessment. The study makes no claim of warning lead time, false-alarm rate, physical lake area or operational capability.')
    add_text(d,'Implementation and source-receipt records are retained with the project materials. Source imagery remains subject to provider terms.',size=8)
    refs=['Cong, Y. et al. (2022). SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery. Advances in Neural Information Processing Systems, 35.',
          'Kennedy, R.E., Yang, Z. and Cohen, W.B. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr—Temporal segmentation algorithms. Remote Sensing of Environment, 114(12), 2897–2910. doi:10.1016/j.rse.2010.07.008.',
          'Kahn, M. et al. (2026). Which Site, and When: A Free-Satellite-Data Test of Himalayan Glacial Lake Bursts, Landslides, and Ice Floods. arXiv:2608.12422. Preprint, not peer reviewed.',
          'Kumar, B., Sathyan, A., Murugesh, T.S. & Krishnan, A. (2020). Design Architecture of Glacier Lake Outburst Flood (GLOF) Early Warning System Using Ultrasonic Sensors. 2020 IEEE Recent Advances in Intelligent Computational Systems. doi:10.1109/RAICS51191.2020.9332472.',
          'NRSC/ISRO (2025). GLOF Risk Assessment of Khangchung Tso Glacial Lake in Brahmaputra River Basin. Technical Report NRSC-RSA-WATER RE-WRD-FEB2025-TR-0002698-V1.0.',
          'Sarfraz, M.S. et al. (2024). Position: Quo vadis, unsupervised time series anomaly detection? Proceedings of the 41st International Conference on Machine Learning, 235, 43461–43476.',
          'Szwarcman, D. et al. (2024). Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications. arXiv:2412.02732.',
          'Tatbul, N. et al. (2018). Precision and Recall for Time Series. Advances in Neural Information Processing Systems, 31.',
          'Taylor, C. et al. (2023). Glacial lake outburst floods threaten millions globally. Nature Communications, 14, 487.',
          'Tseng, G. et al. (2023). Lightweight, Pre-trained Transformers for Remote Sensing Timeseries. arXiv:2304.14065.',
          'Verbesselt, J. et al. (2010). Detecting trend and seasonal changes in satellite image time series. Remote Sensing of Environment, 114(1), 106–115. doi:10.1016/j.rse.2009.08.014.',
          'Wu, R. and Keogh, E. (2021). Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress. IEEE Transactions on Knowledge and Data Engineering, 35(3), 2421–2429. doi:10.1109/TKDE.2021.3112126.',
          'Yu, Y. et al. (2024). Retrospective Analysis of Glacial Lake Outburst Flood (GLOF) Using AI Earth InSAR and Optical Images: A Case Study of South Lhonak Lake, Sikkim. Remote Sensing, 16(13), 2307. doi:10.3390/rs16132307.',
          'Zhu, Z. and Woodcock, C.E. (2014). Continuous change detection and classification of land cover using all available Landsat data. Remote Sensing of Environment, 144, 152–171. doi:10.1016/j.rse.2014.01.011.']
    add_heading(d,'References')
    for ref in refs:add_text(d,ref,size=8.5)
    d.core_properties.title=title;d.core_properties.author='Aditya Shrestha';d.core_properties.comments='Protocol SHA-256: '+protocol_hash
    d.save(out/NAMES['extended'])
    # Full paper Word copy adds all five figures and keeps the same audited content.
    full=setup_doc(title,abstract,keywords)
    content_children=list(d._element.body)
    for child in content_children[7:]:
        if child.tag.endswith('sectPr'):continue
        full._element.body.insert(len(full._element.body)-1,child)
    word_captions=['Observation counts by year and region.','FIT-derived candidate zones; not independently validated physical boundaries.','Valid reflectance distributions with negative values retained.']
    for index,((stem,_),caption) in enumerate(zip(FIGS,word_captions),start=1):
        anchor=next((p for p in full.paragraphs if p.text.startswith(f'Figure {index}.')),None)
        if anchor is None:raise ValueError(f'missing Section 4 interpretation for Figure {index}')
        image_p=full.add_paragraph();image_p.alignment=WD_ALIGN_PARAGRAPH.CENTER;image_p.paragraph_format.keep_with_next=True
        image_p.add_run().add_picture(str(figure_dir/f'{stem}.png'),width=Inches(5.5))
        caption_p=full.add_paragraph();caption_p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        caption_p.paragraph_format.space_after=Pt(5);r=caption_p.add_run(f'Figure {index}. {caption}');r.font.name='Times New Roman';r.font.size=Pt(8.5)
        anchor._p.addnext(image_p._p);image_p._p.addnext(caption_p._p)
    full.core_properties.title=title;full.core_properties.author='Aditya Shrestha';full.core_properties.comments='Protocol SHA-256: '+protocol_hash
    full.save(out/NAMES['fullpaper'])
    # LaTeX source for the two matched PDF artifacts.
    def esc(value):
        text=str(value).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
        for source,target in [('α',r'$\alpha$'),('≤',r'$\leq$'),('≥',r'$\geq$'),('±',r'$\pm$'),
                              ('×',r'$\times$'),('–','--'),('—','---'),('²',r'$^2$'),('’',"'"),('“','``'),('”',"''")]:
            text=text.replace(source,target)
        return text
    tex=[r'\documentclass[9pt,twocolumn]{article}',r'\usepackage[a4paper,margin=0.62in]{geometry}',r'\usepackage{graphicx,booktabs,tabularx,array,url,amsmath}',r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern}',r'\setlength{\columnsep}{0.22in}',r'\setlength{\emergencystretch}{2em}',r'\usepackage[font=small,labelfont=bf]{caption}',r'\title{\bfseries '+esc(title)+'}',r'\author{Aditya Shrestha\\\small Kathmandu University, Nepal\\\small adityashrestha39@gmail.com}',r'\date{}',r'\begin{document}',r'\maketitle',r'\begin{abstract}'+esc(abstract)+r'\end{abstract}',r'\noindent\textbf{Keywords:} '+esc(keywords)]
    site_year_text=', '.join(f"{site.replace('PILOT-','')} {year}: {value}" for (site,year),value in sorted(siteyear.items()))
    n_obs=len([r for r in manifest['observations'] if r.get('status')!='EXPORT_FAILED'])
    sections=[
      ('1. Introduction',[p1,p2]),
      ('2. Related work',[
        'Self-supervised Earth-observation models use satellite-specific temporal and spectral structure: SatMAE applies masked pretraining to temporal or multispectral imagery, Presto learns from remote-sensing pixel time series, and Prithvi-EO is an HLS-pretrained geospatial model (Cong et al., 2022; Tseng et al., 2023; Szwarcman et al., 2024). These works establish relevant model families, but do not validate this study’s two-site optical anomaly scores or event estimands.',
        'Classical satellite time-series change methods provide comparison families: BFAST models trend and seasonal structure, CCDC processes dense Landsat observations for continuous change detection, and LandTrendr segments spectral trajectories (Verbesselt et al., 2010; Zhu and Woodcock, 2014; Kennedy et al., 2010). The present packet does not claim a fair calibrated comparison against these methods.',
        'Anomaly evaluation depends on protocol as well as score: Wu and Keogh analyze benchmark construction problems, Tatbul et al. formalize range-aware precision and recall, and Sarfraz et al. discuss methodological challenges in unsupervised time-series anomaly detection (Wu and Keogh, 2021; Tatbul et al., 2018; Sarfraz et al., 2024). Event labels, exposure, and alarm operating points remain insufficient for E1, E2, and E5.',
        'Optical and SAR observations have been used retrospectively to examine South Lhonak’s 2023 outburst (Yu et al., 2024), alongside regional GLOF risk assessment (Taylor et al., 2023; NRSC/ISRO, 2025). Kumar et al. (2020) describe an ultrasonic early-warning architecture deployed at two other Sikkim lakes. Kahn et al. (2026) is a recent unreviewed preprint that separates site susceptibility from trigger timing and tests simple baselines; it is a novelty challenge, not an independently reproduced benchmark here. This study is optical-only, retrospective, and does not estimate susceptibility, trigger timing, or warning performance.']),
      ('3. Data and methods',[
        f"The manifest contains {n_obs} observational rows, including source-bound 2023 records. For 2018–2022, score-blind selection chose the lowest-cloud item in each equal-duration bin among full-window products. New COG windows use HTTP 206 range reads, and retained receipts bind source bytes, native baseline, BOA scale/offset, grid, and timestamps. The cumulative 10 GB cap includes earlier attempts and conservative interruption reserves.",
        'Each region uses a nearest-pixel-centered 160 × 160 EPSG:32645 window at 20 m resolution, summarized to 100 16 × 16 patches. Clear support is SCL 4/5/6 with at least 25% valid pixels. Float64 patch-mean B03/B08/B11 are unclamped; absolute index denominators below 0.002 are missing with an explicit status. Candidate zones are FIT-only spectral masks, not validated physical shorelines.',
        'FIT comprises 2018–2021 and Jan–Jun 2023; VAL comprises 2022 and Jul–Sep 2023; the uncertain event bracket is excluded; EVAL begins 2023-10-04 18:30 UTC. All fitted preprocessing, PCA, location-matched references, Score-C moments, and optimizer input are FIT-only. The d=64 factorized model has 207,685 parameters, causal temporal attention, AdamW, cosine decay, EMA, and minimum supported VAL masked-MSE checkpointing.',
        'Score A is masked mean squared error on FIT-standardized patch features, with raw A and ln(A) retained when positive. Score B is same-patch latent distance to FIT observations within ±45 calendar days. Score C is the equal-weight combination (α = 0.5) of FIT-only robust-standardized ln(A) and B; these scores do not declare alarms.']),
      ('4. Results',[
        f"The retained site-year counts are {site_year_text}. Table 1 identifies the optical-only modality scope; SAR dates are explicitly recorded as not used.",
        f"The five primary fits had mean selected VAL masked MSE {corrected_summary['cell_summaries'][-1]['mean_best_val_masked_mse']:.4f} (SD {corrected_summary['cell_summaries'][-1]['sample_sd']:.4f}). Across the 50-fit matrix, an independent replay matched 350 target-bearing VAL block losses; 80 blocks without valid targets were recorded as unavailable. These reconstruction losses summarize optimization on one fixed corpus and do not estimate event detection or cross-lake performance.",
        'Event timing, false-alarm exposure, seasonal generalization and incremental value over baselines remain unestimated under available event labels, control exposure and operating-point evidence. The within-scene zone contrast is descriptive because candidate zones lack independent geometry validation.']),
      ('5. Figures and interpretation',[
        'Figure 1 shows source-bound observation counts by year and region; sparse support is visible and no absent acquisition is interpolated.',
        'Figure 2 shows FIT-only spectral candidate zones. It describes the algorithmic mask, not independent physical shoreline accuracy.',
        'Figure 3 displays native valid reflectance values, including negative values, consistent with the unclamped primary pipeline.',
]),
      ('6. Limitations and conclusion',[
        'This is an exploratory retrospective study of two lake regions; earlier inspection of 2023 outcomes precludes confirmatory interpretation. Corrected training and validation results show that checkpoint selection can use the declared target-bearing VAL blocks, but do not establish outburst detection. Event timing, false-alarm exposure, seasonal generalization and incremental value over baselines remain unestimated because independent event/control labels, adequate rolling-origin years and a calibrated common operating point are unavailable. Candidate zones lack independent geometry validation. Corrected-model EVAL scoring has not been performed. Further work requires event/control review, validated lake geometry, fair baseline calibration and independent scientific and statistical assessment.'])]
    for heading,paras in sections:
        tex.append(r'\section*{'+esc(heading)+'}')
        tex.extend(esc(text) for text in paras)
        if heading=='3. Data and methods':
            tex.append(r'\begin{equation}C_t=\tfrac{1}{2}z_{\mathrm{FIT}}(\ln A_t)+\tfrac{1}{2}z_{\mathrm{FIT}}(B_t)\tag{5}\end{equation}')
            tex.append(r'\noindent The equal weight is $\alpha=0.5$; the coefficient is stated in prose and does not compete with the equation number.')
    captions=['Observation counts by year and region.','FIT-derived candidate zones; not independently validated physical boundaries.','Valid reflectance distributions with negative values retained.']
    figure_assets=out/'sr1_true_figures';figure_assets.mkdir(exist_ok=True)
    for (stem,_),caption in zip(FIGS,captions):
        shutil.copy2(figure_dir/f'{stem}.pdf',figure_assets/f'{stem}.pdf')
        asset=f'docs/manuscript/sr1_true_figures/{stem}.pdf'
        tex.append(r'\begin{figure}[htbp]\centering\includegraphics[width=0.96\columnwidth]{\detokenize{'+asset+r'}}\caption{'+esc(caption)+r'}\end{figure}')
    tex.append(r'\begin{table*}[t]\centering\caption{Source-bound observations and modality scope.}\begin{tabularx}{\textwidth}{lXl}\toprule Region & Counts by year & SAR dates\\\midrule')
    for row in counts_by_site:tex.append(' & '.join(map(esc,row))+r'\\')
    tex.append(r'\bottomrule\end{tabularx}\end{table*}')
    tex.append(r'\begin{table*}[t]\centering\caption{Corrected primary FIT/VAL-only five-seed re-execution. All target-bearing VAL blocks were scored and independently replayed.}\begin{tabular}{rrrrrl}\toprule Seed & Epochs & Selected epoch & Recorded VAL MSE & Steps & Stop status\\\midrule')
    for row in seed_rows:tex.append(' & '.join(map(esc,row))+r'\\')
    tex.append(r'\bottomrule\end{tabular}\end{table*}')
    tex.append(r'\begin{table*}[t]\centering\caption{Corrected capacity and factorial validation diagnostics. Mean and SD summarize optimizer seeds on one fixed corpus.}\begin{tabular}{lrrr}\toprule Configuration & Mean VAL MSE (SD) & Target / no-target blocks & Seeds\\\midrule')
    for row in config_rows:
        tex.append(' & '.join(map(esc,row))+r'\\')
    tex.append(r'\bottomrule\end{tabular}\end{table*}')
    tex.append(r'\begin{thebibliography}{14}')
    for i,ref in enumerate(refs,1): tex.append(r'\bibitem{ref'+str(i)+'} '+esc(ref))
    tex.append(r'\end{thebibliography}')
    tex.append(r'\end{document}')
    source='\n'.join(tex)+'\n';(out/'sentinel_gl_ictceel2027.tex').write_text(source);(out/'sentinel_gl_submission.tex').write_text(source)
    compile_dir=a.compile_dir.resolve();compile_dir.mkdir(parents=True,exist_ok=False)
    for run in range(2):
        result=subprocess.run(['/Library/TeX/texbin/pdflatex','-halt-on-error','-interaction=nonstopmode','-output-directory='+str(compile_dir),str((out/'sentinel_gl_ictceel2027.tex').resolve())],cwd=ROOT,capture_output=True,text=True)
        (compile_dir/f'pass_{run+1}.stdout.log').write_text(result.stdout);(compile_dir/f'pass_{run+1}.stderr.log').write_text(result.stderr)
        if result.returncode:raise RuntimeError(f'LaTeX compilation failed; see {compile_dir}')
    for name in ('sentinel_gl_ictceel2027.pdf','sentinel_gl_submission.pdf'):shutil.copy2(compile_dir/'sentinel_gl_ictceel2027.pdf',out/name)
    sums=[]
    for name in [*NAMES.values(),'sentinel_gl_ictceel2027.tex','sentinel_gl_submission.tex','sentinel_gl_ictceel2027.pdf','sentinel_gl_submission.pdf']:
        path=out/name;sums.append(f'{sha(path)}  {name}')
    (out/'SUBMISSION_SHA256SUMS').write_text('\n'.join(sums)+'\n')
    import subprocess as sp
    info=sp.run(['pdfinfo',str(out/'sentinel_gl_ictceel2027.pdf')],capture_output=True,text=True,check=True).stdout
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    if pages>8:raise RuntimeError(f'Compiled PDF has {pages} pages; maximum is eight. Preserve this attempt and revise the document.')
    print(json.dumps({'status':'BUILT_LOCAL_EXPLORATORY_DRAFT','pdf_pages':pages,'protocol_sha256':protocol_hash,
      'files':{n:sha(out/n) for n in [*NAMES.values(),'sentinel_gl_ictceel2027.pdf','sentinel_gl_submission.pdf']}},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
