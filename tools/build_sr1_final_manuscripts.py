#!/usr/bin/env python3
"""Build concise, result-bound SR-1 exploratory manuscript documents."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone

import numpy as np
from docx import Document
from docx.shared import Mm,Pt,Inches,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/manuscript'; FIG=ROOT/'docs/figures/sr1_final_006'
TABLES=ROOT/'docs/tables/sr1_final'; TABLES.mkdir(parents=True,exist_ok=True)
ATT=ROOT/'data/derived/sr1_multiyear/attempt_006'
PKG=ATT/'corrected_eval_001'
SITES=['PILOT-SOUTH-LHONAK','PILOT-KHANGCHUNG-TSHO']
TITLE='Sentinel-GL: Missingness-Aware Sentinel-2 Masked Autoencoding and Multi-Year Surface-Anomaly Diagnostics for South Lhonak and Khangchung Tsho'
NOTICE='Contains modified Copernicus Sentinel data [2018–2023], processed by the author.'
KEYWORDS='glacial lakes; Sentinel-2; masked autoencoders; anomaly detection; missing data'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v): return '—' if v is None or not np.isfinite(v) else f'{float(v):.3g}'
def role(dt):
    d=datetime.fromisoformat(dt.replace('Z','+00:00')).astimezone(timezone.utc)
    if 2018<=d.year<=2021 or (d.year==2023 and d.month<=6):return 'FIT'
    if d.year==2022 or (d.year==2023 and d.month<=9):return 'VAL'
    if d>=datetime(2023,10,4,18,30,tzinfo=timezone.utc):return 'EVAL'
    return 'EVENT_UNCERTAIN'

def set_cell_shading(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);tcPr.append(shd)
def set_cell_margins(cell,top=80,start=90,bottom=80,end=90):
    tc=cell._tc;tcPr=tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
    for name,val in [('top',top),('start',start),('bottom',bottom),('end',end)]:
        node=OxmlElement('w:'+name);node.set(qn('w:w'),str(val));node.set(qn('w:type'),'dxa');m.append(node)
    tcPr.append(m)
def setup(title=True,compact=False):
    d=Document();s=d.sections[0];s.page_width=Mm(210);s.page_height=Mm(297)
    s.top_margin=Mm(25);s.bottom_margin=Mm(25);s.left_margin=Mm(20);s.right_margin=Mm(20)
    normal=d.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10 if compact else 10.5);normal.font.color.rgb=RGBColor(0,0,0)
    normal.paragraph_format.space_after=Pt(3 if compact else 4);normal.paragraph_format.line_spacing=1.0 if compact else 1.04
    for name in ['Heading 1','Heading 2','Heading 3']:
        st=d.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0);st.font.bold=True
        st.font.size=Pt(10.5 if compact else (11 if name=='Heading 1' else 10.5));st.paragraph_format.space_before=Pt(6 if compact else 8);st.paragraph_format.space_after=Pt(2.5 if compact else 3)
    footer=s.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=footer.add_run();fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');run._r.addnext(fld)
    if title:
        p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(5)
        r=p.add_run(TITLE);r.font.name='Times New Roman';r.font.size=Pt(12);r.bold=True;r.font.color.rgb=RGBColor(0,0,0)
        for text,size in [('Aditya Shrestha',10.5),('Department of Computer Science and Engineering (DoCSE), School of Engineering, Kathmandu University, Dhulikhel, Nepal',9.5),('Corresponding author: Aditya Shrestha | Email: adityashrestha39@gmail.com (Institutional: as03233622@student.ku.edu.np)',8.5)]:
            q=d.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_after=Pt(1);rr=q.add_run(text);rr.font.name='Times New Roman';rr.font.size=Pt(size)
    return d
def paragraph(d,text,*,bold=False,italic=False,center=False,size=None):
    p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.widow_control=True
    r=p.add_run(text);r.bold=bold;r.italic=italic;r.font.name='Times New Roman';
    if size:r.font.size=Pt(size)
    return p
def heading(d,text,level=1):
    p=d.add_heading(text,level=level);p.paragraph_format.keep_with_next=True
def refparagraph(d,text,size=6.8):
    p=paragraph(d,text,size=size);p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1.0
    return p
def table(d,caption,headers,rows,font=8):
    p=d.add_paragraph();p.paragraph_format.keep_with_next=True;p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(2)
    r=p.add_run(caption);r.bold=True;r.font.name='Times New Roman';r.font.size=Pt(8.5)
    t=d.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.style='Table Grid';t.autofit=True
    for i,x in enumerate(headers):
        c=t.rows[0].cells[i];c.text=str(x);set_cell_shading(c,'24476B');c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;set_cell_margins(c)
    for row in rows:
        cells=t.add_row().cells
        for i,x in enumerate(row):
            cells[i].text=str(x);cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;set_cell_margins(cells[i])
    for ri,row in enumerate(t.rows):
        pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
        if ri==0:pr.append(OxmlElement('w:tblHeader'))
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.0
                for rr in p.runs:rr.font.name='Times New Roman';rr.font.size=Pt(font);rr.font.color.rgb=RGBColor(255,255,255) if ri==0 else RGBColor(0,0,0);rr.bold=ri==0
            if ri>0 and ri%2==0:set_cell_shading(cell,'F0F4F8')
    return t
def equation(d,text,n,*,keep_with_next=False):
    t=d.add_table(rows=1,cols=3);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=[Inches(.67),Inches(5.35),Inches(.67)]
    for i,width in enumerate(widths):
        t.columns[i].width=width
        for cell in t.columns[i].cells: cell.width=width
    row=t.rows[0];row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    for cell in row.cells:
        tc_pr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
        for edge in ('top','left','bottom','right','insideH','insideV'):
            node=OxmlElement('w:'+edge);node.set(qn('w:val'),'nil');borders.append(node)
        tc_pr.append(borders)
        cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    left,center,right=row.cells
    p=left.paragraphs[0];p.paragraph_format.space_after=Pt(0)
    p= center.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(0);p.paragraph_format.keep_with_next=keep_with_next
    run=p.add_run(text);run.font.name='Times New Roman';run.font.size=Pt(8.2)
    q=right.paragraphs[0];q.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    q.paragraph_format.space_after=Pt(0);q.paragraph_format.keep_with_next=keep_with_next
    q.add_run(f'({n})')
def figure(d,stem,caption,width=6.3):
    p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.paragraph_format.space_after=Pt(1)
    p.add_run().add_picture(str(FIG/(stem+'.png')),width=Inches(width))
    cp=d.add_paragraph();cp.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;cp.paragraph_format.space_after=Pt(4);cp.paragraph_format.keep_together=True
    r=cp.add_run(caption+' '+NOTICE);r.font.name='Times New Roman';r.font.size=Pt(8)

REF_KEYS=['shugar2020','zheng2021','taylor2023','sattar2025','veh2018','sarfraz2024',
 'drusch2012','mcfeeters1996','xu2006','torres2012','yu2024','nrsc2025','kumar2020','kahn2026',
 'vaswani2017','he2022','zerveas2021','cong2022','tseng2023','szwarcman2024','li2023','song2023',
 'verbesselt2010','zhu2014','kennedy2010','liu2008','scholkopf2001','candes2011','lin2010','page1954','wu2021','tatbul2018']

def references():
    reg=json.loads((DOC/'citation_registry_sr1.json').read_text())
    refs={k:reg[k]['reference'] for k in REF_KEYS if k in reg}
    refs['candes2011']='Candès, E.J., Li, X., Ma, Y. & Wright, J. (2011). Robust principal component analysis? Journal of the ACM, 58(3), Article 11, 1–37. https://doi.org/10.1145/1970392.1970395.'
    refs['esa_l2a_dqr']='European Space Agency. (2023). Data Quality Report: Sentinel-2 MSI Level-2A, April 2023. Sentinel Online. https://sentinels.copernicus.eu/documents/d/sentinel/ompc-cs-dqr-002-03-2023-i60r0-msi-l2a-dqr-april-2023.'
    missing=set(REF_KEYS)-set(refs)
    if missing: raise ValueError(f'missing cited-reference metadata: {sorted(missing)}')
    return [refs[k] for k in sorted(refs,key=lambda k:refs[k].casefold())]

def rank2_diagnostic(preds):
    """Post-hoc rank-2 SVD residual from saved standardized replay inputs; never retrains the T-MAE."""
    out=[]
    for seed in range(5):
        path=PKG/'replay'/f'seed_{seed}_score_replay.npz'
        with np.load(path,allow_pickle=False) as z:
            fit=[]
            for i,role in enumerate(z['roles'].tolist()):
                if role!='FIT':continue
                good=z['observed'][i].all(axis=1)&np.isfinite(z['target'][i]).all(axis=1)&np.isin(z['zones'][i],[0,1])
                if good.any():fit.extend(z['target'][i,good].astype(float))
            x=np.asarray(fit,float)
            if x.ndim!=2 or x.shape[0]<3: continue
            center=x.mean(axis=0);_,_,vt=np.linalg.svd(x-center,full_matrices=False);basis=vt[:2].T
            for i,role in enumerate(z['roles'].tolist()):
                good=z['observed'][i].all(axis=1)&np.isfinite(z['target'][i]).all(axis=1)&np.isin(z['zones'][i],[0,1])
                if not good.any():continue
                y=z['target'][i,good].astype(float)-center
                residual=np.linalg.norm(y-(y@basis)@basis.T,axis=1)
                out.append({'site_id':str(z['site_ids'][i]),'role':role,'acquired_at':str(z['acquired_at'][i]),'seed':seed,'value':float(np.median(residual))})
    return out

def main():
    manifest_path=ATT/'observations.sr1.json';manifest=json.loads(manifest_path.read_text())
    preds=json.loads((PKG/'predictions.json').read_text());report=json.loads((PKG/'analysis_report.json').read_text())
    corrected=json.loads((ROOT/'project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_SENSITIVITY_SUMMARY_001.json').read_text())
    rows=[r for r in manifest['observations'] if r.get('status')!='EXPORT_FAILED']
    baseline_counts=Counter(r['processing_baseline'] for r in rows if r.get('processing_baseline'))
    if sum(baseline_counts.values())!=len(rows) or set(baseline_counts)-{'05.00','05.09','05.10'}:
        raise ValueError('native Sentinel-2 processing-baseline inventory is incomplete or unexpected')
    baseline_text=', '.join(f'{count} products at PB {baseline}' for baseline,count in sorted(baseline_counts.items()))
    sc=Counter();clear=defaultdict(list)
    for r in rows:
        rr=role(r['acquired_at']);sc[(r['site_id'],rr)]+=1
        with np.load(ROOT/r['values_file'],allow_pickle=False) as z:
            clear[(r['site_id'],rr)].append(float((np.isin(z['scl'],[4,5,6])&z['valid'].all(axis=0)).mean()))
    table1=[]
    for site in ['PILOT-SOUTH-LHONAK','PILOT-KHANGCHUNG-TSHO']:
        table1.append([site.replace('PILOT-','').replace('-',' ').title(),*[f"{sc[(site,x)]} / {np.median(clear[(site,x)]):.2f}" if sc[(site,x)] else '0 / —' for x in ['FIT','VAL','EVAL']]])
    table2=[]
    for c in sorted(corrected['cell_summaries'],key=lambda c:(c['family'],c['cell'])):
        label=('Primary d=64' if c['family']=='primary' else (f"Capacity d={c['cell'].split('_')[-1]}" if c['family']=='capacity' else f"{c['channels']} ch; {'FIT zones' if c['zones_enabled'] else 'constant zones'}; context {c['context_scenes_max']}"))
        table2.append([label,f"{c['mean_best_val_masked_mse']:.4f} ± {c['sample_sd']:.4f}",f"{c['target_bearing_val_blocks_total']} / {c['no_target_val_blocks_total']}",c['seed_count']])
    method_fields=[('Score A (Zone 0+1)','score_a_zone0_1'),('Score B','score_b_zone0_1'),('Score C','score_c'),('Candidate water area (km²)','water_candidate_area_km2'),('Seasonal residual','baselines.seasonal_climatology_residual'),('Spatial persistence','baselines.spatial_persistence_residual'),('Isolation Forest','baselines.isolation_forest_score'),('One-Class SVM','baselines.one_class_svm_score'),('RPCA full-rank residual','baselines.rpca_score'),('Fixed CUSUM statistic','page_cusum.statistic')]
    def get(r,path):
        v=r
        for x in path.split('.'):v=v.get(x) if isinstance(v,dict) else None
        return v
    def summ(role_names,field):
        vals=[get(r,field) for r in preds if r['role'] in role_names]
        vals=np.array([v for v in vals if v is not None and np.isfinite(v)],float)
        return f"{np.median(vals):.3g} [{np.quantile(vals,.25):.3g}, {np.quantile(vals,.75):.3g}] (n={len(vals)})" if len(vals) else 'not estimable'
    def bysite_scene_values(site,selector,key):
        grouped=defaultdict(list)
        for row in preds:
            if row['site_id']!=site or not selector(row):continue
            value=get(row,key)
            if value is not None and np.isfinite(value):grouped[row['observation_id']].append(float(value))
        return {oid:float(np.median(values)) for oid,values in grouped.items() if values}
    def bysite_summ(site,selector,key):
        vals=np.asarray(list(bysite_scene_values(site,selector,key).values()),float)
        return f'{np.median(vals):.3g} [{np.quantile(vals,.25):.3g}, {np.quantile(vals,.75):.3g}] (n={len(vals)} scenes)' if len(vals) else 'not estimable (n=0 scenes)'
    autumn_fit=lambda r:r['role']=='FIT' and r['acquired_at'][:4] in {'2018','2019','2020','2021'} and r['acquired_at'][5:7] in {'10','11'}
    eval_autumn=lambda r:r['role']=='EVAL' and r['acquired_at'][5:7] in {'10','11'}
    site_specs=[('SL FIT/VAL',SITES[0],lambda r:r['role'] in {'FIT','VAL'}),('SL Autumn 2018–2021',SITES[0],autumn_fit),('SL EVAL',SITES[0],lambda r:r['role']=='EVAL'),('KT FIT/VAL',SITES[1],lambda r:r['role'] in {'FIT','VAL'}),('KT Autumn 2018–2021',SITES[1],autumn_fit),('KT EVAL',SITES[1],lambda r:r['role']=='EVAL')]
    table3=[[name,*[bysite_summ(site,selector,key) for _,site,selector in site_specs]] for name,key in method_fields]
    rank2=rank2_diagnostic(preds)
    def rank2_summ(site,selector):
        grouped=defaultdict(list)
        for r in rank2:
            if r['site_id']==site and selector({'role':r['role'],'acquired_at':r['acquired_at']}):
                grouped[r['acquired_at']].append(float(r['value']))
        vals=np.asarray([np.median(v) for v in grouped.values()],float)
        return f'{np.median(vals):.3g} [{np.quantile(vals,.25):.3g}, {np.quantile(vals,.75):.3g}] (n={len(vals)} scenes)' if len(vals) else 'not estimable (n=0 scenes)'
    table3.append(['Post-hoc rank-2 SVD residual',*[rank2_summ(site,selector) for _,site,selector in site_specs]])
    estimand_names={'E1':'Event-onset detection timing','E2':'False-alert burden per event-free lake-year','E3':'Out-of-sample seasonal transfer','E4':'Within-scene candidate-zone contrast','E5':'Incremental warning value at a common operating point'}
    reason_overrides={
        'E3':'The available checkpoints were fit and validated on the fixed retrospective corpus; no rolling-origin models held out complete prior reference years.',
        'E4':'Candidate FIT-derived spectral zones lack independent physical shoreline validation, and scene patches are spatially and temporally dependent.'}
    table4=[[estimand_names[e],('Conditional descriptive' if e=='E4' else 'Not estimable'),reason_overrides.get(e,report['estimands'][e]['reason'])] for e in ['E1','E2','E3','E4','E5']]
    autumn_scores={}
    for site in SITES:
        autumn_scores[site]={}
        for name,selector in [('FIT',autumn_fit),('EVAL',eval_autumn)]:
            autumn_scores[site][name]={key:bysite_scene_values(site,selector,key) for key in ['score_a_zone0_1','score_b_zone0_1','score_c','water_candidate_area_km2']}
    autumn_comparisons=[]
    for site in SITES:
        for key in ['score_a_zone0_1','score_b_zone0_1','score_c','water_candidate_area_km2']:
            fitv=list(autumn_scores[site]['FIT'][key].values());evalv=list(autumn_scores[site]['EVAL'][key].values())
            if fitv and evalv:
                autumn_comparisons.append((site,key,float(np.median(fitv)),float(np.median(evalv))))
    lower_count=sum(evalv<fitv for _,_,fitv,evalv in autumn_comparisons)
    lower_count_word={0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight'}.get(lower_count,str(lower_count))
    def cmpfmt(site,key):
        item=next(x for x in autumn_comparisons if x[0]==site and x[1]==key)
        digits=3
        return f'{item[2]:.{digits}f}→{item[3]:.{digits}f}'
    sl=SITES[0];kt=SITES[1]
    zone_contrast={}
    for site in SITES:
        zone_contrast[site]={}
        for name,selector in [('FIT',autumn_fit),('EVAL',eval_autumn)]:
            vals=bysite_scene_values(site,selector,'score_a_zone_contrast')
            zone_contrast[site][name]={'median':float(np.median(list(vals.values()))) if vals else None,
                                       'n_scenes':len(vals)}
    contrast_fmt=lambda site,role_name:f"{zone_contrast[site][role_name]['median']:.3f}"
    zone_n=lambda site,role_name:str(zone_contrast[site][role_name]['n_scenes'])
    abstract=(
        'Cloud, snow and irregular revisits limit optical monitoring of Himalayan glacial lakes. '
        'Sentinel-GL is a 207,685-parameter missingness-aware factorized temporal masked autoencoder. '
        'We analyzed 141 Sentinel-2 observations (117 FIT, 17 VAL, seven EVAL) from South Lhonak and Khangchung Tsho, 2018–2023. '
        'The prespecified five-channel model’s five best-validation-epoch checkpoints achieved mean validation masked MSE 0.8498 (SD 0.2371) across 15 target-bearing blocks per seed; '
        'the 50-fit sensitivity matrix included 350 target-bearing and 80 no-target configuration-block instances. '
        'At South Lhonak, Score A medians were 1.399 (9 October), 0.539 (26 October) and 0.528 (13 November); '
        'the seasonal-residual baseline was unavailable on EVAL because complete same-patch FIT support was insufficient. '
        f"Zone 0+1 minus Zone 2 reconstruction-contrast medians were {contrast_fmt(sl,'FIT')}/{contrast_fmt(kt,'FIT')} (FIT n={zone_n(sl,'FIT')}/{zone_n(kt,'FIT')}) "
        f"and {contrast_fmt(sl,'EVAL')}/{contrast_fmt(kt,'EVAL')} (EVAL n={zone_n(sl,'EVAL')}/{zone_n(kt,'EVAL')}), at South Lhonak/Khangchung. "
        f'Six of eight matched autumn Score A, B, C and candidate-area medians were lower in EVAL: South Lhonak {cmpfmt(sl,"score_a_zone0_1")}, {cmpfmt(sl,"score_b_zone0_1")}, {cmpfmt(sl,"score_c")}, and {cmpfmt(sl,"water_candidate_area_km2")} km²; '
        f'Khangchung {cmpfmt(kt,"score_a_zone0_1")}, {cmpfmt(kt,"score_b_zone0_1")}, {cmpfmt(kt,"score_c")}, and {cmpfmt(kt,"water_candidate_area_km2")} km². '
        'This comparison uses seven prior FIT scenes per site versus three/four EVAL scenes; FIT scenes are in-sample. '
        'Zones lack independent geometry validation, and event timing, false-alert burden and warning value remain unestimated.'
    )
    if len(abstract.split())>200:raise ValueError('abstract exceeds 200 words')
    # Abstract
    d=setup();heading(d,'Abstract',1);paragraph(d,abstract);paragraph(d,'Keywords: '+KEYWORDS,bold=True)
    d.core_properties.title=TITLE;d.core_properties.author='Aditya Shrestha';d.save(DOC/'ICT_CEEL_2027_Abstract.docx')
    refs=references()
    # Extended abstract: concise scientific narrative with complete citation parity.
    d=setup();heading(d,'Extended Abstract',1);paragraph(d,abstract)
    heading(d,'1. Introduction',1)
    paragraph(d,'Glacial lakes are expanding and outburst floods pose growing risk across mountain regions (Shugar et al., 2020; Zheng et al., 2021; Taylor et al., 2023). The 2023 Sikkim cascade underscores the local stakes, while satellite reconstructions of the South Lhonak event and earlier Landsat analyses show both the value and limits of retrospective observation (Sattar et al., 2025; Yu et al., 2024; Veh et al., 2018). Unsupervised time-series anomaly scores require especially careful interpretation (Sarfraz et al., 2024).')
    heading(d,'2. Literature',1)
    paragraph(d,'Optical and radar missions provide complementary observations, while water indices support candidate surface masks (Drusch et al., 2012; Torres et al., 2012; McFeeters, 1996; Xu, 2006). Himalayan GLOF studies and regional assessments include Landsat, optical/SAR reconstruction, operational-system proposals, and a 2026 unreviewed preprint (Veh et al., 2018; Yu et al., 2024; Kumar et al., 2020; NRSC/ISRO, 2025; Kahn et al., 2026).')
    paragraph(d,'Attention and masked reconstruction motivate temporal representation learning, and recent work extends these ideas to multivariate sequences and satellite imagery (Vaswani et al., 2017; He et al., 2022; Zerveas et al., 2021; Cong et al., 2022). Earth-observation foundation models and time-series masked autoencoders broaden this design space (Tseng et al., 2023; Szwarcman et al., 2024; Li et al., 2023; Song et al., 2023).')
    paragraph(d,'Classical time-series monitoring includes seasonal-break detection, continuous change detection, and temporal segmentation (Verbesselt et al., 2010; Zhu and Woodcock, 2014; Kennedy et al., 2010). Our comparators draw on isolation forests, one-class support estimation, robust low-rank recovery, and cumulative-sum monitoring (Liu et al., 2008; Schölkopf et al., 2001; Candès et al., 2011; Lin et al., 2010; Page, 1954). Benchmark design and event-based precision/recall remain central cautions (Wu and Keogh, 2021; Tatbul et al., 2018; Sarfraz et al., 2024).')
    heading(d,'3. Data and Methods',1)
    paragraph(d,'We analyzed 141 Sentinel-2 scenes from two 160 × 160 pixel windows at South Lhonak and Khangchung Tsho: 117 FIT, 17 VAL, and 7 EVAL. Collection-1 Level-2A products retain native BOA quantification and additive-offset metadata, which were applied before analysis (European Space Agency, 2023). Candidate zones are FIT-derived spectral regions, not validated lake boundaries. In the 50-fit sensitivity matrix, deterministic validation partitioning yielded 350 target-bearing and 80 no-target configuration-block instances; no-target blocks were excluded from loss aggregation.')
    table(d,'Table 1. Scene support and median clear-pixel fraction by site and split.', ['Region','FIT scenes / clear fraction','VAL scenes / clear fraction','EVAL scenes / clear fraction'],table1)
    table(d,'Table 2. 50-fit FIT/VAL sensitivity matrix.', ['Configuration','Mean VAL MSE ± SD','Target / no-target blocks','Seeds'],table2,font=7)
    figure(d,'fig1_corpus_and_zone_support','Figure 1. Observation timeline and both FIT-derived zone maps.',width=6.1)
    figure(d,'fig2_radiometry_and_tail_guard','Figure 2. Native reflectance distributions and guarded patch-mean indices.',width=6.1)
    heading(d,'4. Results',1)
    paragraph(d,'The primary validation masked MSE was 0.8498 ± 0.2371 across five seeds. Descriptive per-site FIT/VAL and EVAL score distributions and matched autumn summaries are reported in Table 4 and Fig. 5; they do not estimate independent seasonal generalization. The full-rank five-channel robust-PCA residual is near machine precision because its fitted subspace spans the input vectors; a separately labeled rank-2 SVD residual is shown as a sensitivity diagnostic, not a replacement baseline.')
    table(d,'Table 4. Site-specific scene-by-seed medians [IQR], with autumn windows shown separately.', ['Metric','SL FIT/VAL','SL Autumn 18–21','SL EVAL','KT FIT/VAL','KT Autumn 18–21','KT EVAL'],table3,font=5.9)
    figure(d,'fig3_corrective_50fit_diagnostics','Figure 3. Five-seed primary losses and the 50-fit matrix.',width=6.1)
    figure(d,'fig4_multiyear_anomaly_trajectories','Figure 4. Descriptive anomaly trajectories by site; seed bands are not population uncertainty.',width=6.1)
    figure(d,'fig5_zone_and_autumn_contrasts','Figure 5. Site-separated zone contrasts and descriptive October–November autumn scores from the primary model.',width=6.1)
    heading(d,'5. Interpretation and Limitations',1)
    paragraph(d,'Event-onset timing, false-alert burden, seasonal transfer and incremental warning value are not estimable from this corpus: event timing lacks independent review, operational delivery availability and alarm declarations were not measured, Khangchung Tsho is not an independently verified event-free control, and no rolling-origin autumn models were fit. The within-scene zone contrast is conditional on unvalidated candidate zones. Seeds and scenes share the same two-site corpus and do not provide independent lake-level replication.')
    heading(d,'6. Conclusion',1);paragraph(d,'This study provides reproducible descriptive scores for a fixed two-lake corpus. It does not establish a validated warning system, seasonal transfer, event timing, alert burden or added value over fair baselines.')
    heading(d,'Acknowledgement',1);paragraph(d,NOTICE)
    heading(d,'References',1)
    for ref in refs: refparagraph(d,ref)
    d.core_properties.title=TITLE;d.core_properties.author='Aditya Shrestha';d.save(DOC/'ICT_CEEL_2027_Extended_Abstract.docx')
    # Full paper, with the requested hierarchy and figures placed beside the methods/results they support.
    d=setup(compact=True);heading(d,'Abstract',1);paragraph(d,abstract);paragraph(d,'Keywords: '+KEYWORDS,bold=True)
    heading(d,'1. Introduction',1)
    paragraph(d,'Glacial lakes are growing across mountain regions, increasing exposure to outburst floods (Shugar et al., 2020; Zheng et al., 2021; Taylor et al., 2023). The October 2023 Sikkim cascade illustrates the consequences of interacting hazards (Sattar et al., 2025). Optical and radar reconstructions have documented the South Lhonak event, and earlier satellite time-series work shows how surface change can be tracked retrospectively (Veh et al., 2018; Yu et al., 2024).')
    paragraph(d,'Here we evaluate a missingness-aware temporal masked autoencoder on Sentinel-2 observations from South Lhonak and Khangchung Tsho. The study reports descriptive reconstruction and anomaly scores, not a warning capability. Unsupervised time-series anomaly results depend on event definitions, benchmark construction and evaluation scope (Sarfraz et al., 2024).')
    heading(d,'2. Literature',1)
    paragraph(d,'Sentinel-2 and Sentinel-1 offer complementary optical and radar views, while NDWI-family indices provide useful but imperfect candidate water masks (Drusch et al., 2012; Torres et al., 2012; McFeeters, 1996; Xu, 2006). Prior GLOF studies combine satellite records with physical or operational evidence, including work on South Lhonak and Khangchung (Veh et al., 2018; Yu et al., 2024; Kumar et al., 2020; NRSC/ISRO, 2025; Kahn et al., 2026). These studies motivate retrospective surface monitoring, but do not make a spectral change score a direct measure of lake drainage.')
    paragraph(d,'A masked sequence model is an exploratory complement when clouds and snow leave irregular gaps: attention can combine spectral context across a scene with each location’s observed history. Transformers and masked autoencoders provide the architectural basis, while multivariate time-series and satellite adaptations show how these ideas can be applied to incomplete observations (Vaswani et al., 2017; He et al., 2022; Zerveas et al., 2021; Cong et al., 2022; Tseng et al., 2023; Szwarcman et al., 2024; Li et al., 2023; Song et al., 2023). We compare these diagnostics with established change-detection and anomaly methods; event-aware benchmark design remains necessary before interpreting scores as warning performance (Verbesselt et al., 2010; Zhu and Woodcock, 2014; Kennedy et al., 2010; Liu et al., 2008; Schölkopf et al., 2001; Candès et al., 2011; Lin et al., 2010; Page, 1954; Wu and Keogh, 2021; Tatbul et al., 2018; Sarfraz et al., 2024).')
    heading(d,'3. Data and Methods',1)
    heading(d,'3.1 Study Regions and 2018–2023 Sentinel-2 Corpus',2)
    paragraph(d,'The corpus contains 141 Sentinel-2 observations from South Lhonak and Khangchung Tsho, with 117 FIT, 17 VAL and 7 EVAL scenes. The 20 m windows use the study reference coordinates 27.915° N, 88.204° E and 27.988333° N, 88.815° E, respectively, snapped to the nearest 20 m pixel center in EPSG:32645; these reference points are not validated lake outlines. FIT comprises 2018–2021 and January–June 2023; VAL comprises 2022 and July–September 2023; the predefined uncertain 2–4 October 2023 interval is excluded; EVAL begins at the 4 October cutoff. This retrospective partition includes pre-monsoon 2023 observations in FIT and holds 2022 and July–September 2023 out for validation. Because 2022 VAL falls between FIT periods, validation measures reconstruction within this retrospective corpus rather than forward-time prediction; 2018–2021 FIT autumn scenes remain in-sample. Each window is 160 × 160 pixels at 20 m in EPSG:32645, partitioned into 100 non-overlapping 16 × 16 patches. Native metadata records '+baseline_text+'; all are Sentinel-2 Collection-1 baselines. During ingestion, reflectance was derived from each product’s native BOA quantification value and radiometric additive offset in its SAFE metadata, without additional cross-baseline harmonization (European Space Agency, 2023).')
    table(d,'Table 1. Symbols used in the model and score definitions.', ['Symbol','Definition'],[
        ['ρ̄₀₃, ρ̄₀₈, ρ̄₁₁','Clear-pixel patch means of native BOA reflectance for B03, B08 and B11.'],
        ['xₜ,ₚ,𝚌; x̃ₜ,ₚ,𝚌','Observed model feature and its FIT-standardized value at scene t, patch p and channel c.'],
        ['medianFIT,c; MADFIT,c','FIT-only channel median and median absolute deviation.'],
        ['x̂ₜ,ₚ,𝚌','Reconstruction of the masked standardized target.'],
        ['zₜ,ₚ; PCAFIT','Patch-level latent representation and the rank-8 PCA transform fitted to FIT latent vectors.'],
        ['DOYₜ; Δt','Acquisition day of year and revisit gap in days.'],
        ['Ωₜ⁽ᵍ⁾; Cₜ,ₚ','Masked supported patches in zone g and observed target channels for patch p.'],
        ['Pₜ; N₅(t,p)','Eligible Zone 0+1 patches and their five nearest distinct FIT reference acquisitions.'],
        ['α; δ','Fixed Score C weight α = 0.5 and water-index denominator threshold δ = 0.002.']])
    figure(d,'fig1_corpus_and_zone_support','Figure 1. Observation support and separate FIT-derived candidate-zone maps for both regions.',width=6.1)
    heading(d,'3.2 Patch Radiometry, Denominator Guard, and Candidate Zones',2)
    paragraph(d,'B03, B08 and B11 are retained in native signed form. Clear pixels use SCL classes 4, 5 and 6 with at least 25% valid support within a patch. Indices are calculated after patch-level averaging; denominators with absolute magnitude below 0.002 are undefined. Zones 0 and 1 represent FIT-derived core-water and buffer candidates, and Zone 2 is the remaining candidate background. They are not surveyed physical boundaries. Candidate water area is the count of pixels satisfying NDWI > 0 and MNDWI > 0 within Zones 0/1 and clear SCL classes 4/5/6, multiplied by the 20 m pixel area (0.0004 km²); it is a spectral-mask diagnostic, not validated water area or a drainage positive control. Sentinel mission and index definitions follow Drusch et al. (2012), McFeeters (1996) and Xu (2006).')
    equation(d,'NDWI = (ρ̄₀₃ − ρ̄₀₈)/(ρ̄₀₃ + ρ̄₀₈), |ρ̄₀₃ + ρ̄₀₈| ≥ δ; MNDWI = (ρ̄₀₃ − ρ̄₁₁)/(ρ̄₀₃ + ρ̄₁₁), |ρ̄₀₃ + ρ̄₁₁| ≥ δ; δ = 0.002',1)
    figure(d,'fig2_radiometry_and_tail_guard','Figure 2. Native reflectance distributions and denominator-guarded patch-mean indices.',width=6.1)
    heading(d,'3.3 Missingness-Aware Factorized T-MAE and 50-Fit Matrix',2)
    paragraph(d,'The primary model has a 64-dimensional latent representation, four attention heads, three encoder layers (two spatial layers followed by one temporal layer), one spatial decoder layer and 207,685 trainable parameters. Patch features are projected and added to learned patch-position and zone embeddings, a temporal projection and a modality embedding. The spatial encoder attends across observed patch locations within each scene. Its output is rearranged into one sequence per fixed patch location; a causal temporal encoder then attends only to that patch’s observed acquisition times. In notation, Hₜ = SpatialAttention(Xₜ, Mₜ) and Zₚ = CausalTemporalAttention(Hₚ, Mₚ), followed by spatial decoding and channel reconstruction. Unsupported patches are key-masked; masked targets remain visible only as learned mask tokens. The temporal projection is τₜ = [sin(2π(DOYₜ−1)/365.25), cos(2π(DOYₜ−1)/365.25), min(Δt/30, 5)], where Δt is the gap since the preceding acquisition; it represents cyclic season and a capped revisit gap, not monotonic elapsed time.')
    paragraph(d,'Training examples are chronological, site-specific blocks of at most eight scenes. In each training scene, half of the observed patches are sampled at random for masking; validation masks every other observed patch deterministically within each scene. Loss averages only masked channels with observed support; a block with no target is unavailable, not assigned zero loss. We fixed the primary configuration (d = 64, five channels, FIT-derived zones, context length 8) before running the 10-configuration, five-seed sensitivity matrix. Three-channel variants reconstruct B03/B08/B11; five-channel variants also reconstruct NDWI/MNDWI, so their losses average different target spaces and scales and are not directly comparable. Context-1 and context-8 validation windows also have different support: 65 versus 15 target-bearing blocks per fit, with 20 versus zero no-target blocks. Across the 50 fits, deterministic validation partitioning contains 350 target-bearing and 80 no-target configuration-block instances; no-target blocks are excluded from loss aggregation.')
    equation(d,'x̃ₜ,ₚ,𝚌 = (xₜ,ₚ,𝚌 − medianFIT,𝚌) / (1.4826 × MADFIT,𝚌)',2,keep_with_next=True)
    heading(d,'3.4 Anomaly Scores and Baseline Definitions',2)
    paragraph(d,'For Score A, each masked supported patch receives the mean squared error across its observed target channels; the score averages those patch errors within the specified zone. Score A in Table 4 is for candidate Zones 0+1, and Zone 2 provides the background comparator. Score B projects latent patch representations through a rank-8 PCA fitted only to FIT latent vectors, then averages each eligible candidate patch’s five smallest same-patch distances to distinct FIT acquisitions within a circular ±45-day seasonal window; a FIT query is excluded from its own references. Score C equally combines FIT-robust standardized log Score A and Score B using the median and 1.4826 × MAD fitted on FIT scores. These definitions and support sets are given in Equations (3)–(5). The seasonal-residual comparator averages, across supported patches, each patch’s median Euclidean distance to the five closest complete FIT feature vectors from the same patch within ±45 day-of-year; it is unavailable when fewer than five references exist. Spatial persistence is the mean Euclidean distance to the immediately preceding complete feature vector at the same patch. Isolation Forest, one-class SVM, robust PCA and fixed-rule Page CUSUM are additional descriptive comparators (ν = 0.05, robust-PCA tolerance 10⁻⁷, CUSUM κ = 0.5 and h = 4, without resets). Candidate baseline scores can be unavailable where support is insufficient. Thresholds are not operational alarm points.')
    equation(d,'Aₜ⁽ᵍ⁾ = (1 / |Ωₜ⁽ᵍ⁾|) Σₚ∈Ωₜ⁽ᵍ⁾ [(1 / |Cₜ,ₚ|) Σ𝚌∈Cₜ,ₚ (x̃ₜ,ₚ,𝚌 − x̂ₜ,ₚ,𝚌)²]',3)
    equation(d,'Bₜ = (1 / |Pₜ|) Σₚ∈Pₜ [(1 / 5) Σᵢ∈N₅(t,p) ‖PCAFIT(zₜ,ₚ) − PCAFIT(zᵢ,ₚ)‖₂]',4)
    equation(d,'Cₜ = 0.5 zFIT(ln Aₜ⁽⁰⁺¹⁾) + 0.5 zFIT(Bₜ),  zFIT(u) = [u − medianFIT(u)] / [1.4826 × MADFIT(u)]',5)
    heading(d,'4. Results',1)
    table(d,'Table 2. Site and split support; clear fraction is the scene median in each cell.', ['Region','FIT n / clear','VAL n / clear','EVAL n / clear'],table1)
    table(d,'Table 3. FIT/VAL 50-fit exploratory matrix; SD reflects seed variation on the fixed corpus.', ['Configuration','Mean VAL MSE ± SD','Target / no-target blocks','Seeds'],table2,font=7)
    figure(d,'fig3_corrective_50fit_diagnostics','Figure 3. Primary validation losses and the capacity/factorial sensitivity matrix.',width=6.1)
    paragraph(d,'The five seed-specific checkpoints selected at their best validation epochs for the prespecified primary configuration had mean selected masked MSE 0.8498 (SD 0.2371). Inference across five seeds and 141 acquisitions yielded 705 seed–scene records for Scores A, B and C. Classical comparators were fit once using FIT observations and then scored across acquisitions. Unconstrained five-channel robust PCA produced a residual near machine precision because its fitted subspace spans the five-channel input space; Table 4 therefore also reports a separately labeled rank-2 SVD residual as a constrained low-rank sensitivity diagnostic.')
    paragraph(d,'For Table 4, seed values are first reduced to one median per acquisition; displayed medians and IQRs summarize distinct scenes, and n counts scenes. The autumn-matched FIT→EVAL comparison is the primary descriptive contrast: six medians are lower and two higher across the eight site-by-metric contrasts (Scores A, B, C and candidate area). Prior-autumn FIT scenes were used to fit the model, while EVAL scenes were held out; seven versus three/four scenes do not support an inferential test of event separation. FIT/VAL columns pool seasons and are shown only as broader descriptive context.')
    table(d,'Table 4. Scene-level medians [IQR] after collapsing seed scores within each acquisition; n counts distinct scenes.', ['Metric','SL FIT/VAL','SL Autumn 18–21','SL EVAL','KT FIT/VAL','KT Autumn 18–21','KT EVAL'],table3,font=5.9)
    figure(d,'fig4_multiyear_anomaly_trajectories','Figure 4. Descriptive trajectories by site; line and band spread reflects optimizer seeds.',width=6.1)
    figure(d,'fig5_zone_and_autumn_contrasts','Figure 5. Site-separated zone contrasts and October–November score distributions; each autumn point represents one scene after seed aggregation.',width=6.1)
    table(d,'Table 5. Estimand status under the available evidence.', ['Estimand','Status','Reason'],table4,font=7.5)
    heading(d,'5. Interpretation and Limitations',1)
    paragraph(d,'Only within-scene zone contrast is descriptively estimable, conditional on unvalidated spectral zones. Event timing, false-alert burden, seasonal transfer and incremental warning value remain unestimable because independent event/control records and exposure, rolling-origin fits, and a common calibrated operating point are absent. Khangchung is not an independently verified control; two sites, optimizer seeds and dependent patches do not provide independent lake-level replication.')
    paragraph(d,'Across South Lhonak’s three EVAL scenes, Score A medians were 1.399, 0.539 and 0.528 (9 October, 26 October and 13 November); the initial elevation was not sustained. Seasonal-residual scoring was unavailable because five complete same-patch FIT references were unsupported. Zone 0+1 minus Zone 2 medians were 0.121 at South Lhonak (seven prior FIT scenes) and 0.072 at Khangchung (five), versus 0.189 and 0.231 in EVAL. Six of eight matched autumn medians were lower in EVAL; prior FIT scenes were used for training. Candidate area at South Lhonak was 1.144 km² in prior autumns and 1.074 km² in EVAL, but is not validated shoreline or water volume. Snow, ice, shadow and classification effects were not separated.')
    heading(d,'6. Conclusion',1)
    paragraph(d,'Sentinel-GL generated surface diagnostics from this incomplete Sentinel-2 corpus. At South Lhonak, Score A was elevated in the first EVAL scene but fell in the next two; matched autumn summaries did not show uniform increases over in-sample FIT scenes. These results do not establish detection or warning value. Further assessment requires independently validated geometry, event/control records, multiple independent lakes and rolling-origin evaluation at a calibrated operating point.')
    heading(d,'Acknowledgement',1);paragraph(d,NOTICE);paragraph(d,'Sentinel-2 Collection-1 Level-2A native BOA quantification, additive-offset metadata, processing baselines and radiometric tags were retained. Copernicus Sentinel data are provided under a free, full and open policy with attribution; only derived summaries and figures are distributed here.')
    heading(d,'References',1)
    for ref in refs: refparagraph(d,ref,size=7.5)
    d.core_properties.title=TITLE;d.core_properties.author='Aditya Shrestha';d.core_properties.subject='Exploratory retrospective two-lake study'
    d.save(DOC/'ICT_CEEL_2027_Fullpaper.docx')
    # One packet binds source evidence, derived tables, figures and all outputs.
    inputs=[manifest_path,PKG/'analysis_report.json',PKG/'predictions.json',PKG/'baseline_suite.json',ROOT/'project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_EVAL_AUDIT_001.json',ROOT/'project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_EVAL_PROTOCOL_FREEZE_001.json',ROOT/'project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_SENSITIVITY_SUMMARY_001.json',DOC/'citation_registry_sr1.json']
    packet={'schema_version':1,'status':'RESULT_BOUND_EXPLORATORY_MANUSCRIPT_PACKET','title':TITLE,'abstract_word_count':len(abstract.split()),'inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},'figures':json.loads((FIG/'figure_manifest.json').read_text()),'tables':{'site_split_support':table1,'corrected_matrix':table2,'score_distributions':table3,'estimands':table4},'outputs':{n:sha(DOC/n) for n in ['ICT_CEEL_2027_Abstract.docx','ICT_CEEL_2027_Extended_Abstract.docx','ICT_CEEL_2027_Fullpaper.docx']}}
    (TABLES/'result_bindings.json').write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps({'status':packet['status'],'abstract_words':packet['abstract_word_count'],'outputs':packet['outputs']},indent=2))
if __name__=='__main__':main()
