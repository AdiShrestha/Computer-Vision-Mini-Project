"""Compute a sealed, status-aware manuscript packet from actual run artifacts.

No empirical numbers or convergence outcomes are prescribed in this writer.
The old epoch and previously generated manuscripts remain separate evidence.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS={'tmae_score_a':'T-MAE A','tmae_score_b':'T-MAE B','tmae_score_c':'T-MAE C',
 'isolation_forest':'Isolation Forest','psd_rbf_one_class_svm':'One-Class SVM',
 'inexact_alm_rpca':'RPCA residual','cusum':'CUSUM'}
NAMES={'PILOT-SOUTH-LHONAK':'South Lhonak','PILOT-KHANGCHUNG-TSHO':'Khangchung Tsho'}
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def median(a):return float(np.median(a)) if a else None

def bind_results(analysis,observations,figures,sensitivity):
    train=load(analysis/'training/training_report.v2.json');inf=analysis/'inference';summary=load(inf/'analysis_summary.v2.json')
    panel=load(analysis/'panel.v2.json');manifest=load(observations);source_rows=manifest['observations']
    if sha(observations)!=train['source_manifest_sha256'] or train['source_manifest_sha256']!=summary['source_manifest_sha256']:
        raise ValueError('Source/analysis binding mismatch')
    rows=[json.loads(x) for x in (inf/'scores.v2.jsonl').read_text().splitlines()]
    descriptions=[]
    for site in sorted(NAMES):
        for method in METHODS:
            groups=defaultdict(list)
            for row in rows:
                if row['site_id']==site and row['method']==method and row['sensor']=='SENTINEL_2_OPTICAL' and row['score'] is not None:
                    groups[(row['observation_id'],row['acquired_at'],row['event_phase'])].append(row['score'])
            pre=[median(v) for (_,date,phase),v in groups.items() if phase=='PRE_BREACH_CANDIDATE']
            post=[median(v) for (_,date,phase),v in groups.items() if phase=='POST_EVENT_CANDIDATE']
            september=[median(v) for (_,date,phase),v in groups.items() if date.startswith('2023-09')]
            descriptions.append({'site_id':site,'method':method,'pre_n':len(pre),'post_n':len(post),
                'pre_median':median(pre),'post_median':median(post),'september_median':median(september)})
    sites={};baselines=Counter()
    for site in NAMES:
        selected=[r for r in panel['observations'] if r['site_id']==site];opt=[r for r in selected if r['sensor']=='SENTINEL_2_OPTICAL']
        fractions=[]
        for row in opt:
            receipt=load(ROOT/row['receipt_file']);baselines[receipt['native']['baseline']]+=1
            fractions.append(receipt['qa_clear_count']/float(np.prod(row['grid']['shape'])))
        sites[site]={'name':NAMES[site],'observations':len(selected),'optical_observations':len(opt),
            'optical_by_phase':dict(Counter(r['scope']['event_phase'] for r in opt)),
            'optical_wholly_unsupported':sum(not any(r['observed_valid_counts_by_channel']) for r in opt),
            'clear_fraction_min':min(fractions) if fractions else None,'clear_fraction_max':max(fractions) if fractions else None,
            'clear_fraction_median':median(fractions),'grid':selected[0]['grid'],
            'first_post_optical':min((r['acquired_at'] for r in opt if r['scope']['event_phase']=='POST_EVENT_CANDIDATE'),default=None),
            'sar_dates':sorted({r['acquired_at'][:10] for r in selected if r['sensor']=='SENTINEL_1_IW_GRD'})}
    seeds=train['seeds'];counts={};possible=4*len(source_rows)
    for seed in seeds:
        folds=[f for r in rows if r['method']=='tmae_score_a' and r['seed']==seed['seed'] for f in r['score_a_components']['quadrant_folds']]
        available=sum(f['status']=='ESTIMATED' for f in folds)
        counts[str(seed['seed'])]={'estimated':available,'unavailable':possible-available,'possible':possible,
            'unavailable_statuses':dict(Counter(f['status'] for f in folds if f['status']!='ESTIMATED'))}
    losses=[r['selected_monitor_mse'] for r in seeds if r['selected_monitor_mse'] is not None]
    parameter_counts={r['parameter_count'] for r in seeds}
    if len(parameter_counts)!=1:raise ValueError('Seed architecture parameter counts disagree')
    paths={'source_manifest':observations,'panel':analysis/'panel.v2.json','training':analysis/'training/training_report.v2.json',
        'summary':inf/'analysis_summary.v2.json','scores':inf/'scores.v2.jsonl','execution':analysis/'execution_summary.v2.json',
        'figures':figures/'manifest.json','sensitivity':sensitivity,'geometry':observations.parent/'geometry_check.json',
        'protocol':ROOT/'project/domain_2023_protocol.json'}
    packet={'schema_version':2,'scope':'DESCRIPTIVE_TWO_ROI_RETROSPECTIVE_DEVELOPMENT_ANALYSIS',
        'inputs':{k:{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for k,p in paths.items()},
        'observation_count':len(source_rows),'unique_product_count':len({r['product_id'] for r in source_rows}),
        'sensor_counts':dict(Counter(r['sensor'] for r in source_rows)),
        'acquisition_years':sorted({r['acquired_at'][:4] for r in source_rows}),
        'date_min':min(r['acquired_at'] for r in source_rows),'date_max':max(r['acquired_at'] for r in source_rows),
        'sar_unique_products':len({r['product_id'] for r in source_rows if r['sensor']=='SENTINEL_1_IW_GRD'}),
        'sar_unique_dates':sorted({r['acquired_at'][:10] for r in source_rows if r['sensor']=='SENTINEL_1_IW_GRD'}),
        'sites':sites,'optical_processing_baselines':dict(sorted(baselines.items())),
        'parameter_count':parameter_counts.pop(),'fit_observations':train['fit_observation_count'],
        'fit_examples':train['train_examples_with_supported_targets'],'seed_count':len(seeds),'seeds':seeds,
        'selected_loss_min':min(losses) if losses else None,'selected_loss_max':max(losses) if losses else None,
        'model_configuration':train['model_configuration'],'quadrants_by_seed':counts,
        'score_rows':len(rows),'score_nulls':dict(Counter(r['method'] for r in rows if r['score'] is None)),
        'pre_post_descriptive':descriptions,
        'rpca':{k:{j:v for j,v in st['inexact_alm_rpca'].items() if j in ['status','algorithm','iterations','rank','primal_residual','dual_residual','kkt_residuals','tolerance','fit_observation_ids','fit_rows']}
                for k,st in summary['baseline_states'].items()},
        'calibration':summary['calibration'],'geometry':load(paths['geometry']),
        'score_b_reference_configurations':[{"seed":r['seed'],"states":{key:{k:v for k,v in st.items() if k in ['status','k','pca_rank','distinct_fit_acquisitions']} for key,st in r['states'].items()}} for r in summary['score_b_reference_states']],
        'index_tail_sensitivity':load(sensitivity),'figure_manifest':load(figures/'manifest.json'),
        'protocol':load(paths['protocol']),
        'null_policy':'Unavailable values remain null and render as unavailable; nonconvergence is a reportable status, not a writer failure.'}
    return packet

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--inference-dir',type=Path)
    ap.add_argument('--observations',type=Path,default=Path('data/derived/expansion_2023_recentered/observations.v2.json'))
    ap.add_argument('--figures',type=Path,default=Path('docs/figures/2023_recentered'))
    ap.add_argument('--sensitivity',type=Path,default=Path('data/derived/expansion_2023_recentered/index_tail_sensitivity.json'))
    ap.add_argument('--output',type=Path,default=Path('docs/tables/2023_recentered'));a=ap.parse_args()
    if a.inference_dir and (ROOT/a.inference_dir).resolve()!=(ROOT/a.analysis/'inference').resolve():
        ap.error('Use the bound analysis inference directory; separate overrides are not accepted')
    packet=bind_results(ROOT/a.analysis,ROOT/a.observations,ROOT/a.figures,ROOT/a.sensitivity)
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    (out/'result_bindings.json').write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
    with (out/'pre_post_descriptive.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(packet['pre_post_descriptive'][0]));w.writeheader();w.writerows(packet['pre_post_descriptive'])
    print(json.dumps({'status':'BOUND_ACTUAL_RESULTS_WITH_NULL_AND_SOLVER_STATUS_SEMANTICS','score_rows':packet['score_rows'],'sha256':sha(out/'result_bindings.json')}))
if __name__=='__main__':main()
