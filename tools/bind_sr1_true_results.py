#!/usr/bin/env python3
"""Create a hash-bound result packet for the retained true-SR1 execution."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--attempt',default='attempt_004'); ap.add_argument('--analysis',default='analysis_001')
    ap.add_argument('--figure-dir',type=Path,default=Path('docs/figures/sr1_true'))
    ap.add_argument('--output-dir',type=Path,default=Path('docs/tables/sr1_true'))
    args=ap.parse_args()
    source=ROOT/'data/derived/sr1_multiyear'/args.attempt/'observations.sr1.json'
    analysis=source.parent/args.analysis
    report=json.loads((analysis/'analysis_report.json').read_text())
    predictions=json.loads((analysis/'predictions.json').read_text())
    histories=json.loads((analysis/'training_histories.json').read_text())
    figure_manifest=ROOT/args.figure_dir/'figure_manifest.json'
    figures=json.loads(figure_manifest.read_text())
    manifest=json.loads(source.read_text())
    out=ROOT/args.output_dir; out.mkdir(parents=True,exist_ok=False)
    active=[r for r in manifest['observations'] if r.get('status')!='EXPORT_FAILED']
    role_counts=Counter(p['role'] for p in predictions)
    score_summary={}
    for role in ('FIT','VAL','EVENT_UNCERTAIN','EVAL'):
        rows=[p for p in predictions if p['role']==role]
        values={}
        for key in ('score_a_zone0_1','score_b_zone0_1','score_c'):
            finite=[float(p[key]) for p in rows if p.get(key) is not None and np.isfinite(p[key])]
            values[key]={'median':float(np.median(finite)) if finite else None,'row_count':len(finite)}
        score_summary[role]={'unique_scene_count':len({p['observation_id'] for p in rows}),
                             'seed_rows':len(rows),'scores':values}
    site_year=Counter((r['site_id'],r['acquired_at'][:4]) for r in active)
    seeds=sorted({int(p['seed']) for p in predictions})
    packets={
        'schema_version':1,
        'status':'LOCAL_EXPLORATORY_EXECUTION_WITH_OPEN_RESEARCH_BLOCKERS',
        'origin':'observational; 2023 rows retained with exact parent source-manifest timestamps and receipts',
        'protocol_sha256':report['protocol_sha256'],
        'attempt':args.attempt,'analysis':args.analysis,
        'inputs':{},
        'observations':{
            'retained_source_manifest_rows':len(active),
            'failed_selected_rows':manifest.get('failed_scene_count',0),
            'role_prediction_rows_across_seeds':dict(role_counts),
            'unique_sites':sorted({r['site_id'] for r in active}),
            'site_year_acquisition_counts':{f'{s}__{y}':n for (s,y),n in sorted(site_year.items())},
            'provider_range_bytes':manifest['transfer_received_bytes'],
            'transfer_cap_bytes':manifest['transfer_cap_bytes'],
            'cumulative_received_and_reserved_bytes':manifest['cumulative_received_and_reserved_bytes'],
        },
        'training':{
            'seed_ids':seeds,'seed_count':len(seeds),
            'parameter_count':report['model_parameter_count'],
            'device':report['device'],
            'per_seed':[{k:h['history'][k] for k in ('seed','best_epoch','best_val_loss','epochs_completed',
                        'optimizer_steps','stopping_status','device','parameter_count')} for h in histories],
        },
        'score_summary':score_summary,
        'estimands':report['estimands'],
        'interpretation_limits':report['limitations'],
        'files':{
            'source_manifest':rel(source),
            'protocol':'project/sr1_protocol_true_v4.json',
            'analysis_report':rel(analysis/'analysis_report.json'),
            'predictions':rel(analysis/'predictions.json'),
            'training_histories':rel(analysis/'training_histories.json'),
            'figure_manifest':rel(figure_manifest),
            'fit_scaler':report['scaler_path'],
            'zones':{site:info['path'] for site,info in report['zones'].items()},
            'arithmetic_replay':{str(item['seed']):item['path'] for item in report['arithmetic_replay_files']},
            'implementation_sources':{
                'acquisition':'source/runners/run_sr1_multiyear.py',
                'analysis':'source/runners/analyze_sr1_true_pipeline.py',
                'features':'source/sentinel_gl/sr1_features.py',
                'model':'source/sentinel_gl/sr1_model.py',
                'audit':'tools/audit_sr1_true_execution.py',
            },
        },
    }
    for name,path in packets['files'].items():
        if isinstance(path,str):
            p=ROOT/path; packets['inputs'][name]={'path':path,'sha256':sha(p)}
        elif isinstance(path,dict) and name=='implementation_sources':
            for component,p in path.items():packets['inputs'][f'implementation_{component}']={'path':p,'sha256':sha(ROOT/p)}
    for k,p in packets['files']['zones'].items():packets['inputs'][f'zone_{k}']={'path':p,'sha256':sha(ROOT/p)}
    for k,p in packets['files']['arithmetic_replay'].items():packets['inputs'][f'replay_seed_{k}']={'path':p,'sha256':sha(ROOT/p)}
    for figure in figures['figures']:
        for fmt in ('png','pdf'):
            path=figure[fmt];packets['inputs'][f"figure_{figure['stem']}_{fmt}"]={'path':path,'sha256':sha(ROOT/path)}
    packet_path=out/'result_bindings.json'
    packet_path.write_text(json.dumps(packets,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':packets['status'],'sha256':sha(packet_path),'output':rel(packet_path),
                      'observations':len(active),'scores':len(predictions),'seeds':seeds},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
