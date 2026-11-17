"""Measured index tails and a fixed five-seed representation sensitivity study."""
from __future__ import annotations
import argparse,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def med(values):return float(np.median(values)) if values else None
def residual_records(analysis):
    rows=[json.loads(x) for x in (analysis/'inference/scores.v2.jsonl').read_text().splitlines()]
    output=[]
    for row in rows:
        if row['method']!='tmae_score_a' or row['sensor']!='SENTINEL_2_OPTICAL':continue
        sums={'reflectance_only_components':0.,'all_objective_components':0.};counts={k:0 for k in sums}
        for fold in row['score_a_components']['quadrant_folds']:
            if fold['status']!='ESTIMATED':continue
            for i,(target,pred) in enumerate(zip(fold['target_values'],fold['predicted_values'])):
                if target is None:continue
                error=(pred-target)**2
                sums['all_objective_components']+=error;counts['all_objective_components']+=1
                if i<3:sums['reflectance_only_components']+=error;counts['reflectance_only_components']+=1
        output.append({'observation_id':row['observation_id'],'site_id':row['site_id'],'seed':row['seed'],
          'event_phase':row['event_phase'],**{k:sums[k]/counts[k] if counts[k] else None for k in sums},
          'channel_quadrant_counts':counts})
    return output

def summaries(records):
    groups=defaultdict(list)
    for row in records:
        for metric in ['reflectance_only_components','all_objective_components']:
            if row[metric] is not None:groups[(row['site_id'],row['observation_id'],row['event_phase'],metric)].append(row[metric])
    values=defaultdict(list)
    for (site,oid,phase,metric),scores in groups.items():values[(site,phase,metric)].append(med(scores))
    return [{'site_id':site,'event_phase':phase,'metric':metric,'acquisitions':len(scores),'median_residual':med(scores)}
            for (site,phase,metric),scores in sorted(values.items())]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--companion',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    primary=ROOT/args.analysis;companion=ROOT/args.companion
    panel=load(primary/'panel.v2.json');transform=load(primary/'training/fit_transform.v2.json')
    fitted=set(transform['fit_observation_ids']);rr,cc=np.indices((499,499));parity=(rr//16+cc//16)%2==0
    values=defaultdict(list)
    for row in panel['observations']:
        if row['sensor']!='SENTINEL_2_OPTICAL' or row['observation_id'] not in fitted:continue
        if sha(ROOT/row['values_file'])!=row['sha256']:raise ValueError('Source hash mismatch')
        with np.load(ROOT/row['values_file'],allow_pickle=False) as z:
            for ch in ['NDWI','MNDWI']:
                i=row['channels'].index(ch);valid=z['observed_valid'][i]&parity
                values[ch].append(z['values'][i][valid].astype(np.float64))
    tails={}
    for ch,parts in values.items():
        a=np.concatenate(parts);i=transform['channel_order'].index(ch);mean=transform['means'][i];outside=np.abs(a)>1;deviations=(a-mean)**2;total=float(deviations.sum())
        tails[ch]={'fit_parity':0,'fit_pixels':int(a.size),'outside_minus1_plus1':int(outside.sum()),
            'outside_fraction':float(outside.mean()),'squared_deviation_share_percent':float(100*deviations[outside].sum()/total) if total else None,
            'minimum':float(a.min()),'maximum':float(a.max()),'fitted_mean':mean,'fitted_scale':transform['scales'][i],
            'all_squared_deviations_about_fitted_mean':total}
    left=residual_records(primary);right=residual_records(companion)
    lt=load(primary/'training/training_report.v2.json');rt=load(companion/'training/training_report.v2.json')
    assert lt['model_configuration']['optical_representation']=='five_channel'
    assert rt['model_configuration']['optical_representation']=='reflectance_only'
    assert lt['source_manifest_sha256']==rt['source_manifest_sha256']
    left_keys={(r['observation_id'],r['seed']) for r in left if r['reflectance_only_components'] is not None}
    right_keys={(r['observation_id'],r['seed']) for r in right if r['reflectance_only_components'] is not None}
    common=left_keys&right_keys
    result={'schema_version':1,'status':'MEASURED_DESCRIPTIVE_REPRESENTATION_SENSITIVITY','source_manifest_sha256':lt['source_manifest_sha256'],
        'index_tails':tails,'primary':{'analysis':args.analysis.as_posix(),'representation':'five_channel',
          'score_sha256':sha(primary/'inference/scores.v2.jsonl'),'summaries':summaries(left)},
        'companion':{'analysis':args.companion.as_posix(),'representation':'reflectance_only','seeds':[r['seed'] for r in rt['seeds']],
          'score_sha256':sha(companion/'inference/scores.v2.jsonl'),'summaries':summaries(right)},
        'matched_raw_band_components':{'common_seed_acquisition_pairs':len(common),
          'primary':summaries([r for r in left if (r['observation_id'],r['seed']) in common]),
          'companion':summaries([r for r in right if (r['observation_id'],r['seed']) in common])},
        'per_seed_acquisition_residuals':{'primary':left,'companion':right},
        'interpretation':['Each model was actually retrained with its declared channel representation; same architecture, seeds, budget and selection rule.',
          'Primary raw-band components use the primary model predictions; companion raw-band residuals come from separate reflectance-only fits.',
          'All residuals use pre-period standardized channels; compare shared raw-band components for a common target, not differently scaled objectives as independent performance metrics.',
          'Seven-channel model architecture retained; index inputs/presence and targets are excluded in the companion. Unused index standardizer entries do not enter its tokens or objective.',
          'No clamping, fabricated inputs, causal attribution, independent event replication or operational warning inference.']}
    (ROOT/args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'tails':tails,'matched_pairs':len(common)},indent=2))
if __name__=='__main__':main()
