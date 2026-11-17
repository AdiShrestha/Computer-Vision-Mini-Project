#!/usr/bin/env python3
"""Recompute descriptive FIT/VAL aggregates from audited corrected SR-1 fits."""
from __future__ import annotations
import hashlib, json, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('data/derived/sr1_multiyear/attempt_006/reexecution_001/run_manifest.json')
AUDIT=Path('project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_REEXECUTION_AUDIT_001.json')
OUT=Path('project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_SENSITIVITY_SUMMARY_001.json')
def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
 run=json.loads((ROOT/RUN).read_text()); audit=json.loads((ROOT/AUDIT).read_text())
 if run.get('status')!='PASS_COMPLETE_VAL' or audit.get('status')!='PASS_ALL_50_CHECKPOINT_VAL_REPLAYS':
  raise ValueError('corrected execution or independent VAL replay is not complete')
 if audit['inputs'].get(RUN.as_posix())!=sha(RUN) or audit.get('scope',{}).get('eval_rows_scored')!=0:
  raise ValueError('audit bindings or EVAL exclusion mismatch')
 grouped=defaultdict(list)
 for row in audit['results']:
  if not row['status'].startswith('PASS_'): raise ValueError('audit includes a failed checkpoint')
  key=(row['family'],row['cell'],row['channels'],row['zones_enabled'],row['context_scenes_max'])
  grouped[key].append(row)
 summaries=[]
 for key, rows in sorted(grouped.items()):
  vals=[float(r['recorded_best_val_masked_mse']) for r in rows]
  summaries.append({'family':key[0],'cell':key[1],'channels':key[2],'zones_enabled':key[3],
    'context_scenes_max':key[4],'seed_count':len(vals),'mean_best_val_masked_mse':statistics.mean(vals),
    'sample_sd':statistics.stdev(vals) if len(vals)>1 else None,
    'target_bearing_val_blocks_total':sum(r['target_bearing_val_blocks'] for r in rows),
    'no_target_val_blocks_total':sum(len(r['no_target_val_blocks']) for r in rows),
    'valid_channel_targets_total':sum(r['valid_channel_targets'] for r in rows),
    'checkpoint_hashes':[r['checkpoint_sha256'] for r in rows]})
 out={'schema_version':1,'status':'PASS_DESCRIPTIVE_ARITHMETIC_ONLY','inputs':{RUN.as_posix():sha(RUN),AUDIT.as_posix():sha(AUDIT)},
  'scope':{'fit_rows_used':117,'validation_rows_used':17,'eval_rows_read':0,'checkpoint_replay_passed':50,
  'interpretation':'Descriptive checkpoint-selected VAL loss arithmetic only. Seeds are optimizer repeats on one corpus, not independent lakes or events. No inferential test, feature-importance, superiority, generalization or alarm claim is made.'},
  'cell_summaries':summaries,'no_target_val_blocks_total':sum(len(r['no_target_val_blocks']) for r in audit['results']),
  'finite_target_bearing_val_blocks_total':sum(r['target_bearing_val_blocks'] for r in audit['results'])}
 (ROOT/OUT).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'path':OUT.as_posix(),'sha256':sha(OUT),'status':out['status'],'cells':len(summaries)},indent=2))
if __name__=='__main__': main()
