"""Independent arithmetic replay from source pixels and recorded predictions.

Does not acquire data, train, edit checkpoints or assert warning validity.
"""
from pathlib import Path
import argparse,hashlib,json,math
from collections import Counter
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--inference-dir',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 base=ROOT/args.analysis;inf=ROOT/args.inference_dir if args.inference_dir else base/'inference'
 manifest=ROOT/'data/derived/expansion_2023/observations.v2.json';doc=read(manifest);records={r['observation_id']:r for r in doc['observations']}
 train=read(base/'training/training_report.v2.json');transform=read(base/'training/fit_transform.v2.json');summary=read(inf/'analysis_summary.v2.json')
 assert train['source_manifest_sha256']==summary['source_manifest_sha256']==sha(manifest)
 channels=transform['channel_order'];fits=set(transform['fit_observation_ids']);cut=transform['fit_end_exclusive_utc']
 rr,cc=np.indices((499,499));par0=(rr//16+cc//16)%2==0;par1=~par0
 quads=[(rr<249)&(cc<249),(rr<249)&(cc>=249),(rr>=249)&(cc<249),(rr>=249)&(cc>=249)]
 assert np.all(np.sum(quads,axis=0)==1) and not np.any(par0&par1)
 sums=np.zeros(7);squares=np.zeros(7);counts=np.zeros(7,dtype=np.int64);pooled={};hashes=[]
 for oid,r in records.items():
  p=ROOT/r['values_file'];rp=ROOT/r['receipt_file'];receipt=read(rp);assert sha(p)==r['sha256']==receipt['sha256']
  hashes.extend([{'path':r['values_file'],'sha256':sha(p)},{'path':r['receipt_file'],'sha256':sha(rp)}])
  if oid in fits:assert datetime.fromisoformat(r['acquired_at'].replace('Z','+00:00'))<datetime.fromisoformat(cut.replace('Z','+00:00'))
  with np.load(p,allow_pickle=False) as z:v=z['values'];m=z['observed_valid'];assert v.dtype==np.float32 and m.dtype==bool
  assert v.shape==m.shape==(len(r['channels']),499,499) and np.isfinite(v[m]).all() and np.isnan(v[~m]).all()
  pool=[]
  for q,area in enumerate(quads):
   target=[None]*7;den=[None]*7
   for local,ch in enumerate(r['channels']):
    idx=channels.index(ch);mask=m[local]&par1&area;den[idx]=int(mask.sum())
    if den[idx] and transform['scales'][idx] is not None:
     a=v[local][mask].astype(np.float64);target[idx]=float(np.float32(np.mean((a-transform['means'][idx])/transform['scales'][idx],dtype=np.float64)))
   pool.append({'target':target,'denominators':den})
  pooled[oid]=pool
  if oid in fits:
   for local,ch in enumerate(r['channels']):
    idx=channels.index(ch);a=v[local][m[local]&par0].astype(np.float64)
    counts[idx]+=len(a);sums[idx]+=np.sum(a,dtype=np.float64);squares[idx]+=np.dot(a,a)
 means=sums/counts;scales=np.sqrt(squares/counts-means**2)
 assert counts.tolist()==transform['counts']
 meanerror=float(np.max(np.abs(means-np.array(transform['means']))));scaleerror=float(np.max(np.abs(scales-np.array(transform['scales']))))
 assert meanerror<=1e-12 and scaleerror<=1e-12
 rows=[json.loads(line) for line in (inf/'scores.v2.jsonl').read_text().splitlines()];bykey={(r['seed'],r['observation_id'],r['method']):r for r in rows}
 maxscoreerror=0.;maxtrutherror=0.;folds=0;unsupported=0;foldcover=Counter();cstates={r['seed']:r for r in summary['score_c_fit_standardizers']}
 for r in rows:
  assert r['query_parity']==1 and r['fitting_parity']==0
  if r['method']=='tmae_score_a':
   comp=r['score_a_components'];oid=r['observation_id'];assert len(comp['quadrant_folds'])==4
   assert {f['quadrant'] for f in comp['quadrant_folds']}==set(range(4));n=0;e=0.
   for f in comp['quadrant_folds']:
    expected=pooled[oid][f['quadrant']];assert f['source_pixel_denominators']==expected['denominators']
    mask=np.array([x is not None for x in expected['target']]);count=int(mask.sum());assert count==f['target_channel_count']
    if not count:assert f['mean_squared_error'] is None;unsupported+=1;continue
    assert f['status']=='ESTIMATED';folds+=1;foldcover[r['seed']]+=1
    for i in range(7):
     if mask[i]:
      diff=abs(f['target_values'][i]-expected['target'][i]);maxtrutherror=max(maxtrutherror,diff);assert diff<=1e-7
      e+=(f['predicted_values'][i]-f['target_values'][i])**2;n+=1
     else:assert f['target_values'][i] is None and f['predicted_values'][i] is None
   assert n==r['denominator']
   if n:maxscoreerror=max(maxscoreerror,abs(e/n-r['score']));assert abs(e/n-r['score'])<=1e-10
   else:assert r['score'] is None
  if r['method']=='tmae_score_c':
   a=bykey[(r['seed'],r['observation_id'],'tmae_score_a')]['score'];b=bykey[(r['seed'],r['observation_id'],'tmae_score_b')]['score'];state=cstates[r['seed']]
   if r['score'] is not None:
    ast=state['score_a_standardizer'];bst=state['score_b_standardizer']
    expected=.5*(a-ast['mean'])/ast['scale']+.5*(b-bst['mean'])/bst['scale'];assert abs(expected-r['score'])<=1e-10
   elif a is None or b is None:assert r['status'].startswith('UNAVAILABLE')
 for seed in train['seeds']:
  p=base/f"training/seed_{seed['seed']}/selected_checkpoint.pt";assert sha(p)==seed['selected_checkpoint_sha256'];assert seed['selected_checkpoint_replay_absolute_error']==0
 att=read(inf/'representative_seed0_attention.json');path=inf/'representative_seed0_attention.npz';N=len(att['context_token_order']);row_sum_error=0.;weights=0
 with np.load(path,allow_pickle=False) as z:
  for key in z.files:
   a=z[key];digest=hashlib.sha256();digest.update(str(a.dtype).encode());digest.update(json.dumps(list(a.shape),separators=(',',':')).encode());digest.update(np.ascontiguousarray(a).tobytes())
   assert digest.hexdigest()==att['attention_arrays_sha256'][key]
   if key.startswith('mean_'):continue
   a=z[key];assert a.ndim==4 and a.shape[-2:]==(N,N) and np.isfinite(a).all() and np.min(a)>=0
   row_sum_error=max(row_sum_error,float(np.max(np.abs(a.astype(np.float64).sum(axis=-1)-1))));weights+=a.size
 assert row_sum_error<1e-5
 plot_means_error=0.
 with np.load(ROOT/'docs/figures/2023/attention_plot_means.npz',allow_pickle=False) as plotted,np.load(path,allow_pickle=False) as raw:
  assert att['target_folds'][0]==0
  for kind in ['encoder','decoder']:
   expected=raw[kind+'_layer_0'][0].astype(np.float64).mean(axis=0)
   difference=float(np.max(np.abs(expected-plotted[kind+'_layer_0_fold_0'])));plot_means_error=max(plot_means_error,difference)
   assert difference<=1e-8
 for state in summary['baseline_states'].values():
  rp=state['inexact_alm_rpca']
  if rp['status']=='CONVERGED':assert rp['primal_residual']<=rp['tolerance'] and rp['dual_residual']<=rp['tolerance']
 assert summary['calibration']['threshold'] is None and summary['calibration']['exposure_lake_years'] is None
 report={'status':'PASS_INDEPENDENT_SOURCE_AND_ARITHMETIC_REPLAY','observations':len(records),'score_rows':len(rows),'source_bindings':len(hashes),'normalizer_mean_max_abs_error':meanerror,'normalizer_scale_max_abs_error':scaleerror,'score_a_max_absolute_error':maxscoreerror,'source_target_max_absolute_error':maxtrutherror,'supported_fold_seed_pairs':folds,'unsupported_fold_seed_pairs':unsupported,'supported_quadrants_per_seed':dict(foldcover),'attention_weights_checked':weights,'attention_row_sum_max_abs_error':row_sum_error,'attention_plot_float64_mean_max_abs_error':plot_means_error,'manifest_sha256':sha(manifest),'scores_sha256':sha(inf/'scores.v2.jsonl'),'limitations':['No independent model retraining, latent embedding recomputation or human/external scientific replication in this arithmetic audit','Source pixels, saved targets/predictions, parity, fitted moments, fusion arithmetic, checkpoint digests and attention support inspected; calibrated warning metrics remain unavailable'],'inputs':hashes}
 (ROOT/args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))
if __name__=='__main__':main()
