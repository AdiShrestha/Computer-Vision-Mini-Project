"""Replay RPCA fit-matrix binding and monitor residuals from source ROI pixels."""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def array_hash(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(a.tobytes());return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,required=True);parser.add_argument('--observations',type=Path,default=Path('data/derived/expansion_2023_recentered/observations.v2.json'));parser.add_argument('--inference-dir',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 BASE=ROOT/args.analysis;INF=ROOT/args.inference_dir if args.inference_dir else BASE/'inference'
 manifest=read(ROOT/args.observations);records={r['observation_id']:r for r in manifest['observations']};transform=read(BASE/'training/fit_transform.v2.json');summary=read(INF/'analysis_summary.v2.json')
 rr,cc=np.indices((499,499));parities=[(rr//16+cc//16)%2==i for i in [0,1]];quads=[(rr<249)&(cc<249),(rr<249)&(cc>=249),(rr>=249)&(cc<249),(rr>=249)&(cc>=249)];means={}
 for oid,r in records.items():
  with np.load(ROOT/r['values_file'],allow_pickle=False) as z:v=z['values'];m=z['observed_valid']
  for p,parity in enumerate(parities):
   vector=[];mask=[];pixel_count=0
   for local,ch in enumerate(r['channels']):
    idx=transform['channel_order'].index(ch);weighted=0.;total=0
    for quad in quads:
     support=m[local]&parity&quad;n=int(support.sum())
     if n:
      x=v[local][support].astype(np.float64);normalized=(x-transform['means'][idx])/transform['scales'][idx]
      pooled=float(np.float32(np.mean(normalized,dtype=np.float64)));weighted+=pooled*n;total+=n
    vector.append(weighted/total if total else 0.);mask.append(total>0);pixel_count+=total
   means[(oid,p)]=(np.asarray(vector,dtype=np.float64),np.asarray(mask,dtype=bool),pixel_count)
 report=[];maxscoreerror=0.;checked=0
 scores=[json.loads(line) for line in (INF/'scores.v2.jsonl').read_text().splitlines()]
 for sensor,state in summary['baseline_states'].items():
  rp=state['inexact_alm_rpca'];D=np.stack([means[(oid,0)][0] for oid in rp['fit_observation_ids']]);assert array_hash(D)==rp['fit_values_sha256']
  L=np.array(rp['low_rank']);S=np.array(rp['sparse']);primal=float(np.linalg.norm(D-L-S)/np.linalg.norm(D));assert abs(primal-rp['primal_residual'])<=1e-14
  if rp['status']=='CONVERGED':assert primal<=rp['tolerance'] and rp['dual_residual']<=rp['tolerance']
  else:assert rp['status']=='NONCONVERGED_MAX_ITERATIONS'
  _,singular,vt=np.linalg.svd(L,full_matrices=False);rank=int(np.sum(singular>rp['tolerance']*max(float(singular[0]),1.)));assert rank==rp['rank'];basis=vt[:rank].T
  for r in scores:
   if r['method']!='inexact_alm_rpca' or r['sensor']!=sensor:continue
   row,mask,pixels=means[(r['observation_id'],1)];n=int(mask.sum());assert r['query_pixel_channel_count']==pixels
   if n<=rank:assert r['score'] is None;continue
   design=basis[mask];coefficient=np.linalg.lstsq(design,row[mask],rcond=None)[0];residual=row[mask]-design@coefficient
   score=float(np.linalg.norm(residual)/math.sqrt(n));difference=abs(score-r['score']);assert difference<=1e-12;maxscoreerror=max(maxscoreerror,difference);checked+=1
  report.append({'sensor':sensor,'fit_rows':len(D),'rank':rank,'independent_primal_residual':primal,'reported_dual_residual':rp['dual_residual'],'iterations':rp['iterations']})
 output={'status':'PASS_SOURCE_BOUND_FIT_MATRIX_AND_MONITOR_PROJECTION_REPLAY','groups':report,'monitor_scores_recomputed':checked,'maximum_score_abs_difference':maxscoreerror,'limits':'Dual/KKT statistics inspected from recorded solver output; full dual matrices were not retained, so no independent dual-certificate reconstruction is claimed.'}
 (ROOT/args.output).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
