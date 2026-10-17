"""Source-bound 2023 descriptive tables and compact manuscript figures.

This script renders recorded outputs only. It cannot train or acquire data.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.family':'DejaVu Sans',
 'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import SymLogNorm
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures/2023';TABLE=ROOT/'docs/tables/2023'
SOUTH='PILOT-SOUTH-LHONAK';KHANG='PILOT-KHANGCHUNG-TSHO'
NAMES={SOUTH:'South Lhonak',KHANG:'Khangchung Tsho'}
COLORS={SOUTH:'#175c92',KHANG:'#bc591d'}
CUT=datetime(2023,10,3,tzinfo=timezone.utc)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def stamp(value):return datetime.fromisoformat(value.replace('Z','+00:00'))
def arrays(row):
 p=ROOT/row['values_file']
 if sha(p)!=row['sha256']:raise ValueError('Source array hash changed')
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(fig,name):
 OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.04);plt.close(fig)
def csvfile(name,rows):
 TABLE.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (TABLE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def event(ax):
 ax.axvspan(datetime(2023,10,3,tzinfo=timezone.utc),datetime(2023,10,5,tzinfo=timezone.utc),color='#6b4651',alpha=.2,lw=0)
 ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,3,5,7,9,11]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
 ax.set_xlim(datetime(2023,1,1,tzinfo=timezone.utc),datetime(2023,11,20,tzinfo=timezone.utc));ax.grid(alpha=.18)
def source_figures(rows):
 table=[]
 for r in rows:
  z=arrays(r);v=z['values'];m=z['observed_valid'];d={k:r[k] for k in ['observation_id','site_id','sensor','product_id','acquired_at','sha256']}
  d.update({'qa_clear_fraction':r.get('qa_clear_count',0)/249001 if r['sensor']=='SENTINEL_2_OPTICAL' else None,
    'pixel_area_m2':400,'ROI_area_km2':99.6004})
  for i,ch in enumerate(r['channels']):
   values=v[i][m[i]];d[ch+'_valid']=int(values.size);d[ch+'_median']=float(np.median(values)) if values.size else None
   d[ch+'_mean']=float(np.mean(values,dtype=np.float64)) if values.size else None
   d[ch+'_min']=float(values.min()) if values.size else None;d[ch+'_max']=float(values.max()) if values.size else None
  table.append(d)
 csvfile('observation_measurements.csv',table)
 fig,axes=plt.subplots(2,1,figsize=(7.05,2.7),sharex=True,layout='constrained')
 for ax,site in zip(axes,[SOUTH,KHANG]):
  opt=[r for r in rows if r['site_id']==site and r['sensor']=='SENTINEL_2_OPTICAL']
  ax.plot([stamp(r['acquired_at']) for r in opt],[r['qa_clear_count']/249001 for r in opt],'-o',color=COLORS[site],lw=.8,ms=3,label='Optical SCL 4/5/6 fraction')
  sar=[r for r in rows if r['site_id']==site and r['sensor']=='SENTINEL_1_IW_GRD']
  ax.scatter([stamp(r['acquired_at']) for r in sar],[-.09]*len(sar),marker='D',color='#333333',s=20,label='SAR acquisition (QA axis N/A)')
  ax.axhline(0,color='.7',lw=.5);event(ax);ax.set_ylim(-.18,1.03);ax.set_yticks([0,.5,1]);ax.set_ylabel('Clear fraction');ax.set_title(NAMES[site],loc='left',weight='bold')
 axes[0].legend(loc='upper left',ncols=2,frameon=False)
 save(fig,'fig1_observation_timeline')
 optical=[r for r in rows if r['site_id']==SOUTH and r['sensor']=='SENTINEL_2_OPTICAL']
 pre=max((r for r in optical if stamp(r['acquired_at'])<CUT),key=lambda r:r['acquired_at'])
 post=min((r for r in optical if stamp(r['acquired_at'])>=datetime(2023,10,5,tzinfo=timezone.utc)),key=lambda r:r['acquired_at'])
 zp,zq=arrays(pre),arrays(post)
 fig,axes=plt.subplots(2,4,figsize=(7.05,4.15),layout='constrained')
 for ax,z,r,letter in [(axes[0,0],zp,pre,'a'),(axes[0,1],zq,post,'b')]:
  rgb=np.moveaxis(z['values'][[2,1,0]],0,-1);valid=np.all(z['observed_valid'][:3],axis=0)
  rgb=np.clip(rgb/.4,0,1);rgb[~valid]=1
  ax.imshow(rgb,extent=[0,9.98,0,9.98]);ax.set_title(f'({letter}) B11/B08/B03\n'+r['acquired_at'][:10]);ax.set_xlabel('Easting offset (km)');ax.set_ylabel('Northing offset (km)')
 def ecdf(ax,z,index,label,color,style):
  a=z['values'][index][z['observed_valid'][index]].astype(np.float64);a.sort()
  if len(a):
   ids=np.linspace(0,len(a)-1,min(1500,len(a))).astype(int);ax.plot(a[ids],(ids+1)/len(a),style,color=color,lw=.8,label=label)
 for i,ch,c in [(0,'B03','#258039'),(1,'B08','#5e4090'),(2,'B11','#a55f20')]:
  ecdf(axes[0,2],zp,i,ch+' pre',c,'-');ecdf(axes[0,2],zq,i,ch+' post',c,'--')
 axes[0,2].set_box_aspect(1);axes[0,2].set_title('(c) BOA reflectance');axes[0,2].set_xlabel('Reflectance');axes[0,2].set_ylabel('Empirical CDF');axes[0,2].legend(frameon=False,fontsize=5.4)
 for i,ch,c in [(3,'NDWI','#167180'),(4,'MNDWI','#923a71')]:
  ecdf(axes[0,3],zp,i,ch+' pre',c,'-');ecdf(axes[0,3],zq,i,ch+' post',c,'--')
 axes[0,3].set_box_aspect(1);axes[0,3].set_xscale('symlog',linthresh=1);axes[0,3].set_title('(d) Spectral indices');axes[0,3].set_xlabel('Index (symlog)');axes[0,3].set_ylabel('Empirical CDF');axes[0,3].legend(frameon=False,fontsize=5.4)
 sar=next(r for r in rows if r['site_id']==SOUTH and r['sensor']=='SENTINEL_1_IW_GRD');zs=arrays(sar)
 for col,(pol,index,key,validkey) in enumerate([('VV',0,'raw_db','raw_db_valid'),('VV',0,'signed_values','signed_valid'),('VH',1,'raw_db','raw_db_valid'),('VH',1,'signed_values','signed_valid')]):
  ax=axes[1,col];a=np.ma.array(zs[key][index],mask=~zs[validkey][index]);raw=key=='raw_db'
  im=ax.imshow(a,extent=[0,9.98,0,9.98],cmap='gray' if raw else 'coolwarm',
   **({'vmin':-35,'vmax':5} if raw else {'norm':SymLogNorm(linthresh=1e-4,vmin=-1,vmax=1)}))
  ax.set_title(f'({chr(101+col)}) {pol} '+('raw dB' if raw else 'signed'));ax.set_xlabel('Easting offset (km)');ax.set_ylabel('Northing offset (km)')
  fig.colorbar(im,ax=ax,shrink=.7,pad=.025,**({} if raw else {'ticks':[-1,-.01,0,.01,1]}))
 save(fig,'fig2_source_distributions')
 return {'pre_image':pre['observation_id'],'post_image':post['observation_id'],'sar_image':sar['observation_id'],'table':table}
def result_figures(analysis,inference=None):
 inference=inference or analysis/'inference'
 train=load(analysis/'training/training_report.v2.json');summary=load(inference/'analysis_summary.v2.json')
 scores=[json.loads(x) for x in (inference/'scores.v2.jsonl').read_text().splitlines()]
 fig,axes=plt.subplots(1,2,figsize=(7.05,2.15),layout='constrained')
 seedrows=[]
 for seed in train['seeds']:
  history=load(analysis/f"training/seed_{seed['seed']}/fit_monitor_history.v2.json")['trace'];x=[r['epoch'] for r in history]
  for ax,key in zip(axes,['fit_masked_mse','development_monitor_masked_mse']):
   y=[r[key] for r in history];ax.plot(x,y,lw=.85,label=f"Seed {seed['seed']}");ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Masked MSE');ax.grid(alpha=.15)
  seedrows.append({k:v for k,v in seed.items() if not isinstance(v,(list,dict))})
 axes[0].set_title('(a) Parity-0 optimization');axes[1].set_title('(b) Parity-1 development monitor');axes[0].legend(ncols=2,frameon=False)
 save(fig,'fig3_optimization');csvfile('training_by_seed.csv',seedrows)
 methods=['tmae_score_a','tmae_score_b','tmae_score_c','isolation_forest','psd_rbf_one_class_svm','inexact_alm_rpca','cusum']
 labels=['T-MAE A (MSE)','T-MAE B (latent distance)','T-MAE C (fit-standardized)','Isolation Forest','One-Class SVM','RPCA residual','CUSUM (IF input)']
 fig,axes=plt.subplots(4,2,figsize=(7.05,5.4),sharex=True,layout='constrained');axes=axes.ravel()
 for ax,method,label in zip(axes,methods,labels):
  for site in [SOUTH,KHANG]:
   eligible=[r for r in scores if r['method']==method and r['site_id']==site and r['sensor']=='SENTINEL_2_OPTICAL']
   byobs={}
   for r in eligible:byobs.setdefault(r['observation_id'],[]).append(r)
   times=[];med=[];lo=[];hi=[]
   for oid,rs in sorted(byobs.items(),key=lambda x:x[1][0]['acquired_at']):
    vals=[r['score'] for r in rs if r['score'] is not None]
    times.append(stamp(rs[0]['acquired_at']));med.append(float(np.median(vals)) if vals else np.nan);lo.append(min(vals) if vals else np.nan);hi.append(max(vals) if vals else np.nan)
   ax.plot(times,med,'-o',color=COLORS[site],lw=.8,ms=2,label=NAMES[site]);ax.fill_between(times,lo,hi,color=COLORS[site],alpha=.14,lw=0)
  event(ax);ax.set_title(label,loc='left');ax.tick_params(labelbottom=True)
 axes[0].set_yscale('log')
 axes[2].set_yscale('symlog',linthresh=1)
 axes[5].set_yscale('log')
 axes[7].axis('off');axes[7].legend(*axes[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.01,.98),frameon=False)
 axes[7].text(.01,.45,'Band: range of five training seeds.\nGaps retain missing observations.\nOct 3–4 bracket shaded; all dates parity 1.\nMethods have different numerical scales.',transform=axes[7].transAxes,fontsize=7,va='top')
 save(fig,'fig4_anomaly_trajectories')
 att=summary['representative_seed0_attention'];meta=load(inference/att['metadata_path']);p=inference/att['arrays_path']
 with np.load(p,allow_pickle=False) as z:captured={k:z[k] for k in z.files}
 fig,axes=plt.subplots(1,2,figsize=(7.05,2.9),layout='constrained')
 available=[k for k in captured if not k.startswith('mean_')];plot_means={}
 for ax,kind in zip(axes,['encoder','decoder']):
  key=sorted(k for k in available if kind in k)[0];a=captured[key]
  matrix=a[0].astype(np.float64).mean(axis=0);plot_means[kind+'_layer_0_fold_0']=matrix
  im=ax.imshow(matrix,cmap='magma',aspect='auto',vmin=0);n=matrix.shape[0]
  positions=np.unique(np.linspace(0,n-1,min(6,n)).astype(int));tokens=meta['context_token_order']
  names=[tokens[i]['acquired_at'][5:10]+' q'+str(tokens[i]['quadrant']) for i in positions]
  ax.set_xticks(positions,names,rotation=45,ha='right');ax.set_yticks(positions,names)
  ax.set_xlabel('Key token');ax.set_ylabel('Query token');ax.set_title(f'Seed 0: {kind} layer 0, q0 fold');fig.colorbar(im,ax=ax,shrink=.8,pad=.015)
 save(fig,'fig5_attention');np.savez_compressed(OUT/'attention_plot_means.npz',**plot_means)
 csvfile('scores.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in scores])
 return train,summary,scores,meta

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path);parser.add_argument('--inference-dir',type=Path);args=parser.parse_args()
 manifest=ROOT/'data/derived/expansion_2023/observations.v2.json';doc=load(manifest);source=source_figures(doc['observations'])
 record={'source_manifest_sha256':sha(manifest),'source_figure_selection':{k:v for k,v in source.items() if k!='table'},'figures':[]}
 if args.analysis:
  analysis=args.analysis if args.analysis.is_absolute() else ROOT/args.analysis
  inference=ROOT/args.inference_dir if args.inference_dir else analysis/'inference'
  train,summary,scores,attention=result_figures(analysis,inference)
  record.update({'analysis_path':str(analysis.relative_to(ROOT)),'training_report_sha256':sha(analysis/'training/training_report.v2.json'),'inference_summary_sha256':sha(inference/'analysis_summary.v2.json'),'scores_sha256':sha(inference/'scores.v2.jsonl'),'attention_query':attention['query_observation_id'],'inference_path':str(inference.relative_to(ROOT))})
 for p in sorted(OUT.glob('fig*.*')):record['figures'].append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 record['scientific_limits']=['Two developmental ROIs, same-scene spatial monitoring','All preperiod parameter fitting is development-exposed','Single September SAR acquisition, no SAR change sequence','Candidate query mask, no independently validated lake boundary','No event/control labels, calibrated annual exposure, lead-time or detection-performance claim','Attention is internal distribution, not causal explanation']
 (OUT/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'figures':len(record['figures']),'source_observations':len(doc['observations'])}))
if __name__=='__main__':main()
