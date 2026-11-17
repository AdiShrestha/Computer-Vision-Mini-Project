#!/usr/bin/env python3
"""Create five source-bound figures for the corrected exploratory evaluation."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"source"))
from sentinel_gl.sr1_features import compute_guarded_indices

ATT=ROOT/"data/derived/sr1_multiyear/attempt_006"
PKG=ATT/"corrected_eval_001"
OUT=ROOT/"docs/figures/sr1_final_006"
SITES=["PILOT-SOUTH-LHONAK","PILOT-KHANGCHUNG-TSHO"]
LABEL={SITES[0]:"South Lhonak",SITES[1]:"Khangchung Tsho"}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight',pad_inches=.16)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.16)
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',pad_inches=.16)
    plt.close(fig)

def main():
    if OUT.exists(): raise SystemExit(f"immutable figure directory exists: {OUT}")
    OUT.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})
    manifest_path=ATT/'observations.sr1.json'; manifest=json.loads(manifest_path.read_text())
    pred=json.loads((PKG/'predictions.json').read_text())
    report=json.loads((PKG/'analysis_report.json').read_text())
    rows=[r for r in manifest['observations'] if r.get('status')!='EXPORT_FAILED']
    # Figure 1: corpus timeline, support, candidate zones and actual paired scenes.
    fig=plt.figure(figsize=(10.2,6.2)); gs=fig.add_gridspec(2,2,height_ratios=[.8,1],hspace=.3,wspace=.16)
    ax_time=fig.add_subplot(gs[0,:]); ax_sl=fig.add_subplot(gs[1,0]); ax_kt=fig.add_subplot(gs[1,1])
    colors={'FIT':'#3274a1','VAL':'#e1812c','EVAL':'#c44e52'}
    for si,site in enumerate(SITES):
        rr=sorted([r for r in rows if r['site_id']==site],key=lambda r:r['acquired_at'])
        dates=[];clear=[]
        for r in rr:
            dates.append(np.datetime64(r['acquired_at'][:10]))
            with np.load(ROOT/r['values_file'],allow_pickle=False) as d:
                clear.append(float((np.isin(d['scl'],[4,5,6]) & d['valid'].all(axis=0)).mean()))
        ax_time.scatter(dates, np.full(len(dates),si)+np.asarray(clear)*.55, c=[colors.get(next((p['role'] for p in pred if p['observation_id']==r['site_id']+'__'+r['item_id']), 'FIT')) for r in rr],s=14,alpha=.8)
    ax_time.axvspan(np.datetime64('2023-07-01'),np.datetime64('2023-10-04'),color=colors['VAL'],alpha=.08)
    ax_time.axvspan(np.datetime64('2023-10-04'),np.datetime64('2023-11-20'),color=colors['EVAL'],alpha=.10)
    ax_time.set_xlim(np.datetime64('2018-01-01'),np.datetime64('2023-11-20'));ax_time.set_yticks([.25,1.25],['South Lhonak','Khangchung Tsho'])
    ax_time.set_ylabel('Site and clear-pixel fraction');ax_time.set_title('Observation timeline and valid-pixel support')
    ax_time.grid(axis='x',alpha=.2)
    palette=matplotlib.colors.ListedColormap(['#2878b5','#efc44b','#dddddd'])
    for ax,site in [(ax_sl,SITES[0]),(ax_kt,SITES[1])]:
        zone_path=ROOT/report['zones'][site]['path']
        with np.load(zone_path,allow_pickle=False) as z: zones=z['zones']
        ax.imshow(zones,cmap=palette,vmin=0,vmax=2);ax.set_title(f'{LABEL[site]} FIT-derived candidate zones')
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Source-bound two-lake corpus (EVAL begins 4 October 2023)')
    save(fig,OUT/'fig1_corpus_and_zone_support')
    # Figure 2: observed native reflectance and guarded patch-mean indices.
    bands=[[],[],[]];ndwi=[];mndwi=[]
    for r in rows:
        with np.load(ROOT/r['values_file'],allow_pickle=False) as d:
            refl=d['reflectance'];valid=d['valid'];scl=d['scl']
            for i in range(3): bands[i].append(refl[i][valid[i]&np.isfinite(refl[i])])
            # patch means are calculated using the declared clear-pixel rule.
            for y in range(0,160,16):
                for x in range(0,160,16):
                    clear=np.isin(scl[y:y+16,x:x+16],[4,5,6])
                    means=[]
                    for i in range(3):
                        mask=clear&valid[i,y:y+16,x:x+16]&np.isfinite(refl[i,y:y+16,x:x+16])
                        means.append(float(np.mean(refl[i,y:y+16,x:x+16][mask])) if mask.mean()>=.25 else np.nan)
                    a,b=compute_guarded_indices(np.array(means[0]),np.array(means[1]),np.array(means[2]),delta=.002)
                    if np.isfinite(a): ndwi.append(float(a))
                    if np.isfinite(b): mndwi.append(float(b))
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.4))
    for i,(name,color) in enumerate(zip(['B03','B08','B11'],['#2878b5','#4c9f70','#d08031'])):
        axs[0].hist(np.concatenate(bands[i]),bins=70,density=True,histtype='step',lw=1.3,label=name,color=color)
    axs[0].set_xlabel('Native BOA reflectance');axs[0].set_ylabel('Density');axs[0].legend(frameon=False);axs[0].set_title('Negative values retained')
    axs[1].hist(ndwi,bins=60,density=True,alpha=.55,label=f'NDWI, n={len(ndwi)}',color='#2878b5')
    axs[1].hist(mndwi,bins=60,density=True,alpha=.55,label=f'MNDWI, n={len(mndwi)}',color='#d08031')
    axs[1].set_xlabel('Patch-mean index; |denominator| ≥ 0.002');axs[1].set_ylabel('Density');axs[1].legend(frameon=False);axs[1].set_title('Guarded patch means')
    fig.suptitle('Radiometry and denominator-guarded indices')
    save(fig,OUT/'fig2_radiometry_and_tail_guard')
    # Figure 3: retained corrected selected losses and complete corrected matrix summary.
    run=json.loads((ATT/'reexecution_001/run_manifest.json').read_text())
    hist=[r for r in run['jobs'] if r['family']=='primary']
    summary=json.loads((ROOT/'project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_SENSITIVITY_SUMMARY_001.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(9.8,3.8))
    xs=np.arange(len(hist));axs[0].bar(xs,[r['best_val_loss'] for r in hist],color='#3274a1')
    axs[0].set_xticks(xs,[str(r['seed']) for r in hist]);axs[0].set_xlabel('Optimizer seed');axs[0].set_ylabel('Selected VAL masked MSE');axs[0].set_title('Five primary checkpoints (d=64)')
    cells=summary['cell_summaries'];labels=[];means=[];sds=[]
    for c in cells:
        if c['family']=='capacity': lab=f"d={c['cell'].split('_')[-1]}"
        elif c['family']=='primary': lab='Primary d=64'
        else: lab=f"{c['channels']}ch z{int(c['zones_enabled'])} c{c['context_scenes_max']}"
        labels.append(lab);means.append(c['mean_best_val_masked_mse']);sds.append(c['sample_sd'])
    order=np.argsort(means);axs[1].barh(np.arange(len(order)),np.array(means)[order],xerr=np.array(sds)[order],color='#5b9a8b',alpha=.85)
    axs[1].set_yticks(np.arange(len(order)),[labels[i] for i in order],fontsize=6.5);axs[1].set_xlabel('Mean selected VAL masked MSE ± seed SD');axs[1].set_title('50-fit capacity and factorial matrix')
    fig.tight_layout();save(fig,OUT/'fig3_corrective_50fit_diagnostics')
    # Figure 4: corrected trajectories; five seeds are shown as individual paths,
    # with range bands to expose optimizer variation rather than scene uncertainty.
    fields=[('score_a_zone0_1','Score A'),('score_b_zone0_1','Score B'),('score_c','Score C'),('water_candidate_area_km2','Candidate water area (km²)'),('baselines.seasonal_climatology_residual','Seasonal residual'),('baselines.isolation_forest_score','Isolation Forest'),('baselines.one_class_svm_score','One-Class SVM'),('baselines.rpca_score','RPCA'),('page_cusum.statistic','Fixed CUSUM (Score C)')]
    def val(p,path):
        v=p
        for key in path.split('.'):v=v.get(key) if isinstance(v,dict) else None
        return v
    fig,axs=plt.subplots(3,3,figsize=(12,8),sharex=True)
    for ax,(field,label) in zip(axs.flat,fields):
        for site in SITES:
            rr=sorted([p for p in pred if p['site_id']==site],key=lambda p:p['acquired_at'])
            seed_curves=[]
            for seed in range(5):
                sr=[p for p in rr if p['seed']==seed]; dates=[]; ys=[]
                for p in sr:
                    v=val(p,field)
                    if v is not None and np.isfinite(v):dates.append(np.datetime64(p['acquired_at'][:10]));ys.append(float(v))
                if ys:
                    ax.plot(dates,ys,alpha=.23,lw=.65,color='#2878b5' if site==SITES[0] else '#d08031')
                    seed_curves.append((dates,ys))
            # Full scene grid is common; interpolate only for the seed min/max envelope.
            if seed_curves:
                all_dates=sorted({x for ds,_ in seed_curves for x in ds}); xx=np.array([x.astype('datetime64[D]').astype(int) for x in all_dates])
                arr=[]
                for ds,ys in seed_curves:
                    dx=np.array([x.astype('datetime64[D]').astype(int) for x in ds]);arr.append(np.interp(xx,dx,ys,left=np.nan,right=np.nan))
                mat=np.asarray(arr);lo=np.nanmin(mat,axis=0);hi=np.nanmax(mat,axis=0)
                ax.fill_between(all_dates,lo,hi,color='#2878b5' if site==SITES[0] else '#d08031',alpha=.13)
        ax.axvline(np.datetime64('2023-10-04'),color='black',ls='--',lw=.8)
        ax.axvspan(np.datetime64('2023-07-01'),np.datetime64('2023-10-04'),color='#e1812c',alpha=.06)
        ax.axvspan(np.datetime64('2023-10-04'),np.datetime64('2023-11-20'),color='#c44e52',alpha=.07)
        ax.set_title(label,fontsize=8);ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color='#2878b5',lw=2,label='South Lhonak'),
             Line2D([0],[0],color='#d08031',lw=2,label='Khangchung Tsho'),
             Line2D([0],[0],color='black',lw=1,ls='--',label='4 Oct 2023 EVAL boundary')]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.945))
    fig.subplots_adjust(top=.86)
    fig.supxlabel('Acquisition date (UTC)');fig.suptitle('Descriptive trajectories; thin lines are optimizer seeds, bands show seed range',y=.99)
    save(fig,OUT/'fig4_multiyear_anomaly_trajectories')
    # Figure 5: observed conditional zone contrasts and autumn score distributions.
    fig=plt.figure(figsize=(11.4,6.4)); gs=fig.add_gridspec(2,3,height_ratios=[1,1],hspace=.48,wspace=.38)
    ax_zone=fig.add_subplot(gs[:,0]);axs_right=[fig.add_subplot(gs[r,c]) for r in range(2) for c in (1,2)]
    zones=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    for seed in range(5):
        p=PKG/'replay'/f'seed_{seed}_score_replay.npz'
        with np.load(p,allow_pickle=False) as d:
            for i,role in enumerate(d['roles'].tolist()):
                z=d['zones'][i].astype(int);obs=d['observed'][i].astype(bool);mask=d['masked'][i].astype(bool)
                target=d['target'][i];predv=d['prediction'][i];err=np.full(100,np.nan)
                for j in np.flatnonzero(mask):
                    good=obs[j]&np.isfinite(target[j])
                    if good.any():err[j]=np.mean((predv[j,good]-target[j,good])**2)
                for name,sel in [('Zone 0+1',np.isin(z,[0,1])),('Zone 2',z==2)]:
                    vals=err[sel&np.isfinite(err)]
                    if vals.size:
                        oid=str(d['observation_ids'][i])
                        zones[d['site_ids'][i]][role][name][oid].append(float(np.median(vals)))
    roles=['FIT','VAL','EVAL'];x=np.arange(3)
    site_colors={SITES[0]:'#2878b5',SITES[1]:'#d08031'}
    for site in SITES:
        for name,style in [('Zone 0+1','-'),('Zone 2','--')]:
            scene_values={r:[float(np.median(seed_values)) for seed_values in zones[site][r][name].values()]
                          for r in roles}
            med=[np.median(scene_values[r]) if scene_values[r] else np.nan for r in roles]
            ax_zone.plot(x,med,marker='o',ls=style,color=site_colors[site],lw=1.5,
                         label=f'{LABEL[site]} · {name}')
            for xi,r in zip(x,roles):
                vals=scene_values[r]
                if vals:ax_zone.scatter(np.full(len(vals),xi),vals,s=8,color=site_colors[site],alpha=.35,zorder=3)
    ax_zone.set_xticks(x,roles);ax_zone.set_ylabel('Scene median of patch masked MSE (log scale)');ax_zone.set_yscale('log');ax_zone.set_title('(a) Zone contrast by site');ax_zone.grid(axis='y',alpha=.18)
    ax_zone.legend(frameon=False,fontsize=6.7,loc='best')
    metric_defs=[('score_a_zone0_1','Score A'),('score_b_zone0_1','Score B'),('score_c','Score C'),('water_candidate_area_km2','Candidate water area (km²)')]
    autumn_groups=[('2018–2021 FIT',lambda p:p['role']=='FIT' and p['acquired_at'][:4] in {'2018','2019','2020','2021'} and p['acquired_at'][5:7] in {'10','11'}),('2023 EVAL',lambda p:p['role']=='EVAL' and p['acquired_at'][5:7] in {'10','11'})]
    for ax,(field,label) in zip(axs_right,metric_defs):
        vals=[];positions=[];xt=[]
        for site_i,site in enumerate(SITES):
            for season_i,(season,select) in enumerate(autumn_groups):
                by_scene=defaultdict(list)
                for p in pred:
                    if p['site_id']==site and select(p) and p[field] is not None and np.isfinite(p[field]):
                        by_scene[p['observation_id']].append(float(p[field]))
                v=[float(np.median(values)) for values in by_scene.values()]
                posx=site_i*3+season_i+1;positions.append(posx);vals.append(v);xt.append(f'{"SL" if site_i==0 else "KT"}\n{"18–21" if season_i==0 else "2023"}')
        ax.boxplot(vals,positions=positions,widths=.62,showfliers=False,medianprops={'color':'black','lw':1.3},patch_artist=True,
                   boxprops={'facecolor':'#8db6d8','alpha':.75},whiskerprops={'color':'#526b7a'},capprops={'color':'#526b7a'})
        for xx,v in zip(positions,vals):
            if v:ax.scatter(np.full(len(v),xx),v,s=5,alpha=.27,color='#253746',zorder=3)
        ax.set_xticks(positions,xt,fontsize=6.5);ax.set_title(label,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('October–November scene summaries; seeds collapsed within each acquisition',y=.99)
    save(fig,OUT/'fig5_zone_and_autumn_contrasts')
    figlist=[]
    for stem in ['fig1_corpus_and_zone_support','fig2_radiometry_and_tail_guard','fig3_corrective_50fit_diagnostics','fig4_multiyear_anomaly_trajectories','fig5_zone_and_autumn_contrasts']:
        figlist.append({'stem':stem,**{ext:{'path':f'docs/figures/sr1_final_006/{stem}.{ext}','sha256':sha(OUT/f'{stem}.{ext}')} for ext in ['png','pdf','svg']}})
    idx={'schema_version':1,'status':'GENERATED_FROM_SOURCE_BOUND_CORRECTED_EVAL_EXPLORATORY','corrected_eval_report_sha256':sha(PKG/'analysis_report.json'),'corrected_eval_audit_sha256':sha(ROOT/'project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_EVAL_AUDIT_001.json'),'source_manifest_sha256':sha(manifest_path),'figures':figlist,'limits':['No harmonic or ridge outcomes are plotted because they are not in the frozen parent analysis.','No bootstrap intervals are shown for patch/scene clusters.','Every autumn scene contributes one seed-aggregated point; contrasts remain descriptive and rolling-origin E3 remains unestimated.']}
    (OUT/'figure_manifest.json').write_text(json.dumps(idx,indent=2)+'\n')
    print(json.dumps({'status':idx['status'],'figures':len(figlist),'out':OUT.relative_to(ROOT).as_posix()},indent=2))

if __name__=='__main__':main()
