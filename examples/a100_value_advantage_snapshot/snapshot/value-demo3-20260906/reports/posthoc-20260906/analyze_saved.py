"""CPU-only post-hoc audit of frozen holdout predictions; never selects a new training best."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
protocol=json.loads((args.input/'protocol.json').read_text())
logs={r['step']:r for r in map(json.loads,(args.input/'metrics.jsonl').read_text().splitlines()) if r.get('event')=='EVAL'}
steps=[]; matrices={}; records=[]; episode_rows=[]; manifest=[]; id_ref=None; grid_ref=None
scale=protocol['return_scale']; fps=30.0
sec=scale/fps
for path in sorted(args.input.glob('predictions-*.json')):
    step=int(path.stem.split('-')[-1]); obj=json.loads(path.read_text()); a=np.asarray(obj['rows'],dtype=float)
    assert obj['columns']==['episode_slot','frame','target','prediction']
    a=a[np.lexsort((a[:,1],a[:,0]))]; ids=obj['episode_ids']
    if id_ref is None: id_ref=ids; grid_ref=a[:,:3].copy()
    assert ids==id_ref and np.array_equal(a[:,:3],grid_ref), 'Unpaired evaluation grid'
    assert np.isfinite(a).all()
    errors=a[:,3]-a[:,2]; groups=[a[a[:,0]==j] for j in range(len(ids))]
    absmeans=[]; msemeans=[]; taus=[]; rhos=[]; reversals=[]; delta_errors=[]; final=[]; initial=[]
    for eid,g in zip(ids,groups):
        y,p=g[:,2],g[:,3]; e=p-y
        assert len(g)==32 and np.all(np.diff(g[:,1])>0)
        ma=float(np.abs(e).mean()); ms=float((e*e).mean())
        rho=float(spearmanr(y,p).statistic) if np.ptp(p)>0 else 0.
        tau=float(kendalltau(y,p).statistic) if np.ptp(p)>0 else 0.
        de=np.diff(p)-np.diff(y)
        absmeans.append(ma); msemeans.append(ms); rhos.append(rho); taus.append(tau)
        reversals.append(float(np.mean(np.diff(p)<0))); delta_errors.append(float(np.abs(de).mean()))
        final.append(float(abs(e[-1]))); initial.append(float(abs(e[0])))
        episode_rows.append(dict(step=step,episode=eid,mae=ma,rmse=ms**.5,bias=float(e.mean()),p95_absolute_error=float(np.quantile(abs(e),.95)),spearman=rho,kendall=tau,sampled_reverse_fraction=reversals[-1],sampled_delta_mae=delta_errors[-1],terminal_abs_error=final[-1]))
    am=np.asarray(absmeans); mm=np.asarray(msemeans)
    boot=np.random.default_rng(1000).integers(0,len(ids),size=(10000,len(ids)))
    ci=np.quantile(am[boot].mean(1),[.025,.975]).tolist()
    rmci=np.quantile(np.sqrt(mm[boot].mean(1)),[.025,.975]).tolist()
    log=logs[step]
    assert np.isclose(am.mean(),log['test']['episode_mae'],atol=1e-8)
    assert np.isclose(np.sqrt(mm.mean()),log['test']['rmse'],atol=1e-8)
    stages=[]
    for q in range(4):
        vals=[]
        for g in groups:
            progress=g[:,1]/g[-1,1]; mask=(progress>=q/4)&((progress<(q+1)/4) if q<3 else progress<=1)
            vals.append(np.abs(g[mask,3]-g[mask,2]).mean())
        stages.append(float(np.mean(vals)))
    rec=dict(step=step,train_ce=log['train']['ce'],test_ce=log['test']['ce'],train_mae=log['train']['mae'],test_mae=float(am.mean()),train_rmse=log['train']['rmse'],test_rmse=float(np.sqrt(mm.mean())),mae_ci95=ci,rmse_ci95=rmci,mae_seconds=float(am.mean()*sec),bias=float(errors.mean()),p95_abs_error=float(np.quantile(abs(errors),.95)),terminal_mae=float(np.mean(final)),initial_mae=float(np.mean(initial)),macro_spearman=float(np.mean(rhos)),macro_kendall=float(np.mean(taus)),sampled_reverse_fraction=float(np.mean(reversals)),sampled_delta_mae=float(np.mean(delta_errors)),quartile_mae=stages,rmse_skill_vs_train_constant_mean=float(1-mm.mean()/log['test']['baseline_rmse']**2))
    matrices[step]=(am,mm); records.append(rec); steps.append(step)
    manifest.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
best_mae=min(records,key=lambda r:r['test_mae'])['step']; best_rmse=min(records,key=lambda r:r['test_rmse'])['step']; best_ce=min(records,key=lambda r:r['test_ce'])['step']
candidates=sorted(set([s for s in [1500,3000,4750,best_mae,best_rmse,steps[-1]] if s in steps]))
comparisons=[]
for old in candidates:
    for new in candidates:
        if new<=old: continue
        am1,mm1=matrices[old]; am2,mm2=matrices[new]
        delta=am2-am1; dist=delta[boot].mean(1)
        rd=np.sqrt(mm2[boot].mean(1))-np.sqrt(mm1[boot].mean(1))
        comparisons.append(dict(older=old,newer=new,delta_mae_new_minus_old=float(delta.mean()),paired_mae_ci95=np.quantile(dist,[.025,.975]).tolist(),delta_mae_seconds=float(delta.mean()*sec),episodes_improved=int((delta<0).sum()),episodes_worsened=int((delta>0).sum()),delta_rmse_new_minus_old=float(np.sqrt(mm2.mean())-np.sqrt(mm1.mean())),paired_rmse_ci95=np.quantile(rd,[.025,.975]).tolist()))
result=dict(protocol='posthoc-saved-predictions-v1',original_selection_unchanged=protocol['selection'],episodes=len(id_ref),frames_per_episode=32,fps=fps,return_scale=scale,bootstrap='10000 paired whole-episode resamples seed1000; descriptive conditional intervals, not multiple-selection-adjusted; assumes episode independence',limitations=['Holdout used for selection, not unbiased test.','Ranks and reversals compare time-derived labels, not human physical-progress labels.','Sampled delta uses actual variable frame gaps, NOT 50-step advantages.','No logits persisted: cannot infer probability calibration, entropy or interval coverage from means.','No failure/rollout labels: no failure AUROC or policy-success evaluation.','Possible capture-session dependence remains unverified.'],observed_best=dict(ce=best_ce,mae=best_mae,rmse=best_rmse),metrics=records,paired=comparisons,input_sha256=manifest)
(args.output/'saved_prediction_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for name,rows in [('checkpoint_metrics.csv',records),('episode_metrics.csv',episode_rows),('paired_comparisons.csv',comparisons)]:
    with (args.output/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(13,8),dpi=160)
for ax,key,title in zip(axes.flat,['ce','mae','rmse','sampled_delta_mae'],['Cross-entropy','Value MAE','Value RMSE','Temporal-difference error (sparse grid)']):
    if key!='sampled_delta_mae':
        for split,col in [('train','#2563eb'),('test','#ea580c')]:
            ax.plot(steps,[r[split+'_'+key] for r in records],'.-',color=col,label=split)
        if key in ('mae','rmse'):
            ci=np.array([r[key+'_ci95'] for r in records]); ax.fill_between(steps,ci[:,0],ci[:,1],color='#ea580c',alpha=.15,label='episode bootstrap 95%')
    else: ax.plot(steps,[r[key] for r in records],'.-',color='#7c3aed',label='holdout; variable gaps, not 50 frames')
    ax.set_title(title,loc='left'); ax.set_xlabel('Step'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('PiperX value | post-hoc audit of 56 heldout episodes x 32 fixed frames',fontsize=15)
fig.text(.05,.012,'Bands are conditional/descriptive, not selection-adjusted. Temporal labels do not independently measure physical progress.',fontsize=9)
fig.tight_layout(rect=[0,.035,1,.95]); fig.savefig(args.output/'diagnostics.png'); plt.close(fig)
by_step={r['step']:r for r in records}
lines=['# PiperX saved-prediction diagnostics','',f'Observed checkpoints: {steps[0]}–{steps[-1]}; 56 holdout episodes, 32 fixed frames each.','', 'This is post-hoc analysis, not a change to training selection. See JSON for limitations.','', '|Step|Train CE|Holdout CE|MAE|RMSE|MAE seconds|Spearman (time proxy)|Sampled delta MAE|','|---|---|---|---|---|---|---|---|']
for s in candidates:
    r=by_step[s]; lines.append(f"|{s}|{r['train_ce']:.4f}|{r['test_ce']:.4f}|{r['test_mae']:.6f}|{r['test_rmse']:.6f}|{r['mae_seconds']:.3f}|{r['macro_spearman']:.4f}|{r['sampled_delta_mae']:.6f}|")
lines+=['','## Paired differences (newer minus older; negative is improvement)','']
for r in comparisons: lines.append(f"- {r['older']} → {r['newer']}: MAE change {r['delta_mae_new_minus_old']:.8f}, 95% CI {r['paired_mae_ci95']}; improved episodes {r['episodes_improved']}/56.")
(args.output/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(best=result['observed_best'],candidates=[by_step[s] for s in candidates],paired=comparisons),indent=2))
