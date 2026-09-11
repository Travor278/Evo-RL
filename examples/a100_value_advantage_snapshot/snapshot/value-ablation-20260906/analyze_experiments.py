"""CPU-only tables, curves and paired-episode uncertainty; no training mutation.

On A100: python analyze_experiments.py
On a plotting host: python analyze_experiments.py --plot-only curves.json --out DIR
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

VARIANTS=['full','vision_only','frozen_language','frozen_both','task_only']
LABELS={'full':'Image + task + state','vision_only':'Vision only','frozen_language':'Frozen language','frozen_both':'Frozen encoders','task_only':'Image + task (no state)'}

def read(path): return json.loads(path.read_text())
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvout(path,rows):
    if not rows: return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def ranks(a):
    _,inv,counts=np.unique(a,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-counts/2-0.5)[inv]
def trajectory(data):
    rows=np.array(data['rows'],dtype=float); result=[]
    for slot,ep in enumerate(data['episode_ids']):
        r=rows[rows[:,0]==slot]; r=r[np.argsort(r[:,1])]
        y,p=r[:,2],r[:,3]; err=p-y
        r1,r2=ranks(y),ranks(p)
        correlation=float(np.corrcoef(r1,r2)[0,1]) if np.std(r1)*np.std(r2)>0 else 0.0
        result.append(dict(episode=int(ep),mae=float(np.mean(abs(err))),rmse=float(np.sqrt(np.mean(err**2))),bias=float(np.mean(err)),p95_abs_error=float(np.quantile(abs(err),.95)),time_spearman=correlation,sparse_pair_delta_mae=float(np.mean(abs(np.diff(p)-np.diff(y)))),sparse_reverse_fraction=float(np.mean(np.diff(p)<0)),first_quarter_mae=float(np.mean(abs(err[:max(1,len(err)//4)]))),last_quarter_mae=float(np.mean(abs(err[-max(1,len(err)//4):])))))
    return result
def bootstrap(a,b):
    assert [x['episode'] for x in a]==[x['episode'] for x in b]
    d=np.array([x['mae']-y['mae'] for x,y in zip(a,b)])
    rng=np.random.default_rng(2001)
    samples=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(mae_a_minus_b=float(d.mean()),ci95_low=float(lo),ci95_high=float(hi),episodes_better=int(sum(d<0)),episodes_worse=int(sum(d>0)))

def plot(curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(VARIANTS,plt.get_cmap('tab10').colors[:5]))
    for horizon in sorted({c['horizon'] for c in curves}):
        fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
        for c in curves:
            if c['horizon']!=horizon or not c['records']: continue
            for ax,key in zip(axes,['ce','mae']):
                x=[r['step'] for r in c['records']]
                for split,style in [('train','--'),('test','-')]:
                    ax.plot(x,[r[split][key] for r in c['records']],style,color=colors[c['variant']],label=LABELS[c['variant']]+(' / train' if split=='train' else ' / held-out'),linewidth=1.7,marker='o',markersize=3)
                ax.set(xlabel='Optimizer step',ylabel='Cross-entropy loss' if key=='ce' else 'Normalized return MAE')
                ax.grid(alpha=.22)
        axes[0].legend(fontsize=7,ncol=2)
        fig.suptitle(f'PiperX: fixed {horizon}-step LR schedule | seed1000 | dashed=train, solid=held-out\nSame 56 episodes per evaluation split, 32 fixed frames/episode; not an independent final test',fontsize=10)
        fig.savefig(out/f'loss-mae-fixed{horizon}.png',dpi=180)
        plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('/data/experiments/value-ablation-20260906')); ap.add_argument('--plot-only',type=Path); ap.add_argument('--out',type=Path)
    args=ap.parse_args(); out=args.out or args.root/'reports'; out.mkdir(exist_ok=True,parents=True)
    if args.plot_only:
        plot(read(args.plot_only),out); return
    curves=[]; summary=[]; endpoints=[]; trajectories={}; per_episode=[]
    for path in sorted((args.root/'runs').glob('*/protocol.json')):
        protocol=read(path); run=path.parent
        if protocol['arguments'].get('smoke'): continue
        variant=protocol['variant']; horizon=protocol['arguments']['steps']
        records=[]
        if (run/'metrics.jsonl').exists():
            for line in (run/'metrics.jsonl').read_text().splitlines():
                try: records.append(json.loads(line))
                except json.JSONDecodeError: pass # Only a concurrently appended final partial line.
        curves.append(dict(run=run.name,variant=variant,horizon=horizon,records=records))
        final=run/'final.json'
        for r in records:
            row=dict(run=run.name,variant=variant,fixed_horizon=horizon,step=r['step'],effective_epoch=r['effective_epoch'],is_fixed_endpoint=bool(final.exists() and r['step']==horizon))
            for split in ['train','test']:
                for key in ['ce','mae','rmse','episode_mae','crps','bias','entropy','interval90_coverage','interval90_width']:
                    row[split+'_'+key]=r[split].get(key)
            summary.append(row)
        if not final.exists(): continue
        assert read(final)['step']==horizon and records[-1]['step']==horizon
        dense=run/f'dense-review-{horizon:06d}-frames128.json'
        if not dense.exists(): continue
        metrics=read(dense)['test']; training=records[-1]['train']
        row=dict(run=run.name,variant=variant,fixed_steps=horizon,train_ce_32=training['ce'],train_mae_32=training['mae'],holdout_ce_128=metrics['ce'],holdout_mae_128=metrics['mae'],holdout_rmse_128=metrics['rmse'],holdout_crps_128=metrics['crps'],holdout_interval90_coverage=metrics['interval90_coverage'],mae_seconds=metrics['mae_remaining_seconds'],wall_seconds_protocol_to_complete=(run/'training_complete.json').stat().st_mtime-path.stat().st_mtime,checkpoint=str(run/read(final)['file']))
        endpoints.append(row)
        traj=trajectory(read(run/f'dense-predictions-{horizon:06d}-frames128.json')); trajectories[run.name]=traj
        per_episode.extend([dict(run=run.name,**e) for e in traj])
    pairs=[]
    keys=sorted(trajectories)
    for i,name in enumerate(keys):
        for other in keys[i+1:]: pairs.append(dict(run_a=name,run_b=other,**bootstrap(trajectories[name],trajectories[other])))
    dump(out/'curves.json',curves); dump(out/'endpoint_results.json',endpoints)
    csvout(out/'all_evaluations.csv',summary); csvout(out/'fixed_endpoint_table.csv',endpoints); csvout(out/'trajectory_metrics.csv',per_episode); csvout(out/'paired_episode_bootstrap.csv',pairs)
    lines=['# Value architecture and fixed-horizon comparison','', 'Only final-step models appear below. Train metrics use32 frames/episode; dense held-out metrics use128. Intermediate evaluations are in all_evaluations.csv.','', '| Architecture | Fixed steps | Train CE | Held-out CE | Train MAE | Held-out MAE | Held-out RMSE |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in endpoints: lines.append(f"| {LABELS[r['variant']]} | {r['fixed_steps']} | {r['train_ce_32']:.4f} | {r['holdout_ce_128']:.4f} | {r['train_mae_32']:.6f} | {r['holdout_mae_128']:.6f} | {r['holdout_rmse_128']:.6f} |")
    if not endpoints: lines+=['','No formal fixed endpoint completed yet; smoke metrics are intentionally excluded.']
    probe=args.root/'state-probe-step3000/COMPLETE.json'
    if probe.exists():
        result=read(probe)['results']; probe_rows=[]
        lines+=['','## State reliance of the original step3000 model','','| State intervention | Held-out CE | Held-out MAE |','| --- | ---: | ---: |']
        for name,m in result.items():
            probe_rows.append(dict(intervention=name,ce=m['ce'],mae=m['mae'],rmse=m['rmse'],mae_delta_from_original=m['mae']-result['none']['mae']))
            lines.append(f"| {name} | {m['ce']:.6f} | {m['mae']:.8f} |")
        csvout(out/'state_reliance.csv',probe_rows)
        lines+=['','Identical images and labels, altered state tokens only. Interventions can be out of distribution; this is reliance evidence, not proof that a retrained no-state model wins.']
    lines+=['','The10% held-out set is reused for selection (validation). Bootstrap intervals are conditional on this split and are not selection-adjusted. Samples within episodes are not independent. Time ordering/difference metrics use sparse frame gaps, not fixed50-frame Bellman residuals. A single-seed ranking is provisional; it cannot prove real robot success or a global optimum.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(formal_runs=len(curves),evaluations=len(summary),dense_endpoints=len(endpoints),reports=str(out))))

if __name__=='__main__': main()
