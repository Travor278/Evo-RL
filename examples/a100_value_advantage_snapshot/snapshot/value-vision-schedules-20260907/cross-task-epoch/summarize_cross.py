"""CPU-only epoch-aligned curves and within-task paired final comparisons."""
import csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
def read(p):return json.loads(p.read_text())
def main():
    plan=read(ROOT/'plan.json')
    refs=[dict(short='piperx',reference_steps=1500,steps=1500,train_frames=1132703,path='/data/experiments/value-ablation-20260906/runs/piperx_insert_copper_screw-vision_only-fixed1500-seed1000-v1'),dict(short='piperx',reference_steps=3000,steps=3000,train_frames=1132703,path=str(PARENT/'runs/piperx-visionlean-cosine-fixed3000-seed1000'))]
    jobs=refs+[dict(j,path=str(ROOT/'runs'/j['run'])) for j in plan['jobs']]
    curves=[];endpoints=[];pairs={}
    for j in jobs:
        p=Path(j['path']);step=j['steps']
        records=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()] if (p/'metrics.jsonl').exists() else []
        log=PARENT/(p.name+'.log') if j['short']=='piperx' else ROOT/'logs'/(p.name+'.log')
        latest=None
        if log.exists():
            for line in log.read_text(errors='replace').splitlines():
                if line.startswith('{'):
                    try:x=json.loads(line)
                    except ValueError:continue
                    if x.get('event')=='TRAIN':latest=x
        item=dict(task=j['short'],reference_steps=j['reference_steps'],budget=step,train_frames=j['train_frames'],evaluations=records,latest_train=latest)
        review=p/f'dense-review-{step:06d}-frames128.json'
        pred=p/f'dense-predictions-{step:06d}-frames128.json'
        if review.exists():
            d=read(review);assert d['checkpoint_reload']=='strict_ok'
            item['final_dense']=d
            m=d['test'];tr=records[-1]['train'] if records else {}
            endpoints.append(dict(task=j['short'],reference_steps=j['reference_steps'],steps=step,effective_epoch=step*32/j['train_frames'],train_ce_32=tr.get('ce'),train_mae_32=tr.get('mae'),heldout_ce_128=m['ce'],heldout_mae_128=m['episode_mae'],heldout_rmse_128=m['rmse'],heldout_crps_128=m['crps'],mae_seconds=m['mae_remaining_seconds'],baseline_mae=m['baseline_mae'],coverage90=m['interval90_coverage'],checkpoint=str(p/d['checkpoint'])))
            r=read(pred);rows=np.asarray(r['rows']);epis=r['episode_ids'];episode={}
            for i,id_ in enumerate(epis):
                selected=rows[rows[:,0]==i];episode[int(id_)]=float(np.abs(selected[:,2]-selected[:,3]).mean())
            pairs.setdefault(j['short'],{})[j['reference_steps']]=episode
        curves.append(item)
    comparisons=[]
    for task,d in pairs.items():
        if not {1500,3000}<=d.keys():continue
        ids=sorted(d[1500]);assert ids==sorted(d[3000])
        short=np.array([d[1500][i] for i in ids]);long=np.array([d[3000][i] for i in ids]);delta=long-short
        rng=np.random.default_rng(1000);boot=delta[rng.integers(len(ids),size=(10000,len(ids)))].mean(1)
        low,high=np.quantile(boot,[.025,.975])
        comparisons.append(dict(task=task,episodes=len(ids),short_mae=float(short.mean()),long_mae=float(long.mean()),long_minus_short=float(delta.mean()),relative_improvement_percent=float(100*(short.mean()-long.mean())/short.mean()),ci95_low=float(low),ci95_high=float(high),interpretation='long_lower' if high<0 else ('short_lower' if low>0 else 'inconclusive'),scope='conditional on this seed and heldout episodes; not RL performance'))
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    report=dict(generated_at=time.time(),runs=curves,paired= comparisons,completed_endpoints=len(endpoints),expected_endpoints=6)
    (out/'cross-task-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,rows in [('endpoints.csv',endpoints),('paired-mae.csv',comparisons)]:
        if rows:
            with (out/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(completed_endpoints=len(endpoints),expected_endpoints=6,paired=comparisons,latest=[dict(task=x['task'],reference_steps=x['reference_steps'],latest_train=x['latest_train'],latest_eval=x['evaluations'][-1] if x['evaluations'] else None) for x in curves])))
if __name__=='__main__':main()
