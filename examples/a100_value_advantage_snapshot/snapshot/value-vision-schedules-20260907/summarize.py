"""CPU-only status and paired episode endpoint review; never starts training."""
import csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path('/data/experiments/value-vision-schedules-20260907')
OLD=Path('/data/experiments/value-ablation-20260906/runs/piperx_insert_copper_screw-vision_only-fixed1500-seed1000-v1')
def read(p):return json.loads(p.read_text())
def main():
    runs=[];endpoints={}
    paths=[OLD]+sorted((ROOT/'runs').glob('*'))
    for p in paths:
        if 'smoke' in p.name or not (p/'protocol.json').exists():continue
        protocol=read(p/'protocol.json');a=protocol['arguments'];budget=a['steps']
        metrics=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()] if (p/'metrics.jsonl').exists() else []
        last_train=None;log=ROOT/(p.name+'.log')
        if log.exists():
            for line in log.read_text(errors='replace').splitlines():
                if line.startswith('{'):
                    try:r=json.loads(line)
                    except ValueError:continue
                    if r.get('event')=='TRAIN':last_train=r
        final=read(p/'final.json') if (p/'final.json').exists() else None
        dense=p/f'dense-review-{budget:06d}-frames128.json'
        result=dict(name=p.name,reference=p==OLD,gpu_layout='8x4' if p==OLD else '4x8',schedule=a.get('lr_schedule','cosine'),budget=budget,latest_train=last_train,evaluations=metrics,final=final,dense=read(dense) if dense.exists() else None)
        runs.append(result)
        pred=p/f'dense-predictions-{budget:06d}-frames128.json'
        if final and pred.exists():
            d=read(pred);rows=np.asarray(d['rows']);epis=d['episode_ids'];e={}
            for i,id_ in enumerate(epis):
                r=rows[rows[:,0]==i];e[int(id_)]=float(np.abs(r[:,2]-r[:,3]).mean())
            endpoints[p.name]=e
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    (out/'curves-status.json').write_text(json.dumps(dict(generated_at=time.time(),runs=runs),indent=2)+'\n')
    paired=[];names=list(endpoints);rng=np.random.default_rng(1000)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            ids=sorted(endpoints[a]);assert ids==sorted(endpoints[b])
            diffs=np.array([endpoints[a][k]-endpoints[b][k] for k in ids])
            boot=diffs[rng.integers(len(diffs),size=(10000,len(diffs)))].mean(1)
            paired.append(dict(a=a,b=b,metric='episode_MAE_a_minus_b',difference=float(diffs.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    (out/'paired-episode-mae.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(json.dumps([dict(name=r['name'],budget=r['budget'],latest_train=r['latest_train'],latest_eval=r['evaluations'][-1] if r['evaluations'] else None,final=r['final'],dense=r['dense']) for r in runs]))
if __name__=='__main__':main()
