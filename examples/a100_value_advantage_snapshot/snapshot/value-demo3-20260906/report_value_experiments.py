"""Read-only metric analysis; derived reports never alter training checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path('/data/experiments/value-demo3-20260906')


def summarize(run):
    metrics = run / 'metrics.jsonl'
    if not metrics.exists():
        return None
    records = [json.loads(s) for s in metrics.read_text().splitlines() if s.strip()]
    # Recovery can append an evaluation before a failed save; retain latest per step.
    records = sorted({r['step']: r for r in records}.values(), key=lambda r:r['step'])
    if not records:
        return None
    best = min(records, key=lambda r:(r['test']['episode_mae'],r['step']))
    last = records[-1]
    protocol = json.loads((run/'protocol.json').read_text())
    actual_best = json.loads((run/'best.json').read_text()) if (run/'best.json').exists() else None
    result = dict(run=run.name,task=protocol['arguments']['task'],last_evaluated_step=last['step'],
                  best_observed_step=best['step'],best_effective_epoch=best['effective_epoch'],
                  best_train=best['train'],best_holdout=best['test'],
                  last_holdout=last['test'],evaluations=len(records),
                  checkpoint_pointer=actual_best,
                  checkpoint_matches_observed_best=bool(actual_best and actual_best['step']==best['step']),
                  beats_mean_baseline=best['test']['mae']<best['test']['baseline_mae'],
                  best_near_boundary=best['step']>=last['step']-protocol['arguments']['eval_every'],
                  holdout_usage='checkpoint selection; not untouched final test',
                  evaluation_sampling=protocol['evaluation'])
    dense = run / f'dense-review-{best["step"]:06d}-frames128.json'
    pred_file = run / f'dense-predictions-{best["step"]:06d}-frames128.json'
    if dense.exists() and pred_file.exists():
        result['dense_review'] = json.loads(dense.read_text())
        p = json.loads(pred_file.read_text())
        rows = np.array(p['rows'], dtype=np.float64)
        errors = [float(np.mean(np.abs(rows[rows[:,0]==e,2]-rows[rows[:,0]==e,3]))) for e in range(len(p['episode_ids']))]
        assert np.isfinite(errors).all()
        rng=np.random.default_rng(1000)
        samples=np.asarray(errors)[rng.integers(len(errors),size=(2000,len(errors)))].mean(1)
        result['episode_bootstrap_mae_95_interval']=np.quantile(samples,[.025,.975]).tolist()
        result['bootstrap_caveat']='episode resampling uncertainty conditional on this selected checkpoint; not selection-adjusted or session-independent'
    complete=run/'training_complete.json'
    if complete.exists(): result['training_stage']=json.loads(complete.read_text())
    return result


def plot_svg(run, destination):
    records=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines() if s.strip()]
    records=sorted({r['step']:r for r in records}.values(),key=lambda r:r['step'])
    if not records: return
    width,height=1000,540
    panels=[('CE loss','ce'),('Return MAE','mae')]
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="30" y="28" font-size="16">{run.name}</text>']
    colors={'train':'#1677b8','test':'#c54032'}
    maxstep=max(r['step'] for r in records)
    for j,(title,key) in enumerate(panels):
        left,top,pw,ph=65+j*490,65,405,380
        ymax=max(r[s][key] for r in records for s in colors)*1.1 or 1
        parts.append(f'<text x="{left}" y="{top-12}" font-size="15">{title}</text>')
        for t in range(6):
            y=top+ph-t*ph/5
            parts.append(f'<path d="M{left},{y}h{pw}" stroke="#ddd"/><text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11">{ymax*t/5:.3f}</text>')
            x=left+t*pw/5
            parts.append(f'<text x="{x}" y="{top+ph+20}" text-anchor="middle" font-size="11">{maxstep*t/5:.0f}</text>')
        for split,color in colors.items():
            pts=' '.join(f'{left+r["step"]/maxstep*pw:.2f},{top+ph-r[split][key]/ymax*ph:.2f}' for r in records)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{left+pw/2}" y="{top+ph+43}" text-anchor="middle" font-size="12">optimizer step</text>')
    parts.extend(['<text x="65" y="515" fill="#1677b8" font-size="13">Train holdout-free eval subset</text>',
                  '<text x="340" y="515" fill="#c54032" font-size="13">10% episode holdout (used for selection)</text>','</svg>'])
    destination.write_text('\n'.join(parts))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default=str(BASE/'reports'))
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    results=[]
    for run in sorted((BASE/'runs').glob('*-pilot8k-v*')):
        result=summarize(run)
        if result:
            results.append(result)
            plot_svg(run,out/(run.name+'.svg'))
    (out/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    lines=['# Value experiment interim report','',
           'Fixed episode-disjoint 90/10 split; the 10% holdout is reused for checkpoint selection, not an untouched final test. Steps below are best OBSERVED so far, not a proven global optimum.','',
           '| Task | Latest eval step | Best observed step | Effective epoch | Train CE | Holdout CE | Holdout MAE | Beats mean baseline |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        lines.append(f'| {r["task"]} | {r["last_evaluated_step"]} | {r["best_observed_step"]} | {r["best_effective_epoch"]:.4f} | {r["best_train"]["ce"]:.4f} | {r["best_holdout"]["ce"]:.4f} | {r["best_holdout"]["mae"]:.5f} | {r["beats_mean_baseline"]} |')
    if not results: lines.extend(['','No formal experiment evaluation has completed yet.'])
    lines.extend(['','All selected trajectories are successful demonstrations. These results cannot establish failure discrimination or real-robot RL benefits.'])
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(formal_runs=len(results),report=str(out/'summary.md'))))


if __name__=='__main__': main()
