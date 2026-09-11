"""Offline held-out takeover diagnostics, not causal action-quality ground truth.

All ROC/AP metrics use only a fixed uniform autonomous-frame grid, preserving
that grid's event prevalence. Additional dense pre-event frames serve event plots
and negative fractions, never inflate classification prevalence. Report explicit
penalty separately: a negative full A can be mechanically label-induced.
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import Subset
sys.path.insert(0,'/data/experiments/value-vision-schedules-20260907')
import train_lean_value as u
from mixed_data import ROOT,Pool,sha,put

def classification(labels,scores):
    y=np.asarray(labels,dtype=bool);s=np.asarray(scores,dtype=float)
    if not y.any() or y.all():return dict(auroc=None,average_precision=None)
    order=np.argsort(-s,kind='stable');y=y[order];s=s[order]
    end=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[end];fp=np.cumsum(~y)[end]
    recall=tp/y.sum();fpr=fp/(~y).sum();precision=tp/(tp+fp)
    return dict(auroc=float(np.trapezoid(np.r_[0,recall],np.r_[0,fpr])),average_precision=float(np.sum(np.diff(np.r_[0,recall])*precision)))

def plan_points(d):
    records=[];needed=set()
    for slot,e in enumerate(d.ids):
        n=int(d.rows[e]['length']);T=n-1;mask=d.masks[e]
        grid=set(t for t in range(0,T,5) if not mask[t])
        dense=set()
        for event in np.flatnonzero(d.events[e]):
            # events index is last autonomous frame; onset=event+1.
            dense.update(t for t in range(max(0,int(event)-49),int(event)+1) if not mask[t])
        for t in sorted(grid|dense):
            for horizon in [15,30,50]:
                end=min(t+horizon,T);count=int(d.events[e][t:end].sum())
                records.append(dict(episode=e,slot=slot,frame=t,end=end,horizon=horizon,grid=t in grid,event_count=count,source=next(r['source_batch'] for r in d.manifest['hil']['provenance'] if r['output_episode_index']==e)))
                for f in [t,end]:
                    if f!=T:needed.add(int(d.starts[slot])+f)
    return records,sorted(needed)

def summarize(records,seed=1000):
    output={}
    for n in [15,30,50]:
        grid=[r for r in records if r['grid'] and r['horizon']==n]
        y=np.array([r['event_count']>0 for r in grid]);eps=np.array([r['episode'] for r in grid])
        entry=dict(autonomous_grid_frames=len(grid),event_prevalence=float(y.mean()),horizon_seconds=n/30,scores={})
        for component in ['full','time_value_delta','explicit_penalty']:
            a=np.array([r[component] for r in grid]);stats=classification(y,-a)
            stats.update(negative_fraction_pre_event=float((a[y]<0).mean()) if y.any() else None,negative_fraction_no_event=float((a[~y]<0).mean()) if (~y).any() else None)
            rng=np.random.default_rng(seed);unique=np.unique(eps);boot=[]
            for _ in range(300):
                selected=rng.choice(unique,len(unique),replace=True);idx=np.concatenate([np.flatnonzero(eps==e) for e in selected])
                m=classification(y[idx],-a[idx])
                if m['auroc'] is not None:boot.append([m['auroc'],m['average_precision']])
            if boot:
                lo,hi=np.quantile(boot,[.025,.975],axis=0);stats['episode_bootstrap95']={'auroc':[float(lo[0]),float(hi[0])],'average_precision':[float(lo[1]),float(hi[1])]}
            entry['scores'][component]=stats
        output[str(n)]=entry
    return output

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--checkpoint',default='checkpoint-001500.pt');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--batch-size',type=int,default=8);args=ap.parse_args()
    assert Path(args.run).name==args.run and Path(args.checkpoint).name==args.checkpoint
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));assert world==4
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2);dist.init_process_group('nccl',timeout=timedelta(minutes=40))
    m=json.loads((ROOT/'manifest.json').read_text());d=Pool(m,'hil','test');records,needed=plan_points(d)
    config=u.Pistar06Config(dtype='float32',use_gradient_checkpointing=False,device='cuda',camera_features=m['hil']['camera_features'])
    model=u.VisionOnlyModel(config).to(device);saved=torch.load(ROOT/'runs'/args.run/args.checkpoint,map_location='cpu',weights_only=False)
    assert saved['protocol']['manifest_sha256']==sha(ROOT/'manifest.json') and saved['protocol']['return_scale']==15337
    model.load_state_dict(saved['model'],strict=True);step=saved['step'];del saved;model.eval();centers=u.build_bin_centers(201,-1,0,device)
    predictions=[]
    with torch.no_grad():
        for batch in u.loader(d,args,indices=needed[rank::world]):
            inputs,y=u.prepare(batch,None,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs)
            pred=(logits.float().softmax(-1)*centers).sum(-1);assert torch.isfinite(pred).all()
            predictions.extend((d.ids[slot],frame,p) for slot,frame,p in zip(batch['episode'].tolist(),batch['frame'].tolist(),pred.cpu().tolist()))
    all_predictions=[None]*world;dist.all_gather_object(all_predictions,predictions)
    if rank==0:
        v={(e,f):p for part in all_predictions for e,f,p in part};assert len(v)==len(needed)
        for e in d.ids:v[e,int(d.rows[e]['length'])-1]=0.
        for r in records:
            e,t,end=r['episode'],r['frame'],r['end'];nt=end-t
            r['value_before']=v[e,t];r['value_after']=v[e,end]
            r['time_value_delta']=-nt/d.scale+v[e,end]-v[e,t]
            r['explicit_penalty']=-m['penalty']*r['event_count']/d.scale
            r['full']=r['time_value_delta']+r['explicit_penalty']
            oracle=(d.reward[e][t:end].sum()+d.returns[e][end]-d.returns[e][t])/d.scale
            assert abs(oracle)<1e-12
        out=ROOT/'runs'/args.run
        put(out/f'advantage-{step:06d}-records.json',dict(step=step,records=records))
        result=dict(status='ok',step=step,episodes=len(d.ids),predicted_frames=len(needed),windows=summarize(records),interpretation='Offline event-proximity proxy. Full advantage includes actual recorded penalty; even time+value delta uses future images and is not an online takeover detector. No causal action-quality or policy-improvement claim. Perfect realized returns telescope to zero residual.',threshold='fixed A<0; no threshold tuned for negativity',bootstrap='300 resamples of whole held-out episodes',inference='strict checkpoint reload; terminal V=0')
        put(out/f'advantage-{step:06d}-summary.json',result);print(json.dumps(result),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
