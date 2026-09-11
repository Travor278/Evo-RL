"""Inference-only checkpoint comparison, invoked serially at the pipeline's dense-review boundary.

No optimizer, gradient, policy labels, source-data writes, or best-pointer changes.
"""
import copy, json, os
from pathlib import Path
import numpy as np


def distribution_rows(prob, y, centers):
    """Continuous-target ordinal distribution diagnostics, no fitted calibration transform."""
    assert prob.ndim==2 and len(y)==len(prob)
    assert np.isfinite(prob).all() and np.all(prob>=0)
    np.testing.assert_allclose(prob.sum(1),1,atol=2e-6)
    mean=prob@centers; cdf=np.cumsum(prob,axis=1)
    entropy=-(prob*np.log(np.maximum(prob,1e-30))).sum(1)
    std=np.sqrt(np.maximum(prob@(centers**2)-mean**2,0))
    quantiles={q:centers[np.minimum((cdf<q).sum(1),len(centers)-1)] for q in [.05,.1,.25,.5,.75,.9,.95]}
    coverage=(y>=quantiles[.05])&(y<=quantiles[.95])
    # Discrete support CRPS integral against the observed scalar, not two-hot entropy.
    crps=np.diff(centers)[0]*((cdf-(centers[None,:]>=y[:,None]))**2).sum(1)
    return dict(prediction=mean,entropy=entropy,std=std,crps=crps,coverage90=coverage.astype(float),width90=quantiles[.95]-quantiles[.05],quantiles=quantiles)


def summarize_array(a, prob, centers, scale, fps):
    # columns: slot, frame, target, mean, ce, entropy, std, crps, coverage90, width90
    ids=sorted(set(map(int,a[:,0]))); ep=[]
    d=distribution_rows(prob,a[:,2],centers)
    for slot in ids:
        z=a[a[:,0]==slot]; err=z[:,3]-z[:,2]
        ep.append(dict(slot=slot,mae=float(np.abs(err).mean()),mse=float((err**2).mean()),ce=float(z[:,4].mean()),entropy=float(z[:,5].mean()),std=float(z[:,6].mean()),crps=float(z[:,7].mean()),coverage90=float(z[:,8].mean()),width90=float(z[:,9].mean()),bias=float(err.mean())))
    result={key:float(np.mean([e[key] for e in ep])) for key in ep[0] if key!='slot'}
    result['rmse']=result.pop('mse')**.5
    result['mae_seconds']=result['mae']*scale/fps
    result['frames']=len(a); result['episodes']=len(ids)
    result['quantile_observed_cdf']={str(q):float(np.mean(a[:,2]<=v)) for q,v in d['quantiles'].items()}
    result['episode_metrics']=ep
    return result


def paired_rows(a, metadata, scale, fps):
    lookup={(int(r[0]),int(r[1])):r for r in a}; out=[]
    for slot,frames in metadata.items():
        for t in frames:
            before=lookup[(slot,t)]; after=lookup[(slot,t+50)]
            # Current Evo implementation gamma=1, successful nonterminal reward=-1/scale.
            residual=float(after[3]-before[3]-50/scale)
            target_residual=float(after[2]-before[2]-50/scale)
            assert abs(target_residual)<1e-6
            out.append([slot,t,residual,residual*scale/fps])
    return np.asarray(out,dtype=float)


def run_extended_review(model, manifest, train, test, train_eval, args, tokenizer, device, centers, rank, world, out):
    import torch
    import torch.distributed as dist
    from value_experiment import loader, prepare, atomic_json, digest, project_values_to_bins
    request=Path('/data/experiments/value-demo3-20260906/extended-review-request.json')
    if not request.exists(): return
    config=json.loads(request.read_text()); review=out/'extended-review-v1'
    if rank==0: review.mkdir(parents=True,exist_ok=True)
    if world>1: dist.barrier()
    if (review/'COMPLETE.json').exists():
        if rank==0: print('EXTENDED_REVIEW_ALREADY_COMPLETE',flush=True)
        return
    plan_file=review/'plan.json'
    if rank==0 and not plan_file.exists():
        logs=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
        choices=set(config['fixed_candidates'].get(args.task,[]))
        for key in ['ce','episode_mae','rmse']: choices.add(min(logs,key=lambda r:r['test'][key])['step'])
        choices=sorted(s for s in choices if (out/f'checkpoint-{s:06d}.pt').exists())
        plan=dict(candidates=choices,task=args.task,source_protocol_sha256=digest(out/'protocol.json'),request_sha256=digest(request),test_base_frames=128,train_base_frames=32,pair_anchors_per_episode=16,n_step=50,gamma=1,description='Post-hoc candidate comparison; does not alter original best/early-stop. All success MC target has identically zero realized Bellman residual; pair residual is NOT ground-truth action-quality accuracy.',limitations=['10% holdout reused for model selection','90% intervals describe projected returns, not epistemic confidence','No failure labels or true physical-progress preferences','No downstream policy evaluation','Macro aggregation gives episodes equal weight'])
        atomic_json(plan_file,plan)
    if world>1: dist.barrier()
    plan=json.loads(plan_file.read_text()); assert plan['source_protocol_sha256']==digest(out/'protocol.json')
    # Shallow views retain decoded metadata/state arrays, alter only deterministic indices.
    datasets={}; grids={}; pairmeta={}
    for split,original,nbase in [('train',train_eval,32),('test',test,128)]:
        view=copy.copy(original); union=[]; base_keys=[]; pairs={}
        for slot,(start,length) in enumerate(zip(view.starts,view.lengths)):
            n=int(length); base=np.unique(np.linspace(0,n-1,min(n,nbase),dtype=np.int64))
            anchors=np.unique(np.linspace(0,n-51,min(16,n-50),dtype=np.int64)) if n>50 else np.array([],dtype=np.int64)
            pairs[slot]=anchors.tolist(); frames=np.unique(np.r_[base,anchors,anchors+50])
            union.extend((start+frames).tolist()); base_keys.extend((slot,int(t)) for t in base)
        view.eval_indices=np.asarray(union,dtype=np.int64)
        datasets[split]=view; grids[split]=set(base_keys); pairmeta[split]=pairs
    c=centers.detach().cpu().numpy().astype(float); summaries=[]; all_test_pairs={}
    for step in plan['candidates']:
        dest=review/f'checkpoint-{step:06d}'; done=dest/'summary.json'
        if done.exists():
            if rank==0:
                summaries.append(json.loads(done.read_text())); all_test_pairs[step]=np.load(dest/'test-pairs50.npy')
            continue
        if rank==0: dest.mkdir(parents=True,exist_ok=True); print(f'EXTENDED_REVIEW_START step={step}',flush=True)
        if world>1: dist.barrier()
        saved=torch.load(out/f'checkpoint-{step:06d}.pt',map_location='cpu',weights_only=False,mmap=True)
        assert saved['step']==step and saved['protocol']['manifest_sha256']==json.loads((out/'protocol.json').read_text())['manifest_sha256']
        model.load_state_dict(saved['model'],strict=True); del saved
        model.eval(); result=dict(step=step,checkpoint_reload='strict_ok',checkpoint=f'checkpoint-{step:06d}.pt'); pairs_for={}
        for split,dataset in datasets.items():
            arrs=[]; probs=[]; indices=list(range(rank,len(dataset),world))
            with torch.inference_mode():
                for index,batch in enumerate(loader(dataset,args,indices=indices)):
                    inputs,y=prepare(batch,tokenizer,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(**inputs)
                    logits=logits.float(); assert torch.isfinite(logits).all()
                    p=logits.softmax(-1); targets=project_values_to_bins(y,centers)
                    ce=-(targets*logits.log_softmax(-1)).sum(-1)
                    pn=p.cpu().numpy(); yn=y.cpu().numpy(); d=distribution_rows(pn.astype(float),yn,c)
                    # Cross-check inference and CPU expectation computation.
                    np.testing.assert_allclose(d['prediction'],(p*centers).sum(-1).cpu().numpy(),atol=1e-6)
                    a=np.column_stack([batch['episode'].numpy(),batch['frame'].numpy(),yn,d['prediction'],ce.cpu().numpy(),d['entropy'],d['std'],d['crps'],d['coverage90'],d['width90']])
                    arrs.append(a); probs.append(pn)
                    if rank==0 and index%100==0: print(f'EXTENDED_REVIEW_PROGRESS step={step} split={split} rank0_batch={index}',flush=True)
            payload=(np.concatenate(arrs),np.concatenate(probs))
            if world>1:
                gathered=[None]*world if rank==0 else None
                dist.gather_object(payload,gathered,dst=0)
            else: gathered=[payload]
            if rank==0:
                a=np.concatenate([g[0] for g in gathered]); p=np.concatenate([g[1] for g in gathered]); order=np.lexsort((a[:,1],a[:,0])); a=a[order]; p=p[order]
                assert len(a)==len(dataset) and len(set((int(r[0]),int(r[1])) for r in a))==len(a)
                mask=np.asarray([(int(r[0]),int(r[1])) in grids[split] for r in a])
                assert int(mask.sum())==len(grids[split])
                np.savez_compressed(dest/f'{split}-distributions.npz',rows=a,probabilities=p,centers=c,base_mask=mask,episode_ids=dataset.ids,columns=np.array(['episode_slot','frame','target','prediction','ce','entropy','std','crps','coverage90','width90']))
                result[split]=summarize_array(a[mask],p[mask],c,dataset.scale,dataset.info['fps'])
                pairs=paired_rows(a,pairmeta[split],dataset.scale,dataset.info['fps']); pairs_for[split]=pairs
                np.save(dest/f'{split}-pairs50.npy',pairs)
                result[split]['pairs50']=dict(count=len(pairs),residual_mae=float(abs(pairs[:,2]).mean()),residual_rmse=float(np.sqrt((pairs[:,2]**2).mean())),residual_bias=float(pairs[:,2].mean()),residual_mae_seconds=float(abs(pairs[:,3]).mean()))
            if world>1: dist.barrier()
        if rank==0:
            # Threshold fitted on fixed TRAIN cohort only; labels are diagnostics, not dataset writes.
            threshold=float(np.quantile(pairs_for['train'][:,2],.7))
            result['diagnostic_top30_train_threshold']=threshold
            result['holdout_positive_fraction']=float(np.mean(pairs_for['test'][:,2]>=threshold))
            all_test_pairs[step]=pairs_for['test']; atomic_json(done,result); summaries.append(result)
            print(json.dumps(dict(event='EXTENDED_REVIEW_CHECKPOINT_COMPLETE',step=step,test={k:v for k,v in result['test'].items() if k!='episode_metrics'})),flush=True)
        if world>1: dist.barrier()
    if rank==0:
        comparisons=[]
        for i,s1 in enumerate(summaries):
            for s2 in summaries[i+1:]:
                p1=all_test_pairs[s1['step']]; p2=all_test_pairs[s2['step']]; assert np.array_equal(p1[:,:2],p2[:,:2])
                positive1=p1[:,2]>=s1['diagnostic_top30_train_threshold']; positive2=p2[:,2]>=s2['diagnostic_top30_train_threshold']
                comparisons.append(dict(a=s1['step'],b=s2['step'],pair_label_agreement=float(np.mean(positive1==positive2)),positive_jaccard=float((positive1&positive2).sum()/max(1,(positive1|positive2).sum())),warning='Agreement measures stability, NOT accuracy; no external action-quality labels.'))
        atomic_json(review/'COMPLETE.json',dict(plan=plan,checkpoints=summaries,label_stability=comparisons,status='ok'))
        print('EXTENDED_REVIEW_COMPLETE',flush=True)
    if world>1: dist.barrier()
