"""Frozen step1500; all56 DEMO test episodes, global frame ranking, three fixed previews."""
import json, sys, math, os
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import torch
import av
sys.path.insert(0,'/data/experiments/value-vision-schedules-20260907')
from lean_model import LeanVisionValue
from mixed_data import ROOT, Pool, put, sha

OUT=ROOT/'video-demo-global30-step1500'
RUN=ROOT/'runs/piperx-vision-mixed1to1-c150-z15337-fixed1500-seed1000'
CKPT=RUN/'checkpoint-001500.pt'
PREVIEW=[11,23,40]

def frames(path,queries):
    with av.open(str(path)) as con:
        st=con.streams.video[0];st.thread_type='AUTO';st.thread_count=2
        con.seek(max(0,round(float(queries[0])/st.time_base)-1),stream=st,backward=True)
        it=iter(con.decode(st));prev=None;cur=next(it)
        for q in queries:
            while float(cur.pts*st.time_base)<q:
                prev=cur;nxt=next(it,None)
                if nxt is None:break
                cur=nxt
            best=min([v for v in (prev,cur) if v is not None],key=lambda v:abs(float(v.pts*st.time_base)-q))
            assert abs(float(best.pts*st.time_base)-q)<1e-4
            yield best.to_ndarray(format='rgb24')

def worker(rank):
    torch.set_num_threads(2)
    m=json.loads((ROOT/'manifest.json').read_text());manifest_sha=sha(ROOT/'manifest.json')
    ids=sorted(m['demo']['splits']['test']['episode_indices'])[rank::8]
    m['demo']['splits']['test']['episode_indices']=ids
    d=Pool(m,'demo','test');device=torch.device(f'cuda:{rank}')
    pending=[e for e in ids if not (OUT/f'episode-{e:03d}.json').exists()]
    model=None
    if any(e!=11 for e in pending):
        model=LeanVisionValue(SimpleNamespace(use_gradient_checkpointing=False)).to(device)
        saved=torch.load(CKPT,map_location='cpu',weights_only=False)
        assert saved['step']==1500 and saved['protocol']['manifest_sha256']==manifest_sha
        model.load_state_dict(saved['model'],strict=True);del saved;model.eval()
    centers=torch.linspace(-1,0,201,device=device)
    old=json.loads((RUN/'regular-001500-test_demo-predictions.json').read_text())
    for ep in ids:
        path=OUT/f'episode-{ep:03d}.json'
        if path.exists():
            cache=json.loads(path.read_text());assert cache['manifest_sha256']==manifest_sha and cache['checkpoint']==str(CKPT)
            print('REUSE',ep,flush=True);continue
        row=d.rows[ep];n=int(row['length']);cams=[]
        for camera in d.cameras:
            pre='videos/'+camera
            p=d.root/d.info['video_path'].format(video_key=camera,chunk_index=row[pre+'/chunk_index'],file_index=row[pre+'/file_index'])
            q=float(row[pre+'/from_timestamp'])+d.timestamps[ep].astype(np.float64)
            cams.append((str(p),q.tolist()))
        values=[];ces=[]
        if ep==11:
            native=json.loads((ROOT/'video-demo-heldout-mixed-predictions/mixed1500_native_predictions.json').read_text())
            assert native['manifest_sha256']==manifest_sha and native['episode']==11 and native['step']==1500 and native['root']==str(d.root)
            values=native['values'];ces=native['cross_entropy']
        else:
            batch=[];bids=[]
            with torch.inference_mode():
                for i,images in enumerate(zip(*(frames(p,np.array(q)) for p,q in cams))):
                    batch.append(torch.from_numpy(np.stack(images)).permute(0,3,1,2));bids.append(i)
                    if len(batch)==8 or i==n-1:
                        x=torch.stack(batch).to(device)
                        target=torch.tensor(d.returns[ep][bids]/d.scale,device=device,dtype=torch.float32)
                        pos=(target+1)*200;lo=pos.floor().long().clamp(0,200);hi=(lo+1).clamp(0,200);w=pos-lo
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            logits=model(images=x,image_attention_mask=torch.ones(x.shape[:2],device=device,dtype=torch.bool))
                        logp=logits.float().log_softmax(-1)
                        values.extend((logp.exp()*centers).sum(-1).cpu().tolist())
                        ce=-(1-w)*logp.gather(1,lo[:,None]).squeeze(1)-w*logp.gather(1,hi[:,None]).squeeze(1)
                        ces.extend(ce.cpu().tolist());batch=[];bids=[]
                    if (i+1)%2000==0:print('PROGRESS',rank,ep,i+1,n,flush=True)
        v=np.array(values);assert len(v)==n and np.isfinite(v).all() and not d.events[ep].any()
        end=np.minimum(np.arange(n)+50,n-1);boot=v.copy();boot[-1]=0
        a=-(end-np.arange(n))/d.scale+boot[end]-v;a[-1]=0
        slot=old['episode_ids'].index(ep)
        diffs=[abs(v[int(r[1])]-r[3]) for r in old['rows'] if r[0]==slot]
        assert len(diffs)==32 and max(diffs)<.002
        put(path,dict(episode=ep,step=1500,root=str(d.root),checkpoint=str(CKPT),manifest_sha256=manifest_sha,
                      action_state_sha256=d.fingerprints[ep],value_common_scale=v.tolist(),advantage_full=a.tolist(),
                      cross_entropy=ces,target_value=(d.returns[ep]/d.scale).tolist(),human_control=[True]*n,
                      camera_specs=cams,fps=30,scale=d.scale,horizon=50,full_frame=True,
                      max_diff_existing_eval=max(diffs),mae=float(np.mean(abs(v-d.returns[ep]/d.scale))),
                      ce=float(np.mean(ces))))
        print('EP_COMPLETE',rank,ep,n,flush=True)
    return ids

def aggregate():
    m=json.loads((ROOT/'manifest.json').read_text());ids=sorted(m['demo']['splits']['test']['episode_indices'])
    records=[json.loads((OUT/f'episode-{ep:03d}.json').read_text()) for ep in ids]
    arrays=[np.array(r['advantage_full'])[:-1] for r in records]
    raw=np.concatenate(arrays);assert np.isfinite(raw).all()
    k=math.ceil(len(raw)*.30)
    top=np.argsort(-raw,kind='stable')[:k];top=top[raw[top]>0]
    bottom=np.argsort(raw,kind='stable')[:k]
    masks={g:np.zeros(len(raw),dtype=bool) for g in ['top30','bottom30']}
    masks['top30'][top]=True;masks['bottom30'][bottom]=True
    thresholds=dict(top30=float(raw[top].min()),bottom30=float(raw[bottom].max()))
    vv=np.concatenate([r['value_common_scale'] for r in records])
    vlo=math.floor((vv.min()-.01)*20)/20;vhi=min(0.,math.ceil((vv.max()+.01)*20)/20)
    amp=math.ceil(max(abs(raw.min()),abs(raw.max()))*100)/100+.005
    jobs=[];counts=[];offset=0
    for r,a in zip(records,arrays):
        ep=r['episode'];count=dict(episode=ep,eligible_frames=len(a),mae=r['mae'],ce=r['ce'])
        for group in ['top30','bottom30']:
            selected=np.flatnonzero(masks[group][offset:offset+len(a)])
            count[group]=len(selected)
            put(OUT/f'episode-{ep:03d}-{group}-indices.json',dict(episode=ep,group=group,frames=selected.tolist(),global_threshold=thresholds[group]))
            if ep in PREVIEW and len(selected):
                jobs.append(dict(key='mixed1500',group=group,name=f'ep{ep:03d}_mixed1500_{group}',
                                 out=str(OUT/'videos'),timeline=str(OUT/f'episode-{ep:03d}.json'),
                                 selected_frames=selected.tolist(),label='TOP30% & A>0' if group=='top30' else 'BOTTOM30%',
                                 scope='across ALL 56 test episodes',global_threshold=thresholds[group],
                                 camera_specs=r['camera_specs'],value_limits=[vlo,vhi],advantage_limits=[-amp,amp]))
        counts.append(count);offset+=len(a)
    assert offset==len(raw) and len(ids)==56 and len(jobs)==6
    summary=dict(status='ok',checkpoint=str(CKPT),episode_count=56,total_frames=len(vv),eligible_frames=len(raw),
                 fraction=.30,requested_each=k,selected_top=len(top),selected_bottom=len(bottom),
                 thresholds=thresholds,positive_frames=int((raw>0).sum()),negative_frames=int((raw<0).sum()),
                 ranking='Frame-weighted pooled all56, not episode-balanced. Stable ties by episode then frame. Exclude terminal.',
                 preview_episodes=PREVIEW,preview_selection='First three sorted test IDs fixed before inference, not score-selected',
                 estimator='Existing n50 raw A unchanged. Playback contains selected frames only, no chunk expansion.',
                 per_episode=counts,render_specs=jobs,mae=float(np.mean(np.concatenate([np.abs(np.array(r['value_common_scale'])-r['target_value']) for r in records]))))
    put(OUT/'global_summary.json',summary)
    print('GLOBAL_THRESHOLDS',thresholds,'ELIGIBLE',len(raw),'COUNTS',len(top),len(bottom),flush=True)
    return jobs

def main():
    OUT.mkdir(exist_ok=True);(OUT/'videos').mkdir(exist_ok=True)
    m=json.loads((ROOT/'manifest.json').read_text())
    assert len(m['demo']['splits']['test']['episode_indices'])==56
    assert not set(m['demo']['splits']['test']['episode_indices']) & set(m['demo']['splits']['train']['episode_indices'])
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:list(pool.map(worker,range(8)))
    jobs=aggregate()
    from render_demo_raw10_frames import render
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:results=list(pool.map(render,jobs))
    put(OUT/'complete.json',dict(status='ok',videos=results))
    print('ALL_GLOBAL30_COMPLETE',flush=True)

if __name__=='__main__':main()
