"""Fill missing full-frame train502 predictions; never train or overwrite datasets."""
import json,sys,os
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import torch
sys.path.insert(0,'/data/experiments/value-vision-schedules-20260907')
from lean_model import LeanVisionValue
from mixed_data import ROOT,Pool,sha,put
from infer_demo_global30 import frames,CKPT,RUN
OUT=ROOT/'demo558-top10-a50-w20-build'
PRED=OUT/'predictions'

def worker(rank):
    torch.set_num_threads(2)
    m=json.loads((ROOT/'manifest.json').read_text());msha=sha(ROOT/'manifest.json')
    ids=sorted(m['demo']['splits']['train']['episode_indices'])[rank::8]
    m['demo']['splits']['train']['episode_indices']=ids
    d=Pool(m,'demo','train');device=torch.device(f'cuda:{rank}')
    model=LeanVisionValue(SimpleNamespace(use_gradient_checkpointing=False)).to(device)
    saved=torch.load(CKPT,map_location='cpu',weights_only=False)
    assert saved['step']==1500 and saved['protocol']['manifest_sha256']==msha
    model.load_state_dict(saved['model'],strict=True);del saved;model.eval()
    centers=torch.linspace(-1,0,201,device=device)
    old=json.loads((RUN/'regular-001500-train_demo-predictions.json').read_text())
    for ep in ids:
        path=PRED/f'episode-{ep:03d}.json'
        if path.exists():
            r=json.loads(path.read_text());assert r['manifest_sha256']==msha and r['checkpoint']==str(CKPT) and r['full_frame']
            print('REUSE_TRAIN',rank,ep,flush=True);continue
        row=d.rows[ep];n=int(row['length']);cams=[]
        for cam in d.cameras:
            pre='videos/'+cam
            p=d.root/d.info['video_path'].format(video_key=cam,chunk_index=row[pre+'/chunk_index'],file_index=row[pre+'/file_index'])
            q=float(row[pre+'/from_timestamp'])+d.timestamps[ep].astype(np.float64)
            cams.append((str(p),q.tolist()))
        values=[];batch=[]
        with torch.inference_mode():
            for i,images in enumerate(zip(*(frames(p,np.array(q)) for p,q in cams))):
                batch.append(torch.from_numpy(np.stack(images)).permute(0,3,1,2))
                if len(batch)==8 or i==n-1:
                    x=torch.stack(batch).to(device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(images=x,image_attention_mask=torch.ones(x.shape[:2],device=device,dtype=torch.bool))
                    values.extend((logits.float().softmax(-1)*centers).sum(-1).cpu().tolist());batch=[]
        v=np.array(values);assert len(v)==n and np.isfinite(v).all() and not d.events[ep].any()
        end=np.minimum(np.arange(n)+50,n-1);boot=v.copy();boot[-1]=0
        a=-(end-np.arange(n))/d.scale+boot[end]-v;a[-1]=0
        diff=None
        if ep in old['episode_ids']:
            slot=old['episode_ids'].index(ep)
            diffs=[abs(v[int(r[1])]-r[3]) for r in old['rows'] if r[0]==slot]
            assert len(diffs)==32 and max(diffs)<.002;diff=max(diffs)
        put(path,dict(episode=ep,step=1500,root=str(d.root),checkpoint=str(CKPT),manifest_sha256=msha,
                      action_state_sha256=d.fingerprints[ep],value_common_scale=v.tolist(),advantage_full=a.tolist(),
                      camera_specs=cams,fps=30,scale=15337,horizon=50,full_frame=True,
                      value_model_split='train',max_diff_existing_eval=diff))
        print('EP_COMPLETE',rank,ep,n,flush=True)
    return ids

def main():
    OUT.mkdir(exist_ok=True);PRED.mkdir(exist_ok=True)
    m=json.loads((ROOT/'manifest.json').read_text());msha=sha(ROOT/'manifest.json')
    train=m['demo']['splits']['train']['episode_indices'];test=m['demo']['splits']['test']['episode_indices']
    assert len(train)==502 and len(test)==56 and sorted(train+test)==list(range(558))
    for ep in test:
        src=ROOT/'video-demo-global30-step1500'/f'episode-{ep:03d}.json'
        r=json.loads(src.read_text());assert r['manifest_sha256']==msha and r['step']==1500 and r['full_frame']
        dst=PRED/src.name
        if not dst.exists():os.symlink(src,dst)
        else:assert dst.resolve()==src
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:list(pool.map(worker,range(8)))
    rows=[json.loads((PRED/f'episode-{e:03d}.json').read_text()) for e in range(558)]
    assert sum(len(r['advantage_full']) for r in rows)==1255729
    put(OUT/'inference_complete.json',dict(status='ok',episodes=558,frames=1255729,train_new=502,test_reused=56,checkpoint=str(CKPT)))
    print('ALL558_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
