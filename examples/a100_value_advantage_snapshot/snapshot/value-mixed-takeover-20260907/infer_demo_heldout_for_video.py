"""Frozen mixed value checkpoints on a preselected pure-teleoperation test episode."""
import json,sys,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,'/data/experiments/value-vision-schedules-20260907')
from lean_model import LeanVisionValue
from mixed_data import ROOT,Pool,put,sha
from render_episode_value import camera_frames

OUT=ROOT/'video-demo-heldout-mixed-predictions'
RUN=ROOT/'runs/piperx-vision-mixed1to1-c150-z15337-fixed1500-seed1000'

def main():
    OUT.mkdir(exist_ok=True);m=json.loads((ROOT/'manifest.json').read_text())
    test=sorted(m['demo']['splits']['test']['episode_indices']);train=m['demo']['splits']['train']['episode_indices']
    assert len(test)==56 and not set(test)&set(train)
    ep=min(test);selected=json.loads(json.dumps(m));selected['demo']['splits']['test']['episode_indices']=[ep]
    d=Pool(selected,'demo','test');row=d.rows[ep];n=int(row['length']);T=n-1;Z=m['scale'];assert Z==15337 and not d.events[ep].any()
    cams=[]
    for camera in d.cameras:
        p='videos/'+camera;path=d.root/d.info['video_path'].format(video_key=camera,chunk_index=row[p+'/chunk_index'],file_index=row[p+'/file_index'])
        queries=float(row[p+'/from_timestamp'])+d.timestamps[ep].astype(np.float64)
        cams.append((str(path),queries.tolist()))
    provenance={'source':'pure human teleoperation demonstrations','root':str(d.root),'episode':ep,'episode_selection':'minimum test episode index, selected before inference, not by scores','frames':n,'fps':30,'test_ids':test,'manifest_sha256':sha(ROOT/'manifest.json'),'action_state_sha256':d.fingerprints[ep],'cameras':d.cameras,'camera_specs':cams,'scale':Z,'penalty':150,'takeover_events':0,'note':'This DEMO episode11 is a different physical trajectory from the previous HIL episode11.'}
    put(OUT/'source_manifest.json',provenance)
    test_metrics={}
    for step in [1000,1500]:
        metrics=json.loads((RUN/f'regular-{step:06d}-metrics.json').read_text())
        old=json.loads((RUN/f'regular-{step:06d}-test_demo-predictions.json').read_text())
        assert old['episode_ids']==test and len(old['rows'])==56*32
        test_metrics[str(step)]={'metrics':metrics['metrics']['test_demo'],'frames_per_episode':32,'provenance':'existing frozen checkpoint evaluation, reused and checked; not a new full-frame pass over all56 episodes','metrics_path':str(RUN/f'regular-{step:06d}-metrics.json')}
    put(OUT/'all56_test_metrics.json',test_metrics)
    print('SOURCE',str(d.root),'DEMO_EP',ep,'FRAMES',n,flush=True)
    torch.set_num_threads(4);device=torch.device('cuda:0');models={};values={};ces={}
    for step in [1000,1500]:
        key=f'mixed{step}';cache=OUT/f'{key}_native_predictions.json'
        if cache.exists():
            result=json.loads(cache.read_text());assert result['manifest_sha256']==provenance['manifest_sha256'] and result['root']==str(d.root) and result['step']==step
            values[key]=result['values'];ces[key]=result['cross_entropy'];continue
        model=LeanVisionValue(SimpleNamespace(use_gradient_checkpointing=False)).to(device)
        saved=torch.load(RUN/f'checkpoint-{step:06d}.pt',map_location='cpu',weights_only=False)
        assert saved['step']==step and saved['protocol']['manifest_sha256']==provenance['manifest_sha256']
        model.load_state_dict(saved['model'],strict=True);del saved
        models[key]=model.eval();values[key]=[];ces[key]=[]
    centers=torch.linspace(-1,0,201,device=device);batch=[];batch_ids=[]
    if models:
        with torch.inference_mode():
            for i,images in enumerate(zip(*(camera_frames(Path(p),np.array(q)) for p,q in cams))):
                batch.append(torch.from_numpy(np.stack(images)).permute(0,3,1,2));batch_ids.append(i)
                if len(batch)==8 or i==T:
                    x=torch.stack(batch).to(device);targets=torch.tensor(d.returns[ep][batch_ids]/Z,device=device,dtype=torch.float32)
                    positions=(targets+1)*200;low=positions.floor().long().clamp(0,200);high=(low+1).clamp(0,200);weight=positions-low
                    for key,model in models.items():
                        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images=x,image_attention_mask=torch.ones(x.shape[:2],device=device,dtype=torch.bool))
                        logp=logits.float().log_softmax(-1);pred=(logp.exp()*centers).sum(-1)
                        ce=-(1-weight)*logp.gather(1,low[:,None]).squeeze(1)-weight*logp.gather(1,high[:,None]).squeeze(1)
                        values[key].extend(pred.cpu().tolist());ces[key].extend(ce.cpu().tolist())
                    batch=[];batch_ids=[]
                    if (i+1)%400==0 or i==T:print('INFERENCE_DEMO',i+1,n,flush=True)
        for key in models:
            put(OUT/f'{key}_native_predictions.json',{'root':str(d.root),'episode':ep,'step':int(key[5:]),'manifest_sha256':provenance['manifest_sha256'],'values':values[key],'cross_entropy':ces[key]})
    models.clear();torch.cuda.empty_cache()
    specs=[];vv=[];aa=[];episode_metrics={}
    for step in [1000,1500]:
        key=f'mixed{step}';v=np.asarray(values[key]);assert len(v)==n and np.isfinite(v).all()
        ends=np.minimum(np.arange(n)+50,T);boot=v.copy();boot[-1]=0
        a=-(ends-np.arange(n))/Z+boot[ends]-v;a[-1]=0
        target=d.returns[ep]/Z;episode_metrics[key]={'episode':ep,'frames':n,'ce':float(np.mean(ces[key])),'mae':float(np.mean(np.abs(v-target))),'positive_nonterminal_frames':int((a[:-1]>0).sum()),'negative_nonterminal_frames':int((a[:-1]<0).sum())}
        old=json.loads((RUN/f'regular-{step:06d}-test_demo-predictions.json').read_text());slot=old['episode_ids'].index(ep)
        diffs=[abs(v[int(r[1])]-r[3]) for r in old['rows'] if r[0]==slot]
        assert len(diffs)==32 and max(diffs)<.002
        episode_metrics[key]['max_diff_vs_existing32_frame_eval']=float(max(diffs))
        put(OUT/f'{key}_timeline.json',{'episode':ep,'step':step,'root':str(d.root),'checkpoint':str(RUN/f'checkpoint-{step:06d}.pt'),'value_common_scale':v.tolist(),'advantage_full':a.tolist(),'time_value_delta':a.tolist(),'explicit_penalty':[0.]*n,'human_control':[True]*n,'takeover_onset_frames':[],'scale':Z,'fps':30,'reward':'nonterminal -1, terminal0; pure teleoperation has no takeover events; C150 retained but event contribution0','prediction':'every source frame, no interpolation','target_value':target.tolist()})
        vv.extend(v.tolist());aa.extend(a.tolist());specs.append({'key':key,'camera_specs':cams})
    vlo=math.floor((min(vv)-.01)*20)/20;vhi=min(0.,math.ceil((max(vv)+.01)*20)/20);amp=math.ceil(max(abs(min(aa)),abs(max(aa)))*100)/100+.005
    for spec in specs:spec.update(value_limits=[vlo,vhi],advantage_limits=[-amp,amp])
    put(OUT/'comparison_manifest.json',{'source_manifest':provenance,'render_specs':specs,'episode_metrics':episode_metrics})
    print('INFERENCE_COMPLETE',json.dumps(episode_metrics),flush=True)

if __name__=='__main__':main()
