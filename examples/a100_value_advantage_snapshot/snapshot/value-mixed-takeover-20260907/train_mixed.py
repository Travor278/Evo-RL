"""Standalone mixed value pilot; never edits installed LeRobot/Evo sources."""
import argparse,json,math,os,random,sys,time
from datetime import timedelta
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
PARENT=Path('/data/experiments/value-vision-schedules-20260907')
sys.path.insert(0,str(PARENT))
import train_lean_value as u
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset
from mixed_data import ROOT,Pool,BalancedBatches,put,sha

def learning_rate(step,steps,warmup=200):
    if step<=warmup:return 5e-5*step/max(warmup,1)
    return 1e-6+0.5*(5e-5-1e-6)*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))

def clean_metric(m):
    m=dict(m);m['mae_cost_equivalent_seconds']=m.pop('mae_remaining_seconds')
    return m

def combine(parts,weights):
    out={}
    for key in ['ce','mae','episode_mae','bias','crps','entropy','interval90_coverage','interval90_width','mae_cost_equivalent_seconds']:
        out[key]=sum(w*m[key] for w,m in zip(weights,parts))
    out['rmse']=math.sqrt(sum(w*m['rmse']**2 for w,m in zip(weights,parts)))
    out['aggregation']='specified source weights, not all-frame pooled metric'
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--steps',type=int,default=1500)
    ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--warmup',type=int,default=200);ap.add_argument('--eval-every',type=int,default=250)
    ap.add_argument('--eval-frames',type=int,default=32);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--evaluate-checkpoint');args=ap.parse_args()
    assert Path(args.run).name==args.run and args.batch_size==8
    assert (args.steps==2 and args.warmup==1 and args.eval_every==2) if args.smoke else (args.steps==1500 and args.warmup==200 and args.eval_every==250)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    assert world==4,'Protocol global32 requires four ranks'
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=40));torch.set_num_threads(2)
    random.seed(1000+rank);np.random.seed(1000+rank);torch.manual_seed(1000)
    manifest=json.loads((ROOT/'manifest.json').read_text());preflight=json.loads((ROOT/'preflight.json').read_text())
    assert preflight['status']=='ok' and preflight['manifest_sha256']==sha(ROOT/'manifest.json')
    cfg=json.loads((ROOT/'reward_config.json').read_text())
    assert cfg['normalization_multiplier']==1 and manifest['scale']==15337 and manifest['penalty']==150
    demo=Pool(manifest,'demo','train');hil=Pool(manifest,'hil','train');train=ConcatDataset([demo,hil])
    baseline={'demo':demo.baseline(),'hil':hil.baseline()}
    eval_pools={
        'train_demo':Pool(manifest,'demo','train',args.eval_frames,56),
        'train_hil':Pool(manifest,'hil','train',args.eval_frames,14),
        'test_demo':Pool(manifest,'demo','test',args.eval_frames),
        'test_hil_hf':Pool(manifest,'hil','test',args.eval_frames,source_batch='hf_clean'),
        'test_hil_4090a':Pool(manifest,'hil','test',args.eval_frames,source_batch='4090a_20260903')}
    config=u.Pistar06Config(vision_repo_id=str(u.BASE/'models/siglip'),language_repo_id=str(u.BASE/'models/gemma'),camera_features=manifest['demo']['camera_features'],dtype='float32',use_gradient_checkpointing=True,device='cuda',freeze_language_model=False,freeze_vision_encoder=False)
    model=u.VisionOnlyModel(config).to(device);assert not hasattr(model,'language_model')
    centers=u.build_bin_centers(201,-1,0,device)
    out=ROOT/'runs'/args.run
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    protocol=dict(arguments=vars(args),manifest_sha256=sha(ROOT/'manifest.json'),reward_config_sha256=sha(ROOT/'reward_config.json'),runner_sha256=sha(__file__),data_code_sha256=sha(ROOT/'mixed_data.py'),helper_sha256=sha(PARENT/'train_lean_value.py'),lean_model_sha256=sha(PARENT/'lean_model.py'),initial_sha256=sha(PARENT/'assets/initial-seed1000.safetensors'),global_batch=32,per_gpu_batch=8,world=4,source_ratio='exactly demo4+hil4 per rank each step',parameter_dtype='float32',autocast='bfloat16',state_input=False,language_input=False,return_scale=15337,penalty=150,optimizer=dict(type='AdamW',betas=[0.9,0.999],eps=1e-8,weight_decay=1e-5,clip_grad_norm=10),training_stop='fixed1500; no early stopping',selection='final endpoint primary; intermediate holdout and advantage diagnostics secondary',test_note='held-out used for model comparison, not untouched final test')
    exists=[(out/'protocol.json').exists() if rank==0 else None];dist.broadcast_object_list(exists,src=0)
    if exists[0]:
        assert args.resume or args.evaluate_checkpoint,'Do not overwrite an existing run'
        previous=json.loads((out/'protocol.json').read_text())
        for key in ['manifest_sha256','reward_config_sha256','data_code_sha256','runner_sha256','helper_sha256','lean_model_sha256','initial_sha256','global_batch','return_scale','penalty']:
            assert previous[key]==protocol[key],('Resume protocol mismatch',key)
        for key in ['steps','batch_size','warmup','eval_every','smoke']:
            assert previous['arguments'][key]==vars(args)[key]
        if not args.evaluate_checkpoint:assert previous['arguments']['eval_frames']==args.eval_frames
    elif rank==0:
        assert not args.evaluate_checkpoint;put(out/'protocol.json',protocol)
    dist.barrier()

    def evaluation(step,label='regular'):
        metrics={}
        for name,d in eval_pools.items():
            m,pred=u.evaluate(model,d,args,None,device,centers,rank,world,baseline[d.source]);metrics[name]=clean_metric(m)
            if rank==0:put(out/f'{label}-{step:06d}-{name}-predictions.json',dict(columns=['episode_slot','frame','target','prediction'],episode_ids=d.ids,rows=pred))
        # HIL held-out origin weights by evaluated frame count (fixed frames per ep).
        hparts=[metrics['test_hil_hf'],metrics['test_hil_4090a']];total=sum(x['frames'] for x in hparts)
        metrics['test_hil']=combine(hparts,[x['frames']/total for x in hparts])
        metrics['train_balanced']=combine([metrics['train_demo'],metrics['train_hil']],[.5,.5])
        metrics['test_balanced']=combine([metrics['test_demo'],metrics['test_hil']],[.5,.5])
        record=dict(event='EVAL',step=step,evaluation_frames_per_episode=args.eval_frames,kind=label,metrics=metrics)
        if rank==0:
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            put(out/f'{label}-{step:06d}-metrics.json',record);print(json.dumps(record),flush=True)
        return record

    if args.evaluate_checkpoint:
        assert Path(args.evaluate_checkpoint).name==args.evaluate_checkpoint
        saved=torch.load(out/args.evaluate_checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);step=saved['step'];del saved
        evaluation(step,'dense');dist.destroy_process_group();return
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-5)
    start=0
    if args.resume:
        last=json.loads((out/'last.json').read_text());saved=torch.load(out/last['file'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);start=saved['step'];u.restore_rng(saved['rng'][rank]);del saved
    assert start<args.steps,'Already at requested endpoint; do not retrain'
    wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],find_unused_parameters=False)
    sampler=BalancedBatches([len(demo),len(hil)],args.batch_size,rank,world,start,args.steps)
    if rank==0:print(json.dumps(dict(event='READY',step=start,global_batch=32,train_demo=len(demo),train_hil=len(hil),scale=15337,penalty=150,total_parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    sums=torch.zeros(3,device=device);tick=time.monotonic()
    for step,batch in enumerate(u.loader(train,args,sampler=sampler),start+1):
        model.train();lr=learning_rate(step,args.steps,args.warmup)
        for group in optimizer.param_groups:group['lr']=lr
        inputs,y=u.prepare(batch,None,device);optimizer.zero_grad(set_to_none=True)
        assert torch.all((y>=-1)&(y<=0))
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=wrapped(**inputs)
        loss=-(u.project_values_to_bins(y,centers)*logits.float().log_softmax(-1)).sum(-1).mean()
        assert torch.isfinite(loss),'Nonfinite loss';loss.backward()
        if args.smoke:
            for module in [model.vision_encoder,model.value_head]:assert any(p.grad is not None and p.grad.abs().max()>0 for p in module.parameters())
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step()
        sums+=torch.stack([loss.detach(),grad.detach(),torch.ones((),device=device)])
        if step%10==0 or args.smoke:
            dist.all_reduce(sums)
            if rank==0:print(json.dumps(dict(event='TRAIN',step=step,loss=float(sums[0]/sums[2]),grdn=float(sums[1]/sums[2]),lr=lr,seconds=time.monotonic()-tick,draw_epoch_demo=step*16/len(demo),draw_epoch_hil=step*16/len(hil),memory_gb=torch.cuda.max_memory_allocated()/1e9)),flush=True)
            sums.zero_();tick=time.monotonic()
        if step%args.eval_every and step!=args.steps:continue
        evaluation(step)
        states=[None]*world;dist.all_gather_object(states,u.rng_state())
        if rank==0:
            filename=f'checkpoint-{step:06d}.pt';assert not (out/filename).exists()
            tmp=out/(filename+'.part');assert not tmp.exists(),'Preserve previous incomplete checkpoint'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rng=states,protocol=protocol),tmp);os.replace(tmp,out/filename)
            put(out/'last.json',dict(file=filename,step=step))
        dist.barrier()
        if args.smoke:
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):expected=model(**inputs).float()
            saved=torch.load(out/f'checkpoint-{step:06d}.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(**inputs).float()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            if rank==0:put(out/'smoke_ok.json',dict(step=step,forward_backward_optimizer_save_reload='ok'));print('SMOKE_SAVE_RELOAD_OK',flush=True)
    if rank==0:put(out/'final.json',dict(step=step,file=f'checkpoint-{step:06d}.pt',status='training_complete'));print('TRAINING_COMPLETE',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
