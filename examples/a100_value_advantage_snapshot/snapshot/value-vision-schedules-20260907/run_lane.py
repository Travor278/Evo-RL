"""Bounded experiment lane: smoke gate, exact schedule runs, strict final review."""
import argparse,json,os,subprocess,fcntl,time
from pathlib import Path
ROOT=Path('/data/experiments/value-vision-schedules-20260907')
PY='/data/experiments/value-demo3-20260906/venv-py312-torch210-cu128/bin/python'
def put(p,d):
    tmp=p.with_suffix('.part');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('lane',choices=['constant','cosine','cosine4500']);ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args()
    if a.lane=='cosine4500':
        raise SystemExit('USER_PAUSED_4500: explicit user approval required before any restart')
    lock=(ROOT/(a.lane+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'assets/export_validation.json').read_text())['status']=='ok'
    schedule='constant' if a.lane=='constant' else 'cosine'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3' if a.lane in ('constant','cosine4500') else '4,5,6,7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_HOME='/data/cache/huggingface',NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    port={'constant':'29731','cosine':'29732','cosine4500':'29733'}[a.lane]
    def run(steps,smoke=False):
        name=f'piperx-visionlean-{schedule}-fixed{steps}-seed1000'+('-smoke' if smoke else '')
        if smoke and a.lane=='cosine4500':name+='-shared4500'
        out=ROOT/'runs'/name
        cmd=[PY,'-m','torch.distributed.run','--nproc_per_node=4','--master_port='+port,'--',str(ROOT/'train_lean_value.py'),'--task','piperx_insert_copper_screw','--variant','vision_only','--run',name,'--steps',str(steps),'--lr-schedule',schedule,'--seed','1000','--batch-size','8','--workers','2','--peak-lr','5e-5','--min-lr','1e-6','--warmup','1' if smoke else '200','--eval-every','2' if smoke else '250','--eval-frames','2' if smoke else '32']
        if smoke:cmd+=['--smoke']
        if not (out/'training_complete.json').exists():
            if (out/'last.json').exists():cmd+=['--resume']
            elif (out/'protocol.json').exists():raise RuntimeError('Existing incomplete run without checkpoint; inspect before retry')
            put(ROOT/(a.lane+'-active.json'),dict(stage='smoke' if smoke else 'training',name=name,steps=steps,command=cmd,started_at=time.time()))
            with (ROOT/(name+'.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        final=json.loads((out/'final.json').read_text());assert final['step']==steps
        if smoke:
            assert 'SMOKE_SAVE_RELOAD_OK' in (ROOT/(name+'.log')).read_text()
            put(ROOT/(a.lane+'-smoke-ok.json'),dict(status='ok',steps=steps,global_batch=32,gpus=env['CUDA_VISIBLE_DEVICES']))
            put(ROOT/(a.lane+'-active.json'),dict(stage='smoke_complete',name=name,at=time.time()))
            return
        review=out/f'dense-review-{steps:06d}-frames128.json'
        if not review.exists():
            cmd=[x for x in cmd if x!='--resume'];cmd[cmd.index('--eval-frames')+1]='128';cmd+=['--evaluate-checkpoint',final['file']]
            put(ROOT/(a.lane+'-active.json'),dict(stage='dense_review',name=name,steps=steps))
            with (ROOT/(name+'-dense.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert json.loads(review.read_text())['checkpoint_reload']=='strict_ok'
    try:
        run(2,True)
        if a.smoke_only:return
        # Formal work starts only after BOTH simultaneous smoke jobs passed.
        assert (ROOT/'constant-smoke-ok.json').exists() and (ROOT/'cosine-smoke-ok.json').exists()
        for steps in {'constant':[8000],'cosine':[3000],'cosine4500':[4500]}[a.lane]:run(steps)
        put(ROOT/(a.lane+'-complete.json'),dict(status='complete',finished_at=time.time()))
        put(ROOT/(a.lane+'-active.json'),dict(stage='complete'))
    except Exception as e:
        put(ROOT/(a.lane+'-failure.json'),dict(error=type(e).__name__+': '+str(e),at=time.time()))
        raise
if __name__=='__main__':main()
