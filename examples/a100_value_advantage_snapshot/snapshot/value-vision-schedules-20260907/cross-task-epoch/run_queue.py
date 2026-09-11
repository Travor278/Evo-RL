"""Bounded four-job queue. Wait for current lane completion; never share GPUs."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
PY='/data/experiments/value-demo3-20260906/venv-py312-torch210-cu128/bin/python'

def put(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def command(job,lane,smoke=False,dense=False):
    name=job['run'] if not smoke else job['run']+'-smoke'
    steps=2 if smoke else job['steps']
    cmd=[PY,'-m','torch.distributed.run','--nproc_per_node=4',f'--master_port={29841 if lane=="constant" else 29842}','--',str(ROOT/'train_cross_value.py'),'--task',job['task'],'--variant','vision_only','--run',name,'--steps',str(steps),'--lr-schedule','cosine','--seed','1000','--batch-size','8','--workers','2','--peak-lr','5e-5','--min-lr','1e-6','--warmup','1' if smoke else str(job['warmup']),'--eval-every','2' if smoke else str(job['eval_every']),'--eval-frames','2' if smoke else ('128' if dense else '32')]
    if smoke:cmd+=['--smoke']
    if dense:cmd+=['--evaluate-checkpoint',f'checkpoint-{steps:06d}.pt']
    return cmd,name,steps

def execute(job,lane,env,smoke=False,dense=False):
    cmd,name,steps=command(job,lane,smoke,dense)
    out=ROOT/'runs'/name
    final=out/'final.json'
    review=out/f'dense-review-{steps:06d}-frames128.json'
    log=ROOT/'logs'/(name+('-dense' if dense else '')+'.log')
    if dense:
        assert json.loads(final.read_text())['step']==steps
        if review.exists():
            assert json.loads(review.read_text())['checkpoint_reload']=='strict_ok'
            return
    elif (out/'training_complete.json').exists():
        assert json.loads(final.read_text())['step']==steps
        if smoke:assert 'SMOKE_SAVE_RELOAD_OK' in log.read_text()
        return
    elif (out/'last.json').exists():cmd+=['--resume']
    elif (out/'protocol.json').exists():raise RuntimeError('Incomplete run has no checkpoint; preserve and inspect before any fresh restart')
    put(ROOT/f'{lane}-active.json',dict(stage='dense_review' if dense else ('smoke' if smoke else 'training'),job=job,command=cmd,gpus=env['CUDA_VISIBLE_DEVICES'],at=time.time()))
    with log.open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    if dense:assert json.loads(review.read_text())['checkpoint_reload']=='strict_ok'
    else:
        assert json.loads(final.read_text())['step']==steps
        if smoke:assert 'SMOKE_SAVE_RELOAD_OK' in log.read_text()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('lane',choices=['constant','cosine']);ap.add_argument('--dry-run',action='store_true');ap.add_argument('--retry-failed',action='store_true');args=ap.parse_args()
    plan=json.loads((ROOT/'plan.json').read_text());assert plan['status']=='ok' and len(plan['jobs'])==4
    if args.dry_run:
        for j in plan['jobs']:print(json.dumps(dict(job=j,command=command(j,args.lane)[0])))
        return
    for name in ['logs','state','locks','runs']:(ROOT/name).mkdir(exist_ok=True)
    lane_lock=(ROOT/'locks'/f'{args.lane}.lock').open('a');fcntl.flock(lane_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_ids='0,1,2,3' if args.lane=='constant' else '4,5,6,7'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu_ids,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_HOME='/data/cache/huggingface',NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    dependency=PARENT/f'{args.lane}-complete.json'
    put(ROOT/f'{args.lane}-active.json',dict(stage='waiting_for_existing_lane_final_review',dependency=str(dependency),gpus=gpu_ids,at=time.time()))
    print('WAITING_FOR_EXISTING_LANE '+str(dependency),flush=True)
    while not dependency.exists():time.sleep(30)
    assert json.loads(dependency.read_text())['status']=='complete'
    for job in plan['jobs']:
        lock=(ROOT/'locks'/(job['run']+'.lock')).open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close();continue
        try:
            state=ROOT/'state'/(job['run']+'.json')
            previous=json.loads(state.read_text()) if state.exists() else {}
            if previous.get('status')=='complete':continue
            if previous.get('status')=='failed' and not args.retry_failed:continue
            # No user/other-job GPU sharing: wait until this group is actually free.
            while True:
                raw=subprocess.check_output(['nvidia-smi','-i',gpu_ids,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                if all(int(x)<1000 for x in raw.split()):break
                put(ROOT/f'{args.lane}-active.json',dict(stage='waiting_for_free_gpu_group',gpus=gpu_ids,at=time.time()))
                time.sleep(30)
            put(state,dict(status='running',lane=args.lane,job=job,at=time.time()))
            try:
                execute(job,args.lane,env,smoke=True)
                execute(job,args.lane,env)
                execute(job,args.lane,env,dense=True)
                put(state,dict(status='complete',lane=args.lane,job=job,at=time.time()))
                print('CROSS_JOB_COMPLETE '+job['run'],flush=True)
            except Exception as e:
                put(state,dict(status='failed',lane=args.lane,job=job,error=type(e).__name__+': '+str(e),at=time.time()))
                raise
        finally:lock.close()
    put(ROOT/f'{args.lane}-active.json',dict(stage='lane_queue_exhausted',at=time.time()))
    print('LANE_QUEUE_EXHAUSTED',flush=True)
if __name__=='__main__':main()
