"""One authorized mixed run: wait for verified transfer, preflight, smoke, train.
Old user-cancelled experiments are never dependencies and never restarted.
"""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY='/data/experiments/value-demo3-20260906/venv-py312-torch210-cu128/bin/python'
RUN='piperx-vision-mixed1to1-c150-z15337-fixed1500-seed1000'

def put(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def command(smoke=False,dense=False):
    cmd=[PY,'-m','torch.distributed.run','--nproc_per_node=4','--master_port=29861','--',str(ROOT/'train_mixed.py'),'--run',RUN+('-smoke' if smoke else ''),'--steps','2' if smoke else '1500','--batch-size','8','--workers','2','--warmup','1' if smoke else '200','--eval-every','2' if smoke else '250','--eval-frames','2' if smoke else ('128' if dense else '32')]
    if smoke:cmd+=['--smoke']
    if dense:cmd+=['--evaluate-checkpoint','checkpoint-001500.pt']
    return cmd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    if a.dry_run:
        for smoke,dense in [(True,False),(False,False),(False,True)]:print(json.dumps(command(smoke,dense)))
        return
    lock=(ROOT/'experiment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HOME='/data/cache/huggingface',NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    (ROOT/'logs').mkdir(exist_ok=True)
    def state(stage,**kw):put(ROOT/'active.json',dict(stage=stage,at=time.time(),**kw));print(stage,flush=True)
    def execute(cmd,log):
        with (ROOT/'logs'/log).open('a') as f:subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        state('waiting_for_verified_transfer')
        while not (ROOT/'transfer_verified.json').exists():time.sleep(30)
        assert json.loads((ROOT/'transfer_verified.json').read_text())['status']=='ok'
        if not (ROOT/'preflight.json').exists():
            state('data_preflight');execute([PY,str(ROOT/'preflight.py')],'preflight.log')
        assert json.loads((ROOT/'preflight.json').read_text())['status']=='ok'
        state('waiting_for_free_gpu_group',gpus='0,1,2,3')
        while True:
            raw=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if all(int(x)<1000 for x in raw.split()):break
            time.sleep(30)
        for smoke,dense in [(True,False),(False,False),(False,True)]:
            cmd=command(smoke,dense);name=RUN+('-smoke' if smoke else '');out=ROOT/'runs'/name
            if dense and (out/'dense-001500-metrics.json').exists():continue
            if not dense and (out/'final.json').exists():
                if smoke:assert (out/'smoke_ok.json').exists()
                continue
            if not dense and (out/'last.json').exists():cmd+=['--resume']
            elif not dense and (out/'protocol.json').exists():raise RuntimeError('Existing run without checkpoint; preserve and inspect')
            state('dense_review' if dense else ('smoke' if smoke else 'training'),command=cmd)
            execute(cmd,name+('-dense' if dense else '')+'.log')
            if smoke:assert json.loads((out/'smoke_ok.json').read_text())['forward_backward_optimizer_save_reload']=='ok'
        state('value_training_and_dense_complete')
        put(ROOT/'training_dense_complete.json',dict(status='ok',run=RUN,step=1500,advantage_evaluation_required=True))
        advantage=ROOT/'runs'/RUN/'advantage-001500-summary.json'
        if not advantage.exists():
            cmd=[PY,'-m','torch.distributed.run','--nproc_per_node=4','--master_port=29861','--',str(ROOT/'evaluate_advantage.py'),'--run',RUN]
            state('advantage_evaluation',command=cmd);execute(cmd,'advantage-001500.log')
        assert json.loads(advantage.read_text())['status']=='ok'
        state('complete_pending_human_review')
        put(ROOT/'complete.json',dict(status='ok',run=RUN,step=1500,charts_and_interpretation_pending=True))
    except Exception as e:
        state('failed',error=type(e).__name__+': '+str(e));raise

if __name__=='__main__':main()
