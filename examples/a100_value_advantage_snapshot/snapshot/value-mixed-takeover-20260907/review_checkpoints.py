"""Read-only checkpoint inference; no training or modification of existing outputs."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY='/data/experiments/value-demo3-20260906/venv-py312-torch210-cu128/bin/python'
RUN='piperx-vision-mixed1to1-c150-z15337-fixed1500-seed1000'
def main():
    lock=(ROOT/'review-checkpoints.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'complete.json').read_text())['status']=='ok'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HOME='/data/cache/huggingface',NCCL_DEBUG='WARN')
    for step in [1000,750,500,1250,250]:
        out=ROOT/'runs'/RUN
        report=out/f'advantage-{step:06d}-summary.json'
        if report.exists():
            assert json.loads(report.read_text())['status']=='ok'
            continue
        assert not (out/f'advantage-{step:06d}-records.json').exists(),'Partial review exists; inspect rather than overwrite'
        assert (out/f'checkpoint-{step:06d}.pt').is_file()
        raw=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert all(int(x)<1000 for x in raw.split()),'GPU group is occupied'
        print(json.dumps(dict(event='REVIEW_START',step=step,at=time.time())),flush=True)
        cmd=[PY,'-m','torch.distributed.run','--nproc_per_node=4','--master_port=29862','--',str(ROOT/'evaluate_advantage.py'),'--run',RUN,'--checkpoint',f'checkpoint-{step:06d}.pt']
        with (ROOT/'logs'/f'advantage-{step:06d}.log').open('a') as f:
            subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert json.loads(report.read_text())['status']=='ok'
        print(json.dumps(dict(event='REVIEW_COMPLETE',step=step,at=time.time())),flush=True)
    print('ALL_CHECKPOINT_REVIEWS_COMPLETE',flush=True)
if __name__=='__main__':main()
