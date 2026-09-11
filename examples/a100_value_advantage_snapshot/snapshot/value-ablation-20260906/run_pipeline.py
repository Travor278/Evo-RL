"""Single eight-GPU queue. Fixed endpoints only; never overwrite prior artifacts."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/data/experiments/value-ablation-20260906')
PY='/data/experiments/value-demo3-20260906/venv-py312-torch210-cu128/bin/python'
TASK='piperx_insert_copper_screw'
VARIANTS=['frozen_both','frozen_language','task_only','vision_only','full']

def load(path): return json.loads(path.read_text())
def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(temp,path)
def idle():
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not procs.strip(), 'Other GPU process detected; fail closed without stopping it'
def launch(script,args,log):
    idle()
    command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--',str(ROOT/script),*map(str,args)]
    write(ROOT/'active.json',dict(started=time.time(),command=command,log=str(log)))
    print('LAUNCH '+json.dumps(command),flush=True)
    with log.open('a') as f: subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    idle()

def run(variant,steps,smoke=False):
    name=f'{TASK}-{variant}-fixed{steps}-seed1000'+('-smoke-v1' if smoke else '-v1')
    out=ROOT/'runs'/name
    complete=out/'training_complete.json'
    if complete.exists():
        assert load(complete)['step']==steps and load(complete)['fixed_budget']
        assert (out/load(out/'final.json')['file']).is_file()
    else:
        args=['--task',TASK,'--run',name,'--variant',variant,'--steps',steps,'--batch-size',4,'--workers',2,'--warmup',1 if smoke else 200,'--eval-every',steps if smoke else 250,'--eval-frames',2 if smoke else 32,'--seed',1000,'--peak-lr',5e-5,'--min-lr',1e-6]
        if smoke: args+=['--smoke']
        if (out/'protocol.json').exists():
            assert (out/'last.json').exists(), f'{name} failed before first checkpoint; inspect before relaunch'
            args+=['--resume']
        launch('ablation_train.py',args,ROOT/(name+'.log'))
        assert load(complete)['step']==steps
    if smoke:
        assert 'SMOKE_SAVE_RELOAD_OK' in (ROOT/(name+'.log')).read_text()
    else:
        dense=out/f'dense-review-{steps:06d}-frames128.json'
        if not dense.exists():
            launch('ablation_train.py',['--task',TASK,'--run',name,'--variant',variant,'--steps',steps,'--batch-size',4,'--workers',2,'--eval-frames',128,'--evaluate-checkpoint',load(out/'final.json')['file']],ROOT/(name+'-dense.log'))
        assert load(dense)['step']==steps and load(dense)['checkpoint_reload']=='strict_ok'
        return dict(variant=variant,steps=steps,run=name,final_metrics=load(dense)['test'])

def main():
    os.chdir(ROOT)
    lock=(ROOT/'pipeline.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(HF_HOME='/data/cache/huggingface',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    assert subprocess.check_output([PY,'-c','import torch;print(torch.__version__)'],text=True).strip()=='2.10.0+cu128'
    for variant in VARIANTS: run(variant,2,smoke=True)
    write(ROOT/'smokes_complete.json',dict(variants=VARIANTS,steps=2,forward_backward_frozen_gradients_save_reload='passed'))
    if not (ROOT/'state-probe-step3000/COMPLETE.json').exists(): launch('state_probe.py',[],ROOT/'state-probe.log')
    screen=[run(variant,1500) for variant in VARIANTS]
    ranked=sorted(screen,key=lambda x:(x['final_metrics']['episode_mae'],x['final_metrics']['rmse']))
    selection=dict(screen=screen,selected_variants=[x['variant'] for x in ranked[:2]],rule='Provisional: two lowest FINAL step1500 holdout episode-MAE at 128 fixed frames/episode, RMSE tie-break. No intermediate-best selection. A slow starter may be missed.',holdout_caveat='The same 10% used for experiment selection is validation, not an unbiased final test.')
    write(ROOT/'screen_selection.json',selection)
    print('SCREEN_COMPLETE '+json.dumps(selection),flush=True)
    results=list(screen)
    for steps in (2500,3000):
        for variant in selection['selected_variants']: results.append(run(variant,steps))
    write(ROOT/'fixed_horizon_results.json',dict(results=results,requires_analysis=True,repeat_seed_required_for_close_results=True,other_tasks_not_yet_tested=True))
    write(ROOT/'PIPELINE_COMPLETE.json',dict(time=time.time(),fixed_horizon_runs=len(results),requires_model_selection_review=True))
    print('FIXED_HORIZON_EXPERIMENTS_COMPLETE',flush=True)

if __name__=='__main__': main()
