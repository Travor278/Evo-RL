"""CPU-only data/normalization/decode audit and fixed epoch budget manifest."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from train_lean_value import BASE, Frames, atomic_json
import numpy as np

TASKS={
 'piperx_insert_copper_screw':('piperx',1132703,502,56,14882,'172a0c3b0393b32b4d5085e2e8ef84d74312fa96263c26afb3fc0850c6043769'),
 'so101_fold_clothes_left_stack_right':('fold',1125359,97,11,35886,'3417b9a079ee522b08ac31e7a01d54f710299ccb01d86277e659e0ca2d04e4d8'),
 'so101_put_stationery_table_into_bag':('stationery',1129226,357,40,17936,'17b9b3a5e85b4848d860031c3fec38d7af3cddee072fc51fb693c8796e34a6ad'),
}
def main():
    audits=[];jobs=[]
    for task,(short,n,tr,va,scale,sha) in TASKS.items():
        p=BASE/'split90_10/manifests'/f'{task}.json'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
        manifest=json.loads(p.read_text())
        assert len(manifest['splits']['train']['episode_indices'])==tr
        assert len(manifest['splits']['test']['episode_indices'])==va
        assert not set(manifest['splits']['train']['episode_indices']) & set(manifest['splits']['test']['episode_indices'])
        assert manifest['splits']['train']['frames']==n
        decoded=[]
        if short!='piperx':
            train=Frames(manifest,'train',variant='vision_only')
            test=Frames(manifest,'test',scale=train.scale,variant='vision_only')
            assert len(train)==n and train.scale==scale
            for split,ds in [('train',train),('heldout',test)]:
                # Real first/middle/last samples, all 3 cameras, exact tolerance1e-4.
                for index in [0,len(ds)//2,len(ds)-1]:
                    row=ds[index]
                    assert row['images'].shape[0]==3 and row['prompt']==''
                    assert -1<=row['target']<=0
                    decoded.append(dict(split=split,index=index,shape=list(row['images'].shape),target=row['target']))
            stats=json.loads((BASE/'split90_10/train-only-stats'/f'{task}.json').read_text())
            for key in ['action','observation.state']:
                for q in ['q01','q10','q50','q90','q99']:
                    assert q in stats[key] and np.isfinite(stats[key][q]).all()
            del train,test
            for refsteps in [1500,3000]:
                steps=math.ceil(refsteps*n/1132703)
                jobs.append(dict(task=task,short=short,run=f'{short}-visionlean-eq{refsteps}-steps{steps}-seed1000',reference_steps=refsteps,steps=steps,warmup=math.ceil(200*n/1132703),eval_every=math.ceil(250*n/1132703),train_frames=n,target_epoch=refsteps*32/1132703,actual_epoch=steps*32/n,manifest_sha256=sha,return_scale=scale))
        audits.append(dict(task=task,train_frames=n,train_episodes=tr,heldout_episodes=va,return_scale=scale,manifest_sha256=sha,camera_features=manifest['camera_features'],decoded=decoded))
        print(json.dumps(dict(event='PREFLIGHT_TASK_OK',task=task,train_frames=n,decoded=len(decoded))),flush=True)
    result=dict(status='ok',global_batch=32,reference_train_frames=1132703,seed=1000,tasks=audits,jobs=jobs)
    dest=ROOT/'plan.json'
    if dest.exists():assert json.loads(dest.read_text())==result,'Refusing to change established plan'
    else:atomic_json(dest,result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
