"""Read-only paired state interventions on the existing step-3000 checkpoint."""
import json
import os
from argparse import Namespace
from datetime import timedelta
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from ablation_train import BASE,EXPERIMENT,Frames,ValueModel,Pistar06Config,build_bin_centers,evaluate,atomic_json

def main():
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)))
    torch.cuda.set_device(device); torch.set_num_threads(2)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=40))
    manifest=json.loads((BASE/'split90_10/manifests/piperx_insert_copper_screw.json').read_text())
    train=Frames(manifest,'train')
    test=Frames(manifest,'test',scale=train.scale,evaluation_frames=32)
    baseline=float(-sum(n*(n-1)/2 for n in train.lengths)/sum(train.lengths)/train.scale)
    cfg=Pistar06Config(vision_repo_id=str(BASE/'models/siglip'),language_repo_id=str(BASE/'models/gemma'),camera_features=manifest['camera_features'],dtype='float32',device='cuda')
    model=ValueModel(cfg).to(device)
    checkpoint=BASE/'runs/piperx_insert_copper_screw-bf16-gb32-seed1000-pilot8k-v1/checkpoint-003000.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['step']==3000
    model.load_state_dict(saved['model'],strict=True); del saved
    tokenizer=AutoTokenizer.from_pretrained(str(BASE/'models/gemma'),padding_side='right',local_files_only=True)
    centers=build_bin_centers(201,-1,0,device)
    args=Namespace(workers=2,batch_size=4)
    out=EXPERIMENT/'state-probe-step3000'
    if rank==0: out.mkdir(exist_ok=True)
    results={}
    for intervention in ('none','zero','permute'):
        test.state_intervention=intervention
        metrics,predictions=evaluate(model,test,args,tokenizer,device,centers,rank,world,baseline)
        results[intervention]=metrics
        if rank==0:
            atomic_json(out/(intervention+'.json'),dict(checkpoint=str(checkpoint),state_intervention=intervention,metrics=metrics,episode_ids=test.ids,columns=['episode_slot','frame','target','prediction'],rows=predictions))
            print(json.dumps(dict(event='STATE_PROBE',intervention=intervention,metrics=metrics)),flush=True)
    if rank==0:
        atomic_json(out/'COMPLETE.json',dict(step=3000,results=results,caution='Inference interventions may be out of distribution: measure reliance, not retrained necessity. All-success countdown labels are not measured physical progress.'))
    if world>1: dist.destroy_process_group()

if __name__=='__main__': main()
