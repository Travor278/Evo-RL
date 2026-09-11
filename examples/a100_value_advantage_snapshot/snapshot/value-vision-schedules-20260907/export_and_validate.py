"""One-time, non-destructive export and real-image equivalence check."""
import sys,json,hashlib,gc,time
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
sys.path.insert(0,'/data/experiments/value-ablation-20260906')
import ablation_train as old
from lean_model import LeanVisionValue,ASSETS

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def active(s):
    return {k:t.detach().cpu().contiguous() for k,t in s.items() if not k.startswith('vision_encoder.text_model.') and k not in ('vision_encoder.logit_scale','vision_encoder.logit_bias')}
def main():
    torch.set_num_threads(4)
    ASSETS.mkdir(parents=True,exist_ok=True)
    assert not (ASSETS/'export_validation.json').exists(), 'Export already finished; do not overwrite'
    cfg=old.Pistar06Config(vision_repo_id=str(old.BASE/'models/siglip'),language_repo_id=str(old.BASE/'models/gemma'),dtype='float32',use_gradient_checkpointing=True,device='cuda',freeze_language_model=False,freeze_vision_encoder=False)
    torch.manual_seed(1000)
    model=old.VisionOnlyModel(cfg)
    save_file({'cpu_rng':torch.get_rng_state()},str(ASSETS/'initial-rng.safetensors'))
    save_file(active(model.state_dict()),str(ASSETS/'initial-seed1000.safetensors'))
    old.atomic_json(ASSETS/'vision_config.json',model.vision_encoder.config.vision_config.to_dict())
    old.atomic_json(ASSETS/'export_manifest.json',dict(initial_sha256=sha(ASSETS/'initial-seed1000.safetensors'),seed=1000,source_runner_sha256=old.digest(old.__file__)))
    initial=active(model.state_dict())
    lean=LeanVisionValue(cfg)
    for k,v in lean.state_dict().items(): torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
    del initial
    source=old.EXPERIMENT/'runs/piperx_insert_copper_screw-vision_only-fixed1500-seed1000-v1/checkpoint-001500.pt'
    ckpt=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(ckpt['model'],strict=True)
    lean.load_state_dict(active(ckpt['model']),strict=True)
    exported=ASSETS/'vision-only-step1500-model.safetensors'
    save_file(active(ckpt['model']),str(exported))
    lean.load_state_dict(load_file(str(exported)),strict=True)
    del ckpt;gc.collect()
    manifest=json.loads((old.BASE/'split90_10/manifests/piperx_insert_copper_screw.json').read_text())
    ds=old.Frames(manifest,'test',evaluation_frames=3,episode_limit=2,variant='vision_only')
    model=model.cuda().eval();lean=lean.cuda().eval()
    diffs={}
    for dtype in ('float32','bfloat16'):
        largest=0.
        for i in range(len(ds)):
            item=ds[i]
            inputs={'images':item['images'][None].cuda(),'image_attention_mask':torch.ones(1,3,dtype=torch.bool,device='cuda')}
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype=='bfloat16'):
                a=model(**inputs);b=lean(**inputs)
            largest=max(largest,float((a.float()-b.float()).abs().max()))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        diffs[dtype]=largest
    # Eval-mode gradient comparison removes Dropout randomness, not gradient flow.
    # BF16 fused-attention backward is not a bitwise equivalence oracle.
    # Validate the mathematical gradients in FP32 with the same math backend.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        for m in (model,lean):
            m.zero_grad(set_to_none=True)
            logits=m(**inputs)
            loss=logits.float().square().mean();loss.backward()
    grads=0;maxdiff=0.
    oldp=dict(model.named_parameters())
    for k,p in lean.named_parameters():
        assert p.grad is not None and oldp[k].grad is not None,k
        torch.testing.assert_close(p.grad,oldp[k].grad,rtol=1e-5,atol=1e-6)
        maxdiff=max(maxdiff,float((p.grad-oldp[k].grad).abs().max()));grads+=p.numel()
    result=dict(status='ok',real_frames=len(ds),cameras=3,initial_tensors_exact=True,final_reload_strict=True,logits_max_abs_diff=diffs,gradient_validation='FP32 math SDPA; fused BF16 backward is not asserted bitwise',gradient_max_abs_diff=maxdiff,gradient_parameters=grads,old_total_parameters=sum(p.numel() for p in model.parameters()),lean_total_parameters=sum(p.numel() for p in lean.parameters()),inference_file_bytes=exported.stat().st_size,exported_checkpoint=str(exported),source_checkpoint=str(source),created_at=time.time())
    old.atomic_json(ASSETS/'export_validation.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
