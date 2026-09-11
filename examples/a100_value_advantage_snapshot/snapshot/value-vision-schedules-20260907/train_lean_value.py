"""Episode-disjoint demo value experiment; standalone extension of official LeRobot.

Upstream value architecture is imported unmodified. No source dataset is written.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', '/data/cache/huggingface')
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

BASE = Path('/data/experiments/value-demo3-20260906')
EXPERIMENT = Path('/data/experiments/value-vision-schedules-20260907')
import lerobot
import lerobot.utils
import lerobot.lerobot_types
# Upstream 0.4.4 processor.core types moved to lerobot_types in 0.6.1.
# Scope this import alias to this experiment process; do not patch installed files.
sys.modules.setdefault('lerobot.processor.core', lerobot.lerobot_types)
REFERENCE = BASE / 'code/Evo-RL-reference/src/lerobot'
lerobot.__path__.append(str(REFERENCE))
lerobot.utils.__path__.append(str(REFERENCE / 'utils'))
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.modeling_pistar06 import Pistar06Model, build_bin_centers, project_values_to_bins
from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


class ValueModel(Pistar06Model):
    """Handle Transformers 5 structured image outputs without changing features."""
    def _encode_images(self, images):
        out = super()._encode_images(images)
        if isinstance(out, torch.Tensor):
            return out
        pooled = getattr(out, 'pooler_output', None)
        if pooled is None:
            raise TypeError(f'Unsupported image feature output: {type(out).__name__}')
        return pooled

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_vision_encoder: self.vision_encoder.eval()
        if self.cfg.freeze_language_model and hasattr(self,'language_model'): self.language_model.eval()
        return self


from lean_model import LeanVisionValue as VisionOnlyModel


class Frames(Dataset):
    def __init__(self, manifest, split, scale=None, evaluation_frames=None, episode_limit=None, variant='full', state_intervention='none'):
        self.variant=variant
        self.state_intervention=state_intervention
        self.manifest = manifest
        self.root = Path(manifest['root'])
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        ids = sorted(manifest['splits'][split]['episode_indices'])
        if episode_limit:
            ids = sorted(ids, key=lambda e: hashlib.sha256(f'train-eval-v1:{e}'.encode()).hexdigest())[:episode_limit]
        selected = set(ids)
        self.rows = {}
        for p in sorted((self.root / 'meta/episodes').rglob('*.parquet')):
            for row in pq.read_table(p).to_pylist():
                if row['episode_index'] in selected:
                    self.rows[int(row['episode_index'])] = row
        assert set(self.rows) == selected
        self.ids = sorted(ids)
        self.lengths = np.array([self.rows[e]['length'] for e in self.ids])
        self.scale = float(scale or (2 * max(self.lengths)))
        # Same successful-return denominator as upstream c_fail_coef=1, but fit TRAIN only.
        assert max(self.lengths) - 1 <= self.scale, 'Target out of fixed support; do not silently clip'
        self.ends = np.cumsum(self.lengths)
        self.starts = np.r_[0, self.ends[:-1]]
        self.states = {}
        self.timestamps = {}
        files = {}
        for e in self.ids:
            row = self.rows[e]
            p = self.root / self.info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
            files.setdefault(p, []).append(e)
        for p, eps in files.items():
            t = pq.read_table(p, columns=['episode_index', 'frame_index', 'timestamp', 'observation.state'])
            epids = t['episode_index'].to_numpy()
            frame = t['frame_index'].to_numpy()
            states = np.stack(t['observation.state'].to_pylist()).astype(np.float32)
            timestamps = t['timestamp'].to_numpy()
            for e in eps:
                mask = epids == e
                n = self.rows[e]['length']
                assert np.array_equal(frame[mask], np.arange(n))
                self.states[e] = states[mask]
                self.timestamps[e] = timestamps[mask]
        stats_path = BASE / 'split90_10/train-only-stats' / (manifest['task'] + '.json')
        stats = json.loads(stats_path.read_text())['observation.state']
        self.q01 = np.array(stats['q01'], dtype=np.float32)
        self.denom = np.array(stats['q99'], dtype=np.float32) - self.q01
        self.denom[self.denom == 0] = 1e-8
        tasks = pq.read_table(self.root / 'meta/tasks.parquet').to_pylist()
        assert len(tasks) == 1
        self.task_text = tasks[0].get('task', tasks[0].get('__index_level_0__'))
        assert isinstance(self.task_text, str)
        self.eval_indices = None
        if evaluation_frames:
            self.eval_indices = np.concatenate([
                start + np.unique(np.linspace(0, n - 1, min(n, evaluation_frames), dtype=np.int64))
                for start, n in zip(self.starts, self.lengths)
            ])

    def __len__(self):
        return len(self.eval_indices) if self.eval_indices is not None else int(self.ends[-1])

    def __getitem__(self, index):
        if self.eval_indices is not None:
            index = int(self.eval_indices[index])
        slot = int(np.searchsorted(self.ends, index, side='right'))
        frame = int(index - self.starts[slot])
        e = self.ids[slot]
        row = self.rows[e]
        stamp = float(self.timestamps[e][frame])
        images = []
        for camera in self.manifest['camera_features']:
            prefix = f'videos/{camera}'
            path = self.root / self.info['video_path'].format(video_key=camera, chunk_index=row[prefix + '/chunk_index'], file_index=row[prefix + '/file_index'])
            images.append(decode_video_frames(path, [float(row[prefix + '/from_timestamp']) + stamp], 1e-4, backend='pyav', return_uint8=True)[0])
        robot_state=self.states[e][frame]
        if self.state_intervention=='permute':
            # Deterministic marginal resampling; same images/labels, different held-out state.
            donor=int.from_bytes(hashlib.sha256(f'state-intervention-v1:{e}:{frame}'.encode()).digest()[:8],'little')%int(self.ends[-1])
            donor_slot=int(np.searchsorted(self.ends,donor,side='right'))
            robot_state=self.states[self.ids[donor_slot]][donor-int(self.starts[donor_slot])]
        state = 2 * (robot_state - self.q01) / self.denom - 1
        if self.state_intervention=='zero': state=np.zeros_like(state)
        state = np.pad(state, (0, 32 - len(state)))
        bins = np.linspace(-1, 1, 257, dtype=np.float32)[:-1]
        discretized = np.digitize(state, bins) - 1
        task = self.task_text.strip().replace('_', ' ').replace('\n', ' ').strip()
        prompt = f'Task: {task}, State: ' + ' '.join(map(str, discretized.tolist())) + '\nValue: '
        if self.variant=='task_only': prompt=f'Task: {task}\nValue: '
        if self.variant=='vision_only': prompt=''
        target = -(int(row['length']) - frame - 1) / self.scale
        assert -1 <= target <= 0
        return {'images': torch.stack(images), 'prompt': prompt, 'target': target, 'episode': slot, 'frame': frame}


class StepBatches(Sampler):
    """Uniform frame sampling with replacement, reproducible across resume/prefetch."""
    def __init__(self, length, batch, rank, world, start, stop, seed):
        self.length, self.batch, self.rank, self.world = length, batch, rank, world
        self.start, self.stop, self.seed = start, stop, seed

    def __iter__(self):
        for step in range(self.start, self.stop):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            indices = rng.integers(self.length, size=self.batch * self.world)
            yield indices[self.rank * self.batch:(self.rank + 1) * self.batch].tolist()

    def __len__(self):
        return self.stop - self.start


def prepare(batch, tokenizer, device):
    inputs={}
    if tokenizer is not None:
        tokens = tokenizer(batch['prompt'], max_length=200, padding='max_length', truncation=True, return_tensors='pt')
        inputs = {k: tokens[k].to(device) for k in ('input_ids', 'attention_mask')}
    inputs['images'] = batch['images'].to(device, non_blocking=True)
    inputs['image_attention_mask'] = torch.ones(inputs['images'].shape[:2], dtype=torch.bool, device=device)
    return inputs, batch['target'].float().to(device)


def loader(dataset, args, sampler=None, indices=None):
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    # Worker initialization must not consume the model/dropout RNG on resume.
    kw = dict(num_workers=args.workers, pin_memory=True,
              generator=torch.Generator().manual_seed(2000 + int(os.environ.get('RANK',0))))
    if args.workers:
        kw.update(prefetch_factor=1)
    if sampler is not None:
        return DataLoader(dataset, batch_sampler=sampler, **kw)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kw)


@torch.no_grad()
def evaluate(model, dataset, args, tokenizer, device, centers, rank, world, baseline):
    model.eval()
    aggregate = torch.zeros((len(dataset.ids), 12), dtype=torch.float64, device=device)
    indices = list(range(rank, len(dataset), world))
    predictions = []
    for batch in loader(dataset, args, indices=indices):
        inputs, y = prepare(batch, tokenizer, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(**inputs)
        logits = logits.float()
        target = project_values_to_bins(y, centers)
        ce = -(target * logits.log_softmax(-1)).sum(-1)
        probs=logits.softmax(-1)
        cdf=probs.cumsum(-1)
        pred = (probs * centers).sum(-1)
        err = pred - y
        crps=((cdf-(centers[None,:]>=y[:,None]).float()).square()).sum(-1)*(centers[1]-centers[0])
        entropy=-(probs*logits.log_softmax(-1)).sum(-1)
        lo=centers[(cdf<0.05).sum(-1).clamp_max(200)]
        hi=centers[(cdf<0.95).sum(-1).clamp_max(200)]
        covered=((y>=lo)&(y<=hi)).float()
        assert torch.isfinite(ce).all() and torch.isfinite(err).all()
        vals = torch.stack([torch.ones_like(y), ce, err.abs(), err.square(), (baseline-y).abs(), (baseline-y).square(), target.sum(-1),err,crps,entropy,covered,hi-lo], dim=-1)
        aggregate.index_add_(0, batch['episode'].to(device), vals.double())
        predictions.extend(zip(batch['episode'].tolist(), batch['frame'].tolist(), y.cpu().tolist(), pred.cpu().tolist()))
    if world > 1:
        dist.all_reduce(aggregate)
    a = aggregate.cpu().numpy()
    assert np.all(a[:, 0] > 0)
    n = float(a[:, 0].sum())
    result = dict(frames=int(n), episodes=len(dataset.ids), ce=float(a[:,1].sum()/n), mae=float(a[:,2].sum()/n), rmse=float(np.sqrt(a[:,3].sum()/n)), episode_mae=float(np.mean(a[:,2]/a[:,0])), baseline_mae=float(a[:,4].sum()/n), baseline_rmse=float(np.sqrt(a[:,5].sum()/n)))
    if world > 1:
        all_predictions = [None] * world
        dist.all_gather_object(all_predictions, predictions)
        predictions = sum(all_predictions, [])
    result['mae_remaining_seconds'] = result['mae'] * dataset.scale / dataset.info['fps']
    result.update(bias=float(a[:,7].sum()/n),crps=float(a[:,8].sum()/n),entropy=float(a[:,9].sum()/n),interval90_coverage=float(a[:,10].sum()/n),interval90_width=float(a[:,11].sum()/n))
    return result, sorted(predictions)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--variant',choices=['full','vision_only','frozen_language','frozen_both','task_only'],required=True)
    ap.add_argument('--seed',type=int,default=1000)
    ap.add_argument('--lr-schedule',choices=['constant','cosine'],default='cosine')
    ap.add_argument('--peak-lr',type=float,default=5e-5)
    ap.add_argument('--min-lr',type=float,default=1e-6)
    ap.add_argument('--state-intervention',choices=['none','zero','permute'],default='none')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--eval-frames', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--min-steps', type=int, default=2000)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--evaluate-checkpoint', default=None, help="'best' or an existing checkpoint filename; no training")
    args = ap.parse_args()
    assert args.variant == 'vision_only', 'This experiment is vision-only'
    assert args.seed == 1000, 'Initial fixture is seed1000'
    assert args.state_intervention=='none' or (args.evaluate_checkpoint and args.variant in ('full','frozen_language','frozen_both'))
    assert 0<=args.warmup<=args.steps
    rank, world = int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1))
    local_rank = int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda',local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=40))
    torch.set_num_threads(2)
    random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed)
    manifest_path = BASE / 'split90_10/manifests' / (args.task+'.json')
    manifest = json.loads(manifest_path.read_text())
    train = Frames(manifest,'train',variant=args.variant)
    test = Frames(manifest,'test',scale=train.scale,evaluation_frames=args.eval_frames,variant=args.variant,state_intervention=args.state_intervention)
    train_eval = Frames(manifest,'train',scale=train.scale,evaluation_frames=args.eval_frames,episode_limit=len(test.ids),variant=args.variant)
    baseline = float(-sum(n*(n-1)/2 for n in train.lengths)/sum(train.lengths)/train.scale)
    out = EXPERIMENT / 'runs' / args.run
    if rank == 0:
        out.mkdir(parents=True,exist_ok=True)
    if world > 1: dist.barrier()
    config = Pistar06Config(vision_repo_id=str(BASE/'models/siglip'),language_repo_id=str(BASE/'models/gemma'),camera_features=manifest['camera_features'],dtype='float32',use_gradient_checkpointing=True,device='cuda',freeze_language_model=args.variant in ('frozen_language','frozen_both'),freeze_vision_encoder=args.variant=='frozen_both')
    protocol = dict(arguments=vars(args),world_size=world,global_batch=args.batch_size*world,manifest_sha256=digest(manifest_path),stats_sha256=digest(BASE/'split90_10/train-only-stats'/(args.task+'.json')),value_source_commit='6f2db449a21e1bac750b996f2e27cac6739aa63f',amp='bfloat16',parameter_dtype='float32',return_scale=train.scale,return_formula='-(episode_length-frame_index-1)/(2*max_train_episode_length)',failure_data=False,selection='minimum holdout episode MAE',early_stop='after min_steps, patience evaluations without 0.5% relative improvement',frame_sampler='uniform with replacement, per-step stateless seed1000; effective_epoch counts draws not exact coverage',evaluation='fixed linspace frames per whole episode, no augmentation; holdout reused for checkpoint selection',train_constant_baseline=baseline,model_config=config.to_dict() if hasattr(config,'to_dict') else str(config))
    protocol_path = out/'protocol.json'
    protocol.update(variant=args.variant,experiment='lean-vision-schedules-v1',training_stop='EXACT steps, no early stopping; final checkpoint is primary comparison',early_stop='disabled',selection='final-step endpoint is primary; intermediate minima are secondary',runner_sha256=digest(__file__),pure_vision_removes_state=args.variant=='vision_only',language_receives_state=args.variant in ('full','frozen_language','frozen_both'),lr_schedule=f'warmup{args.warmup} then {args.lr_schedule}; cosine floor at fixed final step',initial_fixture_sha256=json.loads((EXPERIMENT/'assets/export_manifest.json').read_text())['initial_sha256'],lean_model_sha256=digest(Path(__file__).with_name('lean_model.py')))
    # Decide existence once on rank0 BEFORE any rank creates protocol.json.
    # Otherwise late ranks can mistake this launch's new file for an old run.
    protocol_exists = [protocol_path.exists() if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(protocol_exists, src=0)
    if protocol_exists[0]:
        assert args.resume or args.evaluate_checkpoint, 'Refusing to reuse existing experiment without --resume'
        old = json.loads(protocol_path.read_text())
        for k in ('manifest_sha256','stats_sha256','return_scale','amp'):
            assert old[k] == protocol[k], ('Incompatible resume',k)
        if not args.evaluate_checkpoint:
            assert old['global_batch'] == protocol['global_batch']
            for k in ('steps','warmup','eval_every','eval_frames','variant','seed','peak_lr','min_lr','lr_schedule'):
                assert old['arguments'][k] == vars(args)[k], ('Schedule changed',k)
    elif rank == 0:
        assert not args.evaluate_checkpoint, 'Evaluation requires an existing experiment'
        atomic_json(protocol_path,protocol)
    if world > 1:
        dist.barrier()
    model = (VisionOnlyModel(config) if args.variant=='vision_only' else ValueModel(config)).to(device)
    tokenizer = None if args.variant=='vision_only' else AutoTokenizer.from_pretrained(str(BASE/'models/gemma'),padding_side='right',local_files_only=True)
    if args.variant=='vision_only': assert not hasattr(model,'language_model')
    if config.freeze_language_model: assert not any(p.requires_grad for p in model.language_model.parameters())
    if config.freeze_vision_encoder: assert not any(p.requires_grad for p in model.vision_encoder.parameters())
    if rank==0: print(json.dumps(dict(event='ARCHITECTURE',variant=args.variant,total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),state_input=protocol['language_receives_state'])),flush=True)
    centers = build_bin_centers(201,-1,0,device)
    if args.evaluate_checkpoint:
        filename = json.loads((out/'best.json').read_text())['file'] if args.evaluate_checkpoint=='best' else args.evaluate_checkpoint
        assert Path(filename).name == filename, 'Checkpoint must be inside this run'
        saved = torch.load(out/filename,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        checkpoint_step = saved['step']
        del saved
        metrics,predictions = evaluate(model,test,args,tokenizer,device,centers,rank,world,baseline)
        if rank==0:
            result=dict(event='DENSE_REVIEW',step=checkpoint_step,checkpoint=filename,frames_per_episode=args.eval_frames,test=metrics,checkpoint_reload='strict_ok')
            suffix='' if args.state_intervention=='none' else '-state-'+args.state_intervention
            result['state_intervention']=args.state_intervention
            atomic_json(out/f'dense-review-{checkpoint_step:06d}-frames{args.eval_frames}{suffix}.json',result)
            atomic_json(out/f'dense-predictions-{checkpoint_step:06d}-frames{args.eval_frames}{suffix}.json',dict(columns=['episode_slot','frame','target','prediction'],episode_ids=test.ids,rows=predictions))
            print(json.dumps(result),flush=True)
        if world>1: dist.destroy_process_group()
        return
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.peak_lr,weight_decay=1e-5)
    start,best,best_step,meaningful,bad = 0,float('inf'),0,float('inf'),0
    if args.resume:
        last = json.loads((out/'last.json').read_text())
        state = torch.load(out/last['file'],map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
        start,best,best_step,meaningful,bad = [state[k] for k in ('step','best','best_step','meaningful','bad')]
        restore_rng(state['rng'][rank]); del state
    if start >= args.steps:
        if rank==0: print('ALREADY_AT_REQUESTED_STEP',flush=True)
        if world>1: dist.destroy_process_group()
        return
    wrapped = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=False) if world>1 else model
    sampler = StepBatches(len(train),args.batch_size,rank,world,start,args.steps,args.seed)
    train_loader = loader(train,args,sampler=sampler)
    if rank == 0:
        print(json.dumps(dict(event='READY',task=args.task,step=start,train_frames=len(train),test_frames=len(test),global_batch=args.batch_size*world,parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    interval_sum = torch.zeros(3,device=device)
    tick = time.monotonic()
    for step,batch in enumerate(train_loader,start+1):
        model.train()
        lr = args.peak_lr*step/max(1,args.warmup) if step<=args.warmup else (args.peak_lr if args.lr_schedule=='constant' else args.min_lr+0.5*(args.peak_lr-args.min_lr)*(1+math.cos(math.pi*(step-args.warmup)/max(1,args.steps-args.warmup))))
        for group in optimizer.param_groups: group['lr']=lr
        inputs,y = prepare(batch,tokenizer,device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits = wrapped(**inputs)
        logits = logits.float()
        loss = -(project_values_to_bins(y,centers)*logits.log_softmax(-1)).sum(-1).mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward()
        if args.smoke:
            assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.value_head.parameters()), 'Head has no gradient'
            for name, frozen in [('vision_encoder',config.freeze_vision_encoder),('language_model',config.freeze_language_model)]:
                module=getattr(model,name,None)
                if module is None: continue
                if frozen:
                    assert all(p.grad is None for p in module.parameters()),name
                    assert not module.training,name
                else:
                    assert any(p.grad is not None and p.grad.abs().max()>0 for p in module.parameters()),name
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),10.0,error_if_nonfinite=True)
        optimizer.step()
        interval_sum += torch.stack([loss.detach(),grad.detach(),torch.ones((),device=device)])
        if step%10==0 or args.smoke:
            report=interval_sum.clone()
            if world>1: dist.all_reduce(report)
            if rank==0:
                print(json.dumps(dict(event='TRAIN',step=step,effective_epoch=step*args.batch_size*world/len(train),loss=float(report[0]/report[2]),grdn=float(report[1]/report[2]),lr=lr,seconds=time.monotonic()-tick,memory_gb=torch.cuda.max_memory_allocated()/1e9)),flush=True)
            interval_sum.zero_(); tick=time.monotonic()
        if step%args.eval_every != 0 and step!=args.steps: continue
        test_metrics,predictions = evaluate(model,test,args,tokenizer,device,centers,rank,world,baseline)
        train_metrics,train_predictions = evaluate(model,train_eval,args,tokenizer,device,centers,rank,world,baseline)
        score = test_metrics['episode_mae']
        if score<best: best,best_step=score,step
        if score < meaningful*0.995: meaningful,bad=score,0
        else: bad+=1
        states=[None]*world
        if world>1: dist.all_gather_object(states,rng_state())
        else: states[0]=rng_state()
        if rank==0:
            record=dict(event='EVAL',step=step,effective_epoch=step*args.batch_size*world/len(train),train=train_metrics,test=test_metrics,best_step=best_step,best_episode_mae=best,bad_evaluations=bad)
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            name=f'checkpoint-{step:06d}.pt'
            tmp=out/(name+'.part')
            assert not (out/name).exists(), 'Refusing to overwrite checkpoint'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,best_step=best_step,meaningful=meaningful,bad=bad,rng=states,protocol=protocol),tmp)
            os.replace(tmp,out/name)
            atomic_json(out/'last.json',dict(file=name,step=step))
            if best_step==step: atomic_json(out/'best.json',dict(file=name,step=step,episode_mae=best))
            atomic_json(out/f'predictions-{step:06d}.json',dict(columns=['episode_slot','frame','target','prediction'],episode_ids=test.ids,rows=predictions))
            atomic_json(out/f'train-predictions-{step:06d}.json',dict(columns=['episode_slot','frame','target','prediction'],episode_ids=train_eval.ids,rows=train_predictions))
        if world>1: dist.barrier()
        if args.smoke:
            # Verify actual saved tensors can restore and reproduce inference.
            inputs,y=prepare(batch,tokenizer,device)
            model.eval()
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16): expected=model(**inputs).float()
            saved=torch.load(out/f'checkpoint-{step:06d}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True); del saved
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): actual=model(**inputs).float()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            if rank==0: print('SMOKE_SAVE_RELOAD_OK',flush=True)
    if rank==0:
        assert step==args.steps
        atomic_json(out/'final.json',dict(file=f'checkpoint-{step:06d}.pt',step=step,variant=args.variant,endpoint_primary=True))
        atomic_json(out/'training_complete.json',dict(step=step,best_step=best_step,best_episode_mae=best,fixed_budget=True,early_stopping=False,dense_final_review_required=True))
        print('TRAINING_STAGE_COMPLETE',flush=True)
    if world>1: dist.destroy_process_group()


if __name__=='__main__':
    main()

