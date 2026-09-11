"""Read-only full-episode value labels. No language/state inputs or dataset writes."""
import hashlib, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

ROOT=Path('/data/experiments/value-mixed-takeover-20260907')
DEMO_MANIFEST=Path('/data/experiments/value-demo3-20260906/split90_10/manifests/piperx_insert_copper_screw.json')
HIL_ROOT=Path('/data/datasets/piperx-full-intervention-141-20260903')

def decode_rgb_precise(path,timestamp,tolerance=1e-4):
    """Strict PyAV nearest-frame lookup in float64; never round or relax queries.

    Long concatenated video timestamps exceed float32's 0.1 ms resolution.
    Preserve the metadata query and compare against actual decoded packet PTS.
    No upstream library or dataset is modified.
    """
    import av
    import math
    assert math.isfinite(timestamp) and timestamp>=0 and tolerance==1e-4
    best=None;best_error=float('inf');best_pts=None
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        container.seek(round(timestamp/stream.time_base)-1,backward=True,any_frame=False,stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:continue
            pts=float(frame.pts*stream.time_base)
            error=abs(pts-timestamp)
            if error<best_error:best,best_error,best_pts=frame,error,pts
            if pts>=timestamp:break
        if best is None or not best_error<tolerance:
            raise RuntimeError(f'Strict float64 PTS mismatch: path={path}, query={timestamp!r}, nearest={best_pts!r}, error={best_error!r}, tolerance={tolerance}')
        return torch.from_numpy(best.to_ndarray(format='rgb24')).permute(2,0,1).contiguous()

def put(path,value):
    import os
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def rewards(mask,c=150):
    mask=np.asarray(mask,dtype=bool).reshape(-1); n=len(mask)
    assert n>=2
    r=np.full(n,-1.,dtype=np.float64);r[-1]=0
    events=np.zeros(n,dtype=bool)
    events[:-1]=(~mask[:-1])&mask[1:]
    # Manuscript b_T=0: no takeover penalty on a transition to the terminal.
    events[-2]=False
    r[events]-=c
    g=np.cumsum(r[::-1])[::-1].copy()
    assert g[-1]==0
    return r,g,events

def make_manifest():
    demo=json.loads(DEMO_MANIFEST.read_text())
    provenance=[json.loads(x) for x in (HIL_ROOT/'meta/source_episodes.jsonl').read_text().splitlines()]
    def identity(r):return str(r.get('source_episode_uuid') or (r['source_dataset']+':'+str(r['source_episode_index'])))
    assert len(provenance)==141 and len(set(map(identity,provenance)))==141
    ranked=sorted(provenance,key=lambda r:hashlib.sha256(('value-mixed-takeover-split-v1:1000:'+identity(r)).encode()).hexdigest())
    test={r['output_episode_index'] for r in ranked[:14]}
    assert sorted(test)==[11,17,18,22,35,42,51,53,82,91,114,118,121,136]
    for r in provenance:
        assert r['source_episode_success'] is True or (r['source_batch']=='4090a_20260903' and r['source_episode_success'] is None)
    hil=dict(root=str(HIL_ROOT),camera_features=['observation.images.right_environment_1','observation.images.left_wrist','observation.images.right_wrist'],splits={s:dict(episode_indices=sorted(r['output_episode_index'] for r in provenance if (r['output_episode_index'] in test)==(s=='test'))) for s in ['train','test']},provenance=provenance)
    return dict(demo=demo,hil=hil,seed=1000,penalty=150,scale=15337,normalization_multiplier=1,demo_manifest_sha256=sha(DEMO_MANIFEST),hil_provenance_sha256=sha(HIL_ROOT/'meta/source_episodes.jsonl'),success_provenance={'demo558':'user confirmed successful previously','hf_clean66':'source metadata success=true','4090a75':'user 2026-09-07: should all be successful; confirmation not original metadata'},split_note='Whole source episodes; held-out reused for checkpoint comparison, not an untouched final test')

class Pool(Dataset):
    def __init__(self,manifest,source,split,evaluation_frames=None,episode_limit=None,source_batch=None):
        self.source=source;self.manifest=manifest; self.scale=float(manifest['scale'])
        cfg=manifest[source];self.root=Path(cfg['root']);self.cameras=cfg['camera_features']
        self.info=json.loads((self.root/'meta/info.json').read_text());assert self.info['fps']==30
        ids=sorted(cfg['splits'][split]['episode_indices'])
        prov={r['output_episode_index']:r for r in cfg.get('provenance',[])}
        if source_batch:ids=[e for e in ids if prov[e]['source_batch']==source_batch]
        if episode_limit:
            ids=sorted(sorted(ids,key=lambda e:hashlib.sha256(f'mixed-train-eval-v1:{source}:{e}'.encode()).hexdigest())[:episode_limit])
        self.ids=ids;self.rows={}
        for p in sorted((self.root/'meta/episodes').rglob('*.parquet')):
            for row in pq.read_table(p).to_pylist():
                if int(row['episode_index']) in ids:self.rows[int(row['episode_index'])]=row
        assert set(self.rows)==set(ids) and ids
        self.lengths=np.array([self.rows[e]['length'] for e in ids],dtype=np.int64)
        self.ends=np.cumsum(self.lengths);self.starts=np.r_[0,self.ends[:-1]]
        self.timestamps={};self.returns={};self.reward={};self.events={};self.masks={};self.fingerprints={}
        files={}
        for e in ids:
            r=self.rows[e];p=self.root/self.info['data_path'].format(chunk_index=r['data/chunk_index'],file_index=r['data/file_index'])
            files.setdefault(p,[]).append(e)
        for p,eps in files.items():
            cols=['episode_index','frame_index','timestamp','action','observation.state']+(['intervention'] if source=='hil' else [])
            table=pq.read_table(p,columns=cols);epids=table['episode_index'].to_numpy();frames=table['frame_index'].to_numpy();ts=table['timestamp'].to_numpy()
            action=np.asarray(table['action'].to_pylist(),dtype='<f4');state=np.asarray(table['observation.state'].to_pylist(),dtype='<f4')
            mask=np.array(table['intervention'].to_pylist(),dtype=bool).reshape(-1) if source=='hil' else np.zeros(len(ts),dtype=bool)
            for e in eps:
                select=epids==e;n=int(self.rows[e]['length']);assert np.array_equal(frames[select],np.arange(n))
                assert np.isfinite(action[select]).all() and np.isfinite(state[select]).all()
                assert np.allclose(ts[select],np.arange(n)/30,atol=1e-4,rtol=0)
                self.timestamps[e]=ts[select];self.masks[e]=mask[select]
                r,g,b=rewards(mask[select],manifest['penalty'])
                assert -g[0]<=self.scale,('Target out of fixed support; no clipping',source,e,-g[0],self.scale)
                self.reward[e]=r;self.returns[e]=g;self.events[e]=b
                h=hashlib.sha256();h.update(action[select].tobytes());h.update(state[select].tobytes())
                self.fingerprints[e]=h.hexdigest()
        self.eval_indices=None
        if evaluation_frames:
            self.eval_indices=np.concatenate([s+np.unique(np.linspace(0,n-1,min(n,evaluation_frames),dtype=np.int64)) for s,n in zip(self.starts,self.lengths)])

    def __len__(self):return len(self.eval_indices) if self.eval_indices is not None else int(self.ends[-1])

    def locate(self,index):
        if self.eval_indices is not None:index=int(self.eval_indices[index])
        slot=int(np.searchsorted(self.ends,index,side='right'));return slot,self.ids[slot],int(index-self.starts[slot])

    def __getitem__(self,index):
        slot,e,frame=self.locate(index);row=self.rows[e];images=[]
        for camera in self.cameras:
            prefix=f'videos/{camera}';path=self.root/self.info['video_path'].format(video_key=camera,chunk_index=row[prefix+'/chunk_index'],file_index=row[prefix+'/file_index'])
            timestamp=float(row[prefix+'/from_timestamp'])+float(self.timestamps[e][frame])
            images.append(decode_rgb_precise(path,timestamp))
        return dict(images=torch.stack(images),target=self.returns[e][frame]/self.scale,episode=slot,frame=frame,prompt='')

    def baseline(self):return float(sum(g.sum() for g in self.returns.values())/sum(self.lengths)/self.scale)

class BalancedBatches(Sampler):
    def __init__(self,lengths,batch,rank,world,start,stop,seed=1000):
        assert len(lengths)==2 and batch%2==0
        self.lengths=lengths;self.batch=batch;self.rank=rank;self.world=world;self.start=start;self.stop=stop;self.seed=seed
    def __len__(self):return self.stop-self.start
    def __iter__(self):
        for step in range(self.start,self.stop):
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,step]))
            for rank in range(self.world):
                a=rng.integers(self.lengths[0],size=self.batch//2)
                b=self.lengths[0]+rng.integers(self.lengths[1],size=self.batch//2)
                indices=rng.permutation(np.r_[a,b]).tolist()
                if rank==self.rank:result=indices
            yield result
