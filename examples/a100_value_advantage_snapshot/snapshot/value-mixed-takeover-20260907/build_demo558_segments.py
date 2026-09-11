"""Materialize top10 A50 / union20 pure-DEMO clips into a standalone LeRobot v3 dataset."""
import os,json,math,argparse,hashlib,shutil
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import av
from mixed_data import ROOT,put,sha
from infer_demo_global30 import frames

BUILD=ROOT/'demo558-top10-a50-w20-build'
SOURCE=Path('/data/datasets/RW-RL-Dataset-piperx')
DEST=Path('/data/datasets/piperx-demo558-value1500-a50-top10-union20-v1')
H=20; FPS=30

def atom_parquet(table,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() and not path.with_suffix('.part.parquet').exists()
    part=path.with_suffix('.part.parquet');pq.write_table(table,part,compression='snappy');os.replace(part,path)

def plan(smoke=False):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    ids=[11,23,40] if smoke else list(range(558))
    if not smoke:assert json.loads((BUILD/'inference_complete.json').read_text())['status']=='ok'
    records=[json.loads((BUILD/'predictions'/f'episode-{e:03d}.json').read_text()) for e in ids]
    meta={int(r['episode_index']):r for f in sorted((SOURCE/'meta/episodes').rglob('*.parquet')) for r in pq.read_table(f).to_pylist()}
    a=[]
    for r in records:
        assert r['step']==1500 and r['full_frame'] and r['root']==str(SOURCE) and r['scale']==15337
        assert r['manifest_sha256']==sha(ROOT/'manifest.json')
        v=np.array(r['value_common_scale']);t=np.arange(len(v));end=np.minimum(t+50,len(v)-1);boot=v.copy();boot[-1]=0
        expected=-(end-t)/15337+boot[end]-v;expected[-1]=0
        assert np.allclose(expected,r['advantage_full'],rtol=0,atol=1e-12)
        assert len(v)==meta[r['episode']]['length']
        a.append(expected[:-1])
    scores=np.concatenate(a);requested=math.ceil(.10*len(scores));selected=np.argsort(-scores,kind='stable')[:requested];selected=selected[scores[selected]>0]
    mask=np.zeros(len(scores),dtype=bool);mask[selected]=True;threshold=float(scores[selected].min())
    dst=DEST.with_name(DEST.name+'-smoke') if smoke else DEST
    audit=BUILD/('smoke' if smoke else 'release');audit.mkdir(exist_ok=True)
    clips=[];sources=[];off=0;global_index=0
    for r,scores_ep in zip(records,a):
        ep=r['episode'];n=len(scores_ep)+1
        anchors=np.flatnonzero(mask[off:off+len(scores_ep)]);off+=len(scores_ep)
        if not len(anchors):continue
        cover=np.zeros(n,dtype=bool)
        for t in anchors:cover[t:min(t+H,n)]=True
        edges=np.diff(np.r_[False,cover,False].astype(int));starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
        localoffset=0;these=[]
        for lo,hi in zip(starts,stops):
            aa=anchors[(anchors>=lo)&(anchors<hi)];assert len(aa) and aa[0]==lo
            assert hi==min(aa[-1]+H,n)
            c=dict(episode_index=len(clips),source_episode=ep,source_from=int(lo),source_to=int(hi),length=int(hi-lo),
                   dataset_from_index=global_index,dataset_to_index=global_index+int(hi-lo),video_from_index=localoffset,
                   anchor_frames=aa.tolist(),anchor_advantages=[float(r['advantage_full'][i]) for i in aa],
                   source_value_model_split='test' if ep in manifest['demo']['splits']['test']['episode_indices'] else 'train')
            assert c['length']>=2;clips.append(c);these.append(c);global_index+=c['length'];localoffset+=c['length']
        sources.append(dict(source_episode=ep,meta=meta[ep],prediction_path=str(BUILD/'predictions'/f'episode-{ep:03d}.json'),
                            clips=these,selected_source_frames=np.flatnonzero(cover).tolist(),anchors=anchors.tolist()))
    p=dict(status='planned',smoke=smoke,source=str(SOURCE),destination=str(dst),audit=str(audit),
           value_checkpoint=records[0]['checkpoint'],manifest_sha256=sha(ROOT/'manifest.json'),
           source_episodes_evaluated=len(ids),source_total_frames=sum(len(r['advantage_full']) for r in records),
           eligible_frames=len(scores),top_fraction=.10,requested_anchor_count=requested,selected_anchor_count=len(selected),
           threshold=threshold,advantage_horizon=50,window_length=20,scale=15337,positive_required=True,
           exported_source_episodes=len(sources),output_episodes=len(clips),output_frames=global_index,
           output_hours=global_index/108000,encoder=dict(codec='libx264',crf=18,preset='fast',pix_fmt='yuv420p',fps=30),
           clips=clips,sources=sources,
           note='Frame-weighted global top10 across all558 (smoke uses3); selected starts expanded20, overlaps/adjacency merged; discontinuities are separate episodes. Interior frames need not be top10. Segments are not labeled successful/terminal demonstrations.')
    path=audit/'plan.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:put(path,p)
    print('PLAN',len(ids),len(clips),global_index,threshold,flush=True);return path

def export_source(job):
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.utils.utils import flatten_dict
    plan_path,source=job;p=json.loads(Path(plan_path).read_text());dst=Path(p['destination']);audit=Path(p['audit'])
    ep=source['source_episode'];marker=audit/f'source-{ep:03d}-complete.json'
    if marker.exists():
        m=json.loads(marker.read_text());assert m['plan_sha256']==sha(plan_path)
        assert all((dst/f['path']).stat().st_size==f['bytes'] for f in m['files']);return m
    r=json.loads(Path(source['prediction_path']).read_text());info=json.loads((SOURCE/'meta/info.json').read_text());meta=source['meta']
    table=pq.read_table(SOURCE/info['data_path'].format(chunk_index=meta['data/chunk_index'],file_index=meta['data/file_index']))
    table=table.filter(pa.compute.equal(table['episode_index'],ep));assert len(table)==meta['length']
    assert table['frame_index'].to_pylist()==list(range(len(table)))
    original={key:np.array(table[key].to_pylist(),dtype='<f4') for key in ['action','observation.state']}
    fingerprint=hashlib.sha256(original['action'].tobytes()+original['observation.state'].tobytes()).hexdigest()
    assert fingerprint==r['action_state_sha256']
    ch=ep//1000;fi=ep%1000;frames_selected=np.array(source['selected_source_frames']);parts=[];ep_rows=[];annotations=[]
    for c in source['clips']:
        lo,hi=c['source_from'],c['source_to'];n=hi-lo;new=table.slice(lo,n).replace_schema_metadata(None)
        replacements=dict(timestamp=pa.array(np.arange(n)/FPS,type=pa.float32()),frame_index=pa.array(np.arange(n),type=pa.int64()),
                          episode_index=pa.array([c['episode_index']]*n,type=pa.int64()),index=pa.array(np.arange(c['dataset_from_index'],c['dataset_to_index']),type=pa.int64()))
        for name,arr in replacements.items():new=new.set_column(new.schema.get_field_index(name),name,arr)
        parts.append(new)
        stat=compute_episode_stats({key:original[key][lo:hi] for key in original},{key:info['features'][key] for key in original})
        row=dict(episode_index=c['episode_index'],length=n,tasks=meta['tasks'],
                 **{'data/chunk_index':ch,'data/file_index':fi,'dataset_from_index':c['dataset_from_index'],'dataset_to_index':c['dataset_to_index'],
                    'meta/episodes/chunk_index':0,'meta/episodes/file_index':0})
        for key in info['features']:
            if info['features'][key]['dtype']=='video':
                pre='videos/'+key;row.update({pre+'/chunk_index':ch,pre+'/file_index':fi,pre+'/from_timestamp':c['video_from_index']/FPS,pre+'/to_timestamp':(c['video_from_index']+n)/FPS})
        row.update({'stats/'+k:np.asarray(v).tolist() for k,v in flatten_dict(stat).items()});ep_rows.append(row)
        aset=set(c['anchor_frames'])
        annotations.extend(dict(index=c['dataset_from_index']+i-lo,episode_index=c['episode_index'],source_episode_index=ep,source_frame_index=i,
                                predicted_value=r['value_common_scale'][i],advantage50=r['advantage_full'][i],selected_anchor=i in aset) for i in range(lo,hi))
    packed=pa.concat_tables(parts);data_path=dst/info['data_path'].format(chunk_index=ch,file_index=fi)
    files=[]
    # A source worker owns its paths. Do not overwrite previously materialized files.
    if not data_path.exists():atom_parquet(packed,data_path)
    else:assert pq.read_table(data_path).equals(packed)
    for key,arr in original.items():assert np.array_equal(np.array(packed[key].to_pylist(),dtype='<f4'),arr[frames_selected])
    for cam,(source_path,queries) in zip([k for k in info['features'] if info['features'][k]['dtype']=='video'],r['camera_specs']):
        assert cam in source_path
        path=dst/info['video_path'].format(video_key=cam,chunk_index=ch,file_index=fi);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            part=path.with_suffix('.part.mp4');assert not part.exists(),'Preserve partial; inspect before recovery'
            con=av.open(str(part),'w');st=con.add_stream('libx264',rate=FPS);st.width=640;st.height=480;st.pix_fmt='yuv420p';st.time_base=Fraction(1,FPS)
            st.options={'crf':'18','preset':'fast','threads':'2'}
            for i,rgb in enumerate(frames(source_path,np.array(queries)[frames_selected])):
                frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,FPS)
                for packet in st.encode(frame):con.mux(packet)
            for packet in st.encode():con.mux(packet)
            con.close();os.replace(part,path)
        count=0;maxerr=0.;pts=[]
        with av.open(str(path)) as con:
            st=con.streams.video[0];assert st.codec_context.name=='h264' and st.width==640 and st.height==480 and float(st.average_rate)==30
            assert st.start_time==0
            for frame in con.decode(st):
                t=float(frame.pts*frame.time_base);maxerr=max(maxerr,abs(t-count/FPS));pts.append(frame.pts);count+=1
        assert count==len(frames_selected) and maxerr<1e-4 and len(pts)==len(set(pts))
        files.append(dict(path=str(path.relative_to(dst)),bytes=path.stat().st_size,frames=count,pts_max_error=maxerr))
    files.append(dict(path=str(data_path.relative_to(dst)),bytes=data_path.stat().st_size,rows=len(packed)))
    annotation_path=dst/'selection/frames'/f'source-{ep:03d}.parquet'
    if not annotation_path.exists():atom_parquet(pa.Table.from_pylist(annotations),annotation_path)
    result=dict(source_episode=ep,plan_sha256=sha(plan_path),files=files,episodes=ep_rows,rows=len(packed),action_state_exact=True)
    put(marker,result);print('SOURCE_EXPORTED',ep,len(source['clips']),len(packed),flush=True);return result

def validate_reader_shard(job):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import default_collate
    import torch
    torch.set_num_threads(2)
    dst,rows=job
    dataset=LeRobotDataset('local/'+Path(dst).name,root=dst,video_backend='pyav',tolerance_s=1e-4,
                           delta_timestamps={'action':[i/FPS for i in range(20)]})
    tests=0;batch=[]
    for row in rows:
        lo=row['dataset_from_index'];n=row['length']
        for i in sorted({lo,lo+n//2,lo+n-1}):
            sample=dataset[i]
            assert int(sample['episode_index'])==row['episode_index'] and sample['action'].shape==(20,14)
            for cam in dataset.meta.video_keys:assert tuple(sample[cam].shape)==(3,480,640) and torch.isfinite(sample[cam]).all()
            assert torch.isfinite(sample['action']).all()
            if i==lo+n-1:assert sample['action_is_pad'][1:].all()
            if len(batch)<4:batch.append(sample)
            tests+=1
    if batch:
        collated=default_collate(batch);assert collated['action'].shape==(len(batch),20,14)
    return tests

def finish_dataset(plan_path,results):
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.utils.utils import unflatten_dict
    p=json.loads(Path(plan_path).read_text());dst=Path(p['destination']);audit=Path(p['audit'])
    rows=sorted([r for result in results for r in result['episodes']],key=lambda r:r['episode_index'])
    assert [r['episode_index'] for r in rows]==list(range(p['output_episodes']))
    assert sum(r['length'] for r in rows)==p['output_frames']
    meta_path=dst/'meta/episodes/chunk-000/file-000.parquet'
    if not meta_path.exists():atom_parquet(pa.Table.from_pylist(rows),meta_path)
    info=json.loads((SOURCE/'meta/info.json').read_text());info.update(total_episodes=len(rows),total_frames=p['output_frames'],duration=p['output_frames']/FPS,
            splits={'train':f'0:{len(rows)}'},total_tasks=1,selection_method='mixed1500 A50 global top10 positive starts + union20')
    info['data_files_size_in_mb']=sum(f.stat().st_size for f in (dst/'data').rglob('*.parquet'))/1e6
    info['video_files_size_in_mb']=sum(f.stat().st_size for f in (dst/'videos').rglob('*.mp4'))/1e6
    put(dst/'meta/info.json',info)
    if not (dst/'meta/tasks.parquet').exists():shutil.copy2(SOURCE/'meta/tasks.parquet',dst/'meta/tasks.parquet')
    stats=[unflatten_dict({k[6:]:np.array(v) for k,v in r.items() if k.startswith('stats/')}) for r in rows]
    write_stats(aggregate_stats(stats),dst)
    put(dst/'selection/segments.json',p['clips'])
    put(dst/'selection/protocol.json',{k:v for k,v in p.items() if k not in ['clips','sources']})
    repo='local/'+dst.name
    dataset=LeRobotDataset(repo,root=dst,video_backend='pyav',tolerance_s=1e-4,delta_timestamps={'action':[i/FPS for i in range(20)]})
    recompute_stats(dataset,skip_image_video=True,relative_action=False,chunk_size=20)
    for key in ['action','observation.state']:
        for q in ['q01','q10','q50','q90','q99']:
            x=np.asarray(dataset.meta.stats[key][q]);assert x.shape==(14,) and np.isfinite(x).all()
    # Decode first/middle/last of EVERY clip, with bounded independent reader processes.
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        tests=sum(pool.map(validate_reader_shard,[(str(dst),rows[i::6]) for i in range(6)]))
    readme=f'''# PiperX advantage-selected teleoperation segments\n\nOnly pure human demonstrations. Value checkpoint step1500 (mixed demo+HIL); no HIL frames in this export.\n\nA50 ranked globally across {p['source_episodes_evaluated']} source episodes, top10% AND A>0. Every selected start expands to [t,t+20); overlaps and adjacency merge. Each disconnected component is a separate output episode. Interior frames need not themselves be top10%.\n\nOutput: {len(rows)} segments, {p['output_frames']} frames, {p['output_hours']:.6f} hours at30FPS.\n\nThree camera streams and action/state are physically sliced; files pack segments from one source episode, with metadata separating episode boundaries. Videos re-encoded H264 CRF18/fast 640x480; PTS and full decode validated. Action/state retained exactly. Stats recalculated using LeRobot v0.6.1, absolute actions, includes q01/q10/q50/q90/q99.\n\nTraceability: selection/segments.json, selection/frames/*.parquet, selection/protocol.json. Original Value train/test membership retained per segment. This all558 dataset is not an untouched Value test set. Clip endings are cut boundaries, NOT task-success labels. No automatic policy training or upload was performed.\n\nUsing every frame as a policy training start is not the same as restricting starts to selected anchors; use selection annotations if that is required. Policy action horizon remains a separate training setting; this dataset does not change it.\n'''
    (dst/'README.md').write_text(readme)
    report=dict(status='ok',dataset=str(dst),repo_id=repo,episodes=len(rows),frames=p['output_frames'],hours=p['output_hours'],
                source_episodes=p['exported_source_episodes'],source_evaluated=p['source_episodes_evaluated'],anchors=p['selected_anchor_count'],threshold=p['threshold'],
                actual_lerobot_read_checks=tests,all_video_full_decode=True,action_state_exact=True,quantiles_verified=True,
                total_bytes=sum(f.stat().st_size for f in dst.rglob('*') if f.is_file()))
    put(audit/'complete.json',report);put(dst/'selection/validation.json',report)
    print('DATASET_COMPLETE',json.dumps(report),flush=True)

def main():
    os.environ['HF_HUB_OFFLINE']='1'
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    path=plan(args.smoke);p=json.loads(path.read_text());dst=Path(p['destination']);dst.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(export_source,[(str(path),s) for s in p['sources']]))
    finish_dataset(path,results)

if __name__=='__main__':main()
