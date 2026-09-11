"""Train-only numeric normalization and episode identity audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

base = Path('/data/experiments/value-demo3-20260906/split90_10')
summary = []
for manifest in sorted((base / 'manifests').glob('*.json')):
    if manifest.name.endswith('.sampling-original.json'):
        continue
    cfg = json.loads(manifest.read_text())
    root = Path(cfg['root'])
    info = json.loads((root / 'meta/info.json').read_text())
    selected = set(cfg['selected_episode_indices'])
    train = set(cfg['splits']['train']['episode_indices'])
    split_of = {ep:s for s,group in cfg['splits'].items() for ep in group['episode_indices']}
    meta = {}
    for path in sorted((root / 'meta/episodes').rglob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            if row['episode_index'] in selected:
                meta[row['episode_index']] = row
    files = {}
    for ep,row in meta.items():
        path = root / info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
        files.setdefault(path, set()).add(ep)
    stats_data = {'observation.state': [], 'action': []}
    identity = {}
    seen = set()
    for path,episodes in sorted(files.items()):
        table = pq.read_table(path, columns=['episode_index','frame_index','timestamp','observation.state','action'])
        epids = table['episode_index'].to_numpy()
        frame = table['frame_index'].to_numpy()
        stamp = table['timestamp'].to_numpy()
        values = {k:np.stack(table[k].to_pylist()).astype(np.float32) for k in stats_data}
        for ep in sorted(episodes):
            mask = epids == ep
            n = int(mask.sum())
            assert ep not in seen and n == meta[ep]['length'], (ep,n,meta[ep]['length'])
            seen.add(ep)
            assert np.array_equal(frame[mask], np.arange(n)), (ep,'frame index')
            assert np.all(np.diff(stamp[mask]) > 0), (ep,'timestamp')
            digest = hashlib.sha256()
            for k,value in values.items():
                arr = np.ascontiguousarray(value[mask])
                assert np.isfinite(arr).all(), (ep,k,'nonfinite')
                digest.update(k.encode()); digest.update(arr.tobytes())
                if ep in train:
                    stats_data[k].append(arr)
            identity.setdefault(digest.hexdigest(), []).append(ep)
    assert seen == selected
    duplicates = [eps for eps in identity.values() if len(eps)>1]
    cross_split = [eps for eps in duplicates if len({split_of[e] for e in eps})>1]
    assert not cross_split, ('Duplicate trajectories leak across splits',cross_split)
    stats = {}
    for key,chunks in stats_data.items():
        data = np.concatenate(chunks)
        assert len(data) == cfg['splits']['train']['frames']
        quantiles = np.quantile(data, [.01,.1,.5,.9,.99], axis=0)
        stats[key] = {name:q.tolist() for name,q in zip(['q01','q10','q50','q90','q99'],quantiles)}
        stats[key].update(min=data.min(0).tolist(),max=data.max(0).tolist(),mean=data.mean(0,dtype=np.float64).tolist(),std=data.std(0,dtype=np.float64).tolist(),count=[len(data)])
    out = base / 'train-only-stats'
    out.mkdir(exist_ok=True)
    target = out / (cfg['task']+'.json')
    content = json.dumps(stats,indent=2)+'\n'
    if target.exists():
        assert target.read_text()==content
    else:
        target.write_text(content)
    result = {'task':cfg['task'],'episodes_checked':len(seen),'train_frames':cfg['splits']['train']['frames'],'nonfinite':0,'duplicate_episode_groups':duplicates,'cross_split_duplicates':0,'stats_file':str(target),'status':'ok'}
    summary.append(result)
    print(json.dumps(result),flush=True)
(base/'numeric_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
