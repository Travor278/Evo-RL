"""Freeze whole-episode holdouts before value-model training. Never edit sources."""
import hashlib
import json
import math
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path('/data/experiments/value-demo3-20260906')
ARCHIVE = BASE / 'rwrl_so101_piperx_duration_matched_seed1000_v1.tar.gz'
TASKS = {
    'piperx_insert_copper_screw': (Path('/data/datasets/RW-RL-Dataset-piperx'), 558, 1255729),
    'so101_fold_clothes_left_stack_right': (Path('/data/datasets/RW-RL-Dataset-SO101/so101_fold_clothes_left_stack_right'), 108, 1258509),
    'so101_put_stationery_table_into_bag': (Path('/data/datasets/RW-RL-Dataset-SO101/so101_put_stationery_table_into_bag'), 397, 1256576),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, obj):
    content = json.dumps(obj, indent=2, ensure_ascii=False) + '\n'
    if path.exists():
        assert path.read_text() == content, f'Refusing to change frozen manifest: {path}'
    else:
        path.write_text(content)

BASE.mkdir(parents=True, exist_ok=True)
reports = []
with tarfile.open(ARCHIVE) as archive:
    for task, (root, expected_n, expected_frames) in TASKS.items():
        info = json.loads((root / 'meta/info.json').read_text())
        rows = []
        for path in sorted((root / 'meta/episodes').rglob('*.parquet')):
            rows.extend(pq.read_table(path).to_pylist())
        by_id = {int(row['episode_index']): row for row in rows}
        assert len(rows) == len(by_id) == info['total_episodes']
        if task.startswith('so101'):
            member = 'piperx_duration_matched_seed1000_v1/manifests/' + task + '.seed1000.json'
            original = json.load(archive.extractfile(member))
            selected = sorted(original['selection']['episode_indices'])
        else:
            original = None
            selected = sorted(by_id)
        assert len(selected) == len(set(selected)) == expected_n
        assert sum(by_id[e]['length'] for e in selected) == expected_frames
        # Rank identities rather than frames: an episode is indivisible.
        def rank(e):
            key = ['evorl-value-demo-split-v1', 1000, task, e]
            return hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest()
        ranked = sorted(selected, key=rank)
        n_test = n_val = math.floor(expected_n * .1 + .5)
        groups = {'test': ranked[:n_test], 'validation': ranked[n_test:n_test+n_val], 'train': ranked[n_test+n_val:]}
        assert set(groups['train']).isdisjoint(groups['validation'])
        assert set(groups['train']).isdisjoint(groups['test'])
        assert set(groups['validation']).isdisjoint(groups['test'])
        assert set(sum(groups.values(), [])) == set(selected)
        cameras = [k for k,v in info['features'].items() if v['dtype'] == 'video']
        missing = []
        paths = set()
        for e in selected:
            row = by_id[e]
            paths.add(root / info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index']))
            for cam in cameras:
                paths.add(root / info['video_path'].format(video_key=cam, chunk_index=row[f'videos/{cam}/chunk_index'], file_index=row[f'videos/{cam}/file_index']))
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
        assert not missing, missing[:10]
        report = {
            'task': task, 'root': str(root), 'sampling_archive_sha256': sha(ARCHIVE),
            'info_sha256': sha(root / 'meta/info.json'), 'seed': 1000,
            'split_algorithm': 'sha256 identity rank, namespace evorl-value-demo-split-v1; rounded 10% test, 10% validation, remainder train',
            'selected_episode_indices': selected, 'episodes': expected_n, 'frames': expected_frames,
            'splits': {name: {'episode_indices': sorted(ids), 'episodes': len(ids), 'frames': sum(by_id[e]['length'] for e in ids)} for name, ids in groups.items()},
            'camera_features': cameras, 'state_dim': info['features']['observation.state']['shape'][0],
            'referenced_files': len(paths), 'missing_files': 0,
            'success_label_status': 'explicit terminal-success field not yet verified; do not silently use failure defaults',
            'normalization': 'must fit on train episodes only; existing full-data stats not approved for holdout experiments',
            'split_scope': 'episode disjoint; cross-episode/session duplicates require further audit',
        }
        out = BASE / 'manifests'
        out.mkdir(exist_ok=True)
        save(out / (task + '.json'), report)
        if original:
            save(out / (task + '.sampling-original.json'), original)
        summary = {k:v for k,v in report.items() if k not in ['selected_episode_indices', 'splits']}
        summary['splits'] = {k:{kk:vv for kk,vv in v.items() if kk != 'episode_indices'} for k,v in report['splits'].items()}
        reports.append(summary)
save(BASE / 'data_preflight_summary.json', reports)
print(json.dumps(reports, ensure_ascii=False, indent=2))
