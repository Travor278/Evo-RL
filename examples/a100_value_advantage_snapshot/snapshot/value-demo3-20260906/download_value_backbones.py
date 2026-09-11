"""Pinned value-backbone downloads; authentication stays in the HF cache."""
import os
from pathlib import Path

os.environ['HF_HOME'] = '/data/cache/huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download

root = Path('/data/experiments/value-demo3-20260906/models')
models = [
    ('google/siglip-so400m-patch14-384', '9fdffc58afc957d1a03a25b10dba0329ab15c2a3', 'siglip'),
    ('google/gemma-3-270m', '9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1', 'gemma'),
]
for repo, revision, name in models:
    print(f'DOWNLOAD_START repo={repo} revision={revision}', flush=True)
    try:
        snapshot_download(repo_id=repo, revision=revision, local_dir=root / name,
                          allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'],
                          max_workers=3)
    except Exception as exc:
        print(f'DOWNLOAD_FAILED repo={repo} type={type(exc).__name__}', flush=True)
        raise SystemExit(1) from None
    print(f'DOWNLOAD_COMPLETE repo={repo}', flush=True)
print('ALL_BACKBONES_COMPLETE', flush=True)
