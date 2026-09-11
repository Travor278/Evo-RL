"""Verify archived bytes without importing or executing any experiment code."""
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        root = Path('\\\\?\\' + str(root))
    manifest = json.loads((root / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    files = 0
    sources = 0
    for row in manifest['files']:
        if row['disposition'] != 'included':
            continue
        path = root / row['repository_path']
        stored = path.read_bytes()
        assert hashlib.sha256(stored).hexdigest() == row['stored_sha256'], path
        raw = gzip.decompress(stored) if row.get('encoding') == 'gzip' else stored
        if row.get('encoding') == 'lfs-pointer-json':
            raw = json.loads(stored)['git_lfs_pointer'].encode('ascii')
        assert hashlib.sha256(raw).hexdigest() == row['sha256'], path
        assert len(raw) == row['size'], path
        if row['path'].endswith('.py'):
            ast.parse(raw, filename=row['path'])
            sources += 1
        files += 1
    print(json.dumps({'verified_files': files, 'parsed_python_files': sources}))


if __name__ == '__main__':
    main()
