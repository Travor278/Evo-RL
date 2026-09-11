"""Restore verified archival files to a NEW local directory; never run jobs."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', help='New directory; must not already exist')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    dest = Path(args.destination).resolve()
    if os.name == 'nt':
        root = Path('\\\\?\\' + str(root))
        dest = Path('\\\\?\\' + str(dest))
    manifest = json.loads((root / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    dest.mkdir(parents=True, exist_ok=False)
    restored = 0
    for row in manifest['files']:
        if row['disposition'] != 'included':
            continue
        relative = PurePosixPath(row['path']).relative_to('/data/experiments')
        assert '..' not in relative.parts
        stored = (root / row['repository_path']).read_bytes()
        assert hashlib.sha256(stored).hexdigest() == row['stored_sha256']
        raw = gzip.decompress(stored) if row.get('encoding') == 'gzip' else stored
        if row.get('encoding') == 'lfs-pointer-json':
            raw = json.loads(stored)['git_lfs_pointer'].encode('ascii')
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        target.chmod(row['mode'] & 0o777)
        restored += 1
    # Materialize file aliases from already verified, included targets.
    # Directory aliases (official LeRobot) are already copied as real directories.
    aliases = 0
    for row in manifest['files']:
        if not row.get('link'):
            continue
        import posixpath
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(row['path']), row['link']))
        if not resolved.startswith('/data/experiments/'):
            continue
        source = dest / PurePosixPath(resolved).relative_to('/data/experiments')
        target = dest / PurePosixPath(row['path']).relative_to('/data/experiments')
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
            aliases += 1
    print(json.dumps({'restored_files': restored, 'materialized_aliases': aliases, 'destination': str(dest)}))


if __name__ == '__main__':
    main()
