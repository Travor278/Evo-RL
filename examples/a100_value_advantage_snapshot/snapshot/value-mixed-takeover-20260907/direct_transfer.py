"""Scoped rsync sender and independent destination SHA verifier. No dataset deletion."""
import argparse, hashlib, json, os, pathlib, shlex, subprocess, time, fcntl

SOURCE = pathlib.Path('/home/zhaobo/.cache/huggingface/lerobot/evorl-piperx-copper-screw-full-episodes-with-intervention-20260903')
DEST = pathlib.Path('/data/datasets/piperx-full-intervention-141-20260903')
ROOT = pathlib.Path('/data/experiments/value-mixed-takeover-20260907')
WORK = pathlib.Path('/home/zhaobo/value-direct-transfer-20260907')
MARKER = 'value-direct-transfer-20260907'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def atomic(p, obj):
    tmp = p.with_name(p.name + '.tmp-' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def prepare():
    WORK.mkdir(exist_ok=True, mode=0o700)
    rows = []
    for p in sorted(SOURCE.rglob('*')):
        assert not p.is_symlink()
        if not p.is_file() or '.cache' in p.relative_to(SOURCE).parts: continue
        rows.append(dict(path=p.relative_to(SOURCE).as_posix(), bytes=p.stat().st_size, sha256=sha(p)))
    assert sum(r['path'].endswith('.mp4') for r in rows) == 249
    manifest = dict(records=rows, files=len(rows), bytes=sum(r['bytes'] for r in rows))
    atomic(WORK/'source_manifest.json', manifest)
    (WORK/'files.list0').write_bytes(b''.join(r['path'].encode()+b'\0' for r in rows))
    print(json.dumps(dict(event='SOURCE_HASHED', files=len(rows), bytes=manifest['bytes'])), flush=True)

def send():
    lock = (WORK/'send.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UserKnownHostsFile='+str(WORK/'known_hosts'), '-o', 'IdentitiesOnly=yes',
           '-o', 'Compression=no', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
           '-i', str(WORK/'key')]
    base = ['rsync', '-rt', '--ignore-existing', '--partial', '--partial-dir=.rsync-partial',
            '--timeout=120', '--info=progress2', '--stats', '-e', shlex.join(ssh)]
    print('DIRECT_RSYNC_START '+str(time.time()), flush=True)
    subprocess.run(base + ['--from0', '--files-from='+str(WORK/'files.list0'), str(SOURCE)+'/', 'root@36.212.230.67:./'], check=True)
    receipt = WORK/'rsync_complete.json'
    atomic(receipt, dict(status='rsync_complete', manifest_sha256=sha(WORK/'source_manifest.json'), at=time.time()))
    subprocess.run(base + [str(receipt), 'root@36.212.230.67:.transfer-control/'], check=True)
    for name in ('key', 'key.pub'):
        p=WORK/name
        assert p.is_file() and not p.is_symlink()
        p.unlink()
    print('DIRECT_RSYNC_COMPLETE SOURCE_TEMP_KEY_REMOVED', flush=True)

def verify():
    lock = (ROOT/'direct_verify.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = DEST/'.transfer-control/rsync_complete.json'
    print('WAITING_DIRECT_RSYNC_RECEIPT', flush=True)
    while not receipt.exists(): time.sleep(10)
    assert json.loads(receipt.read_text())['manifest_sha256'] == sha(ROOT/'direct_source_manifest.json')
    m = json.loads((ROOT/'direct_source_manifest.json').read_text())
    assert m['files'] == len(m['records']) and sum(r['path'].endswith('.mp4') for r in m['records']) == 249
    assert len({r['path'] for r in m['records']}) == len(m['records'])
    for i,r in enumerate(m['records'],1):
        rel = pathlib.Path(r['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        p = DEST/rel
        assert p.resolve().is_relative_to(DEST) and not p.is_symlink()
        assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256'], str(rel)
        print(json.dumps(dict(event='SHA_OK', files=i, path=str(rel))), flush=True)
    # Revoke only this task's authorized key; retain all dataset files and partials.
    auth = pathlib.Path('/root/.ssh/authorized_keys')
    with auth.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = f.readlines()
        matches = [l for l in lines if l.strip().endswith(' '+MARKER)]
        assert len(matches) == 1, 'Unexpected temporary key count'
        f.seek(0); f.writelines(l for l in lines if l not in matches); f.truncate(); f.flush(); os.fsync(f.fileno())
    assert MARKER not in auth.read_text()
    report = dict(status='ok', files=m['files'], bytes=m['bytes'], records=m['records'], root=str(DEST),
                  transport='4090A-direct-rsync', temporary_authorized_key_removed=True, at=time.time())
    atomic(ROOT/'transfer_verified.json', report)
    print('TRANSFER_COMPLETE SHA_ALL_OK TEMP_AUTH_REMOVED', flush=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['prepare','send','verify']); args=ap.parse_args()
    {'prepare':prepare,'send':send,'verify':verify}[args.mode]()
