"""Stream a read-only dataset between two SSH hosts without local disk caching.

Receive into a NEW dedicated root. Every file is hash verified before atomic
publication. Existing identical files are retained; mismatches fail closed.
No credentials or SSH authorization files are changed.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['send','receive']); ap.add_argument('root'); ap.add_argument('--report'); a=ap.parse_args()
    root=Path(a.root).resolve()
    if a.mode=='send':
        stream=sys.stdout.buffer
        for p in sorted(root.rglob('*')):
            if p.is_symlink(): raise RuntimeError('Unexpected source symlink')
            if not p.is_file() or '.cache' in p.relative_to(root).parts: continue
            entry=dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
            stream.write((json.dumps(entry)+'\n').encode()); stream.flush()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(4*1024*1024),b''): stream.write(b)
        stream.write(b'{"complete":true}\n'); stream.flush(); return
    assert a.report
    root.mkdir(parents=True,exist_ok=True); records=[]; stream=sys.stdin.buffer
    while True:
        line=stream.readline()
        if not line: raise RuntimeError('Incomplete stream; retain verified files and parts')
        entry=json.loads(line)
        if entry.get('complete'): break
        rel=Path(entry['path']); assert not rel.is_absolute() and '..' not in rel.parts
        dest=root/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        assert dest.resolve().is_relative_to(root) and not dest.is_symlink()
        if dest.exists():
            assert dest.stat().st_size==entry['bytes'] and sha(dest)==entry['sha256'], 'Existing destination mismatch'
        temp=dest.with_name(dest.name+f'.transfer-{os.getpid()}-{time.time_ns()}.part')
        h=hashlib.sha256(); left=entry['bytes']
        output=None if dest.exists() else temp.open('xb')
        try:
            while left:
                b=stream.read(min(left,4*1024*1024))
                if not b: raise RuntimeError('Stream interrupted; partial retained')
                h.update(b); left-=len(b)
                if output: output.write(b)
            if output: output.flush(); os.fsync(output.fileno())
        finally:
            if output: output.close()
        assert h.hexdigest()==entry['sha256'], 'Transfer hash mismatch; do not publish'
        if output: os.replace(temp,dest)
        records.append(entry)
        print(json.dumps(dict(event='FILE_OK',files=len(records),bytes=sum(x['bytes'] for x in records),path=entry['path'])),flush=True)
    assert sum(x['path'].endswith('.mp4') for x in records)==249
    report=dict(status='ok',files=len(records),bytes=sum(x['bytes'] for x in records),records=records,root=str(root))
    p=Path(a.report); tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,p)
    print('TRANSFER_COMPLETE',flush=True)

if __name__=='__main__': main()
