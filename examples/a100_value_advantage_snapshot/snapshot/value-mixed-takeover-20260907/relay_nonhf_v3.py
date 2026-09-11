"""SHA-verified missing-file-only SSH relay. Never deletes dataset files or parts."""
import argparse, fcntl, hashlib, json, os, pathlib, shlex, subprocess, sys, time

ROOT='/data/experiments/value-mixed-takeover-20260907'
DEST='/data/datasets/piperx-full-intervention-141-20260903'
SOURCE='/home/zhaobo/.cache/huggingface/lerobot/evorl-piperx-copper-screw-full-episodes-with-intervention-20260903'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic(p,obj):
    tmp=p.with_name(p.name+'.tmp-'+str(os.getpid())+'-'+str(time.time_ns()))
    with tmp.open('x') as f:
        json.dump(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def path(root,rel):
    r=pathlib.Path(rel); assert not r.is_absolute() and '..' not in r.parts
    p=root/r; assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
    return p

def valid(p,r):
    return p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256']

def manifest(work):
    m=json.loads((work/'direct_source_manifest.json').read_text())
    rows=m['records'];assert len(rows)==m['files']==260
    assert len({r['path'] for r in rows})==260
    assert sum(r['path'].endswith('.mp4') for r in rows)==249
    assert sum(r['bytes'] for r in rows)==m['bytes']==13944913650
    routes=json.loads((work/'hf_routes.json').read_text())['records']
    bypath={r['path']:r for r in rows}
    assert len(routes)==198 and len({r['output'] for r in routes})==198
    for r in routes:
        assert bypath[r['output']]['sha256']==r['sha256'] and bypath[r['output']]['bytes']==r['bytes']
    exclude={r['output'] for r in routes}
    rows=[r for r in rows if r['path'] not in exclude]
    assert len(rows)==62 and sum(r['bytes'] for r in rows)==6474991608
    return dict(records=rows,files=len(rows),bytes=sum(r['bytes'] for r in rows))

def plan(root,work):
    m=manifest(work); keep=[]; missing=[]
    for r in m['records']:
        p=path(root,r['path'])
        if p.exists():
            assert valid(p,r),'Existing file mismatch: '+r['path']
            keep.append(r)
        else: missing.append(r)
    result=dict(records=m['records'],missing=missing,existing_files=len(keep),
                existing_bytes=sum(r['bytes'] for r in keep),total_bytes=m['bytes'],at=time.time())
    atomic(work/'relay_nonhf_plan.json',result)
    print(json.dumps(result),flush=True)

def send(root):
    p=json.load(sys.stdin);out=sys.stdout.buffer
    for r in p['missing']:
        src=path(root,r['path']);assert valid(src,r),'Source changed: '+r['path']
        out.write((json.dumps(r)+'\n').encode());out.flush()
        with src.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):out.write(b)
        out.flush()
    out.write(b'{"complete":true}\n');out.flush()

def receive(root,work):
    lock=(work/'relay_nonhf_receive.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((work/'relay_nonhf_plan.json').read_text());expected={r['path']:r for r in p['missing']}
    seen=set();s=sys.stdin.buffer;started=time.monotonic();last=started;newbytes=0
    print(json.dumps(dict(event='RELAY_START',existing_files=p['existing_files'],existing_bytes=p['existing_bytes'],
                          missing_files=len(expected),remaining_bytes=p['total_bytes']-p['existing_bytes'])),flush=True)
    while True:
        line=s.readline();assert line,'Interrupted stream; retain all parts'
        r=json.loads(line)
        if r.get('complete'):break
        assert r==expected.get(r['path']) and r['path'] not in seen
        dest=path(root,r['path']);assert not dest.exists(),'Concurrent destination writer'
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_name(dest.name+f'.relay-{os.getpid()}-{time.time_ns()}.part')
        left=r['bytes'];h=hashlib.sha256()
        with tmp.open('xb') as f:
            while left:
                b=s.read(min(left,256*1024));assert b,'Interrupted stream; retain part'
                f.write(b);h.update(b);left-=len(b);newbytes+=len(b)
                now=time.monotonic()
                if now-last>=10:
                    print(json.dumps(dict(event='RELAY_PROGRESS',new_bytes=newbytes,
                          total_including_partial=p['existing_bytes']+newbytes,
                          average_MB_s=round(newbytes/(now-started)/1e6,3),elapsed_s=round(now-started,1))),flush=True)
                    last=now
            f.flush();os.fsync(f.fileno())
        assert h.hexdigest()==r['sha256'],'Received SHA mismatch; retain part'
        assert not dest.exists();os.replace(tmp,dest);seen.add(r['path'])
        print(json.dumps(dict(event='FILE_OK',files=p['existing_files']+len(seen),path=r['path'],new_bytes=newbytes)),flush=True)
    assert seen==set(expected),'Missing stream files'
    print('FULL_SHA_RECHECK_START',flush=True)
    m=manifest(work)
    for r in m['records']:assert valid(path(root,r['path']),r),'Final SHA mismatch: '+r['path']
    report=dict(status='ok',files=m['files'],bytes=m['bytes'],records=m['records'],root=str(root),
                transport='Mac-relay-nonHF-v3',at=time.time())
    atomic(work/'relay_nonhf_verified.json',report)
    print('RELAY_NONHF_COMPLETE SHA_ALL_62_OK',flush=True)

def relay():
    lock=pathlib.Path(__file__).with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3']
    target='root@36.212.230.67';source='zhaobo@zhaobo-4090-a.tail160248.ts.net'
    remote=ROOT+'/relay_nonhf_v3.py'
    output=subprocess.check_output(ssh+[target,shlex.join(['python3',remote,'plan','--root',DEST,'--work',ROOT])])
    p=json.loads(output)
    print(json.dumps({k:v for k,v in p.items() if k not in ('records','missing')}),flush=True)
    sender=None;receiver=None
    try:
        sender=subprocess.Popen(ssh+[source,shlex.join(['python3','/home/zhaobo/value_relay_nonhf_v3_20260907.py','send','--root',SOURCE])],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        receiver=subprocess.Popen(ssh+[target,shlex.join(['python3','-u',remote,'receive','--root',DEST,'--work',ROOT])],stdin=sender.stdout)
        sender.stdout.close()
        sender.stdin.write(output);sender.stdin.close()
        source_exit=sender.wait();target_exit=receiver.wait()
        assert source_exit==target_exit==0,(source_exit,target_exit)
    finally:
        for proc in (sender,receiver):
            if proc is not None and proc.poll() is None:proc.terminate()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','send','receive','relay'])
    ap.add_argument('--root',default=DEST);ap.add_argument('--work',default=ROOT);a=ap.parse_args()
    if a.mode=='relay':relay()
    elif a.mode=='send':send(pathlib.Path(a.root))
    else:{'plan':plan,'receive':receive}[a.mode](pathlib.Path(a.root),pathlib.Path(a.work))
