"""Disjoint pinned-HF acquisition and full-manifest training gate; never delete parts."""
import argparse, concurrent.futures, fcntl, json, os, pathlib, time
from urllib.parse import quote
from relay_nonhf_v3 import sha, atomic, path, valid, manifest, ROOT, DEST

def log(event, **kw):
    print(json.dumps(dict(event=event,at=time.time(),**kw)),flush=True)

def config(work):
    full=json.loads((work/'direct_source_manifest.json').read_text())
    nonhf=manifest(work)
    routes=json.loads((work/'hf_routes.json').read_text())
    assert routes['repo']=='Travor278/evorl-piperx-copper-screw-hil-clean'
    assert routes['revision']=='18016184b09929643b6bd055b3f5833bfb2e7b85'
    assert routes['endpoint']=='https://hf-mirror.com'
    assert sum(r['bytes'] for r in routes['records'])==7469922042
    assert not ({r['path'] for r in nonhf['records']} & {r['output'] for r in routes['records']})
    return full,routes

def download(root,work):
    import requests
    lock=(work/'hf_acquire_v3.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    full,cfg=config(work);started=time.monotonic()
    def one(r):
        dest=path(root,r['output']);v=dict(bytes=r['bytes'],sha256=r['sha256'])
        if dest.exists():
            assert valid(dest,v),'Existing SHA mismatch: '+r['output']
            log('HF_SKIP',path=r['output'],bytes=r['bytes']);return
        dest.parent.mkdir(parents=True,exist_ok=True)
        part=dest.with_name(dest.name+'.hf-v3.part')
        assert not part.is_symlink()
        url=cfg['endpoint']+'/datasets/'+cfg['repo']+'/resolve/'+cfg['revision']+'/'+quote(r['source'],safe='/')
        for attempt in range(1,7):
            try:
                offset=part.stat().st_size if part.exists() else 0
                assert offset<=r['bytes'],'Oversized part'
                if offset==r['bytes']:
                    assert sha(part)==r['sha256'],'Part SHA mismatch'
                    assert not dest.exists()
                    os.replace(part,dest);log('HF_FILE_OK',path=r['output'],bytes=r['bytes']);return
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with requests.get(url,headers=headers,stream=True,timeout=(20,60)) as resp:
                    resp.raise_for_status()
                    if offset and resp.status_code==200:
                        # Preserve the unusable-range partial instead of truncating it.
                        os.rename(part,part.with_name(part.name+'.retained-'+str(time.time_ns())))
                        offset=0
                    elif resp.status_code==206:
                        assert resp.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    else: assert resp.status_code==200 and offset==0
                    count=offset;last=time.monotonic()
                    with part.open('ab' if part.exists() else 'xb') as f:
                        for b in resp.iter_content(1024*1024):
                            if not b: continue
                            assert count+len(b)<=r['bytes']
                            f.write(b);count+=len(b)
                            if time.monotonic()-last>=15:
                                log('HF_PROGRESS',path=r['output'],file_bytes=count,total=r['bytes'])
                                last=time.monotonic()
                        f.flush();os.fsync(f.fileno())
                assert valid(part,v),'Downloaded SHA/size mismatch'
                assert not dest.exists()
                os.replace(part,dest);log('HF_FILE_OK',path=r['output'],bytes=r['bytes']);return
            except Exception as e:
                # Do not print exception messages containing signed URLs or credentials.
                log('HF_RETRY',path=r['output'],attempt=attempt,error=type(e).__name__)
                if attempt==6: raise RuntimeError('HF failed: '+r['output']) from None
                time.sleep(min(30,attempt*5))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one,cfg['records']))
    for r in cfg['records']: assert valid(path(root,r['output']),r)
    atomic(work/'hf_verified.json',dict(status='ok',files=198,bytes=7469922042,
        revision=cfg['revision'],manifest_sha256=sha(work/'direct_source_manifest.json'),at=time.time()))
    log('HF_COMPLETE',files=198,elapsed_s=time.monotonic()-started)

def verify(root,work):
    lock=(work/'acquire_verify_v3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    full,cfg=config(work)
    while not all((work/n).exists() for n in ('hf_verified.json','relay_nonhf_verified.json')):
        log('WAIT_BOTH_ROUTES');time.sleep(30)
    hf=json.loads((work/'hf_verified.json').read_text())
    relay=json.loads((work/'relay_nonhf_verified.json').read_text())
    assert hf['status']==relay['status']=='ok'
    assert hf['files']==198 and relay['files']==62
    assert hf['revision']==cfg['revision'] and hf['manifest_sha256']==sha(work/'direct_source_manifest.json')
    assert relay['records']==manifest(work)['records']
    for i,r in enumerate(full['records'],1):
        assert valid(path(root,r['path']),r),'Unified SHA mismatch: '+r['path']
        if i%25==0:log('UNIFIED_SHA',files=i,total=260)
    # Original queue gate is published only after every full-dataset record passes.
    atomic(work/'transfer_verified.json',dict(status='ok',files=260,bytes=13944913650,
        records=full['records'],root=str(root),transport='HF-pinned-plus-Mac-nonHF-v3',
        revision=cfg['revision'],at=time.time()))
    log('TRANSFER_COMPLETE',files=260,bytes=13944913650)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['download','verify'])
    ap.add_argument('--work',default=ROOT);ap.add_argument('--root',default=DEST)
    a=ap.parse_args();globals()[a.mode](pathlib.Path(a.root),pathlib.Path(a.work))
