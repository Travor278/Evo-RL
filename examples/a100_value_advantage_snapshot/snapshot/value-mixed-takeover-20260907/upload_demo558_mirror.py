"""Upload only verified dataset to Elvinky over mirror, without any proxy."""
import os,json,hashlib,logging,re,fcntl,time
from pathlib import Path
for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(key,None)
os.environ.update(HF_HOME='/data/cache/huggingface',HF_ENDPOINT='https://hf-mirror.com',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
from huggingface_hub import HfApi,_commit_api

ROOT=Path('/data/experiments/value-mixed-takeover-20260907')
SRC=Path('/data/datasets/piperx-demo558-value1500-a50-top10-union20-v1')
WORK=ROOT/'hf-upload-demo558-mirror-v1';WORK.mkdir(exist_ok=True)
lock=(WORK/'upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
REPO='Elvinky/piperx-demo558-value1500-a50-top10-union20-v1'
class Redact(logging.Formatter):
    def format(self,record):
        s=super().format(record)
        s=re.sub(r'hf_[A-Za-z0-9]{15,}','[REDACTED]',s)
        return re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[REDACTED]',s)
handler=logging.StreamHandler();handler.setFormatter(Redact('%(levelname)s:%(name)s:%(message)s'))
logging.basicConfig(level=logging.WARNING,handlers=[handler],force=True)
def put(name,data):
    p=WORK/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,p)
original=_commit_api.post_lfs_batch_info
def rewrite(v):
    if isinstance(v,dict):return {k:rewrite(x) for k,x in v.items()}
    if isinstance(v,list):return [rewrite(x) for x in v]
    if isinstance(v,tuple):return tuple(rewrite(x) for x in v)
    if isinstance(v,str):return v.replace('https://hf-mirror.org/','https://hf-mirror.com/')
    return v
_commit_api.post_lfs_batch_info=lambda *a,**kw:rewrite(original(*a,**kw))
def run():
    validation=json.loads((SRC/'selection/validation.json').read_text());assert validation['status']=='ok'
    api=HfApi(endpoint='https://hf-mirror.com');assert api.whoami()['name']=='Elvinky'
    ri=api.repo_info(REPO,repo_type='dataset');assert not ri.private
    assert not (WORK/'complete.json').exists(),'Upload already validated'
    if not (WORK/'baseline.json').exists():put('baseline.json',dict(repo_id=REPO,revision=ri.sha,endpoint=api.endpoint,private=ri.private,created_at=time.time(),route='A100 directly to hf-mirror.com and returned object storage; no proxies'))
    stage=WORK/'upload-view';stage.mkdir(exist_ok=True);manifest=[]
    for p in sorted(SRC.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(SRC);assert not p.is_symlink();assert rel.parts[0] in ['README.md','meta','selection','data','videos'];assert '.cache' not in rel.parts and '.part' not in p.name
        dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(p,dest)
        else:assert os.path.samefile(p,dest)
        n=p.stat().st_size;h=hashlib.sha256();g=hashlib.sha1(f'blob {n}\0'.encode())
        with p.open('rb') as f:
            while chunk:=f.read(1024*1024):h.update(chunk);g.update(chunk)
        manifest.append(dict(path=str(rel),size=n,sha256=h.hexdigest(),git_blob=g.hexdigest()))
    assert len(manifest)==2728
    put('manifest.json',manifest);print('UPLOAD_CANDIDATES',len(manifest),sum(x['size'] for x in manifest),flush=True)
    known={x['path'] for x in manifest}
    existing=set(api.list_repo_files(REPO,repo_type='dataset'));assert existing<=known|{'.gitattributes'},'Unexpected remote content'
    api.upload_large_folder(repo_id=REPO,repo_type='dataset',folder_path=stage,revision='main',num_workers=4,allow_patterns=['README.md','meta/**','selection/**','data/**','videos/**'],ignore_patterns=['.cache/**'],print_report=True,print_report_every=30)
    print('UPLOAD_RETURNED_BEGIN_VERIFY',flush=True)
    ri=api.repo_info(REPO,repo_type='dataset');assert not ri.private
    remote={x.path:x for x in api.list_repo_tree(REPO,repo_type='dataset',revision=ri.sha,recursive=True) if hasattr(x,'size')}
    missing=[];size_bad=[];hash_bad=[]
    for m in manifest:
        x=remote.get(m['path'])
        if x is None:missing.append(m['path']);continue
        if x.size!=m['size']:size_bad.append(m['path'])
        lfs=getattr(x,'lfs',None)
        if lfs:
            actual=lfs.get('sha256') if isinstance(lfs,dict) else lfs.sha256
            if actual!=m['sha256']:hash_bad.append(m['path'])
        elif x.blob_id!=m['git_blob']:hash_bad.append(m['path'])
    extra=sorted(set(remote)-known-{'.gitattributes'})
    report=dict(status='ok' if not(missing or size_bad or hash_bad or extra) else 'failed',repo_id=REPO,private=ri.private,revision=ri.sha,files_expected=len(manifest),files_seen=sum(x['path'] in remote for x in manifest),total_bytes=sum(x['size'] for x in manifest),missing=missing,size_bad=size_bad,hash_bad=hash_bad,unexpected=extra,endpoint=api.endpoint,verification='Remote file tree at fixed revision; LFS SHA256 or git blob SHA1 matched per file',finished_at=time.time())
    put('verification.json',report);assert report['status']=='ok'
    put('complete.json',report);print('UPLOAD_VERIFIED',json.dumps(report),flush=True)
try:run()
except Exception as e:
    failure=dict(error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),time=time.time())
    put('failure.json',failure);print('UPLOAD_FAILED',json.dumps(failure),flush=True);raise SystemExit(2)
