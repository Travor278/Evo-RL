"""Scoped direct rsync with independent SHA verification and temporary-key cleanup."""
import fcntl, json, os, pathlib, shlex, subprocess, sys, time
import relay_nonhf_v3 as common

ROOT = pathlib.Path(common.ROOT)
DEST = pathlib.Path(common.DEST)
WORK = pathlib.Path('/home/zhaobo/value-direct-transfer-20260907')
MARKER = 'value-direct-nonhf-v4-20260907'
AUTH = pathlib.Path('/root/.ssh/authorized_keys')

def log(event, **values):
    print(json.dumps(dict(event=event, at=time.time(), **values)), flush=True)

def install():
    pub = (ROOT/'direct-v4.pub').read_text().strip()
    assert pub.startswith('ssh-ed25519 ') and pub.endswith(' '+MARKER)
    with AUTH.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        original = f.read()
        assert MARKER not in original
        line = 'restrict,command="/usr/bin/rrsync -wo -no-del '+str(DEST)+'" '+pub+'\n'
        f.seek(0, 2)
        if original and not original.endswith('\n'): f.write('\n')
        f.write(line); f.flush(); os.fsync(f.fileno())
    log('SCOPED_KEY_INSTALLED')

def plan():
    common.plan(DEST, ROOT)
    p = json.loads((ROOT/'relay_nonhf_plan.json').read_text())
    common.atomic(ROOT/'direct-v4-plan.json', p)
    log('DIRECT_PLAN', existing=p['existing_files'], missing=len(p['missing']),
        remaining_bytes=sum(r['bytes'] for r in p['missing']))

def send():
    lock=(WORK/'v4-send.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((WORK/'direct-v4-plan.json').read_text())
    missing=p['missing']
    for r in missing:
        assert common.valid(common.path(pathlib.Path(common.SOURCE),r['path']),r),r['path']
    listing=WORK/'v4-files.list0'
    listing.write_bytes(b''.join(r['path'].encode()+b'\0' for r in missing))
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
         '-o','UserKnownHostsFile='+str(WORK/'known_hosts'),'-o','IdentitiesOnly=yes',
         '-o','ProxyCommand=none','-o','ProxyJump=none','-o','Compression=no',
         '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4','-i',str(WORK/'key-v4')]
    base=['rsync','-rt','--ignore-existing','--partial','--partial-dir=.rsync-partial-v4',
          '--timeout=90','--info=progress2','--stats','-e',shlex.join(ssh)]
    log('DIRECT_START',files=len(missing),bytes=sum(r['bytes'] for r in missing))
    for attempt in range(1,9):
        rc=subprocess.call(base+['--from0','--files-from='+str(listing),common.SOURCE+'/',
                                 'root@36.212.230.67:./'])
        if rc==0: break
        log('DIRECT_RETRY',attempt=attempt,returncode=rc)
        if attempt==8: raise RuntimeError('Direct transfer retry limit; retained partials')
        time.sleep(15)
    receipt=WORK/'direct-v4-receipt.json'
    common.atomic(receipt,dict(status='rsync_complete',plan_sha256=common.sha(WORK/'direct-v4-plan.json')))
    subprocess.run(base+[str(receipt),'root@36.212.230.67:.transfer-control/'],check=True)
    for name in ('key-v4','key-v4.pub'):
        key=WORK/name
        assert key.is_file() and not key.is_symlink()
        key.unlink()
    log('DIRECT_COMPLETE_SOURCE_KEYS_REMOVED')

def verify():
    lock=(ROOT/'v4-verify.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=DEST/'.transfer-control/direct-v4-receipt.json'
    log('WAITING_DIRECT_RECEIPT')
    while not receipt.exists(): time.sleep(10)
    assert json.loads(receipt.read_text())['plan_sha256']==common.sha(ROOT/'direct-v4-plan.json')
    m=common.manifest(ROOT)
    for i,r in enumerate(m['records'],1):
        assert common.valid(common.path(DEST,r['path']),r),'Final SHA mismatch: '+r['path']
        log('DIRECT_SHA_OK',files=i,total=62)
    with AUTH.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        lines=f.readlines(); matches=[l for l in lines if l.strip().endswith(' '+MARKER)]
        assert len(matches)==1,'Unexpected temporary key count'
        f.seek(0); f.writelines(l for l in lines if l not in matches); f.truncate()
        f.flush(); os.fsync(f.fileno())
    common.atomic(ROOT/'relay_nonhf_verified.json',dict(status='ok',files=m['files'],bytes=m['bytes'],
        records=m['records'],root=str(DEST),transport='4090A-direct-rsync-v4',
        temporary_authorized_key_removed=True,at=time.time()))
    log('DIRECT_ALL_62_SHA_OK_AUTH_REVOKED')

if __name__=='__main__':
    {'install':install,'plan':plan,'send':send,'verify':verify}[sys.argv[1]]()
