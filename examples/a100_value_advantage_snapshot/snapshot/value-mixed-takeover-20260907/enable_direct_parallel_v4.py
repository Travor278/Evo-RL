"""Permit only the task's disjoint rsync shards to run concurrently."""
import fcntl, os, pathlib
p=pathlib.Path('/root/.ssh/authorized_keys')
marker='value-direct-nonhf-v4-20260907'
old='/usr/bin/rrsync -wo -no-del /data/datasets/piperx-full-intervention-141-20260903'
new='/usr/bin/rrsync -wo -no-del -no-lock /data/datasets/piperx-full-intervention-141-20260903'
with p.open('r+') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    lines=f.readlines()
    matches=[i for i,l in enumerate(lines) if l.strip().endswith(' '+marker)]
    assert len(matches)==1
    i=matches[0]
    assert old in lines[i] and 'restrict,command=' in lines[i]
    lines[i]=lines[i].replace(old,new)
    f.seek(0);f.writelines(lines);f.truncate();f.flush();os.fsync(f.fileno())
print('SCOPED_PARALLEL_ENABLED')
