"""Retire only the old CPU queue after its existing torchrun has exited.

SIGSTOP the checked queue controller, never its GPU children. This prevents
its already-loaded [3000,6000] loop from launching the cancelled 6000 run.
After 3000 exits, retire the controller and use the updated 3000-only lane
for strict final review. Training tensors and worker processes are untouched.
"""
import json, os, signal, subprocess, time
from pathlib import Path
from run_lane import ROOT, PY, put

PARENT=1908829
CHILD=1908832

def info(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
        return dict(pid=pid,state=s[0],ppid=int(s[1]),start=s[19],cmd=cmd)
    except FileNotFoundError:
        return None

def main():
    parent=info(PARENT); child=info(CHILD)
    assert parent and parent['cmd'].endswith('run_lane.py cosine '), parent
    assert child and child['ppid']==PARENT and 'torch.distributed.run' in child['cmd'] and 'fixed3000-seed1000' in child['cmd'], child
    # No process-group signals: only the waiting CPU controller is stopped.
    os.kill(PARENT,signal.SIGSTOP)
    put(ROOT/'cosine-controller-handoff.json',dict(stage='waiting_for_existing_3000',parent=parent,child=child,cancelled_steps=6000,at=time.time()))
    print('OLD_CPU_CONTROLLER_STOPPED; GPU_TRAINING_UNCHANGED',flush=True)
    while True:
        current=info(CHILD)
        if current is None or current['start']!=child['start'] or current['state']=='Z':break
        time.sleep(10)
    current_parent=info(PARENT)
    if current_parent and current_parent['start']==parent['start']:
        assert current_parent['state'] in ('T','t'), current_parent
        children=[]
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                x=info(int(p.name))
                if x and x['ppid']==PARENT and x['state']!='Z':children.append(x)
        assert not children, children
        os.kill(PARENT,signal.SIGKILL)
        for _ in range(50):
            if info(PARENT) is None:break
            time.sleep(.1)
    out=ROOT/'runs/piperx-visionlean-cosine-fixed3000-seed1000'
    if not (out/'training_complete.json').exists():
        put(ROOT/'cosine-controller-handoff.json',dict(stage='training_interrupted_inspect_before_resume',at=time.time()))
        raise RuntimeError('3000 training exited without completion; no automatic restart')
    put(ROOT/'cosine-controller-handoff.json',dict(stage='final_review_3000_only',at=time.time()))
    subprocess.run([PY,str(ROOT/'run_lane.py'),'cosine'],cwd=ROOT,check=True)
    put(ROOT/'cosine-controller-handoff.json',dict(stage='complete',cancelled_steps=6000,at=time.time()))

if __name__=='__main__':main()
