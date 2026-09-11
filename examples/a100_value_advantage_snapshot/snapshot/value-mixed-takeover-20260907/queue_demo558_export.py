"""Wait for the owned inference process; export only after its validated success marker."""
import json,time,subprocess,sys
from pathlib import Path
ROOT=Path('/data/experiments/value-mixed-takeover-20260907')
marker=ROOT/'demo558-top10-a50-w20-build/inference_complete.json'
while not marker.exists():
    alive=subprocess.run(['tmux','has-session','-t','demo558-a50w20-infer'],capture_output=True).returncode==0
    if not alive:
        raise RuntimeError('Inference tmux ended without success marker; do not export incomplete scores')
    time.sleep(15)
r=json.loads(marker.read_text());assert r['status']=='ok' and r['episodes']==558
print('INFERENCE_VERIFIED_BEGIN_EXPORT',flush=True)
subprocess.run([sys.executable,'-u',str(ROOT/'build_demo558_segments.py'),'--workers','6'],check=True,cwd=ROOT)
print('EXPORT_QUEUE_COMPLETE',flush=True)
