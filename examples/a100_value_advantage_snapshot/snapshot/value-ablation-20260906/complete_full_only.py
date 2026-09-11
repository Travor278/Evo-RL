"""User-authorized completion of full/fixed1500 only. No broader queue."""
import fcntl
import json
import os
import subprocess
import time
from run_pipeline import ROOT, PY, run, write

os.chdir(ROOT)
lock = (ROOT/'pipeline.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.environ.update(HF_HOME='/data/cache/huggingface', HF_HUB_OFFLINE='1',
    HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
assert subprocess.check_output([PY,'-c','import torch;print(torch.__version__)'],text=True).strip() == '2.10.0+cu128'
result = run('full', 1500)
subprocess.run([PY, str(ROOT/'analyze_experiments.py')], check=True)
write(ROOT/'FULL1500_COMPLETION.json', dict(completed_at=time.time(), result=result,
    only_authorized_run='full/fixed1500', next_action='Render five identical fixed1500 schedules with explicit Chinese legends. No other training.'))
print('FULL1500_AND_DENSE_REVIEW_COMPLETE ' + json.dumps(result), flush=True)
