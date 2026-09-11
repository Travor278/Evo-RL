#!/usr/bin/env bash
# Sequential, fail-closed experiment stages. Existing results are never overwritten.
set -euo pipefail
experiment_root=/data/experiments/value-demo3-20260906
cd "$experiment_root"
exec 9>pipeline.lock
flock -n 9 || { echo PIPELINE_ALREADY_RUNNING; exit 1; }
export HF_HOME=/data/cache/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python_bin="$experiment_root/venv-py312-torch210-cu128/bin/python"
echo PIPELINE_WAITING_FOR_ENVIRONMENT
while tmux has-session -t value-demo3-env 2>/dev/null; do sleep 30; done
while tmux has-session -t value-demo3-compat8-smoke 2>/dev/null; do sleep 15; done
"$python_bin" - <<'PY'
import importlib.metadata as m
import torch
import lerobot
import value_experiment
expected={'lerobot':'0.6.1','torch':'2.10.0+cu128','torchvision':'0.25.0+cu128','transformers':'5.3.0','accelerate':'1.13.0','datasets':'4.8.5','av':'15.1.0'}
actual={k:m.version(k) for k in expected}
assert actual==expected, actual
assert torch.version.cuda=='12.8' and torch.cuda.is_available()
assert torch.cuda.device_count()==8
print('TARGET_ENVIRONMENT_VERIFIED',actual,flush=True)
PY
"$python_bin" -m pip check
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo GPU_OCCUPIED_ABORTING_PIPELINE
    exit 1
fi
test ! -e runs/targetenv-8gpu-smoke-v1
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=8 -- value_experiment.py \
  --task piperx_insert_copper_screw --run targetenv-8gpu-smoke-v1 \
  --steps 3 --batch-size 4 --workers 2 --eval-every 3 --eval-frames 3 --warmup 3 --smoke \
  >> targetenv-8gpu-smoke-v1.log 2>&1
grep -q '^SMOKE_SAVE_RELOAD_OK' targetenv-8gpu-smoke-v1.log
echo TARGET_8GPU_SMOKE_PASSED
for value_task in piperx_insert_copper_screw so101_fold_clothes_left_stack_right so101_put_stationery_table_into_bag; do
    value_run="${value_task}-bf16-gb32-seed1000-pilot8k-v1"
    test ! -e "runs/$value_run"
    echo "FORMAL_TRAIN_START task=$value_task run=$value_run"
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=8 -- value_experiment.py \
      --task "$value_task" --run "$value_run" \
      --steps 8000 --batch-size 4 --workers 2 --eval-every 250 --eval-frames 32 \
      --warmup 200 --min-steps 2000 --patience 8 >> "$value_run.log" 2>&1
    echo "DENSE_REVIEW_START task=$value_task"
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=8 -- value_experiment.py \
      --task "$value_task" --run "$value_run" --evaluate-checkpoint best \
      --batch-size 4 --workers 2 --eval-frames 128 >> "$value_run-dense.log" 2>&1
    echo "TASK_PILOT_AND_REVIEW_COMPLETE task=$value_task"
done
echo THREE_PILOTS_COMPLETE_REQUIRES_CONVERGENCE_REVIEW
