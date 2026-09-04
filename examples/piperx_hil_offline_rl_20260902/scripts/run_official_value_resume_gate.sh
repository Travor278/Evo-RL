#!/usr/bin/env bash
set -euo pipefail

TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
REPO=$TARGET/src/Evo-RL
TRANSFORMERS_FORK=$TARGET/src/transformers_fix_lerobot_openpi
OVERLAY=$TARGET/env/official_overlay
PY=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pi05_jax_full558_55k_20260901/openpi/.venv/bin/python
OUTPUT=$TARGET/checkpoints/pistar06_value_official_smoke_r1
CONFIG=$OUTPUT/checkpoints/last/pretrained_model/value_train_config.json

test "$(git -C "$REPO" rev-parse HEAD)" = 6f2db449a21e1bac750b996f2e27cac6739aa63f
test "$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)" = dcddb970176382c0fcf4521b0c0e6fc15894dfe0
test -f "$CONFIG"
test "$(readlink "$OUTPUT/checkpoints/last")" = 000100

export PYTHONPATH="$TRANSFORMERS_FORK/src:$OVERLAY:$REPO/src"
export HF_HOME=$TARGET/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$TARGET/cache/huggingface/hub
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

"$PY" -m lerobot.scripts.lerobot_value_train \
  --config_path="$CONFIG" \
  --resume=true \
  --steps=102 \
  --save_freq=1 \
  --log_freq=1 \
  --wandb.enable=false

test "$(readlink "$OUTPUT/checkpoints/last")" = 000102
"$PY" - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
step=json.load(open(root/'checkpoints/000102/training_state/training_step.json'))['step']
scheduler=json.load(open(root/'checkpoints/000102/training_state/scheduler_state.json'))
assert step == 102, step
assert scheduler['last_epoch'] >= 102, scheduler['last_epoch']
print(f'RESUME_GATE_OK step={step} scheduler_last_epoch={scheduler["last_epoch"]}')
PY
