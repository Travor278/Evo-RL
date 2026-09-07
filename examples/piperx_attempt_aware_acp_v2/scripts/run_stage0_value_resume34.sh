#!/usr/bin/env bash
set -euo pipefail

EXP=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906
SSD=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906
REPO="$EXP/src/Evo-RL"
PY="$SSD/stage0_venv/bin/python"
OUTPUT="$EXP/checkpoints/pistar06_value_attempt_v2_stage0_overfit32_r1"
CONFIG="$OUTPUT/checkpoints/last/pretrained_model/value_train_config.json"

test -f "$CONFIG"
test "$(readlink "$OUTPUT/checkpoints/last")" = 000032

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

"$PY" -m lerobot.scripts.lerobot_value_train \
  --config_path="$CONFIG" \
  --resume=true \
  --steps=34 \
  --save_freq=1 \
  --log_freq=1 \
  --wandb.enable=false

test "$(readlink "$OUTPUT/checkpoints/last")" = 000034
test -f "$OUTPUT/checkpoints/000034/pretrained_model/model.safetensors"
test -f "$OUTPUT/checkpoints/000034/training_state/optimizer_state.safetensors"
"$PY" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
step = json.load(open(root / "checkpoints/000034/training_state/training_step.json"))["step"]
scheduler = json.load(open(root / "checkpoints/000034/training_state/scheduler_state.json"))
assert step == 34, step
assert scheduler["last_epoch"] >= 34, scheduler
print(f"STAGE0_RESUME_OK step={step} scheduler_last_epoch={scheduler['last_epoch']}")
PY
