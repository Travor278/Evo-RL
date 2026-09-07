#!/usr/bin/env bash
set -euo pipefail

EXP=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906
SSD=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906
REPO="$EXP/src/Evo-RL"
PY="$SSD/stage0_venv/bin/python"
VIEW="$SSD/attempt_value_view"
SPLITS="$EXP/manifests/attempt_split_manifest_v2.json"
VISION=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902/cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3
LANGUAGE=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902/cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
OUTPUT="$EXP/checkpoints/pistar06_value_attempt_v2_stage0_overfit32_r1"

test -x "$PY"
test -d "$VIEW"
test -f "$SPLITS"
test -f "$VISION/config.json"
test -f "$LANGUAGE/config.json"
test ! -e "$OUTPUT"

TRAIN_EPISODES=$("$PY" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["splits"]["train"]["episode_indices"], separators=(",", ":")))' "$SPLITS")

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

"$PY" -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local/evorl-piperx-copper-screw-attempt-value-v2 \
  --dataset.root="$VIEW" \
  --dataset.episodes="$TRAIN_EPISODES" \
  --dataset.video_backend=pyav \
  --dataset.use_imagenet_stats=false \
  --value.type=pistar06 \
  --value.device=cuda \
  --value.dtype=bfloat16 \
  --value.vision_repo_id="$VISION" \
  --value.language_repo_id="$LANGUAGE" \
  --value.num_bins=201 \
  --value.bin_min=-1.0 \
  --value.bin_max=0.0 \
  --value.freeze_vision_encoder=false \
  --value.freeze_language_model=false \
  --value.use_gradient_checkpointing=true \
  --value.push_to_hub=false \
  --targets.success_field=episode_success \
  --targets.c_fail_coef=1.0 \
  --attempt_sampling.enable=true \
  --attempt_sampling.attempt_field=logical_attempt_id \
  --attempt_sampling.valid_field=logical_transition_valid \
  --attempt_sampling.outcome_known_field=logical_attempt_outcome_known \
  --batch_size=4 \
  --steps=32 \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq=16 \
  --log_freq=1 \
  --seed=20260906 \
  --output_dir="$OUTPUT" \
  --job_name=pistar06_value_attempt_v2_stage0_overfit32_r1 \
  --wandb.enable=false

test "$(readlink "$OUTPUT/checkpoints/last")" = 000032
test -f "$OUTPUT/checkpoints/000016/pretrained_model/model.safetensors"
test -f "$OUTPUT/checkpoints/000032/pretrained_model/model.safetensors"
printf 'STAGE0_OVERFIT32_OK output=%s\n' "$OUTPUT"
