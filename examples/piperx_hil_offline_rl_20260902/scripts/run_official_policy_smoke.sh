#!/usr/bin/env bash
set -euo pipefail

TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
REPO=$TARGET/src/Evo-RL
TRANSFORMERS_FORK=$TARGET/src/transformers_fix_lerobot_openpi
OVERLAY=$TARGET/env/official_overlay
PY=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pi05_jax_full558_55k_20260901/openpi/.venv/bin/python
DATASET=$TARGET/derived/mixed_replay_full558_hil_train60_v1
BASE_POLICY=$TARGET/checkpoints/pi05_base_step50000_policy_compat_6f2db_v2
OUTPUT=/tmp/v2sam-ego2exo-official-v2-policy-smoke-r3
LOG=$TARGET/logs/v2sam-ego2exo-official-v2-policy-smoke-r3.log
ACP_FIELD=complementary_info.acp_indicator_evorl_official_v1
ACP_MASK_FIELD=complementary_info.acp_apply_mask_evorl_official_v1
SOURCE_FIELD=complementary_info.replay_source_evorl_official_v1

test "$(nvidia-smi -L | wc -l)" = 1
test ! -e "$OUTPUT"
test ! -e "$LOG"
test -f "$DATASET/meta/MIXED_REPLAY_COMPLETE.json"
test -f "$BASE_POLICY/model.safetensors"
test "$(sha256sum "$TARGET/scripts/0004-acp-replay-mask-weighted-sampler.patch" | awk '{print $1}')" = 15857546555052fd7e452ef158f6a0c90b4730943d19c37b5338d94f233bdcb6
test "$(sha256sum "$TARGET/scripts/0005-explicit-checkpoint-steps.patch" | awk '{print $1}')" = e7f927b128bd7e11cac0278d93e0e5932eef3d84b4c893e0e0fabccdab206a39
test "$(sha256sum "$TARGET/scripts/0006-strict-policy-numerics.patch" | awk '{print $1}')" = 05699b0e4d8aca354a348577d50e033bf58ee2264fd16bfa3998e35b6b2c4ed7

export PYTHONPATH="$TRANSFORMERS_FORK/src:$OVERLAY:$REPO/src"
export HF_HOME=$TARGET/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$TARGET/cache/huggingface/hub
export TRANSFORMERS_CACHE=$TARGET/cache/huggingface/transformers
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

COMMON_ARGS=(
  --dataset.repo_id=local/piperx-full558-hil-train60-replay-v1
  --dataset.root="$DATASET"
  --dataset.video_backend=pyav
  --dataset.use_imagenet_stats=false
  --policy.path="$BASE_POLICY"
  --use_policy_training_preset=false
  --optimizer.type=sgd
  --optimizer.lr=1e-6
  --optimizer.momentum=0.0
  --optimizer.dampening=0.0
  --optimizer.nesterov=false
  --optimizer.weight_decay=0.0
  --optimizer.grad_clip_norm=1.0
  --scheduler.type=cosine_decay_with_warmup
  --scheduler.num_warmup_steps=0
  --scheduler.num_decay_steps=2
  --scheduler.peak_lr=1e-6
  --scheduler.decay_lr=1e-6
  '--rename_map={"observation.images.camera_top":"observation.images.base_0_rgb","observation.images.camera_wrist_left":"observation.images.left_wrist_0_rgb","observation.images.camera_wrist_right":"observation.images.right_wrist_0_rgb"}'
  --batch_size=1
  --num_workers=2
  --eval_freq=0
  --save_checkpoint=true
  --log_freq=1
  --seed=20260902
  --wandb.enable=false
  --replay_sampling.enable=true
  --replay_sampling.source_field="$SOURCE_FIELD"
  --replay_sampling.hil_value=1
  --replay_sampling.target_hil_fraction=0.25
  --replay_sampling.num_samples=1600639
  --acp.enable=true
  --acp.indicator_field="$ACP_FIELD"
  --acp.apply_mask_field="$ACP_MASK_FIELD"
  --acp.indicator_dropout_prob=0.30
)

exec > >(tee -a "$LOG") 2>&1
echo "SMOKE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -m accelerate.commands.launch --num_processes=1 --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_train \
  "${COMMON_ARGS[@]}" \
  --steps=2 --save_freq=0 '--save_steps=[2]' \
  --output_dir="$OUTPUT" --job_name=v2sam-ego2exo-official-v2-policy-smoke-r3

test "$(readlink "$OUTPUT/checkpoints/last")" = 000002
test -f "$OUTPUT/checkpoints/000002/pretrained_model/model.safetensors"
test -f "$OUTPUT/checkpoints/000002/training_state/optimizer_state.safetensors"
echo "SMOKE_CHECKPOINT_2_OK utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PY" -m accelerate.commands.launch --num_processes=1 --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_train \
  --config_path="$OUTPUT/checkpoints/000002/pretrained_model/train_config.json" \
  --resume=true --steps=3 --save_freq=0 '--save_steps=[3]' --log_freq=1

test "$(readlink "$OUTPUT/checkpoints/last")" = 000003
test -f "$OUTPUT/checkpoints/000003/pretrained_model/model.safetensors"
test -f "$OUTPUT/checkpoints/000003/training_state/optimizer_state.safetensors"
test "$("$PY" -c "import json;print(json.load(open('$OUTPUT/checkpoints/000003/training_state/training_step.json'))['step'])")" = 3
if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|CUDA error|NCCL error|RuntimeError|Traceback' "$LOG"; then
  echo "SMOKE_FAILURE_PATTERN_FOUND" >&2
  exit 2
fi
echo "POLICY_SMOKE_RESUME_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) output=$OUTPUT"
