#!/usr/bin/env bash
set -euo pipefail

TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
REPO=$TARGET/src/Evo-RL
TRANSFORMERS_FORK=$TARGET/src/transformers_fix_lerobot_openpi
OVERLAY=$TARGET/env/official_overlay
PY=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pi05_jax_full558_55k_20260901/openpi/.venv/bin/python
DERIVED=$TARGET/derived/hil_official_value_v1
INVENTORY=$TARGET/manifests/dataset_inventory.json
TRAIN_MANIFEST=$TARGET/manifests/train_episodes.json
ASSETS=$TARGET/manifests/official_asset_revisions.json
OUTPUT=$TARGET/checkpoints/pistar06_value_official_smoke_r1

test "$(git -C "$REPO" rev-parse HEAD)" = 6f2db449a21e1bac750b996f2e27cac6739aa63f
test "$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)" = dcddb970176382c0fcf4521b0c0e6fc15894dfe0
test -f "$INVENTORY"
test -f "$TRAIN_MANIFEST"
test -f "$ASSETS"
test -d "$DERIVED"
test ! -e "$OUTPUT"

export PYTHONPATH="$TRANSFORMERS_FORK/src:$OVERLAY:$REPO/src"
export HF_HOME=$TARGET/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$TARGET/cache/huggingface/hub
export TRANSFORMERS_CACHE=$TARGET/cache/huggingface/transformers
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

mapfile -t PREFLIGHT < <("$PY" - "$INVENTORY" "$TRAIN_MANIFEST" "$ASSETS" <<'PY'
import json, sys
inventory=json.load(open(sys.argv[1]))
failed=[key for key,value in inventory['gates'].items() if not value]
if failed:
    raise SystemExit(f"audit gates failed: {failed}")
train=json.load(open(sys.argv[2]))
assets=json.load(open(sys.argv[3]))['assets']
print(json.dumps(train['episode_indices'],separators=(',',':')))
print(assets['vision']['snapshot_path'])
print(assets['language']['snapshot_path'])
PY
)
TRAIN_EPISODES=${PREFLIGHT[0]}
VISION_PATH=${PREFLIGHT[1]}
LANGUAGE_PATH=${PREFLIGHT[2]}

"$PY" -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local/evorl-piperx-copper-screw-hil-official-value-v1 \
  --dataset.root="$DERIVED" \
  --dataset.episodes="$TRAIN_EPISODES" \
  --dataset.video_backend=pyav \
  --dataset.use_imagenet_stats=false \
  --value.type=pistar06 \
  --value.device=cuda \
  --value.dtype=bfloat16 \
  --value.vision_repo_id="$VISION_PATH" \
  --value.language_repo_id="$LANGUAGE_PATH" \
  --value.freeze_vision_encoder=false \
  --value.freeze_language_model=false \
  --value.use_gradient_checkpointing=true \
  --value.push_to_hub=false \
  --targets.success_field=episode_success \
  --targets.c_fail_coef=1.0 \
  --batch_size=4 \
  --steps=100 \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq=50 \
  --log_freq=5 \
  --seed=20260902 \
  --output_dir="$OUTPUT" \
  --job_name=pistar06_value_official_smoke_r1 \
  --wandb.enable=false
