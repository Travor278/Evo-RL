#!/usr/bin/env bash
set -euo pipefail

TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
REPO=$TARGET/src/Evo-RL
TRANSFORMERS_FORK=$TARGET/src/transformers_fix_lerobot_openpi
OVERLAY=$TARGET/env/official_overlay
PY=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pi05_jax_full558_55k_20260901/openpi/.venv/bin/python
DERIVED=$TARGET/derived/hil_official_policy_v1
VALUE_OUTPUT=$TARGET/checkpoints/pistar06_value_official_smoke_r1
OUTPUT=$TARGET/reports/value_infer_official_smoke_r1

test -d "$VALUE_OUTPUT/checkpoints"
test ! -e "$OUTPUT"

test "$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)" = dcddb970176382c0fcf4521b0c0e6fc15894dfe0
export PYTHONPATH="$TRANSFORMERS_FORK/src:$OVERLAY:$REPO/src"
export HF_HOME=$TARGET/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$TARGET/cache/huggingface/hub
export TRANSFORMERS_CACHE=$TARGET/cache/huggingface/transformers
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

"$PY" -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=local/evorl-piperx-copper-screw-hil-official-policy-v1 \
  --dataset.root="$DERIVED" \
  --dataset.success_field=episode_success \
  --inference.checkpoint_path="$VALUE_OUTPUT" \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=4 \
  --runtime.num_workers=4 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.30 \
  --acp.force_intervention_positive=true \
  --acp.intervention_field=complementary_info.is_intervention \
  --acp.value_field=complementary_info.value_evorl_official_smoke_r1 \
  --acp.advantage_field=complementary_info.advantage_evorl_official_smoke_r1 \
  --acp.indicator_field=complementary_info.acp_indicator_evorl_official_smoke_r1 \
  --acp.c_fail_coef=1.0 \
  --viz.enable=false \
  --seed=20260902 \
  --output_dir="$OUTPUT" \
  --job_name=value_infer_official_smoke_r1
