#!/usr/bin/env bash
set -euo pipefail

EXP=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906
SSD=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906
STEPS=${VALUE_SMOKE_STEPS:-300}
RUN_ID=${VALUE_SMOKE_RUN_ID:-smoke300_r1}
PLATFORM_TASK=${VALUE_SMOKE_PLATFORM_TASK:-v2sam-exo2ego-fusion-official24-resume-e8-20260827-v2-cuda132-r15}
FAILURE_FRACTION=${VALUE_SMOKE_FAILURE_FRACTION:-}
REPO="$EXP/src/Evo-RL"
OLD_TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
TRANSFORMERS_FORK="$OLD_TARGET/src/transformers_fix_lerobot_openpi"
OVERLAY="$OLD_TARGET/env/official_overlay"
OLD_ROOT=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/pi05_full558_044
OLD_WHEELHOUSE="$OLD_ROOT/wheelhouse_v060_py312_ngc2502_locked"
EXACT_WHEELHOUSE="$OLD_TARGET/env/h100_py312_exact_wheels"
VENV="/tmp/evorl-attempt-v2-value-${RUN_ID}-py312"
VIEW="$SSD/attempt_value_view"
SPLITS="$EXP/manifests/attempt_split_manifest_v2.json"
ASSETS="$EXP/manifests/official_asset_revisions.json"
OUTPUT="$EXP/checkpoints/pistar06_value_attempt_v2_${RUN_ID}"
RUN_MANIFEST="$EXP/manifests/value_${RUN_ID}_run.txt"
JOB="v2sam-ego2exo-attempt-v2-value-${RUN_ID}"
PORT=29517

mkdir -p "$EXP/logs" "$EXP/manifests" "$EXP/checkpoints"
exec > >(tee -a "$EXP/logs/${JOB}.log") 2>&1

echo "JOB_START name=$JOB utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
test "$(nvidia-smi -L | wc -l)" = 8
test "$(git -C "$REPO" rev-parse HEAD)" = 2387400a66c3b6e8d21968ec432fc53e0bc4163d
test -z "$(git -C "$REPO" status --porcelain)"
test "$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)" = dcddb970176382c0fcf4521b0c0e6fc15894dfe0
test -d "$VIEW"
test -f "$SPLITS"
test -f "$ASSETS"
test -f "$EXACT_WHEELHOUSE/SHA256SUMS"
test ! -e "$OUTPUT"

python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
PY
python -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --no-index --find-links "$OLD_WHEELHOUSE" -r "$OLD_ROOT/h100_minimal_py312.txt"
"$VENV/bin/python" -m pip uninstall -y transformers
"$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$TRANSFORMERS_FORK"
"$VENV/bin/python" -m pip install --no-index --find-links "$OLD_WHEELHOUSE" --force-reinstall --no-deps \
  accelerate==1.13.0 draccus==0.10.0 huggingface-hub==0.35.3 numpy==1.26.4 safetensors==0.7.0
"$VENV/bin/python" -m pip install --no-index --find-links "$EXACT_WHEELHOUSE" --force-reinstall --no-deps \
  av==17.0.0 datasets==3.6.0 dill==0.3.8 fsspec==2025.3.0 opencv-python-headless==4.11.0.86 \
  pandas==2.2.3 pyarrow==20.0.0 tokenizers==0.21.1

export PYTHONPATH="$TRANSFORMERS_FORK/src:$OVERLAY:$REPO/src"
export HF_HOME="$OLD_TARGET/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$OLD_TARGET/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$OLD_TARGET/cache/huggingface/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

mapfile -t PREFLIGHT < <("$VENV/bin/python" - "$SPLITS" "$ASSETS" <<'PY'
import json
import sys

splits = json.load(open(sys.argv[1]))["splits"]
assets = json.load(open(sys.argv[2]))["assets"]
assert splits["train"]["attempts"] == 386
assert splits["train"]["failure"] == 5
assert splits["validation"]["failure"] == 1
print(json.dumps(splits["train"]["episode_indices"], separators=(",", ":")))
print(assets["vision"]["snapshot_path"])
print(assets["language"]["snapshot_path"])
PY
)
TRAIN_EPISODES=${PREFLIGHT[0]}
VISION_PATH=${PREFLIGHT[1]}
LANGUAGE_PATH=${PREFLIGHT[2]}

"$VENV/bin/python" - <<'PY'
import accelerate, av, datasets, numpy, pandas, pyarrow, torch, transformers
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
assert transformers.__version__ == "4.53.3", transformers.__version__
print("H100_IMPORT_GATE_OK", torch.__version__, torch.version.cuda)
PY
"$VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 "$OLD_ROOT/preflight/nccl_smoke.py"

cat > "$RUN_MANIFEST" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
job=$JOB
platform_task=$PLATFORM_TASK
world_size=8
per_device_batch=8
global_batch=64
steps=$STEPS
seed=20260906
evorl_commit=$(git -C "$REPO" rev-parse HEAD)
dataset_revision=d8c2c0f0115ce3456dbfa38f5529276c0e4bbe16
split_manifest_sha256=$(sha256sum "$SPLITS" | awk '{print $1}')
outcome_contract_sha256=$(sha256sum "$EXP/configs/outcome_contract.yaml" | awk '{print $1}')
value_dataset=$VIEW
base_558_in_value=false
failure_fraction=${FAILURE_FRACTION:-none}
EOF

BALANCE_ARGS=()
if [ -n "$FAILURE_FRACTION" ]; then
  BALANCE_ARGS+=(
    --attempt_sampling.outcome_success_field=logical_attempt_success
    --attempt_sampling.failure_fraction="$FAILURE_FRACTION"
  )
fi

"$VENV/bin/python" -m accelerate.commands.launch \
  --multi_gpu --num_processes=8 --num_machines=1 --mixed_precision=bf16 --main_process_port="$PORT" \
  --module lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local/evorl-piperx-copper-screw-attempt-value-v2 \
  --dataset.root="$VIEW" \
  --dataset.episodes="$TRAIN_EPISODES" \
  --dataset.video_backend=pyav \
  --dataset.use_imagenet_stats=false \
  --value.type=pistar06 \
  --value.device=cuda \
  --value.dtype=bfloat16 \
  --value.vision_repo_id="$VISION_PATH" \
  --value.language_repo_id="$LANGUAGE_PATH" \
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
  --attempt_sampling.terminal_field=logical_attempt_terminal \
  --attempt_sampling.outcome_known_field=logical_attempt_outcome_known \
  "${BALANCE_ARGS[@]}" \
  --batch_size=8 \
  --steps="$STEPS" \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq="$STEPS" \
  --log_freq=10 \
  --seed=20260906 \
  --output_dir="$OUTPUT" \
  --job_name="$JOB" \
  --wandb.enable=false

EXPECTED_STEP=$(printf '%06d' "$STEPS")
test "$(readlink "$OUTPUT/checkpoints/last")" = "$EXPECTED_STEP"
echo "VALUE_SMOKE_PASS run_id=$RUN_ID steps=$STEPS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) output=$OUTPUT"
