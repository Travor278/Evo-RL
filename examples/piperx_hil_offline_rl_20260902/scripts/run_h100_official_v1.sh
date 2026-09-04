#!/usr/bin/env bash
set -euo pipefail

TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
REPO=$TARGET/src/Evo-RL
TRANSFORMERS_FORK=$TARGET/src/transformers_fix_lerobot_openpi
OVERLAY=$TARGET/env/official_overlay
OLD_ROOT=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/pi05_full558_044
OLD_WHEELHOUSE=$OLD_ROOT/wheelhouse_v060_py312_ngc2502_locked
EXACT_WHEELHOUSE=$TARGET/env/h100_py312_exact_wheels
VENV=/tmp/evorl-official-v1-py312
VALUE_DATASET=$TARGET/derived/hil_official_value_v1
POLICY_DATASET=$TARGET/derived/hil_official_policy_v1
INVENTORY=$TARGET/manifests/dataset_inventory.json
TRAIN_MANIFEST=$TARGET/manifests/train_episodes.json
ASSETS=$TARGET/manifests/official_asset_revisions.json
BENCH_OUTPUT=$TARGET/checkpoints/v2sam-ego2exo-official-v1-value-benchmark-200
VALUE_OUTPUT=$TARGET/checkpoints/v2sam-ego2exo-official-v1-value
INFER_OUTPUT=$TARGET/reports/v2sam-ego2exo-official-v1-value-infer
GATE_OUTPUT=$TARGET/reports/v2sam-ego2exo-official-v1-offline-gate.json
RUN_MANIFEST=$TARGET/manifests/v2sam-ego2exo-official-v1-r2-run.txt
JOB=v2sam-ego2exo-official-v1-r2
PORT=29501

mkdir -p "$TARGET/logs" "$TARGET/manifests" "$TARGET/checkpoints" "$TARGET/reports"
exec > >(tee -a "$TARGET/logs/${JOB}.log") 2>&1

echo "JOB_START name=$JOB utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
test "$(nvidia-smi -L | wc -l)" = 8
test "$(git -C "$REPO" rev-parse HEAD)" = 6f2db449a21e1bac750b996f2e27cac6739aa63f
test "$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)" = dcddb970176382c0fcf4521b0c0e6fc15894dfe0
test "$(sha256sum "$TARGET/scripts/0001-pistar06-resume-postprocessor-override.patch" | awk '{print $1}')" = 0c40c4b4c89840a380d920fa04cb74692d348b2040281b495a82adcfe0cdc1b1
test "$(sha256sum "$TARGET/scripts/0002-pistar06-ddp-find-unused.patch" | awk '{print $1}')" = 4641be6c154a6bce41d0bd960d949f0e609fd727243d97fb0f6e349611e42ef5
grep -Fq 'DistributedDataParallelKwargs(find_unused_parameters=True)' "$REPO/src/lerobot/scripts/lerobot_value_train.py"
git -C "$REPO" diff --check
test "$(git -C "$REPO" status --porcelain)" = " M src/lerobot/scripts/lerobot_value_train.py"
test -d "$VALUE_DATASET"
test -d "$POLICY_DATASET"
test -f "$INVENTORY"
test -f "$TRAIN_MANIFEST"
test -f "$ASSETS"
test -f "$EXACT_WHEELHOUSE/SHA256SUMS"
(cd "$EXACT_WHEELHOUSE" && sha256sum -c SHA256SUMS)
test ! -e "$BENCH_OUTPUT"
test ! -e "$VALUE_OUTPUT"
test ! -e "$INFER_OUTPUT"
test ! -e "$GATE_OUTPUT"

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
export HF_HOME=$TARGET/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$TARGET/cache/huggingface/hub
export TRANSFORMERS_CACHE=$TARGET/cache/huggingface/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

mapfile -t PREFLIGHT < <("$VENV/bin/python" - "$INVENTORY" "$TRAIN_MANIFEST" "$ASSETS" <<'PY'
import json, sys
inventory=json.load(open(sys.argv[1]))
failed=[key for key,value in inventory['gates'].items() if not value]
if failed:
    raise SystemExit(f"audit gates failed: {failed}")
train=json.load(open(sys.argv[2]))
assets=json.load(open(sys.argv[3]))['assets']
assert len(train['episode_indices']) == 60
print(json.dumps(train['episode_indices'], separators=(',', ':')))
print(assets['vision']['snapshot_path'])
print(assets['language']['snapshot_path'])
PY
)
TRAIN_EPISODES=${PREFLIGHT[0]}
VISION_PATH=${PREFLIGHT[1]}
LANGUAGE_PATH=${PREFLIGHT[2]}

"$VENV/bin/python" - <<'PY'
import accelerate, av, datasets, draccus, huggingface_hub, numpy, pandas, pyarrow, safetensors, tokenizers, torch, transformers
from importlib.metadata import version
from lerobot.utils.import_utils import _transformers_available
from lerobot.values.pistar06 import modeling_pistar06
expected = {
    'accelerate': '1.13.0', 'av': '17.0.0', 'datasets': '3.6.0',
    'huggingface_hub': '0.35.3', 'numpy': '1.26.4', 'pandas': '2.2.3',
    'pyarrow': '20.0.0', 'safetensors': '0.7.0', 'tokenizers': '0.21.1',
    'transformers': '4.53.3',
}
actual = {
    'accelerate': accelerate.__version__, 'av': av.__version__, 'datasets': datasets.__version__,
    'huggingface_hub': huggingface_hub.__version__, 'numpy': numpy.__version__, 'pandas': pandas.__version__,
    'pyarrow': pyarrow.__version__, 'safetensors': safetensors.__version__, 'tokenizers': tokenizers.__version__,
    'transformers': transformers.__version__,
}
assert actual == expected, (actual, expected)
assert version('transformers') == '4.53.3', version('transformers')
assert _transformers_available is True
assert modeling_pistar06.AutoModel is not None
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
print('H100_IMPORT_GATE_OK', actual, torch.__version__, torch.version.cuda)
PY

"$VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 "$OLD_ROOT/preflight/nccl_smoke.py"

cat > "$RUN_MANIFEST" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
job=$JOB
compute_group=开发区-H100-cuda13.2版本-183核
compute_group_id=lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e
image=docker.sii.shaipower.online/base/ngc-pytorch:25.02-cuda12.8.0-py3
world_size=8
per_device_batch=8
global_batch=64
benchmark_steps=200
formal_value_steps=8000
seed=20260902
evorl_commit=$(git -C "$REPO" rev-parse HEAD)
transformers_fork_commit=$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)
dataset_inventory_sha256=$(sha256sum "$INVENTORY" | awk '{print $1}')
train_manifest_sha256=$(sha256sum "$TRAIN_MANIFEST" | awk '{print $1}')
exact_wheels_sha256=$(sha256sum "$EXACT_WHEELHOUSE/SHA256SUMS" | awk '{print $1}')
value_dataset=$VALUE_DATASET
policy_dataset=$POLICY_DATASET
EOF

COMMON_VALUE_ARGS=(
  --dataset.repo_id=local/evorl-piperx-copper-screw-hil-official-value-v1
  --dataset.root="$VALUE_DATASET"
  --dataset.episodes="$TRAIN_EPISODES"
  --dataset.video_backend=pyav
  --dataset.use_imagenet_stats=false
  --value.type=pistar06
  --value.device=cuda
  --value.dtype=bfloat16
  --value.vision_repo_id="$VISION_PATH"
  --value.language_repo_id="$LANGUAGE_PATH"
  --value.freeze_vision_encoder=false
  --value.freeze_language_model=false
  --value.use_gradient_checkpointing=true
  --value.push_to_hub=false
  --targets.success_field=episode_success
  --targets.c_fail_coef=1.0
  --batch_size=8
  --num_workers=4
  --save_checkpoint=true
  --seed=20260902
  --wandb.enable=false
)

echo "BENCHMARK_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$VENV/bin/python" -m accelerate.commands.launch \
  --multi_gpu --num_processes=8 --num_machines=1 --mixed_precision=bf16 --main_process_port="$PORT" \
  --module lerobot.scripts.lerobot_value_train \
  "${COMMON_VALUE_ARGS[@]}" \
  --steps=200 --save_freq=100 --log_freq=10 \
  --output_dir="$BENCH_OUTPUT" --job_name="${JOB}-value-benchmark-200"
test "$(readlink "$BENCH_OUTPUT/checkpoints/last")" = 000200
echo "BENCHMARK_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "FORMAL_VALUE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$VENV/bin/python" -m accelerate.commands.launch \
  --multi_gpu --num_processes=8 --num_machines=1 --mixed_precision=bf16 --main_process_port="$PORT" \
  --module lerobot.scripts.lerobot_value_train \
  "${COMMON_VALUE_ARGS[@]}" \
  --steps=8000 --save_freq=1000 --log_freq=10 \
  --output_dir="$VALUE_OUTPUT" --job_name="${JOB}-value"
test "$(readlink "$VALUE_OUTPUT/checkpoints/last")" = 008000
echo "FORMAL_VALUE_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "ACP_INFERENCE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$VENV/bin/python" -m accelerate.commands.launch \
  --multi_gpu --num_processes=8 --num_machines=1 --mixed_precision=bf16 --main_process_port="$PORT" \
  --module lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=local/evorl-piperx-copper-screw-hil-official-policy-v1 \
  --dataset.root="$POLICY_DATASET" \
  --dataset.success_field=episode_success \
  --inference.checkpoint_path="$VALUE_OUTPUT" \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=32 \
  --runtime.num_workers=4 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.30 \
  --acp.force_intervention_positive=true \
  --acp.intervention_field=complementary_info.is_intervention \
  --acp.value_field=complementary_info.value_evorl_official_v1 \
  --acp.advantage_field=complementary_info.advantage_evorl_official_v1 \
  --acp.indicator_field=complementary_info.acp_indicator_evorl_official_v1 \
  --acp.c_fail_coef=1.0 \
  --viz.enable=false \
  --seed=20260902 \
  --output_dir="$INFER_OUTPUT" \
  --job_name="${JOB}-value-infer"

"$VENV/bin/python" "$TARGET/scripts/analyze_official_value_smoke.py" \
  --derived-root "$POLICY_DATASET" \
  --manifest-root "$TARGET/manifests" \
  --value-field complementary_info.value_evorl_official_v1 \
  --advantage-field complementary_info.advantage_evorl_official_v1 \
  --indicator-field complementary_info.acp_indicator_evorl_official_v1 \
  --output "$GATE_OUTPUT"

echo "JOB_SUCCESS name=$JOB utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gate=$GATE_OUTPUT"
