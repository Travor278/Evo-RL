#!/usr/bin/env bash
set -euo pipefail

TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
REPO=$TARGET/src/Evo-RL
TRANSFORMERS_FORK=$TARGET/src/transformers_fix_lerobot_openpi
OVERLAY=$TARGET/env/official_overlay
OLD_ROOT=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/pi05_full558_044
OLD_WHEELHOUSE=$OLD_ROOT/wheelhouse_v060_py312_ngc2502_locked
EXACT_WHEELHOUSE=$TARGET/env/h100_py312_exact_wheels
VENV=/tmp/evorl-official-v1-acp-r1-py312
POLICY_DATASET=$TARGET/derived/hil_official_policy_v1
VALUE_OUTPUT=$TARGET/checkpoints/v2sam-ego2exo-official-v1-value
FAILED_INFER_OUTPUT=$TARGET/reports/v2sam-ego2exo-official-v1-value-infer
INFER_OUTPUT=$TARGET/reports/v2sam-ego2exo-official-v1-value-infer-r1
GATE_OUTPUT=$TARGET/reports/v2sam-ego2exo-official-v1-offline-gate.json
RUN_MANIFEST=$TARGET/manifests/v2sam-ego2exo-official-v1-acp-r1-run.txt
JOB=v2sam-ego2exo-official-v1-acp-r1
PORT=29501

mkdir -p "$TARGET/logs" "$TARGET/manifests" "$TARGET/reports"
exec > >(tee -a "$TARGET/logs/${JOB}.log") 2>&1
echo "JOB_START name=$JOB utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

test "$(nvidia-smi -L | wc -l)" = 8
test "$(git -C "$REPO" rev-parse HEAD)" = 6f2db449a21e1bac750b996f2e27cac6739aa63f
test "$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)" = dcddb970176382c0fcf4521b0c0e6fc15894dfe0
test "$(sha256sum "$TARGET/scripts/0003-value-infer-video-backend.patch" | awk '{print $1}')" = 3e1811d22b99ffc4e12c45c3519e075a900bb9cc3429a4454016513702d370c4
grep -Fq 'video_backend: str | None = "pyav"' "$REPO/src/lerobot/configs/value.py"
grep -Fq '"video_backend": cfg.dataset.video_backend' "$REPO/src/lerobot/scripts/lerobot_value_infer.py"
git -C "$REPO" diff --check
test -d "$POLICY_DATASET"
test -f "$VALUE_OUTPUT/checkpoints/008000/pretrained_model/model.safetensors"
test "$(readlink "$VALUE_OUTPUT/checkpoints/last")" = 008000
test -d "$FAILED_INFER_OUTPUT"
test ! -e "$INFER_OUTPUT"
test ! -e "$GATE_OUTPUT"
(cd "$EXACT_WHEELHOUSE" && sha256sum -c SHA256SUMS)

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

"$VENV/bin/python" - "$POLICY_DATASET" <<'PY'
import sys
from importlib.metadata import version
import transformers
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.import_utils import _transformers_available
assert version('transformers') == '4.53.3'
assert transformers.__version__ == '4.53.3'
assert _transformers_available is True
ds = LeRobotDataset(repo_id='local/evorl-piperx-copper-screw-hil-official-policy-v1', root=sys.argv[1], video_backend='pyav')
sample = ds[0]
image_keys = sorted(k for k in sample if k.startswith('observation.images.'))
assert len(image_keys) == 3, image_keys
assert all(tuple(sample[k].shape) == (3, 480, 640) for k in image_keys)
print('PYAV_DATASET_GATE_OK', image_keys)
PY
"$VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 "$OLD_ROOT/preflight/nccl_smoke.py"

cat > "$RUN_MANIFEST" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
job=$JOB
source_job=job-b32f52e3-21c9-4122-803a-570cb7bc9ceb
source_value_checkpoint=$VALUE_OUTPUT/checkpoints/008000
source_value_checkpoint_step=8000
source_training_status=PASS
resume_scope=ACP inference and offline gate only
video_backend=pyav
world_size=8
per_device_batch=32
global_inference_batch=256
seed=20260902
evorl_commit=$(git -C "$REPO" rev-parse HEAD)
transformers_fork_commit=$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)
EOF

echo "ACP_INFERENCE_RESUME_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$VENV/bin/python" -m accelerate.commands.launch \
  --multi_gpu --num_processes=8 --num_machines=1 --mixed_precision=bf16 --main_process_port="$PORT" \
  --module lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=local/evorl-piperx-copper-screw-hil-official-policy-v1 \
  --dataset.root="$POLICY_DATASET" \
  --dataset.video_backend=pyav \
  --dataset.success_field=episode_success \
  --inference.checkpoint_path="$VALUE_OUTPUT" \
  --inference.checkpoint_ref=008000 \
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
