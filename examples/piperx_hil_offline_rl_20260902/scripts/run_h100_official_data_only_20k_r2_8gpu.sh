#!/usr/bin/env bash
set -euo pipefail

MODE=data_only

TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
REPO=$TARGET/src/Evo-RL
TRANSFORMERS_FORK=$TARGET/src/transformers_fix_lerobot_openpi
OVERLAY=$TARGET/env/official_overlay
OLD_ROOT=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/pi05_full558_044
WHEELHOUSE=$OLD_ROOT/wheelhouse_v060_py312_ngc2502_locked
EXACT_WHEELHOUSE=$TARGET/env/h100_py312_exact_wheels
SOURCE_DATASET=$TARGET/derived/mixed_replay_full558_hil_train60_v1
DATASET_MANIFEST=$TARGET/manifests/mixed_replay_full558_hil_train60_v1.json
ACP_GATE=$TARGET/reports/v2sam-ego2exo-official-v1-offline-gate.json
BASE_POLICY=$TARGET/checkpoints/pi05_base_step50000_policy_compat_6f2db_v2
ACP_FIELD=complementary_info.acp_indicator_evorl_official_v1
ACP_MASK_FIELD=complementary_info.acp_apply_mask_evorl_official_v1
SOURCE_FIELD=complementary_info.replay_source_evorl_official_v1
SUPERVISOR=$TARGET/scripts/supervise_policy_log.py
JOB=v2sam-ego2exo-official-v3-20k-r2-8gpu
OUTPUT=$TARGET/checkpoints/${JOB}-data-only-policy
SSD_ARCHIVE=/inspire/ssd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902_recovery/${JOB}-data-only-policy/checkpoints
OFFLOAD_LOG=$TARGET/logs/${JOB}-checkpoint-offload.log
RAM_ROOT=/dev/shm/$JOB
DATASET=$RAM_ROOT/mixed_replay
MANIFEST=$TARGET/manifests/${JOB}-run.json
REPORT=$TARGET/reports/${JOB}.md
LOG=$TARGET/logs/${JOB}.log
VENV=/tmp/${JOB}-py312
PORT=29980

mkdir -p "$TARGET/logs" "$TARGET/manifests" "$TARGET/reports" "$TARGET/checkpoints"
mkdir -p "$SSD_ARCHIVE"
exec > >(tee -a "$LOG") 2>&1
echo "FORMAL_POLICY_JOB_START name=$JOB mode=$MODE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

test "$(nvidia-smi -L | wc -l)" = 8
test "$(git -C "$REPO" rev-parse HEAD)" = 6f2db449a21e1bac750b996f2e27cac6739aa63f
test "$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)" = dcddb970176382c0fcf4521b0c0e6fc15894dfe0
for pair in \
  0004-acp-replay-mask-weighted-sampler.patch:15857546555052fd7e452ef158f6a0c90b4730943d19c37b5338d94f233bdcb6 \
  0005-explicit-checkpoint-steps.patch:e7f927b128bd7e11cac0278d93e0e5932eef3d84b4c893e0e0fabccdab206a39 \
  0006-strict-policy-numerics.patch:05699b0e4d8aca354a348577d50e033bf58ee2264fd16bfa3998e35b6b2c4ed7 \
  0007-direct-pyav-stream-seek.patch:41f3bef2626473e0a39f288923d42a3ed5537ce34259906c12d94ee66a41875d \
  0008-pyav-defer-rgb-conversion.patch:8316913faacb03fcd0545f367c1e7e3aa4854420707d0e53cf4a54103dcdeed2 \
  0009-pi05-uint8-loader-ipc.patch:d431572b19041f48e0aaa5872332b8c8bcedc8846ea753493d12cee553d4f722; do
  file=${pair%%:*}; expected=${pair##*:}
  test "$(sha256sum "$TARGET/scripts/$file" | awk '{print $1}')" = "$expected"
done
grep -Fq 'Non-finite policy loss before backward' "$REPO/src/lerobot/scripts/lerobot_train.py"
grep -Fq 'return_uint8=getattr(cfg.policy, "type", None) == "pi05"' "$REPO/src/lerobot/datasets/factory.py"
grep -Fq 'img = img.to(torch.float32) / 255.0' "$REPO/src/lerobot/policies/pi05/modeling_pi05.py"
git -C "$REPO" diff --check
test -f "$SUPERVISOR" && test -f "$BASE_POLICY/model.safetensors"
test -f "$SOURCE_DATASET/meta/MIXED_REPLAY_COMPLETE.json"
test -f "$DATASET_MANIFEST" && test -f "$ACP_GATE"
test ! -e "$OUTPUT" && test ! -e "$MANIFEST" && test ! -e "$REPORT"
test -z "$(find "$SSD_ARCHIVE" -mindepth 1 -maxdepth 1 -print -quit)"

python - "$SOURCE_DATASET" "$DATASET_MANIFEST" "$ACP_GATE" <<'PY'
import json, sys
from pathlib import Path
root, manifest_path, gate_path = map(Path, sys.argv[1:])
info = json.loads((root / 'meta/info.json').read_text())
marker = json.loads((root / 'meta/MIXED_REPLAY_COMPLETE.json').read_text())
manifest = json.loads(manifest_path.read_text())
gate = json.loads(gate_path.read_text())
assert (info['total_episodes'], info['total_frames']) == (618, 1_600_639)
assert manifest['base_episodes'] == 558 and manifest['hil_train_episodes'] == 60
assert manifest['base_frames'] == 1_255_729 and manifest['hil_train_frames'] == 344_910
assert marker['status'] == 'complete' and all(gate['gates'].values())
links = list((root / 'videos').rglob('*.mp4'))
assert len(links) == 1854 and all(p.is_symlink() and p.resolve(strict=True).is_file() for p in links)
print('MIXED_SOURCE_GATE_OK', {'episodes': info['total_episodes'], 'frames': info['total_frames'], 'video_links': len(links)})
PY

python - "$RAM_ROOT" <<'PY'
import shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
free = shutil.disk_usage('/dev/shm').free
assert free >= 45_000_000_000, free
assert not root.exists(), root
root.mkdir(parents=True)
print('RAM_STAGE_FREE_GATE_OK', free)
PY
stage_start=$(date +%s)
mkdir "$DATASET"
cp -aL "$SOURCE_DATASET"/. "$DATASET"/
stage_seconds=$(($(date +%s)-stage_start))
python - "$SOURCE_DATASET" "$DATASET" "$stage_seconds" <<'PY'
import json, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:3]); seconds = int(sys.argv[3])
for rel in ('meta/info.json','meta/stats.json','meta/tasks.parquet','meta/MIXED_REPLAY_COMPLETE.json'):
    assert (source/rel).read_bytes() == (target/rel).read_bytes(), rel
videos = list((target/'videos').rglob('*.mp4'))
video_bytes = sum(p.stat().st_size for p in videos)
assert len(videos) == 1854 and video_bytes == 27_765_310_647
assert not any(p.is_symlink() for p in videos)
print('MIXED_RAM_STAGE_PASS', {'seconds': seconds, 'video_files': len(videos), 'video_bytes': video_bytes})
PY

python -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$OLD_ROOT/h100_minimal_py312.txt"
"$VENV/bin/python" -m pip uninstall -y transformers
"$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$TRANSFORMERS_FORK"
"$VENV/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" --force-reinstall --no-deps \
  accelerate==1.11.0 datasets==4.1.1 dill==0.4.0 fsspec==2025.9.0 pandas==2.3.3 pyarrow==21.0.0 \
  huggingface-hub==0.35.3 numpy==1.26.4 safetensors==0.7.0 av==15.1.0 draccus==0.10.0
"$VENV/bin/python" -m pip install --no-index --find-links "$EXACT_WHEELHOUSE" --force-reinstall --no-deps tokenizers==0.21.1

export PYTHONPATH="$TRANSFORMERS_FORK/src:$VENV/lib/python3.12/site-packages:$OVERLAY:$REPO/src"
export EVORL_EXPECTED_VENV="$VENV"
export HF_HOME=$TARGET/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$TARGET/cache/huggingface/hub
export TRANSFORMERS_CACHE=$TARGET/cache/huggingface/transformers
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8

"$VENV/bin/python" - <<'PY'
import os
import accelerate, av, datasets, pandas, pyarrow, serial, tokenizers, torch, transformers
import lerobot.policies
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()
actual = {'accelerate': accelerate.__version__, 'datasets': datasets.__version__, 'pandas': pandas.__version__, 'pyarrow': pyarrow.__version__, 'tokenizers': tokenizers.__version__, 'av': av.__version__, 'transformers': transformers.__version__}
expected = {'accelerate':'1.11.0','datasets':'4.1.1','pandas':'2.3.3','pyarrow':'21.0.0','tokenizers':'0.21.1','av':'15.1.0','transformers':'4.53.3'}
assert actual == expected, (actual, expected)
assert accelerate.__file__.startswith(os.environ['EVORL_EXPECTED_VENV']), accelerate.__file__
assert '/env/official_overlay/' in serial.__file__, serial.__file__
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
print('FORMAL_POLICY_IMPORT_GATE_OK', actual, {'accelerate_file': accelerate.__file__, 'serial_file': serial.__file__, 'torch': torch.__version__, 'cuda': torch.version.cuda})
PY

"$VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 --master_port=$PORT "$OLD_ROOT/preflight/nccl_smoke.py"
echo "FORMAL_POLICY_NCCL_GATE_OK ranks=8"

python - "$MANIFEST" "$0" "$JOB" "$MODE" "$DATASET_MANIFEST" "$ACP_GATE" "$BASE_POLICY" "$stage_seconds" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
out, runner = map(Path, sys.argv[1:3]); job, mode = sys.argv[3:5]
dataset_manifest, acp_gate, policy = map(Path, sys.argv[5:8]); stage_seconds = int(sys.argv[8])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
repo = Path('/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902/src/Evo-RL')
payload = {
 'schema':'evorl_formal_policy_run/v1','created_utc':datetime.now(timezone.utc).isoformat(),
 'job':job,'mode':mode,'world_size':8,
 'batch_size_per_rank':8,'global_batch_size':64,'workers_per_rank':4,'default_prefetch_factor':2,'total_prefetch_samples':512,
 'steps':20000,'save_steps':[1000,5000,10000,15000,20000],'seed':20260902,'target_hil_fraction':0.25,'mixed_ram_stage_seconds':stage_seconds,
 'checkpoint_storage':{'final':'gpfs_hdd','intermediate_archive':'gpfs_flash','intermediate_hdd_paths_are_symlinks':True},
 'package_stack':{'accelerate':'1.11.0','datasets':'4.1.1','pandas':'2.3.3','pyarrow':'21.0.0'},
 'evorl_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
 'runner_sha256':sha(runner),'dataset_manifest_sha256':sha(dataset_manifest),'acp_gate_sha256':sha(acp_gate),'initial_model_sha256':sha(policy/'model.safetensors')}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print('FORMAL_POLICY_MANIFEST_WRITTEN', out)
PY

ARGS=(
  --dataset.repo_id=local/piperx-full558-hil-train60-replay-v1
  --dataset.root="$DATASET" --dataset.video_backend=pyav --dataset.use_imagenet_stats=false
  --policy.path="$BASE_POLICY"
  '--rename_map={"observation.images.camera_top":"observation.images.base_0_rgb","observation.images.camera_wrist_left":"observation.images.left_wrist_0_rgb","observation.images.camera_wrist_right":"observation.images.right_wrist_0_rgb"}'
  --batch_size=8 --num_workers=4 --eval_freq=0 --save_checkpoint=true --log_freq=5 --seed=20260902 --wandb.enable=false
  --replay_sampling.enable=true --replay_sampling.source_field="$SOURCE_FIELD" --replay_sampling.hil_value=1
  --replay_sampling.target_hil_fraction=0.25 --replay_sampling.num_samples=1600576
)

MODE_ARGS=(--acp.enable=false)
ARGS+=("${MODE_ARGS[@]}" --steps=20000 --save_freq=0 '--save_steps=[1000,5000,10000,15000,20000]' --output_dir="$OUTPUT" --job_name="$JOB")

echo "FORMAL_POLICY_TRAIN_START name=$JOB mode=$MODE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
train_start=$(date +%s)

offload_checkpoint() {
  local step=$1
  local owner_pid=$2
  local source=$OUTPUT/checkpoints/$step
  local destination=$SSD_ARCHIVE/$step
  local expected_step=$((10#$step))
  while kill -0 "$owner_pid" 2>/dev/null; do
    if [[ -f "$source/pretrained_model/model.safetensors" \
       && -f "$source/training_state/optimizer_state.safetensors" \
       && -f "$source/training_state/training_step.json" \
       && "$(stat -c %s "$source/pretrained_model/model.safetensors")" = 14467170760 \
       && "$(stat -c %s "$source/training_state/optimizer_state.safetensors")" = 25988800348 \
       && "$(python -c "import json;print(json.load(open('$source/training_state/training_step.json'))['step'])")" = "$expected_step" ]]; then
      sleep 10
      test "$(stat -c %s "$source/pretrained_model/model.safetensors")" = 14467170760
      test "$(stat -c %s "$source/training_state/optimizer_state.safetensors")" = 25988800348
      "$VENV/bin/python" - "$source" <<'PY'
import sys
from pathlib import Path
from safetensors import safe_open
root=Path(sys.argv[1])
with safe_open(root/'pretrained_model/model.safetensors',framework='np') as handle: assert len(list(handle.keys())) > 0
with safe_open(root/'training_state/optimizer_state.safetensors',framework='np') as handle: assert len(list(handle.keys())) > 0
PY
      test ! -e "$destination"
      mv -- "$source" "$destination"
      ln -s "$destination" "$source"
      test "$(readlink -f "$source")" = "$destination"
      echo "CHECKPOINT_OFFLOAD_PASS step=$step destination=$destination utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OFFLOAD_LOG"
      return 0
    fi
    sleep 15
  done
  echo "CHECKPOINT_OFFLOAD_FAIL step=$step reason=train_process_exited utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OFFLOAD_LOG"
  return 41
}

"$VENV/bin/python" -m accelerate.commands.launch \
  --multi_gpu --num_processes=8 --num_machines=1 --mixed_precision=bf16 --main_process_port=$((PORT+1)) \
  --module lerobot.scripts.lerobot_train "${ARGS[@]}" &
train_pid=$!
offload_pids=()
for step in 001000 005000 010000 015000; do
  offload_checkpoint "$step" "$train_pid" &
  offload_pids+=("$!")
done
while kill -0 "$train_pid" 2>/dev/null; do
  sleep 60
  kill -0 "$train_pid" 2>/dev/null || break
  set +e
  decision=$("$VENV/bin/python" "$SUPERVISOR" --log "$LOG" --marker FORMAL_POLICY_TRAIN_START --target-steps 20000 --max-projected-seconds 46800 --min-metrics 2)
  decision_rc=$?
  set -e
  echo "FORMAL_POLICY_SUPERVISOR rc=$decision_rc payload=$decision"
  if [[ "$decision_rc" -ne 0 ]]; then
    kill -TERM "$train_pid" 2>/dev/null || true
    sleep 5
    kill -KILL "$train_pid" 2>/dev/null || true
    wait "$train_pid" 2>/dev/null || true
    echo "FORMAL_POLICY_SUPERVISOR_STOP rc=$decision_rc payload=$decision"
    exit "$decision_rc"
  fi
done
wait "$train_pid"
for offload_pid in "${offload_pids[@]}"; do wait "$offload_pid"; done
train_seconds=$(($(date +%s)-train_start))
test "$train_seconds" -le 46800
test "$(readlink "$OUTPUT/checkpoints/last")" = 020000
for step in 001000 005000 010000 015000 020000; do
  test -f "$OUTPUT/checkpoints/$step/pretrained_model/model.safetensors"
  test -f "$OUTPUT/checkpoints/$step/training_state/optimizer_state.safetensors"
  test "$(python -c "import json;print(json.load(open('$OUTPUT/checkpoints/$step/training_state/training_step.json'))['step'])")" = "$((10#$step))"
done
python - "$TARGET/manifests/${JOB}-checkpoint-storage.json" "$OUTPUT" "$SSD_ARCHIVE" <<'PY'
import json, sys
from pathlib import Path
out, output, archive = map(Path, sys.argv[1:])
payload={'schema':'evorl_checkpoint_storage/v1','output':str(output),'ssd_archive':str(archive),'steps':{}}
for step in ('001000','005000','010000','015000','020000'):
    path=output/'checkpoints'/step
    payload['steps'][step]={'is_symlink':path.is_symlink(),'resolved':str(path.resolve(strict=True))}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
sha256sum \
  "$OUTPUT/checkpoints/001000/pretrained_model/model.safetensors" \
  "$OUTPUT/checkpoints/005000/pretrained_model/model.safetensors" \
  "$OUTPUT/checkpoints/010000/pretrained_model/model.safetensors" \
  "$OUTPUT/checkpoints/015000/pretrained_model/model.safetensors" \
  "$OUTPUT/checkpoints/020000/pretrained_model/model.safetensors" \
  > "$TARGET/manifests/${JOB}-checkpoint-sha256.txt"
python - "$REPORT" "$JOB" "$MODE" "$train_seconds" <<'PY'
import sys
from pathlib import Path
report, job, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
seconds = int(sys.argv[4])
report.write_text(
    '# 8×H100 formal 20k data-only policy training\n\n'
    f'- Job: `{job}`\n'
    f'- Mode: `{mode}`\n'
    '- Steps: `20000`\n'
    f'- Wall seconds: `{seconds}`\n'
    '- Checkpoints: `1000`, `5000`, `10000`, `15000`, `20000`\n'
    '- Result: `PASS`\n'
)
PY
echo "FORMAL_POLICY_PASS name=$JOB mode=$MODE seconds=$train_seconds output=$OUTPUT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
