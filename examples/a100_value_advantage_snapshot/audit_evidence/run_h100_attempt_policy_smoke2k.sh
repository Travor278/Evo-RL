#!/usr/bin/env bash
set -euo pipefail

MODE=${POLICY_MODE:?POLICY_MODE must be acp or bc}
if [[ "$MODE" != acp && "$MODE" != bc ]]; then
  echo "POLICY_MODE must be acp or bc" >&2
  exit 64
fi
HIL_FRACTION=${POLICY_HIL_FRACTION:?POLICY_HIL_FRACTION is required}
if [[ "$HIL_FRACTION" != 0.25 && "$HIL_FRACTION" != 0.50 ]]; then
  echo "POLICY_HIL_FRACTION must be 0.25 or 0.50" >&2
  exit 64
fi
RUN_ID=${POLICY_RUN_ID:?POLICY_RUN_ID is required}
PLATFORM_TASK=${POLICY_PLATFORM_TASK:?POLICY_PLATFORM_TASK is required}
EXPECTED_COMMIT=${POLICY_EXPECTED_COMMIT:?POLICY_EXPECTED_COMMIT is required}

EXP=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906
SSD_EXP=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906
REPO="$EXP/src/Evo-RL"
OLD_TARGET=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902
TRANSFORMERS_FORK="$OLD_TARGET/src/transformers_fix_lerobot_openpi"
OVERLAY="$OLD_TARGET/env/official_overlay"
OLD_ROOT=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/pi05_full558_044
WHEELHOUSE="$OLD_ROOT/wheelhouse_v060_py312_ngc2502_locked"
EXACT_WHEELHOUSE="$OLD_TARGET/env/h100_py312_exact_wheels"
SOURCE_DATASET=/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906/mixed_policy_replay_attempt_v2_train
BASE_POLICY="$OLD_TARGET/checkpoints/pi05_base_step50000_policy_compat_6f2db_v2"
JOB="v2sam-ego2exo-attempt-v2-${RUN_ID}"
OUTPUT="$EXP/checkpoints/$JOB"
HDD_CHECKPOINT_LINK="$EXP/checkpoints/$JOB"
LOG="$EXP/logs/$JOB.log"
MANIFEST="$EXP/manifests/$JOB-run.json"
SELECTION_REPORT="$EXP/reports/policy_ratio_smoke_selection_v2.json"
RAM_ROOT="/dev/shm/$JOB"
DATASET="$RAM_ROOT/mixed_replay"
VENV="/tmp/$JOB-py312"
PORT=29650
STEPS=${POLICY_STEPS:-2000}
case "$STEPS" in
  2000)
    RUN_KIND=smoke2k
    SAVE_STEPS='[1000,2000]'
    LAST_STEP=002000
    CHECKPOINT_STEPS=(001000 002000)
    START_MARKER=POLICY_SMOKE_START
    PASS_MARKER=POLICY_SMOKE_PASS
    ;;
  20000)
    RUN_KIND=formal20k
    SAVE_STEPS='[1000,5000,10000,15000,20000]'
    LAST_STEP=020000
    CHECKPOINT_STEPS=(001000 005000 010000 015000 020000)
    START_MARKER=POLICY_FORMAL20K_START
    PASS_MARKER=POLICY_FORMAL20K_PASS
    OUTPUT="$SSD_EXP/formal_checkpoints/$JOB"
    ;;
  *)
    echo "POLICY_STEPS must be 2000 or 20000" >&2
    exit 64
    ;;
esac

mkdir -p "$EXP/logs" "$EXP/manifests" "$EXP/reports" "$EXP/checkpoints"
if [[ "$RUN_KIND" == formal20k ]]; then
  mkdir -p "$SSD_EXP/formal_checkpoints"
fi
exec > >(tee -a "$LOG") 2>&1
echo "$START_MARKER job=$JOB mode=$MODE ratio=$HIL_FRACTION steps=$STEPS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

gpu_count=$(nvidia-smi -L | wc -l)
repo_head=$(git -C "$REPO" rev-parse HEAD)
repo_dirty=$(git -C "$REPO" status --porcelain | wc -l)
transformers_head=$(git -C "$TRANSFORMERS_FORK" rev-parse HEAD)
printf 'POLICY_PREFLIGHT gpu_count=%s repo_head=%s repo_dirty=%s transformers_head=%s source_marker=%s base_model=%s output_exists=%s manifest_exists=%s ram_exists=%s\n' \
  "$gpu_count" "$repo_head" "$repo_dirty" "$transformers_head" \
  "$([ -f "$SOURCE_DATASET/meta/MIXED_REPLAY_COMPLETE.json" ] && printf yes || printf no)" \
  "$([ -f "$BASE_POLICY/model.safetensors" ] && printf yes || printf no)" \
  "$([ -e "$OUTPUT" ] && printf yes || printf no)" \
  "$([ -e "$MANIFEST" ] && printf yes || printf no)" \
  "$([ -e "$RAM_ROOT" ] && printf yes || printf no)"
test "$gpu_count" = 8
test "$repo_head" = "$EXPECTED_COMMIT"
test "$repo_dirty" = 0
test "$transformers_head" = dcddb970176382c0fcf4521b0c0e6fc15894dfe0
test -f "$SOURCE_DATASET/meta/MIXED_REPLAY_COMPLETE.json"
test -f "$BASE_POLICY/model.safetensors"
if [[ "$RUN_KIND" == formal20k ]]; then
  test -f "$SELECTION_REPORT"
  python - "$SELECTION_REPORT" "$HIL_FRACTION" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["held_out_test_used_for_selection"] is False
assert float(p["selected_attempt_fraction"])==float(sys.argv[2])==0.50
print("POLICY_SELECTION_GATE_OK",p["validation_anchor_sha256"])
PY
fi
test ! -e "$OUTPUT"
if [[ "$RUN_KIND" == formal20k ]]; then
  test ! -e "$HDD_CHECKPOINT_LINK"
  formal_free=$(df -B1 --output=avail "$SSD_EXP/formal_checkpoints" | tail -n 1)
  test "$formal_free" -gt 250000000000
  echo "POLICY_CHECKPOINT_STORAGE_GATE_OK medium=ssd_staging free_bytes=$formal_free output=$OUTPUT"
fi
test ! -e "$MANIFEST"
test ! -e "$RAM_ROOT"

python - "$SOURCE_DATASET" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
info=json.loads((root/"meta/info.json").read_text())
marker=json.loads((root/"meta/MIXED_REPLAY_COMPLETE.json").read_text())
assert (info["total_episodes"],info["total_frames"])==(944,1565209)
assert marker["status"]=="complete"
assert info["replay_contract"]["attempt_train_episodes"]==386
links=list((root/"videos").rglob("*.mp4"))
assert len(links)==2832 and all(p.is_symlink() and p.resolve(strict=True).is_file() for p in links)
print("MIXED_SOURCE_GATE_OK",{"episodes":944,"frames":1565209,"video_links":len(links)})
PY

free=$(df -B1 --output=avail /dev/shm | tail -n 1)
source_bytes=$(du -sbL "$SOURCE_DATASET" | cut -f1)
test "$free" -gt $((source_bytes + 20000000000))
mkdir -p "$RAM_ROOT" "$DATASET"
stage_start=$(date +%s)
cp -aL "$SOURCE_DATASET"/. "$DATASET"/
stage_seconds=$(($(date +%s)-stage_start))
python - "$SOURCE_DATASET" "$DATASET" <<'PY'
import sys
from pathlib import Path
source,target=map(Path,sys.argv[1:])
for rel in ("meta/info.json","meta/stats.json","meta/tasks.parquet","meta/MIXED_REPLAY_COMPLETE.json"):
    assert (source/rel).read_bytes()==(target/rel).read_bytes(),rel
videos=list((target/"videos").rglob("*.mp4"))
assert len(videos)==2832 and not any(p.is_symlink() for p in videos)
print("MIXED_RAM_STAGE_PASS",{"video_files":len(videos)})
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
export HF_HOME="$OLD_TARGET/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$OLD_TARGET/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$OLD_TARGET/cache/huggingface/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8

"$VENV/bin/python" - <<'PY'
import os
import accelerate,av,datasets,pandas,pyarrow,torch,transformers
assert accelerate.__version__=="1.11.0"
assert datasets.__version__=="4.1.1"
assert av.__version__=="15.1.0"
assert transformers.__version__=="4.53.3"
assert accelerate.__file__.startswith(os.environ["EVORL_EXPECTED_VENV"])
assert torch.cuda.is_available() and torch.cuda.device_count()==8
print("POLICY_IMPORT_GATE_OK",torch.__version__,torch.version.cuda)
PY
"$VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 --master_port="$PORT" "$OLD_ROOT/preflight/nccl_smoke.py"

python - "$MANIFEST" "$SOURCE_DATASET" "$BASE_POLICY" "$MODE" "$HIL_FRACTION" "$stage_seconds" "$STEPS" "$RUN_KIND" "$SELECTION_REPORT" <<'PY'
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
out,dataset,policy=map(Path,sys.argv[1:4]);mode=sys.argv[4];ratio=float(sys.argv[5]);stage=int(sys.argv[6]);steps=int(sys.argv[7]);run_kind=sys.argv[8];selection=Path(sys.argv[9])
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
repo=Path("/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906/src/Evo-RL")
payload={"schema":"attempt-aware-policy-run/v3","created_utc":datetime.now(timezone.utc).isoformat(),"run_kind":run_kind,"mode":mode,"target_attempt_fraction":ratio,"steps":steps,"world_size":8,"batch_size_per_rank":8,"global_batch_size":64,"seed":20260906,"mixed_ram_stage_seconds":stage,"evorl_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),"dataset_manifest_sha256":sha(dataset/"meta/replay_manifest.json"),"initial_model_sha256":sha(policy/"model.safetensors"),"validation_selection_sha256":sha(selection) if run_kind=="formal20k" else None,"checkpoint_storage":"ssd_staging" if run_kind=="formal20k" else "hdd","sampler":"attempt_balanced","acp_dropout":0.30 if mode=="acp" else None}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

ARGS=(
  --dataset.repo_id=local/piperx-attempt-aware-replay-v2
  --dataset.root="$DATASET" --dataset.video_backend=pyav --dataset.use_imagenet_stats=false
  --policy.path="$BASE_POLICY"
  '--rename_map={"observation.images.camera_top":"observation.images.base_0_rgb","observation.images.camera_wrist_left":"observation.images.left_wrist_0_rgb","observation.images.camera_wrist_right":"observation.images.right_wrist_0_rgb"}'
  --batch_size=8 --num_workers=4 --eval_freq=0 --save_checkpoint=true --save_freq=0
  --save_steps="$SAVE_STEPS" --steps="$STEPS" --log_freq=5 --seed=20260906 --wandb.enable=false
  --replay_sampling.enable=true --replay_sampling.strategy=attempt_balanced
  --replay_sampling.source_field=replay_source --replay_sampling.hil_value=1
  --replay_sampling.target_hil_fraction="$HIL_FRACTION"
  --replay_sampling.attempt_id_field=logical_attempt_id
  --replay_sampling.valid_chunk_field=logical_action_chunk_valid_50
  --output_dir="$OUTPUT" --job_name="$JOB"
)
if [[ "$MODE" == acp ]]; then
  ARGS+=(--acp.enable=true --acp.indicator_field=acp_indicator_attempt_v2 --acp.apply_mask_field=acp_apply_mask --acp.indicator_dropout_prob=0.30)
else
  ARGS+=(--acp.enable=false)
fi

echo "POLICY_TRAIN_START job=$JOB mode=$MODE ratio=$HIL_FRACTION utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
start=$(date +%s)
"$VENV/bin/python" -m accelerate.commands.launch \
  --multi_gpu --num_processes=8 --num_machines=1 --mixed_precision=bf16 --main_process_port=$((PORT+1)) \
  --module lerobot.scripts.lerobot_train "${ARGS[@]}"
seconds=$(($(date +%s)-start))
test "$(readlink "$OUTPUT/checkpoints/last")" = "$LAST_STEP"
for step in "${CHECKPOINT_STEPS[@]}"; do
  test -f "$OUTPUT/checkpoints/$step/pretrained_model/model.safetensors"
  test -f "$OUTPUT/checkpoints/$step/training_state/optimizer_state.safetensors"
done
for step in "${CHECKPOINT_STEPS[@]}"; do
  sha256sum "$OUTPUT/checkpoints/$step/pretrained_model/model.safetensors"
done > "$EXP/manifests/$JOB-checkpoint-sha256.txt"
if [[ "$RUN_KIND" == formal20k ]]; then
  ln -s "$OUTPUT" "$HDD_CHECKPOINT_LINK"
fi
if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|CUDA error|NCCL error|Traceback' "$LOG"; then
  echo "POLICY_FAILURE_PATTERN_FOUND" >&2
  exit 2
fi
echo "$PASS_MARKER job=$JOB mode=$MODE ratio=$HIL_FRACTION steps=$STEPS seconds=$seconds utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
