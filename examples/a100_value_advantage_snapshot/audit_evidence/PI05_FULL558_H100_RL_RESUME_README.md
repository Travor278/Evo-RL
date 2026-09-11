# PI05 PiperX full558 H100 experiment and RL-resume record

This directory archives the preparation, validation, and inference-comparison evidence for the full558 H100 run. Keep this document, the referenced SSD assets, and the complete step-50000 training state before starting RL continuation.

## Provenance

- Training task: `v2sam-exo2ego-fusion-official24-resume-e8-20260827-v2-cuda132-r13`
- Task ID: `job-36da1646-ef45-4821-bb8c-f0263b05f65c`
- LeRobot: customized v0.6.0
- Source commit: `eb74530ea345c942212820952e46b4d186f9a9f0`
- Container image: `ngc-pytorch:25.02-cuda12.8.0-py3`
- Compute: one node, 8 x H100
- Precision: FP32, full-parameter training, AMP disabled
- Topology: 8 ranks x batch 8 x accumulation 1 = global batch 64
- Seed: 1000
- Steps: 50000
- Save frequency: 18000
- Optimizer LR: 2.5e-5
- Scheduler: warmup 1200, decay steps 50000, final LR 2.5e-6
- RGB inputs: top/base, left wrist, right wrist; model input 224 x 224
- Video backend: PyAV

## Immutable data and model inputs

- SSD root: `/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/pi05_full558_044`
- Dataset: `dataset_repo_d2f706127c3f`
- Dataset commit: `d2f706127c3fb91044627908b599c42c3177c050`
- PiperX LFS validation: 2234/2234 present, missing=0, pointer=0
- Base model: `pi05_base_b211f3d44c36`
- Base commit: `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`
- Base `model.safetensors`: 14467165872 bytes
- Base SHA256/LFS OID: `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`
- Tokenizer revision: `35e4f46485b4d07967e7e9935bc3786aad50687c`
- `tokenizer.json` SHA256: `ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563`
- `tokenizer.model` SHA256: `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`

## Training outputs to preserve

- Model payload root: `model_payloads/pi05_piperx_full558_h100_8gpu_50k_20260828_r13`
- Training-state root: `train_state/pi05_piperx_full558_h100_8gpu_50k_20260828_r13`
- Run logs/metrics: `runs/pi05_piperx_full558_h100_8gpu_50k_20260828_r13`
- Launcher: `launch_full558_h100_8gpu_50k.sh`
- LeRobot source snapshot: `lerobot_v060_eb74530`
- Offline wheelhouses: `wheelhouse_v060_py312_ngc2502_locked` and `wheelhouse_py312_ngc2502_locked`
- Tokenizer snapshot: `tokenizers/google--paligemma-3b-pt-224--35e4f46485b4d07967e7e9935bc3786aad50687c`

The complete last checkpoint must include the FP32 model, optimizer, scheduler, RNG state, training step, and topology metadata. Before RL continuation, revalidate `last`, `best`, and `best.json`; do not infer the selected best checkpoint from directory names.

Known step-36000 validation: model 16573760032 bytes, optimizer 25988800372 bytes, training_step=36000, scheduler last_epoch=36000, 8 ranks x batch 8 x accumulation 1. The final step-50000 model SHA256 is `d85c7cd84060a924b6ef10d055491c500c5714a21ddbb49e1dbdc828b7a74147`.

## Evaluation and deployment caveat

- Fixed 50-frame offline MAE originally reported for step 50000: `2.281647205352783`, all finite.
- That historical score used implicit CUDA noise and is not a deterministic cross-GPU comparison.
- With an explicit identical CPU noise tensor, step 50000 reproduced exactly between H100 and `.166` only under the paired stack: Torch 2.7.1+cu126, CUDA 12.6, cuDNN 90501, torchvision 0.22.1+cu126, transformers 5.3.0, NumPy 1.26.4.
- The old `.166` Torch 2.11/CUDA 12.8 stack produced materially different actions. Preserve numerical-stack compatibility for RL evaluation.
- Raw RGB, normalized state, tokenizer IDs, attention masks, and all three internal 224 x 224 image tensors were byte-identical across the paired evaluation environments; image resizing/preprocessing was not the cause of the prior deployment regression.

## RL continuation checklist

1. Re-run source, dataset, base-model, and tokenizer hash checks.
2. Verify the step-50000 model and every continuation-state component is readable and mutually consistent.
3. Record the RL algorithm, reward implementation, rollout dataset/version, random seeds, optimizer changes, and base checkpoint in a new immutable run manifest.
4. Use a saved deterministic noise bank and multiple seeds for offline A/B gates.
5. Never overwrite the supervised step-50000 model/state; write RL outputs to a new run name and directory.
6. Do not trigger physical robot control, public release, or deployment merely from an offline MAE result.

The archived `reports/`, `logs/`, and `probes/` subdirectories contain the original preparation evidence that was previously scattered in the HDD root.
