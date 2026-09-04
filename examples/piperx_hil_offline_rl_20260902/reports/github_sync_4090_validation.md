# Remote-first source synchronization and 4090 validation

Validation date: 2026-09-05 (Asia/Shanghai).

## Source-of-truth comparison

The source of truth was the Qizhi worktree that produced the completed training
runs:

`/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_hil_rl_piperx_20260902/src/Evo-RL`

Its upstream `HEAD` is
`6f2db449a21e1bac750b996f2e27cac6739aa63f`. The stable patch ID of its tracked
source diff is `56692cb1115d4ee8af174e804ae658b4f051541d`. The patch ID and all 11 changed
source-file Git blob IDs, including the untracked-on-Qizhi
`src/lerobot/rl/replay_sampler.py`, matched this branch exactly.

The 44 recorded configs, runners, reports and patch artifacts in this experiment
bundle were also compared against the live Qizhi files one by one with SHA-256.
All matched. The remote checksum manifest is:

`manifests/github_sync_bundle_sha256.txt`

## 4090 environment gate

The public GitHub branch was cloned afresh on the online RTX 4090 instance and
tested from that clone, rather than from the local Windows checkout.

- GPU: NVIDIA GeForce RTX 4090, 48 GB
- PyTorch: `2.7.1+cu126`
- CUDA runtime reported by PyTorch: `12.6`
- Transformers fork: `4.53.3`
- Accelerate: `1.11.0`
- Datasets: `4.1.1`
- PyArrow: `21.0.0`
- Pandas: `2.3.3`
- PyAV: `15.1.0`
- Tokenizers: `0.21.1`
- Safetensors: `0.7.0`
- Draccus distribution: `0.10.0`

The exact imports, CUDA visibility and GitHub-clone import path passed.

## Tests

- ACP prompt/apply-mask, replay sampler and value-inference tests:
  `22 passed, 2 skipped` in 38.34 seconds. The skips are optional local-dataset
  cases, not failures.
- ACP Pi0.5 prompt pipeline, Pi*0.6 configuration/algorithms, value visualization
  and sampler tests: `18 passed` in 4.17 seconds.
- Python compile, shell syntax, clean Git checkout, diff whitespace (excluding
  byte-identical historical patch payloads), and credential scan: passed.
- Real mixed-replay contract: passed on 618 episodes, 1,600,639 frames, 1,854
  video links, 1,255,729 base frames and 344,910 HIL frames. A deterministic
  64,000-sample draw produced a HIL fraction of `0.246796875` against the target
  `0.25`; ACP masking preserved untagged base prompts.

Primary logs on Qizhi:

- `logs/github_sync_4090_import.log`
- `logs/github_sync_4090_pytest.log`
- `logs/github_sync_4090_secondary_pytest.log`
- `logs/github_sync_4090_static_gates.log`
- `logs/github_sync_4090_mixed_replay_contract.log`
