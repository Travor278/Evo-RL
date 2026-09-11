#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-/root/evorl-piperx-intervention}"
code_dir="$project_root/code/lerobot-v0.6.1-official"
venv_dir="$project_root/venv-py312-torch210-cu128"

test -f "$code_dir/pyproject.toml"
python3 -m venv --without-pip "$venv_dir"
PYTHONPATH=/usr/share/python-wheels/pip-24.0-py3-none-any.whl "$venv_dir/bin/python" -m pip install pip==26.0.1

python_bin="$venv_dir/bin/python"
pip_bin="$venv_dir/bin/pip"
export PIP_NO_CACHE_DIR=1

"$pip_bin" install --upgrade pip wheel
"$pip_bin" install setuptools==81.0.0
"$pip_bin" install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0

"$pip_bin" install \
  accelerate==1.13.0 \
  av==15.1.0 \
  cmake==4.1.2 \
  datasets==4.8.5 \
  draccus==0.11.6 \
  einops==0.8.2 \
  gymnasium==1.2.3 \
  huggingface-hub==1.16.1 \
  jsonlines==4.0.0 \
  numpy==2.2.6 \
  opencv-python-headless==4.13.0.92 \
  packaging==25.0 \
  pandas==2.3.3 \
  pillow==12.2.0 \
  pyarrow==24.0.0 \
  requests==2.32.5 \
  safetensors==0.7.0 \
  scipy==1.17.1 \
  sentencepiece==0.2.1 \
  termcolor==3.3.0 \
  torchcodec==0.10.0 \
  tqdm==4.67.3 \
  transformers==5.3.0 \
  wandb==0.24.2

"$pip_bin" install --no-deps --editable "$code_dir"

"$python_bin" - <<'PY'
import importlib.metadata as metadata
import torch
import lerobot
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

expected = {
    "lerobot": "0.6.1",
    "torch": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "transformers": "5.3.0",
    "accelerate": "1.13.0",
    "datasets": "4.8.5",
    "pyarrow": "24.0.0",
    "av": "15.1.0",
}
actual = {name: metadata.version(name) for name in expected}
for name, version in actual.items():
    print(f"{name}=={version}")
for name, version in expected.items():
    if actual[name] != version:
        raise SystemExit(f"version mismatch for {name}: expected {version}, got {actual[name]}")
print(f"torch.version.cuda={torch.version.cuda}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected CUDA runtime 12.8, got {torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"pi05.chunk_size={PI05Config().chunk_size}")
print("official LeRobot/Pi0.5 import smoke test: OK")
PY

"$pip_bin" check
