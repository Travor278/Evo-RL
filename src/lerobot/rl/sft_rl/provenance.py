"""Portable source and numerical-stack identities for reproducible continuation."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def source_fingerprint():
    package = Path(__file__).resolve().parent
    library = package.parents[1]
    files = list(package.glob("*.py")) + [
        library / name
        for name in (
            "configs/train.py",
            "policies/pi05/modeling_pi05.py",
            "optim/schedulers.py",
            "scripts/lerobot_train.py",
            "scripts/lerobot_sft_rl.py",
        )
    ]
    hashes = {
        path.relative_to(library).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in sorted(files)
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    repository = library.parent.parent
    commit, dirty = None, None
    if (repository / ".git").exists():
        try:
            commit = (
                subprocess.check_output(
                    ["git", "-C", str(repository), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            dirty = bool(
                subprocess.check_output(
                    ["git", "-C", str(repository), "status", "--porcelain", "--", "src/lerobot"],
                    stderr=subprocess.DEVNULL,
                )
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "sha256": digest,
        "files": hashes,
        "line_endings": "LF-normalized",
        "git_commit": commit,
        "uncommitted_source": dirty,
    }


def runtime_versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "transformers", "numpy", "av", "datasets", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
