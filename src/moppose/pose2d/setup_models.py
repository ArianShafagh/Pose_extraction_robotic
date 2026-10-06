"""Fetch third-party pose code and weights (run once per computer)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from moppose.pose2d.backends import SAPIENS_CKPT_ROOT, SAPIENS_ROOT

SAPIENS_REPO = "https://github.com/facebookresearch/sapiens2"
SAPIENS_COMMIT = "7e5bae88456ac418ff0e58e74106c9fe192055d4"  # tested version


def setup_sapiens2(size: str = "0.8b") -> Path:
    """Clone Sapiens2 at the tested commit and download the pose checkpoint (not gated)."""
    if not (SAPIENS_ROOT / ".git").exists():
        SAPIENS_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", SAPIENS_REPO, str(SAPIENS_ROOT)], check=True)
    subprocess.run(["git", "-C", str(SAPIENS_ROOT), "fetch", "--quiet", "origin"], check=False)
    subprocess.run(["git", "-C", str(SAPIENS_ROOT), "checkout", "--quiet", SAPIENS_COMMIT], check=True)

    from huggingface_hub import hf_hub_download

    name = f"sapiens2_{size}_pose.safetensors"
    dest = SAPIENS_CKPT_ROOT / "pose"
    dest.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=f"facebook/sapiens2-pose-{size}", filename=name, local_dir=dest)
    return Path(path)


def setup_rtmw() -> None:
    """Download the RTMW-x ONNX model into rtmlib's cache."""
    from moppose.pose2d.backends import RTMWBackend

    RTMWBackend(device="cpu")
