"""2D pose models behind one interface.

Every backend gets rectified person crops (see virtual_cam.py) plus the tight person
box inside each crop, and returns keypoints in crop pixels with scores. The model
enlarges the box by its own training padding (1.25x for both models here).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from moppose.pose2d.skeleton import COCO_WHOLEBODY_TO_BODY_FEET, index_map

REPO = Path(__file__).resolve().parents[3]
SAPIENS_ROOT = Path(os.environ.get("SAPIENS_ROOT", REPO / "third_party" / "sapiens2"))
SAPIENS_CKPT_ROOT = Path(os.environ.get("SAPIENS_CHECKPOINT_ROOT", REPO / "third_party" / "sapiens2_host"))

# RTMW 384x288 trained on the 14-dataset "cocktail14" mix (rtmlib's 'performance' model).
# Not the cocktail13 RTMW-x release: its foot keypoints are broken (toes land near the head).
RTMW_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip")


class PoseBackend:
    name: str
    native_names: list[str]
    to_body_feet: np.ndarray  # indices into native keypoints for skeleton.BODY_FEET

    def infer(self, crops: list[np.ndarray], boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """crops: list of BGR images; boxes (B, 4). Returns kpts (B, J, 2) crop px, scores (B, J)."""
        raise NotImplementedError

    def normalize_conf(self, scores: np.ndarray) -> np.ndarray:
        """Map the model's native scores to a 0..1 confidence (identity by default)."""
        return scores


def _prepare_cuda_for_onnxruntime() -> None:
    """onnxruntime-gpu needs CUDA/cuDNN DLLs; reuse the ones shipped with PyTorch."""
    try:
        import torch  # noqa: F401  (loads CUDA libs)
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except ImportError:
        pass


class RTMWBackend(PoseBackend):
    """RTMW whole-body (133 COCO-WholeBody keypoints), ONNX via rtmlib."""

    name = "rtmw"

    def __init__(self, device: str = "cuda", onnx_model: str = RTMW_URL):
        _prepare_cuda_for_onnxruntime()
        from rtmlib import RTMPose

        self.model = RTMPose(onnx_model=onnx_model, model_input_size=(288, 384), backend="onnxruntime", device=device)
        self.native_names = [f"cwb_{i}" for i in range(133)]
        self.to_body_feet = COCO_WHOLEBODY_TO_BODY_FEET

    def normalize_conf(self, scores):
        # SimCC scores of this model are unbounded (good joints ~3-8); s/(1+s) keeps the
        # ordering and maps them into 0..1 (1 -> 0.5, 4 -> 0.8).
        s = np.clip(scores, 0, None)
        return s / (1.0 + s)

    def infer(self, crops, boxes):
        kp, sc = [], []
        for img, b in zip(crops, boxes):
            k, s = self.model(img, bboxes=[list(map(float, b))])
            kp.append(np.asarray(k)[0])
            sc.append(np.asarray(s)[0])
        return np.stack(kp), np.stack(sc)


class Sapiens2Backend(PoseBackend):
    """Sapiens2 308-keypoint top-down pose (Meta). fp16 on GPU, optional flip test."""

    name = "sapiens2"

    def __init__(self, size: str = "0.8b", device: str = "cuda:0", fp16: bool = True, flip_test: bool = True,
                 checkpoint: Path | None = None):
        if not SAPIENS_ROOT.exists():
            raise FileNotFoundError(f"Sapiens2 code not found at {SAPIENS_ROOT} - run `uv run moppose setup`")
        sys.path.insert(0, str(SAPIENS_ROOT))
        import torch
        from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
        from sapiens.pose.models import init_model

        self.torch = torch
        cfg_dir = SAPIENS_ROOT / "sapiens" / "pose" / "configs"
        config = cfg_dir / "keypoints308" / "shutterstock_goliath_3po" / \
            f"sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py"
        checkpoint = checkpoint or SAPIENS_CKPT_ROOT / "pose" / f"sapiens2_{size}_pose.safetensors"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Sapiens2 checkpoint missing: {checkpoint} - run `uv run moppose setup`")
        self.model = init_model(str(config), str(checkpoint), device=device)
        self.meta = parse_pose_metainfo(dict(from_file=str(cfg_dir / "_base_" / "keypoints308.py")))
        codec_cfg = dict(self.model.cfg.codec)
        codec_cfg.pop("type")
        self.codec = UDPHeatmap(**codec_cfg)
        self.fp16 = fp16 and device.startswith("cuda")
        if self.fp16:
            self.model.half()
        self.flip_test = flip_test
        id2name = self.meta["keypoint_id2name"]
        self.native_names = [id2name[i] for i in range(len(id2name))]
        self.to_body_feet = index_map(self.native_names)

    def infer(self, crops, boxes):
        torch = self.torch
        inputs, metas = [], []
        for img, b in zip(crops, boxes):
            data = self.model.pipeline(dict(img=img, bbox=np.asarray(b, np.float32)[None],
                                            bbox_score=np.ones(1, np.float32)))
            data = self.model.data_preprocessor(data)
            inputs.append(data["inputs"])
            metas.append(data["data_samples"]["meta"])
        x = torch.cat(inputs, dim=0)
        if self.fp16:
            x = x.half()
        with torch.no_grad():
            pred = self.model(x)
            if self.flip_test:
                pf = self.model(x.flip(-1)).flip(-1)[:, self.meta["flip_indices"]]
                pred = (pred + pf) / 2.0
        pred = pred.float().cpu().numpy()
        kp, sc = [], []
        for i, meta in enumerate(metas):
            k, s = self.codec.decode(pred[i])
            k = k / meta["input_size"] * meta["bbox_scale"] + meta["bbox_center"] - 0.5 * meta["bbox_scale"]
            kp.append(k[0])
            sc.append(s[0])
        return np.stack(kp), np.stack(sc)


def make_backend(name: str, **kw) -> PoseBackend:
    if name == "rtmw":
        return RTMWBackend(**{k: v for k, v in kw.items() if k in ("device", "onnx_model")})
    if name == "sapiens2":
        dev = kw.get("device", "cuda")
        return Sapiens2Backend(size=kw.get("size", "0.8b"), device="cuda:0" if dev == "cuda" else dev,
                               fp16=kw.get("fp16", True), flip_test=kw.get("flip_test", True))
    raise ValueError(f"unknown backend {name!r} (rtmw | sapiens2)")
