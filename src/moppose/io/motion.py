"""Where in the image things move: the region a calibration actually has to be good for."""

from __future__ import annotations

import cv2
import numpy as np

from moppose.io.video import iter_frames, open_video


def motion_heatmap(path, n_frames: int = 300, warmup: int = 20) -> np.ndarray:
    """(H, W) count of frames in which each pixel was foreground (MOG2 background subtraction).

    Samples `n_frames` frames spread over the whole video.
    """
    cap = open_video(path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)
    kernel = np.ones((5, 5), np.uint8)
    heat = np.zeros((h, w), np.float64)
    for i, fr in enumerate(iter_frames(path, every_s=max(dur / n_frames, 0.0))):
        fg = mog.apply(cv2.GaussianBlur(fr.image, (5, 5), 0))
        if i >= warmup:
            heat += cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel) > 0
    return heat


def work_area_points(heat: np.ndarray, rel_thresh: float = 0.02, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (x, y) with meaningful motion and their weights, subsampled every `step` px."""
    sub = heat[::step, ::step]
    ys, xs = np.nonzero(sub > rel_thresh * max(sub.max(), 1e-9))
    return np.c_[xs * step, ys * step].astype(np.float32), sub[ys, xs]


def overlay(image: np.ndarray, heat: np.ndarray, hull: np.ndarray | None) -> np.ndarray:
    """Motion heatmap over a frame, with the board-covered area outlined in white."""
    m = (np.clip(heat / max(heat.max(), 1e-9) * 4, 0, 1) * 255).astype(np.uint8)
    out = cv2.addWeighted(image, 0.5, cv2.applyColorMap(m, cv2.COLORMAP_JET), 0.5, 0)
    if hull is not None:
        cv2.polylines(out, [hull.astype(np.int32)], True, (255, 255, 255), 3)
    return out
