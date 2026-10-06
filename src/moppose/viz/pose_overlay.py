"""Draw 2D skeletons on frames."""

from __future__ import annotations

import cv2
import numpy as np

COLORS = {"rtmw": (0, 200, 255), "sapiens2": (255, 80, 255)}


def draw_skeleton(img: np.ndarray, kpts: np.ndarray, conf: np.ndarray, edges: np.ndarray,
                  color=(0, 255, 0), thr: float = 0.3, label: str | None = None) -> np.ndarray:
    out = img
    ok = (conf >= thr) & np.all(np.isfinite(kpts), axis=1)
    for a, b in edges:
        if ok[a] and ok[b]:
            cv2.line(out, tuple(np.round(kpts[a]).astype(int)), tuple(np.round(kpts[b]).astype(int)), color, 2, cv2.LINE_AA)
    for (x, y), good in zip(kpts, ok):
        if good:
            cv2.circle(out, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
    if label and ok.any():
        x, y = kpts[ok].min(axis=0)
        cv2.putText(out, label, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def zoom_on(img: np.ndarray, pts: np.ndarray, size: int = 480, pad: float = 0.35) -> np.ndarray:
    """Square crop around the given points (for preview grids)."""
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    h, w = img.shape[:2]
    if len(pts) == 0:
        return cv2.resize(img, (size, size))
    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
    s = max(x2 - x1, y2 - y1) * (1 + 2 * pad) + 20
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    c, d = int(min(cx + s / 2, w)), int(min(cy + s / 2, h))
    return cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA)
