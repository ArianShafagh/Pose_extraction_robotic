"""Mapping calibration-video frames onto the pixel grid of the scene recordings.

The ChArUco videos may come from a different export path than the scene
recordings (e.g. a 1920x1080 screen/app export with black side bars, while the
camera stream itself is 1280x1024). A calibration is only valid for one pixel
grid, so calibration frames are cropped to the active picture area and resized
to the scene resolution *before* detection. Everything downstream then lives in
scene-video pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from moppose.io.video import open_video, read_frame_at


@dataclass(frozen=True)
class FrameMap:
    crop: tuple[int, int, int, int]   # x, y, w, h of the picture inside the source frame
    out_size: tuple[int, int]         # (width, height) of the scene recordings

    @property
    def is_identity(self) -> bool:
        x, y, w, h = self.crop
        return x == 0 and y == 0 and (w, h) == self.out_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        x, y, w, h = self.crop
        img = image[y:y + h, x:x + w]
        if (w, h) != self.out_size:
            img = cv2.resize(img, self.out_size, interpolation=cv2.INTER_AREA)
        return img

    def to_dict(self) -> dict:
        return {"crop": list(self.crop), "out_size": list(self.out_size)}


def video_size(path) -> tuple[int, int]:
    cap = open_video(path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return size


def active_area(path, n: int = 8, thresh: float = 20.0) -> tuple[int, int, int, int]:
    """Bounding box of the non-black picture area, using the max over n frames."""
    cap = open_video(path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    acc = None
    for k in range(n):
        g = cv2.cvtColor(read_frame_at(path, dur * (k + 0.5) / n).image, cv2.COLOR_BGR2GRAY)
        acc = g if acc is None else np.maximum(acc, g)
    cols = np.flatnonzero(acc.max(0) > thresh)
    rows = np.flatnonzero(acc.max(1) > thresh)
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def auto_map(calib_video, scene_video) -> FrameMap:
    """Crop the black bars of the calibration video and scale it to the scene resolution."""
    out = video_size(scene_video)
    if video_size(calib_video) == out:
        return FrameMap((0, 0, *out), out)
    x, y, w, h = active_area(calib_video)
    if abs((w / h) / (out[0] / out[1]) - 1) > 0.01:
        raise ValueError(
            f"picture area {w}x{h} of {calib_video} has a different aspect ratio than the scene "
            f"video {out[0]}x{out[1]} - the export is cropped or stretched; set calib_crop manually"
        )
    return FrameMap((x, y, w, h), out)


def check_map(fm: FrameMap, calib_video, scene_video, t_calib: float = 1.0, t_scene: float = 1.0) -> dict:
    """Verify the mapping on the static background with SIFT matches.

    Fits an affine transform mapped-calib -> scene; for a correct map it is the
    identity. Returns how far the image corners move under it (px).
    """
    a = cv2.cvtColor(fm.apply(read_frame_at(calib_video, t_calib).image), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(read_frame_at(scene_video, t_scene).image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return {"ok": False, "reason": "no features"}
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
    if len(good) < 20:
        return {"ok": False, "reason": f"only {len(good)} matches"}
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffine2D(pa, pb, ransacReprojThreshold=1.5, maxIters=5000, confidence=0.999)
    w, h = fm.out_size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    shift = np.linalg.norm(corners @ A[:, :2].T + A[:, 2] - corners, axis=1)
    return {"ok": True, "inliers": int(inl.sum()), "matches": len(good),
            "corner_shift_px": float(shift.max()), "affine": A.round(5).tolist()}
