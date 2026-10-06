"""Perspective-rectified crops around a person in a fisheye image.

Pose models are trained on ordinary (pinhole) photos. A person near the edge of a
fisheye image is bent and squeezed, which hurts keypoint accuracy. Instead of
feeding the raw frame, we aim a *virtual pinhole camera* (same optical centre,
rotated) at the person and render what it would see, using the calibrated lens
model. Keypoints found in that crop are mapped back to rays and to raw pixels, so
nothing downstream needs to know about the crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from moppose.calib.camera import Camera


@dataclass
class VirtualView:
    R: np.ndarray            # (3, 3) real-camera frame -> virtual-camera frame
    K: np.ndarray            # (3, 3) virtual pinhole intrinsics
    size: tuple[int, int]    # (width, height) of the crop
    bbox: np.ndarray         # (4,) tight person box x1, y1, x2, y2 in crop pixels

    def crop_to_rays(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) crop pixels -> (N, 3) unit rays in the real camera frame."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        p = np.c_[(uv[:, 0] - self.K[0, 2]) / self.K[0, 0], (uv[:, 1] - self.K[1, 2]) / self.K[1, 1], np.ones(len(uv))]
        r = p @ self.R  # R^T p, row-wise
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def crop_to_raw(self, cam: Camera, uv: np.ndarray) -> np.ndarray:
        """(N, 2) crop pixels -> (N, 2) raw distorted pixels of the real camera."""
        return cam.project_cam(self.crop_to_rays(uv))

    def raw_to_crop(self, cam: Camera, pts: np.ndarray) -> np.ndarray:
        """(N, 2) raw distorted pixels -> (N, 2) crop pixels."""
        p = cam.rays(pts) @ self.R.T
        return np.c_[p[:, 0] / p[:, 2] * self.K[0, 0] + self.K[0, 2], p[:, 1] / p[:, 2] * self.K[1, 1] + self.K[1, 2]]

    def maps(self, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
        """cv2.remap tables: for every crop pixel, where to sample the raw image."""
        w, h = self.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        raw = self.crop_to_raw(cam, np.c_[u.ravel(), v.ravel()])
        return raw[:, 0].reshape(h, w).astype(np.float32), raw[:, 1].reshape(h, w).astype(np.float32)

    def render(self, cam: Camera, image: np.ndarray) -> np.ndarray:
        mx, my = self.maps(cam)
        return cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _outline(bbox: np.ndarray, n: int = 9) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    t = np.linspace(0.0, 1.0, n)
    xs, ys = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
    return np.concatenate([
        np.c_[xs, np.full(n, y1)], np.c_[xs, np.full(n, y2)],
        np.c_[np.full(n, x1), ys], np.c_[np.full(n, x2), ys],
    ])


def _look_rotation(direction: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation whose z axis is `direction` and whose -y axis is as close to `up` as possible."""
    z = direction / np.linalg.norm(direction)
    down = -(up - (up @ z) * z)
    if np.linalg.norm(down) < 1e-6:  # looking straight along `up`: pick any perpendicular
        down = np.cross(z, [1.0, 0.0, 0.0])
    y = down / np.linalg.norm(down)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def make_view(
    cam: Camera,
    bbox_raw: np.ndarray,
    size: tuple[int, int] = (768, 1024),
    margin: float = 1.6,
    up: np.ndarray = np.array([0.0, -1.0, 0.0]),
) -> VirtualView:
    """Virtual pinhole camera framing the person in `bbox_raw` (x1, y1, x2, y2 raw pixels).

    margin : crop size relative to the person extent. The pose models enlarge the person
             box by 1.25x themselves, so the crop must be larger than that.
    up     : direction treated as "up" in the crop, in real-camera coordinates
             (default: image up; use world up once extrinsics are known).
    """
    rays = cam.rays(_outline(np.asarray(bbox_raw, np.float64)))
    d = rays.mean(axis=0)
    R = _look_rotation(d, up)
    # re-centre on the angular midpoint of the outline (the mean ray is biased by distortion)
    for _ in range(2):
        p = rays @ R.T
        xy = p[:, :2] / p[:, 2:3]
        mid = (xy.min(0) + xy.max(0)) / 2
        R = _look_rotation(R.T @ np.array([mid[0], mid[1], 1.0]), up)
    p = rays @ R.T
    xy = p[:, :2] / p[:, 2:3]
    half = np.abs(xy).max(axis=0)  # half extent (tan) along x and y
    w, h = size
    f = min(w / 2 / (half[0] * margin), h / 2 / (half[1] * margin))
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    uv = xy * f + K[:2, 2]
    bbox = np.r_[uv.min(axis=0), uv.max(axis=0)]
    return VirtualView(R=R, K=K, size=(w, h), bbox=bbox)
