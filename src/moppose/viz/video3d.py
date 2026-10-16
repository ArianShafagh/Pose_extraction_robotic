"""Full-length 3D skeleton video, rendered with OpenCV (fast: the whole recording in minutes).

A virtual perspective camera looks at the scene; drawn per frame: floor grid (1 m), the real
cameras, the pelvis trail of the last seconds, the skeleton's floor shadow and the skeleton
(left side red, right side blue).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

LEFT, RIGHT, MID = (40, 40, 220), (220, 120, 30), (90, 90, 90)  # BGR


class ViewCamera:
    def __init__(self, center: np.ndarray, azimuth_deg: float, elevation_deg: float, distance: float,
                 size: tuple[int, int], fov_deg: float = 50.0):
        self.size = size
        self.f = size[0] / 2 / np.tan(np.deg2rad(fov_deg) / 2)
        self.set_view(center, azimuth_deg, elevation_deg, distance)

    def set_view(self, center, azimuth_deg, elevation_deg, distance):
        az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self.C = center + distance * d
        z = center - self.C
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.R = np.stack([x, y, z])

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(..., 3) world -> (..., 2) pixels and a mask of points in front of the viewer."""
        P = (X - self.C) @ self.R.T
        ok = P[..., 2] > 0.1
        z = np.where(ok, P[..., 2], 1.0)
        uv = np.stack([self.f * P[..., 0] / z + self.size[0] / 2, self.f * P[..., 1] / z + self.size[1] / 2], -1)
        return uv, ok & np.all(np.isfinite(X), axis=-1)


def _line(img, cam: ViewCamera, a, b, color, th=1):
    (pa, pb), ok = cam.project(np.stack([a, b]))
    if ok.all():
        cv2.line(img, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), color, th, cv2.LINE_AA)


def _edge_color(names, a, b):
    s = names[a] + names[b]
    return LEFT if "left" in s and "right" not in s else RIGHT if "right" in s and "left" not in s else MID


def render_full_video(X: np.ndarray, t: np.ndarray, edges: np.ndarray, names: list[str], path: Path,
                      cams=None, title: str = "", width: int = 1280, height: int = 720, view: str = "fixed",
                      azimuth_deg: float = -60, elevation_deg: float = 25, distance_m: float = 6.5,
                      trail_s: float = 5.0, show_cameras: bool = True, orbit_deg_per_s: float = 6.0, **_) -> dict:
    """Render every sample of X (N, J, 3) [metres, z up] to an mp4 at the data rate."""
    size = (int(width), int(height))
    fps = float(1.0 / np.median(np.diff(t)))
    pel = np.nanmean(X[:, [names.index("left_hip"), names.index("right_hip")]], axis=1)
    valid = np.all(np.isfinite(pel), axis=1)
    ctr = np.r_[np.nanmedian(pel[valid, :2], axis=0), 0.9]
    # floor grid extent: the walked area (+ cameras) with a margin, whole metres
    pts = [pel[valid, :2]]
    if show_cameras and cams:
        pts.append(np.array([c.center_world[:2] for c in cams]))
    allp = np.concatenate(pts)
    lo, hi = np.floor(allp.min(0) - 1.0), np.ceil(allp.max(0) + 1.0)
    vc = ViewCamera(ctr, azimuth_deg, elevation_deg, distance_m, size)
    trail_n = int(round(trail_s * fps))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not wr.isOpened():
        raise IOError(f"cannot write video {path}")
    for i in tqdm(range(len(t)), desc="3D video", unit="f"):
        if view == "orbit":
            vc.set_view(ctr, azimuth_deg + orbit_deg_per_s * (t[i] - t[0]), elevation_deg, distance_m, )
        img = np.full((size[1], size[0], 3), 245, np.uint8)
        for x in np.arange(lo[0], hi[0] + 1e-9):          # floor grid, 1 m
            _line(img, vc, np.r_[x, lo[1], 0], np.r_[x, hi[1], 0], (210, 210, 210))
        for y in np.arange(lo[1], hi[1] + 1e-9):
            _line(img, vc, np.r_[lo[0], y, 0], np.r_[hi[0], y, 0], (210, 210, 210))
        _line(img, vc, np.zeros(3), np.r_[1, 0, 0], (0, 0, 200), 2)   # world axes at the origin
        _line(img, vc, np.zeros(3), np.r_[0, 1, 0], (0, 160, 0), 2)
        if show_cameras and cams:
            for c in cams:
                p = c.center_world
                _line(img, vc, p, np.r_[p[:2], 0], (180, 180, 180))
                (q,), ok = vc.project(p[None])
                if ok[0]:
                    cv2.drawMarker(img, tuple(q.astype(int)), (60, 60, 60), cv2.MARKER_TRIANGLE_UP, 14, 2)
                    cv2.putText(img, c.name, tuple(q.astype(int) + [8, -6]), 0, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        a = max(0, i - trail_n)                            # pelvis trail on the floor
        tr = np.c_[pel[a:i + 1, :2], np.zeros(i + 1 - a)]
        uv, ok = vc.project(tr)
        for k in range(1, len(tr)):
            if ok[k] and ok[k - 1]:
                cv2.line(img, tuple(uv[k - 1].astype(int)), tuple(uv[k].astype(int)), (150, 150, 150), 2, cv2.LINE_AA)
        x = X[i]
        shadow = np.c_[x[:, :2], np.zeros(len(x))]          # skeleton shadow on the floor
        for e0, e1 in edges:
            _line(img, vc, shadow[e0], shadow[e1], (200, 200, 200), 2)
        for e0, e1 in edges:
            _line(img, vc, x[e0], x[e1], _edge_color(names, e0, e1), 3)
        uv, ok = vc.project(x)
        for q, good in zip(uv, ok):
            if good:
                cv2.circle(img, tuple(np.round(q).astype(int)), 3, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.putText(img, title, (16, 30), 0, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(img, f"t = {t[i]:8.2f} s   frame {i + 1}/{len(t)}", (16, size[1] - 18), 0, 0.6, (40, 40, 40), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "1 m grid | left = red, right = blue", (size[0] - 330, size[1] - 18), 0, 0.5, (90, 90, 90), 1,
                    cv2.LINE_AA)
        wr.write(img)
    wr.release()
    return {"frames": int(len(t)), "fps": fps, "duration_s": float(len(t) / fps), "size": list(size)}
