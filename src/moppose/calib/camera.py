"""Camera model: intrinsics (fisheye or pinhole) + optional world pose.

Convention: X_cam = R @ X_world + t. Pixels are (u, v) with origin top-left.
"Normalized" points are undistorted coordinates on the z=1 plane (x/z, y/z).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

_ITER_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)


@dataclass
class Camera:
    name: str
    model: str                      # "fisheye" (4 coeffs, equidistant) | "pinhole" (rational, 8 coeffs)
    size: tuple[int, int]           # (width, height)
    K: np.ndarray                   # (3, 3)
    D: np.ndarray                   # fisheye: (4,), pinhole: (5|8|12|14,)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rms: float | None = None        # intrinsic calibration reprojection RMS (px)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.model not in ("fisheye", "pinhole"):
            raise ValueError(f"unknown camera model {self.model!r}")
        self.K = np.asarray(self.K, np.float64).reshape(3, 3)
        self.D = np.asarray(self.D, np.float64).reshape(-1)
        self.R = np.asarray(self.R, np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, np.float64).reshape(3)
        self.size = (int(self.size[0]), int(self.size[1]))

    # ---- geometry -------------------------------------------------------
    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix in normalized (undistorted) coordinates: [R | t]."""
        return np.hstack([self.R, self.t[:, None]])

    @property
    def center_world(self) -> np.ndarray:
        return -self.R.T @ self.t

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) distorted pixels -> (N, 2) normalized undistorted coordinates."""
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2))
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(pts, self.K, self.D, criteria=_ITER_CRITERIA)
        else:
            out = cv2.undistortPointsIter(pts, self.K, self.D, None, None, _ITER_CRITERIA)
        return out.reshape(-1, 2)

    def project_cam(self, X_cam: np.ndarray) -> np.ndarray:
        """(N, 3) points/rays in this camera's own frame -> (N, 2) distorted pixels."""
        X = np.asarray(X_cam, np.float64).reshape(-1, 1, 3)
        if len(X) == 0:
            return np.zeros((0, 2))
        zero = np.zeros((3, 1))
        if self.model == "fisheye":
            uv, _ = cv2.fisheye.projectPoints(X, zero, zero, self.K, self.D)
        else:
            uv, _ = cv2.projectPoints(X, zero, zero, self.K, self.D)
        return uv.reshape(-1, 2)

    def rays(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) distorted pixels -> (N, 3) unit viewing rays in the camera frame."""
        n = self.undistort_points(pts)
        r = np.concatenate([n, np.ones((len(n), 1))], axis=1)
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def project(self, X_world: np.ndarray) -> np.ndarray:
        """(N, 3) world points -> (N, 2) distorted pixels."""
        X = np.asarray(X_world, np.float64).reshape(-1, 1, 3)
        if len(X) == 0:
            return np.zeros((0, 2))
        rvec, _ = cv2.Rodrigues(self.R)
        if self.model == "fisheye":
            uv, _ = cv2.fisheye.projectPoints(X, rvec, self.t.reshape(3, 1), self.K, self.D)
        else:
            uv, _ = cv2.projectPoints(X, rvec, self.t, self.K, self.D)
        return uv.reshape(-1, 2)

    def undistort_maps(self, balance: float = 0.0, out_size: tuple[int, int] | None = None, fov_scale: float = 1.0):
        """Remap tables to a rectilinear (perspective) image.

        balance in [0, 1]: 0 = crop to valid pixels only, 1 = keep the whole field of view
        (heavily stretched borders for wide fisheyes).
        Returns (map1, map2, K_new).
        """
        out_size = out_size or self.size
        if self.model == "fisheye":
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, self.size, np.eye(3), balance=balance, new_size=out_size, fov_scale=fov_scale
            )
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), K_new, out_size, cv2.CV_16SC2)
        else:
            K_new, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, self.size, balance, out_size)
            m1, m2 = cv2.initUndistortRectifyMap(self.K, self.D, np.eye(3), K_new, out_size, cv2.CV_16SC2)
        return m1, m2, K_new

    def undistort_image(self, image: np.ndarray, balance: float = 0.0, fov_scale: float = 1.0) -> np.ndarray:
        m1, m2, _ = self.undistort_maps(balance, fov_scale=fov_scale)
        return cv2.remap(image, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # ---- io -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "size": list(self.size),
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "rms": None if self.rms is None else float(self.rms),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(
            name=d["name"], model=d["model"], size=tuple(d["size"]), K=d["K"], D=d["D"],
            R=d.get("R", np.eye(3)), t=d.get("t", np.zeros(3)), rms=d.get("rms"), meta=d.get("meta") or {},
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "Camera":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


def save_rig(cameras: list[Camera], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"cameras": [c.to_dict() for c in cameras]}, f, sort_keys=False)


def load_rig(path: str | Path) -> dict[str, Camera]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {c["name"]: Camera.from_dict(c) for c in data["cameras"]}
