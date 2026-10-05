"""Synthetic fisheye views of a ChArUco board, rendered by ray casting."""

from __future__ import annotations

import cv2
import numpy as np

from moppose.calib.camera import Camera
from moppose.calib.charuco import make_board
from moppose.config import BoardConfig

BOARD = BoardConfig(squares_x=8, squares_y=6, square_len_m=0.06, marker_len_m=0.045, dictionary="DICT_5X5_100")
SIZE = (1280, 960)
# Equidistant fisheye, ~170 deg diagonal field of view.
TRUE_CAM = Camera(
    name="synth", model="fisheye", size=SIZE,
    K=[[420.0, 0, 646.0], [0, 421.0, 478.0], [0, 0, 1]],
    D=[0.05, -0.02, 0.008, -0.002],
)


class Renderer:
    def __init__(self, cam: Camera = TRUE_CAM, cfg: BoardConfig = BOARD, px_per_m: float = 2500.0):
        self.cam = cam
        self.board = make_board(cfg)
        self.bw = cfg.squares_x * cfg.square_len_m
        self.bh = cfg.squares_y * cfg.square_len_m
        self.px_per_m = px_per_m
        self.tex = self.board.generateImage((int(self.bw * px_per_m), int(self.bh * px_per_m)), marginSize=0)
        w, h = cam.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        n = cam.undistort_points(np.stack([u.ravel(), v.ravel()], 1))
        self.rays = np.concatenate([n, np.ones((len(n), 1))], 1)  # (HW, 3) rays in camera frame

    def render(self, rvec: np.ndarray, tvec: np.ndarray, noise: float = 2.0, rng=None) -> np.ndarray:
        """Board pose (board -> camera). Returns a grayscale image (white background)."""
        R, _ = cv2.Rodrigues(np.asarray(rvec, float))
        t = np.asarray(tvec, float).reshape(3)
        normal = R[:, 2]
        denom = self.rays @ normal
        lam = (normal @ t) / np.where(np.abs(denom) < 1e-12, np.nan, denom)
        pts_cam = self.rays * lam[:, None]
        pts_board = (pts_cam - t) @ R  # R^T (p - t)
        bx = pts_board[:, 0] * self.px_per_m
        by = pts_board[:, 1] * self.px_per_m
        bad = ~(lam > 0)
        bx[bad] = -1e6
        by[bad] = -1e6
        w, h = self.cam.size
        img = cv2.remap(self.tex, bx.reshape(h, w).astype(np.float32), by.reshape(h, w).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
        if noise > 0:
            rng = rng or np.random.default_rng(0)
            img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        return img


def random_poses(n: int, seed: int = 0):
    """Board poses spread over the field of view, including the strongly distorted borders."""
    rng = np.random.default_rng(seed)
    cfg = BOARD
    center_off = np.array([cfg.squares_x * cfg.square_len_m / 2, cfg.squares_y * cfg.square_len_m / 2, 0])
    poses = []
    while len(poses) < n:
        # direction of the board centre: angle off-axis up to ~65 deg
        off = np.deg2rad(rng.uniform(0, 65))
        az = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0.35, 0.7)
        c = d * np.array([np.sin(off) * np.cos(az), np.sin(off) * np.sin(az), np.cos(off)])
        # board faces the camera roughly, plus random tilt
        tilt = np.deg2rad(rng.uniform(-35, 35, 3))
        R_face, _ = cv2.Rodrigues(np.array([np.sin(off) * np.sin(az), -np.sin(off) * np.cos(az), 0.0]) * 0.6)
        R_tilt, _ = cv2.Rodrigues(tilt)
        R = R_face @ R_tilt
        t = c - R @ center_off
        rvec, _ = cv2.Rodrigues(R)
        poses.append((rvec.ravel(), t))
    return poses
