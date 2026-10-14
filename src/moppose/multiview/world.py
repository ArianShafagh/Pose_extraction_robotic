"""From the self-calibrated rig frame to a metric, floor-aligned world frame.

* Floor: the heels and toes touch the floor many times during the session, so the
  densest plane through all foot keypoints (RANSAC) is the floor. Z points up (towards
  the head), the origin is the centre of the walked area on the floor, X points from the
  reference camera towards the second camera (projected onto the floor).
* Scale: the self-calibration knows shapes, not sizes. One known length - measured by
  triangulating its two end points - fixes the metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moppose.calib.camera import Camera

FOOT = ("left_heel", "right_heel", "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe")


@dataclass
class WorldTransform:
    A: np.ndarray       # (3, 3) rotation rig -> world
    o: np.ndarray       # (3,) world origin in rig coordinates
    scale: float = 1.0  # metres per rig unit

    def points(self, X: np.ndarray) -> np.ndarray:
        """Rig-frame points (..., 3) -> world metres."""
        return self.scale * ((X - self.o) @ self.A.T)

    def camera(self, cam: Camera) -> Camera:
        """Camera with pose in the world frame: X_cam = R_w X_world + t_w (metres)."""
        R_w = cam.R @ self.A.T
        t_w = self.scale * (cam.R @ self.o + cam.t)
        return Camera(cam.name, cam.model, cam.size, cam.K, cam.D, R=R_w, t=t_w, rms=cam.rms, meta=dict(cam.meta))


def fit_floor(X: np.ndarray, joint_names: list[str], cams: list[Camera], iters: int = 3000, seed: int = 0):
    """X (N, J, 3) rig-frame trajectories. Returns WorldTransform (scale 1) and floor statistics."""
    rng = np.random.default_rng(seed)
    fi = [joint_names.index(n) for n in FOOT]
    P = X[:, fi].reshape(-1, 3)
    P = P[np.all(np.isfinite(P), 1)]
    leg = np.nanmedian(np.linalg.norm(X[:, joint_names.index("left_hip")] - X[:, joint_names.index("left_ankle")], axis=1))
    thr = 0.03 * leg  # ~2.5 cm for a typical leg
    best = (0, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < thr
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    Q = P[best[1]]
    centre = Q.mean(0)
    n = np.linalg.svd(Q - centre)[2][-1]
    head = X[:, joint_names.index("nose")].reshape(-1, 3)
    head = head[np.all(np.isfinite(head), 1)]
    if np.mean((head - centre) @ n) < 0:
        n = -n
    # x axis: reference camera -> second camera, projected onto the floor
    d = cams[1].center_world - cams[0].center_world
    x = d - (d @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    A = np.stack([x, y, n])
    # origin: centre of the walked area on the floor
    o = centre - ((centre - centre) @ n) * n
    heights = (P - centre) @ n
    stats = {"inlier_ratio": best[0] / len(P), "leg_rig_units": float(leg),
             "foot_height_p50": float(np.median(heights)), "foot_height_p05": float(np.percentile(heights, 5))}
    return WorldTransform(A=A, o=o), stats


def triangulated_length(cams: list[Camera], uv_a: np.ndarray, uv_b: np.ndarray) -> float:
    """Length between two points, each given in >= 2 cameras as raw pixels (C, 2) (NaN = not marked)."""
    from moppose.triangulate.dlt import triangulate

    uv = np.stack([uv_a, uv_b], axis=1)  # (C, 2, 2)
    X, _ = triangulate(cams, uv, np.isfinite(uv[..., 0]).astype(float), min_conf=0.5, max_reproj_px=1e9)
    return float(np.linalg.norm(X[0] - X[1]))


def mop_scale(cams: list[Camera], names: list[str], joint_px: dict[str, list[float]], top_wrist: np.ndarray,
              length_m: float, hand_offset_m: float) -> tuple[float, dict]:
    """Metres per rig unit from the mop handle.

    The handle runs from the joint on the mop pad (marked in >= 2 cameras) to its end held in
    the upper hand; the end is taken as the upper wrist plus `hand_offset_m`.
    """
    from moppose.triangulate.dlt import triangulate

    uv = np.full((len(cams), 1, 2), np.nan)
    for n, p in joint_px.items():
        uv[names.index(n), 0] = p
    J, err = triangulate(cams, uv, np.isfinite(uv[..., 0]).astype(float), min_conf=0.5, max_reproj_px=1e9)
    L = float(np.linalg.norm(J[0] - top_wrist))
    s = (length_m - hand_offset_m) / L
    return s, {"handle_rig_units": L, "joint_reproj_px": [float(e) for e in err[:, 0] if np.isfinite(e)],
               "mop_joint_rig": J[0].tolist()}
