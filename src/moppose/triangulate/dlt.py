"""Multi-view triangulation (confidence-weighted DLT) in normalized camera coordinates.

Inputs are undistorted normalized points (Camera.undistort_points), so lens
distortion never enters the linear system.
"""

from __future__ import annotations

import numpy as np

from moppose.calib.camera import Camera


def triangulate_points(
    P: np.ndarray,      # (C, 3, 4) normalized projection matrices [R|t]
    x: np.ndarray,      # (C, N, 2) normalized image points
    w: np.ndarray,      # (C, N) weights, 0 = not observed
) -> np.ndarray:
    """Weighted DLT for N points at once. Returns (N, 3); NaN where < 2 cameras observed."""
    C, N = w.shape
    # rows: w * (x * P[2] - P[0]), w * (y * P[2] - P[1])   -> A: (N, 2C, 4)
    r0 = x[..., 0:1] * P[:, None, 2, :] - P[:, None, 0, :]   # (C, N, 4)
    r1 = x[..., 1:2] * P[:, None, 2, :] - P[:, None, 1, :]
    A = np.concatenate([r0 * w[..., None], r1 * w[..., None]], axis=0)  # (2C, N, 4)
    A = np.nan_to_num(np.transpose(A, (1, 0, 2)))
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    X = Xh[:, :3] / Xh[:, 3:4]
    X[(w > 0).sum(0) < 2] = np.nan
    return X


def reprojection_errors_px(cams: list[Camera], X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """(C, N) pixel distance between observed `uv` (C, N, 2) and projection of X (N, 3)."""
    err = np.full(uv.shape[:2], np.nan)
    ok = np.all(np.isfinite(X), axis=1)
    for c, cam in enumerate(cams):
        if ok.any():
            err[c, ok] = np.linalg.norm(cam.project(X[ok]) - uv[c, ok], axis=1)
    return err


def triangulate(
    cams: list[Camera],
    uv: np.ndarray,                 # (C, N, 2) distorted pixel observations (NaN = missing)
    conf: np.ndarray | None = None,  # (C, N) detector confidence
    min_conf: float = 0.3,
    max_reproj_px: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate N points from C cameras with outlier-camera rejection.

    If a point's worst camera reprojects worse than `max_reproj_px` and > 2 cameras
    see it, every leave-one-camera-out solution is tried and the one whose remaining
    cameras agree best is kept. (A single bad view drags the joint DLT solution, so the
    camera with the largest error is not necessarily the bad one.)
    Returns X (N, 3) and the final per-camera reprojection error (C, N), NaN for unused views.
    """
    C, N, _ = uv.shape
    conf = np.ones((C, N)) if conf is None else np.asarray(conf, float)
    valid = np.all(np.isfinite(uv), axis=2) & (conf >= min_conf)
    w = np.where(valid, conf, 0.0)
    x = np.zeros((C, N, 2))
    for c, cam in enumerate(cams):
        if valid[c].any():
            x[c, valid[c]] = cam.undistort_points(uv[c, valid[c]])
    P = np.stack([cam.P for cam in cams])

    X = triangulate_points(P, x, w)
    for _ in range(C - 2):
        err = reprojection_errors_px(cams, X, uv)
        err[w == 0] = np.nan
        redo = (np.fmax.reduce(err, axis=0) > max_reproj_px) & ((w > 0).sum(0) > 2)
        if not redo.any():
            break
        cols = np.flatnonzero(redo)
        best_drop = np.full(len(cols), -1)
        best_score = np.full(len(cols), np.inf)
        for c in range(C):
            has = w[c, cols] > 0
            if not has.any():
                continue
            wc = w[:, cols].copy()
            wc[c] = 0.0
            e = reprojection_errors_px(cams, triangulate_points(P, x[:, cols], wc), uv[:, cols])
            e[wc == 0] = np.nan
            score = np.fmax.reduce(e, axis=0)
            better = has & (score < best_score)
            best_score[better] = score[better]
            best_drop[better] = c
        ok = best_drop >= 0
        if not ok.any():
            break
        w[best_drop[ok], cols[ok]] = 0.0
        X[cols] = triangulate_points(P, x[:, cols], w[:, cols])
    err = reprojection_errors_px(cams, X, uv)
    err[w == 0] = np.nan
    return X, err
