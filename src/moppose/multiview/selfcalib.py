"""Camera poses (extrinsics) from the person's own keypoints - no calibration object needed.

The same 23 joints are seen by all cameras at the same (synchronised) moments; thousands
of such correspondences over the whole session pin down the relative rotation and
translation of the cameras:

1. per camera pair: essential matrix with RANSAC on undistorted (normalised) keypoints,
   then rotation + translation direction (recoverPose)
2. chain everything into the frame of the reference camera and make the translation
   scales consistent by triangulating the same points with both pairs
3. refine each camera's time offset by minimising the epipolar error
4. bundle adjustment: camera poses and 3D points together, minimising the reprojection
   error in fisheye pixels with a robust loss

The result is defined up to one global scale (fixed later from a known length) and is
expressed in the reference camera's frame (re-oriented to the floor later).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from moppose.calib.camera import Camera
from moppose.triangulate.dlt import triangulate_points


@dataclass
class Correspondences:
    times: np.ndarray       # (N,) reference times
    uv: np.ndarray          # (C, N, J, 2) raw pixels
    conf: np.ndarray        # (C, N, J)


def normalized(cams: list[Camera], uv: np.ndarray) -> np.ndarray:
    """(C, ..., 2) raw pixels -> normalised undistorted coordinates (NaN kept)."""
    out = np.full(uv.shape, np.nan)
    for c, cam in enumerate(cams):
        flat = uv[c].reshape(-1, 2)
        ok = np.all(np.isfinite(flat), axis=1)
        tmp = np.full(flat.shape, np.nan)
        if ok.any():
            tmp[ok] = cam.undistort_points(flat[ok])
        out[c] = tmp.reshape(uv[c].shape)
    return out


def pair_points(xn: np.ndarray, conf: np.ndarray, a: int, b: int, min_conf: float):
    """Flattened normalised point pairs seen confidently by cameras a and b."""
    pa, pb = xn[a].reshape(-1, 2), xn[b].reshape(-1, 2)
    ok = (conf[a].ravel() >= min_conf) & (conf[b].ravel() >= min_conf) & np.all(np.isfinite(pa), 1) & np.all(np.isfinite(pb), 1)
    return pa[ok], pb[ok], np.flatnonzero(ok)


def relative_pose(pa: np.ndarray, pb: np.ndarray, thr: float, max_pts: int = 40000, seed: int = 0):
    """R, t (unit) with x_b ~ R x_a + t, plus inlier ratio."""
    rng = np.random.default_rng(seed)
    if len(pa) > max_pts:
        sel = rng.choice(len(pa), max_pts, replace=False)
        pa, pb = pa[sel], pb[sel]
    E, mask = cv2.findEssentialMat(pa, pb, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.9999, threshold=thr)
    _, R, t, mask2 = cv2.recoverPose(E, pa, pb, focal=1.0, pp=(0.0, 0.0), mask=mask.copy())
    return R, t.ravel(), float(mask.mean())


def sampson(pa: np.ndarray, pb: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """First-order epipolar (Sampson) distance in normalised units for x_b ~ R x_a + t."""
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    xa = np.c_[pa, np.ones(len(pa))]
    xb = np.c_[pb, np.ones(len(pb))]
    Ex = xa @ E.T
    Etx = xb @ E
    num = np.sum(xb * Ex, axis=1) ** 2
    den = Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-18))


# ------------------------------------------------------------------ fisheye projection (vectorised)
def project_fisheye(X_cam: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """OpenCV fisheye (equidistant) projection of camera-frame points, numpy (for least squares)."""
    x, y, z = X_cam[..., 0], X_cam[..., 1], X_cam[..., 2]
    a, b = x / z, y / z
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    th2 = th * th
    thd = th * (1 + D[0] * th2 + D[1] * th2 ** 2 + D[2] * th2 ** 3 + D[3] * th2 ** 4)
    s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
    u = K[0, 0] * s * a + K[0, 1] * s * b + K[0, 2]
    v = K[1, 1] * s * b + K[1, 2]
    return np.stack([u, v], axis=-1)


def _rot(rvec: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]


def bundle_adjust(cams: list[Camera], uv: np.ndarray, w: np.ndarray, X0: np.ndarray, ref: int = 0,
                  loss_scale_px: float = 3.0, verbose: int = 0) -> tuple[list[Camera], np.ndarray, dict]:
    """Refine poses of all cameras except `ref` (fixed at identity) and the 3D points.

    uv (C, P, 2) observations in raw pixels (NaN = missing), w (C, P) weights, X0 (P, 3).
    Gauge: reference camera fixed; global scale fixed by keeping |t| of the first other camera.
    """
    C, P = w.shape
    others = [c for c in range(C) if c != ref]
    first = others[0]
    scale0 = float(np.linalg.norm(cams[first].t))
    obs = [(c, np.flatnonzero((w[c] > 0) & np.all(np.isfinite(uv[c]), 1))) for c in range(C)]

    def unpack(p):
        poses = {}
        k = 0
        for c in others:
            rv, tv = p[k:k + 3], p[k + 3:k + 6]
            if c == first:  # fix the scale of the first translation
                tv = tv / np.linalg.norm(tv) * scale0
            poses[c] = (_rot(rv), tv)
            k += 6
        X = p[k:].reshape(P, 3)
        return poses, X

    def residuals(p):
        poses, X = unpack(p)
        res = []
        for c, idx in obs:
            R, t = (np.eye(3), np.zeros(3)) if c == ref else poses[c]
            Xc = X[idx] @ R.T + t
            proj = project_fisheye(Xc, cams[c].K, cams[c].D)
            res.append(((proj - uv[c, idx]) * np.sqrt(w[c, idx])[:, None]).ravel())
        return np.concatenate(res)

    p0 = []
    for c in others:
        p0 += list(cv2.Rodrigues(cams[c].R)[0].ravel()) + list(cams[c].t)
    p0 = np.r_[p0, X0.ravel()]

    # sparsity: each residual pair depends on its camera's 6 params and its point's 3
    n_res = 2 * sum(len(idx) for _, idx in obs)
    S = lil_matrix((n_res, len(p0)), dtype=np.int8)
    row = 0
    for c, idx in obs:
        for j in range(2):
            rows = row + 2 * np.arange(len(idx)) + j
            if c != ref:
                k = 6 * others.index(c)
                for q in range(6):
                    S[rows, k + q] = 1
            base = 6 * len(others)
            for q in range(3):
                S[rows, base + 3 * idx + q] = 1
        row += 2 * len(idx)

    r0 = residuals(p0)
    sol = least_squares(residuals, p0, jac_sparsity=S, loss="soft_l1", f_scale=loss_scale_px, x_scale="jac",
                        method="trf", max_nfev=60, verbose=verbose)
    poses, X = unpack(sol.x)
    out = []
    for c, cam in enumerate(cams):
        R, t = (np.eye(3), np.zeros(3)) if c == ref else poses[c]
        out.append(Camera(cam.name, cam.model, cam.size, cam.K, cam.D, R=R, t=t, rms=cam.rms, meta=dict(cam.meta)))
    rr = sol.fun.reshape(-1, 2)
    info = {"rms_before_px": float(np.sqrt(np.mean(r0.reshape(-1, 2) ** 2) * 2)),
            "median_err_px": float(np.median(np.linalg.norm(rr, axis=1))),
            "cost": float(sol.cost), "nfev": int(sol.nfev)}
    return out, X, info


def triangulate_all(cams: list[Camera], uv: np.ndarray, w: np.ndarray) -> np.ndarray:
    """DLT of flattened observations uv (C, P, 2) with weights w (C, P)."""
    xn = normalized(cams, uv[:, :, None, :])[:, :, 0, :]
    xn = np.nan_to_num(xn)
    P = np.stack([c.P for c in cams])
    return triangulate_points(P, xn, w)
