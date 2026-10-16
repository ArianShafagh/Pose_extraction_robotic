"""Time offsets between independently recorded cameras.

Coarse: the timing of the person's movements (speed of ankles and wrists, normalised by
the person's size in the image) is the same in every camera even though the views
differ, so the offset is the lag that maximises their correlation.

Fine: once the camera geometry is known (essential matrices), the offset that minimises
the epipolar error of all joints (see selfcalib.refine_offsets).
"""

from __future__ import annotations

import numpy as np

from moppose.multiview.data import Track2D

SYNC_JOINTS = ("left_ankle", "right_ankle", "left_wrist", "right_wrist")


def motion_signal(tr: Track2D, times: np.ndarray) -> np.ndarray:
    """(len(SYNC_JOINTS), N) normalised, de-trended speed signals sampled at `times`."""
    k, c = tr.sample(times, max_gap=1.0)
    scale = np.nanmedian(np.linalg.norm(k[:, tr.joint_names.index("left_shoulder")] -
                                        k[:, tr.joint_names.index("left_ankle")], axis=1))
    out = []
    for name in SYNC_JOINTS:
        p = k[:, tr.joint_names.index(name)]
        p = np.where(np.isfinite(p), p, np.nanmedian(p, axis=0))
        sp = np.hypot(np.gradient(p[:, 0]), np.gradient(p[:, 1])) / scale
        sp = sp - np.convolve(sp, np.ones(31) / 31, mode="same")
        out.append((sp - sp.mean()) / (sp.std() + 1e-12))
    return np.stack(out)


def coarse_offset(ref: Track2D, other: Track2D, max_lag: float = 3.0, step: float = 1 / 60,
                  t_range: tuple[float, float] | None = None) -> tuple[float, float]:
    """Offset d (other's time = ref time + d) maximising motion correlation. Returns (d, peak correlation)."""
    lo = max(ref.t[0], other.t[0]) + max_lag + 1
    hi = min(ref.t[-1], other.t[-1]) - max_lag - 1
    if t_range:
        lo, hi = max(lo, t_range[0]), min(hi, t_range[1])
    times = np.arange(lo, hi, 1 / 30)
    s_ref = motion_signal(ref, times)
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    corr = np.array([np.mean(np.sum(s_ref * motion_signal(other, times + d), axis=1) / len(times)) for d in lags])
    i = int(np.argmax(corr))
    if 0 < i < len(corr) - 1:  # parabolic sub-step peak
        a, b, c = corr[i - 1:i + 2]
        sub = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        sub = 0.0
    return float(lags[i] + sub * step), float(corr[i])


def relative_pose(cam_a, cam_b) -> tuple[np.ndarray, np.ndarray]:
    """R, t with x_b = R x_a + t from two cameras posed in the same world frame."""
    R = cam_b.R @ cam_a.R.T
    return R, cam_b.t - R @ cam_a.t


def refine_offset(ref: Track2D, other: Track2D, cam_ref, cam_other, R: np.ndarray, t: np.ndarray,
                  times: np.ndarray, d0: float, min_conf: float = 0.6,
                  steps=((0.15, 0.01), (0.012, 0.002))) -> tuple[float, float]:
    """Offset d (other's time = ref time + d) minimising the median epipolar error of all joints.

    R, t: relative pose (x_other = R x_ref + t), e.g. from relative_pose() of a calibrated rig.
    Returns (d, median epipolar error in pixels).
    """
    from moppose.multiview.selfcalib import normalized, sampson

    f = float(np.mean([cam_ref.K[0, 0], cam_other.K[0, 0]]))
    ka, ca = ref.sample(times)
    xa = normalized([cam_ref], ka[None])[0]
    best = (np.inf, float(d0))
    for span, step in steps:
        for d in np.arange(best[1] - span, best[1] + span + 1e-9, step):
            kb, cb = other.sample(times + d)
            xb = normalized([cam_other], kb[None])[0]
            ok = (ca >= min_conf) & (cb >= min_conf) & np.all(np.isfinite(xa), -1) & np.all(np.isfinite(xb), -1)
            if ok.sum() < 50:
                continue
            e = float(np.median(sampson(xa[ok], xb[ok], R, t)) * f)
            if e < best[0]:
                best = (e, float(d))
    return best[1], best[0]
