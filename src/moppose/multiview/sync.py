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
