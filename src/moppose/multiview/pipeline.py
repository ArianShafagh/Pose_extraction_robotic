"""2D pose of all cameras -> synchronised, self-calibrated, metric 3D landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

from moppose.calib.camera import Camera
from moppose.multiview.data import Track2D, on_common_clock
from moppose.multiview.selfcalib import (bundle_adjust, normalized, pair_points, relative_pose, sampson,
                                         triangulate_all)
from moppose.multiview.sync import coarse_offset
from moppose.triangulate.dlt import triangulate


@dataclass
class ExtrinsicsResult:
    cams: list[Camera]                       # rig frame of the reference camera, arbitrary scale
    offsets: dict[str, float]                # video time of cam = reference time + offset
    info: dict = field(default_factory=dict)


def calibrate_extrinsics(tracks: dict[str, Track2D], intrinsics: list[Camera], min_conf: float = 0.6,
                         thr_px: float = 2.5, ba_every: int = 6, log=print) -> ExtrinsicsResult:
    names = [c.name for c in intrinsics]
    ref = tracks[names[0]]
    f = float(np.mean([c.K[0, 0] for c in intrinsics]))
    thr = thr_px / f
    info: dict = {}

    # 1. coarse time offsets from motion correlation
    offs = {names[0]: 0.0}
    for n in names[1:]:
        offs[n], corr = coarse_offset(ref, tracks[n])
        log(f"  coarse offset {n}: {offs[n] * 1000:+.1f} ms (motion correlation {corr:.2f})")
    times = ref.t[(ref.t > ref.t[0] + 2) & (ref.t < ref.t[-1] - 2)]

    def stack(o):
        uv, cf = on_common_clock(tracks, o, times)
        return uv, cf, normalized(intrinsics, uv)

    # 2. pairwise relative poses, 3. refine offsets on the epipolar error, re-estimate poses
    uv, cf, xn = stack(offs)
    poses = {b: relative_pose(*pair_points(xn, cf, 0, b, min_conf)[:2], thr)[:2] for b in range(1, len(names))}
    for b in range(1, len(names)):
        n = names[b]
        best = (np.inf, offs[n])
        for span, step in ((0.15, 0.01), (0.012, 0.002)):
            for d in np.arange(best[1] - span, best[1] + span + 1e-9, step):
                k, cc = tracks[n].sample(times + d)
                xb = normalized([intrinsics[b]], k[None])[0]
                ok = (cf[0] >= min_conf) & (cc >= min_conf) & np.all(np.isfinite(xb), -1) & np.all(np.isfinite(xn[0]), -1)
                e = float(np.median(sampson(xn[0][ok], xb[ok], *poses[b])) * f)
                if e < best[0]:
                    best = (e, float(d))
        offs[n] = best[1]
        log(f"  refined offset {n}: {offs[n] * 1000:+.1f} ms (median epipolar error {best[0]:.2f} px)")
    uv, cf, xn = stack(offs)
    rig = [Camera(c.name, c.model, c.size, c.K, c.D, rms=c.rms, meta=dict(c.meta)) for c in intrinsics]
    for b in range(1, len(names)):
        pa, pb, _ = pair_points(xn, cf, 0, b, min_conf)
        R, t, inl = relative_pose(pa, pb, thr)
        e = float(np.median(sampson(pa, pb, R, t)) * f)
        rig[b].R, rig[b].t = R, t
        info[f"pair_{names[0]}_{names[b]}"] = {"points": int(len(pa)), "inliers": inl, "median_epipolar_px": e}
        log(f"  {names[0]}-{names[b]}: {len(pa)} correspondences, {inl * 100:.0f}% inliers, epipolar {e:.2f} px")

    # 4. consistent translation scales: triangulate the same points with pair (0,1) and pair (0,b)
    allv = (cf >= min_conf).all(0) & np.all(np.isfinite(uv), -1).all(0)
    sel = np.flatnonzero(allv.ravel())[::5]
    U = uv.reshape(len(names), -1, 2)[:, sel]
    X01 = triangulate_all(rig[:2], U[[0, 1]], np.ones((2, len(sel))))
    for b in range(2, len(names)):
        X0b = triangulate_all([rig[0], rig[b]], U[[0, b]], np.ones((2, len(sel))))
        ok = np.all(np.isfinite(X01), 1) & np.all(np.isfinite(X0b), 1)
        rig[b].t = rig[b].t * float(np.median(np.linalg.norm(X01[ok], axis=1) / np.linalg.norm(X0b[ok], axis=1)))

    # 5. bundle adjustment on every `ba_every`-th frame
    fsel = np.arange(0, len(times), ba_every)
    Ub = uv[:, fsel].reshape(len(names), -1, 2)
    Wb = cf[:, fsel].reshape(len(names), -1)
    Wb = np.where(Wb >= min_conf, Wb, 0.0)
    keep = (Wb > 0).sum(0) >= 2
    Ub, Wb = Ub[:, keep], Wb[:, keep]
    X0 = triangulate_all(rig, Ub, Wb)
    good = np.all(np.isfinite(X0), 1)
    rig, _, ba = bundle_adjust(rig, Ub[:, good], Wb[:, good], X0[good])
    info["bundle_adjustment"] = ba | {"points": int(good.sum()), "frames": int(len(fsel))}
    log(f"  bundle adjustment: {good.sum()} points, median reprojection error {ba['median_err_px']:.2f} px")
    info["offsets_ms"] = {k: v * 1000 for k, v in offs.items()}
    return ExtrinsicsResult(rig, offs, info)


GROUPS = {
    "arm": ("shoulder", "elbow", "wrist"),
    "leg": ("hip", "knee", "ankle", "heel", "big_toe", "small_toe"),
}
LIMBS = [("shoulder", "elbow", "elbow"), ("elbow", "wrist", "wrist"), ("hip", "knee", "knee"), ("knee", "ankle", "ankle")]


def _group_error(cams, uv, cf, idx, min_conf, max_reproj_px):
    """Mean reprojection error per frame of the joints `idx` (N,) after triangulation."""
    C, N = uv.shape[:2]
    sub = uv[:, :, idx].reshape(C, -1, 2)
    X, err = triangulate(cams, sub, cf[:, :, idx].reshape(C, -1), min_conf=min_conf, max_reproj_px=max_reproj_px)
    e = np.where(np.isfinite(err), err, max_reproj_px * 2)  # a rejected view counts as a large error
    e = np.where(cf[:, :, idx].reshape(C, -1) >= min_conf, e, 0.0)
    return e.reshape(C, N, len(idx)).mean(axis=(0, 2))


def fix_left_right(cams, uv, cf, joint_names, min_conf=0.5, max_reproj_px=15.0, gain=0.6, min_err_px=5.0):
    """Undo left/right label swaps of arms or legs in single camera views.

    For every camera, frame and limb group, the labels in that camera are swapped if this lowers
    the multi-view reprojection error of the group to less than `gain` times the original.
    Returns corrected uv, cf and the number of swaps per camera and group.
    """
    uv, cf = uv.copy(), cf.copy()
    stats = {}
    for g, parts in GROUPS.items():
        L = [joint_names.index(f"left_{p}") for p in parts]
        R = [joint_names.index(f"right_{p}") for p in parts]
        idx = L + R
        for c, cam in enumerate(cams):
            e0 = _group_error(cams, uv, cf, idx, min_conf, max_reproj_px)
            sw_uv, sw_cf = uv.copy(), cf.copy()
            sw_uv[c][:, L], sw_uv[c][:, R] = uv[c][:, R], uv[c][:, L]
            sw_cf[c][:, L], sw_cf[c][:, R] = cf[c][:, R], cf[c][:, L]
            e1 = _group_error(cams, sw_uv, sw_cf, idx, min_conf, max_reproj_px)
            do = (e1 < gain * e0) & (e0 > min_err_px)
            uv[c][do] = sw_uv[c][do]
            cf[c][do] = sw_cf[c][do]
            stats[f"{cam.name}_{g}"] = int(do.sum())
    return uv, cf, stats


def reject_bone_outliers(X, joint_names, tol=0.25):
    """Drop the distal joint where a limb bone deviates more than `tol` from its median length."""
    X = X.copy()
    dropped = {}
    for side in ("left", "right"):
        for a, b, distal in LIMBS:
            ia, ib = joint_names.index(f"{side}_{a}"), joint_names.index(f"{side}_{b}")
            length = np.linalg.norm(X[:, ia] - X[:, ib], axis=1)
            med = np.nanmedian(length)
            bad = np.abs(length - med) > tol * med
            X[bad, joint_names.index(f"{side}_{distal}")] = np.nan
            dropped[f"{side}_{a}-{b}"] = int(bad.sum())
    return X, dropped


def reconstruct(tracks: dict[str, Track2D], cams: list[Camera], offsets: dict[str, float], times: np.ndarray,
                min_conf: float = 0.5, max_reproj_px: float = 15.0, fix_swaps: bool = True):
    """Triangulate every joint at every reference time.

    Returns X (N, J, 3) in the cameras' frame, err (C, N, J), views (N, J) and a stats dict.
    """
    uv, cf = on_common_clock(tracks, offsets, times)
    jn = next(iter(tracks.values())).joint_names
    stats = {}
    if fix_swaps:
        uv, cf, stats["left_right_swaps"] = fix_left_right(cams, uv, cf, jn, min_conf, max_reproj_px)
    C, N, J, _ = uv.shape
    X, err = triangulate(cams, uv.reshape(C, -1, 2), cf.reshape(C, -1), min_conf=min_conf, max_reproj_px=max_reproj_px)
    X = X.reshape(N, J, 3)
    err = err.reshape(C, N, J)
    views = np.isfinite(err).sum(0)
    bad = np.nanmax(np.where(np.isfinite(err), err, -1), axis=0) > max_reproj_px
    X[bad] = np.nan
    return X, err, views, stats


def clean_and_smooth(X: np.ndarray, t: np.ndarray, max_gap_s: float = 0.5, cutoff_hz: float = 4.0):
    """Fill short gaps (linear) and low-pass filter (zero-phase Butterworth) each joint coordinate.

    Returns the smoothed trajectories and a (N, J) mask of samples that were filled in.
    """
    Xs = X.copy()
    filled = np.zeros(X.shape[:2], bool)
    fs = 1.0 / np.median(np.diff(t))
    b, a = butter(4, min(cutoff_hz / (fs / 2), 0.99))
    for j in range(X.shape[1]):
        ok = np.all(np.isfinite(X[:, j]), axis=1)
        if ok.sum() < 10:
            continue
        idx = np.flatnonzero(ok)
        for d in range(3):
            Xs[:, j, d] = np.interp(t, t[idx], X[idx, j, d])
        # do not invent data across long gaps
        gap_mask = np.zeros(len(t), bool)
        holes = np.flatnonzero(~ok)
        if len(holes):
            nxt = np.searchsorted(idx, holes)
            prev_t = np.where(nxt > 0, t[idx[np.maximum(nxt - 1, 0)]], -np.inf)
            next_t = np.where(nxt < len(idx), t[idx[np.minimum(nxt, len(idx) - 1)]], np.inf)
            gap_mask[holes] = (next_t - prev_t) > max_gap_s
        filled[:, j] = ~ok & ~gap_mask
        # filter contiguous segments
        valid = ~gap_mask
        seg_starts = np.flatnonzero(np.diff(np.r_[0, valid.astype(int)]) == 1)
        seg_ends = np.flatnonzero(np.diff(np.r_[valid.astype(int), 0]) == -1) + 1
        for s0, s1 in zip(seg_starts, seg_ends):
            if s1 - s0 > 15:
                for d in range(3):
                    Xs[s0:s1, j, d] = filtfilt(b, a, Xs[s0:s1, j, d])
        Xs[gap_mask, j] = np.nan
    return Xs, filled


def bone_lengths(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(N, E) length of every skeleton edge per frame."""
    return np.linalg.norm(X[:, edges[:, 0]] - X[:, edges[:, 1]], axis=2)
