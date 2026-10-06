"""Per-camera intrinsic calibration from ChArUco videos.

Two lens models are fitted:
  * fisheye  - cv2.fisheye (equidistant, k1..k4). Right model for real fisheye lenses.
  * pinhole  - cv2.calibrateCamera with the rational model (k1..k6, p1, p2). Works for
               moderately wide lenses (< ~120 deg) and is a useful sanity comparison.
Bad views (motion blur, mis-detections) are rejected iteratively by their own RMS.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from moppose.calib.charuco import Detection, Target, is_degenerate
from moppose.calib.frame_map import FrameMap
from moppose.io.video import iter_frames, open_video

MIN_VIEWS = 8
_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-9)


@dataclass
class CalibResult:
    model: str
    K: np.ndarray
    D: np.ndarray
    rms: float                      # RMS over all corners of all used views (px)
    per_view_rms: np.ndarray        # (V,)
    views: list[Detection]
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    rejected: list[Detection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def collect_detections(
    target: Target,
    videos: list[Path],
    every_s: float,
    min_corners: int,
    min_sharpness: float,
    frame_maps: list[FrameMap | None] | None = None,
) -> tuple[list[Detection], tuple[int, int], dict]:
    """Run ChArUco detection on frames sampled every `every_s` seconds.

    `frame_maps[i]` (optional) crops/resizes frames of videos[i] onto the scene pixel grid first.
    """
    dets: list[Detection] = []
    size = None
    stats = {"frames": 0, "with_board": 0, "too_few": 0, "blurry": 0, "degenerate": 0}
    frame_maps = frame_maps or [None] * len(videos)
    for vid, fm in zip(videos, frame_maps):
        cap = open_video(vid)
        vsize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        if fm is not None:
            vsize = fm.out_size
        if size is None:
            size = vsize
        elif size != vsize:
            raise ValueError(f"{vid} has size {vsize}, previous videos {size}; one camera = one resolution")
        total = int(n / fps / every_s) + 1 if n > 0 else None
        for fr in tqdm(iter_frames(vid, every_s=every_s), total=total, desc=f"detect {Path(vid).name}", unit="f"):
            stats["frames"] += 1
            img = fr.image if fm is None else fm.apply(fr.image)
            d = target.detect(img, t=fr.t, frame_index=fr.index)
            if d is None:
                continue
            stats["with_board"] += 1
            if len(d.ids) < min_corners:
                stats["too_few"] += 1
                continue
            if d.sharpness < min_sharpness:
                stats["blurry"] += 1
                continue
            if is_degenerate(target.obj[d.ids]):
                stats["degenerate"] += 1
                continue
            dets.append(d)
    if size is None:
        raise ValueError("no calibration videos given")
    return dets, size, stats


def save_detections(path: Path, dets: list[Detection], size: tuple[int, int], key: str = "") -> None:
    """`key` describes the inputs (videos, frame maps, board); a cache with another key is stale."""
    np.savez_compressed(
        path,
        key=np.array(key),
        size=np.array(size),
        t=np.array([d.t for d in dets]),
        frame_index=np.array([d.frame_index for d in dets]),
        sharpness=np.array([d.sharpness for d in dets]),
        counts=np.array([len(d.ids) for d in dets]),
        corners=np.concatenate([d.corners for d in dets]) if dets else np.zeros((0, 2), np.float32),
        ids=np.concatenate([d.ids for d in dets]) if dets else np.zeros(0, np.int32),
    )


def load_detections(path: Path) -> tuple[list[Detection], tuple[int, int], str]:
    z = np.load(path)
    offs = np.concatenate([[0], np.cumsum(z["counts"])])
    dets = [
        Detection(t=float(z["t"][i]), frame_index=int(z["frame_index"][i]),
                  corners=z["corners"][offs[i]:offs[i + 1]], ids=z["ids"][offs[i]:offs[i + 1]],
                  sharpness=float(z["sharpness"][i]))
        for i in range(len(z["t"]))
    ]
    key = str(z["key"]) if "key" in z.files else ""
    return dets, tuple(int(v) for v in z["size"]), key


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def _per_view_rms(model, obj, img, K, D, rvecs, tvecs) -> np.ndarray:
    errs = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        if model == "fisheye":
            p, _ = cv2.fisheye.projectPoints(o.reshape(-1, 1, 3), r, t, K, D)
        else:
            p, _ = cv2.projectPoints(o.reshape(-1, 1, 3), r, t, K, D)
        errs.append(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1))))
    return np.asarray(errs)


def _total_rms(per_view: np.ndarray, counts: np.ndarray) -> float:
    return float(np.sqrt(np.sum(per_view ** 2 * counts) / np.sum(counts)))


def _fit_fisheye(obj, img, size, K0=None):
    """cv2.fisheye.calibrate; drops views that OpenCV reports as ill-conditioned."""
    idx = list(range(len(obj)))
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
             | cv2.fisheye.CALIB_CHECK_COND)
    if K0 is not None:
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    for _ in range(len(obj)):
        o = [obj[i].reshape(-1, 1, 3) for i in idx]
        m = [img[i].reshape(-1, 1, 2).astype(np.float64) for i in idx]
        K = K0.copy() if K0 is not None else np.zeros((3, 3))
        D = np.zeros((4, 1))
        try:
            _, K, D, rvecs, tvecs = cv2.fisheye.calibrate(o, m, size, K, D, flags=flags, criteria=_CRITERIA)
            return K, D.reshape(-1), list(rvecs), list(tvecs), idx
        except cv2.error as e:
            hit = re.search(r"input array (\d+)", str(e))
            bad = int(hit.group(1)) if hit else int(np.argmin([len(img[i]) for i in idx]))
            idx.pop(bad)
            if len(idx) < MIN_VIEWS:
                raise RuntimeError(f"fisheye calibration failed: too few usable views ({e})") from e
    raise RuntimeError("fisheye calibration failed")


def _fit_pinhole(obj, img, size):
    o = [x.astype(np.float32) for x in obj]
    m = [x.astype(np.float32).reshape(-1, 1, 2) for x in img]
    flags = cv2.CALIB_RATIONAL_MODEL
    _, K, D, rvecs, tvecs = cv2.calibrateCamera(o, m, size, None, None, flags=flags, criteria=_CRITERIA)
    return K, D.reshape(-1), list(rvecs), list(tvecs), list(range(len(obj)))


def calibrate(
    target: Target,
    dets: list[Detection],
    size: tuple[int, int],
    model: str,
    outlier_rms_px: float = 2.0,
    max_iter: int = 15,
) -> CalibResult:
    """Fit `model` ("fisheye" | "pinhole") with iterative rejection of bad views."""
    obj_all = target.obj
    views = list(dets)
    rejected: list[Detection] = []
    K0 = None
    for _ in range(max_iter):
        if len(views) < MIN_VIEWS:
            raise RuntimeError(f"only {len(views)} views left (need {MIN_VIEWS}); record more board views")
        obj = [obj_all[d.ids] for d in views]
        img = [d.corners for d in views]
        if model == "fisheye":
            K, D, rvecs, tvecs, keep = _fit_fisheye(obj, img, size, K0)
        elif model == "pinhole":
            K, D, rvecs, tvecs, keep = _fit_pinhole(obj, img, size)
        else:
            raise ValueError(model)
        if len(keep) != len(views):
            keep_set = set(keep)
            rejected += [d for i, d in enumerate(views) if i not in keep_set]
            views = [views[i] for i in keep]
            obj = [obj[i] for i in keep]
            img = [img[i] for i in keep]
        per_view = _per_view_rms(model, obj, img, K, D, rvecs, tvecs)
        thr = max(outlier_rms_px, 3.0 * float(np.median(per_view)))
        bad = np.flatnonzero(per_view > thr)
        if len(bad) == 0 or len(views) - 1 < MIN_VIEWS:
            break
        # drop at most 10% of the views per round, worst first
        k = min(len(bad), max(1, math.ceil(0.1 * len(views))))
        drop = set(np.argsort(-per_view)[:k].tolist())
        rejected += [views[i] for i in drop]
        views = [v for i, v in enumerate(views) if i not in drop]
        K0 = K if model == "fisheye" else None
    counts = np.array([len(d.ids) for d in views])
    return CalibResult(
        model=model, K=K, D=D, rms=_total_rms(per_view, counts), per_view_rms=per_view,
        views=views, rvecs=rvecs, tvecs=tvecs, rejected=rejected,
    )


def choose_model(results: dict[str, CalibResult]) -> str:
    """Prefer fisheye (the physical lens type) unless pinhole is clearly (>10%) better."""
    if "fisheye" in results and "pinhole" in results:
        return "pinhole" if results["pinhole"].rms < 0.9 * results["fisheye"].rms else "fisheye"
    return next(iter(results))


def validity(model: str, K: np.ndarray, D: np.ndarray, size: tuple[int, int], dets: list[Detection]) -> dict:
    """How much of the image the calibration can be trusted for.

    covered_r   : radius (px from the principal point) inside which 99% of board corners lay
    corner_r    : radius of the farthest image corner
    monotonic_r : radius up to which the distortion curve is still increasing; beyond it the
                  model folds back and is meaningless (classic symptom of missing edge views)
    usable      : covered_r reaches at least 80% of corner_r and the model is monotonic there
    """
    cc = K[:2, 2]
    f = float(K[0, 0])
    pts = np.concatenate([d.corners for d in dets])
    covered_r = float(np.percentile(np.linalg.norm(pts - cc, axis=1), 99))
    w, h = size
    corner_r = float(max(np.linalg.norm(np.array(p, float) - cc) for p in [(0, 0), (w, 0), (0, h), (w, h)]))
    if model == "fisheye":
        th = np.linspace(0.0, np.pi * 0.75, 4000)  # up to 135 deg off-axis
        k = D
        r = f * th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8)
    else:  # radial part of the rational model along the x axis
        x = np.linspace(0.0, 4.0, 4000)
        k = np.zeros(8)
        k[:min(8, len(D))] = D[:8]
        r2 = x * x
        rad = (1 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3) / (1 + k[5] * r2 + k[6] * r2**2 + k[7] * r2**3)
        r = f * x * rad
    # Only the part of the curve that maps into the image matters: it must keep increasing
    # until it reaches the image corner radius.
    reach = np.flatnonzero(r >= corner_r)
    end = reach[0] if len(reach) else len(r) - 1
    fold = np.flatnonzero(np.diff(r[:end + 1]) <= 0)
    monotonic_r = float(r[fold[0]]) if len(fold) else (float("inf") if len(reach) else float(r.max()))
    return {
        "covered_r": covered_r, "corner_r": corner_r, "monotonic_r": monotonic_r,
        "covered_frac": covered_r / corner_r,
        "usable": bool(covered_r >= 0.8 * corner_r and monotonic_r >= corner_r),
    }
