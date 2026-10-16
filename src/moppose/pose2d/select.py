"""Pick the mopping person among all tracked people.

Other people in the room mostly sit, so the mopper is the person whose feet travel
the farthest. Trackers fragment a person into several ids (occlusions, leaving the
view); fragments are chained when one ends close in time and space to where the
next one starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from moppose.pose2d.people import Tracks


@dataclass
class TrackSummary:
    track_id: int
    t0: float
    t1: float
    n: int
    path_px: float          # distance travelled by the bbox bottom-centre (feet)
    start: np.ndarray       # bottom-centre at t0
    end: np.ndarray         # bottom-centre at t1


def _feet(bbox: np.ndarray) -> np.ndarray:
    return np.c_[(bbox[:, 0] + bbox[:, 2]) / 2, bbox[:, 3]]


def summarize(tr: Tracks) -> dict[int, TrackSummary]:
    out = {}
    for tid in np.unique(tr.track_id[tr.track_id >= 0]):
        m = tr.track_id == tid
        order = np.argsort(tr.t[m])
        t, feet = tr.t[m][order], _feet(tr.bbox[m][order])
        # median-smooth before measuring path so box jitter does not count as walking
        k = 5
        if len(feet) > k:
            pad = np.pad(feet, ((k // 2, k // 2), (0, 0)), mode="edge")
            feet = np.stack([np.median(pad[i:i + k], axis=0) for i in range(len(feet))])
        path = float(np.sum(np.linalg.norm(np.diff(feet, axis=0), axis=1))) if len(feet) > 1 else 0.0
        out[int(tid)] = TrackSummary(int(tid), float(t[0]), float(t[-1]), int(m.sum()), path, feet[0], feet[-1])
    return out


def select_mopper(
    tr: Tracks,
    seed_id: int | None = None,
    max_gap_s: float = 3.0,
    max_jump_px: float = 200.0,
) -> list[int]:
    """Track ids belonging to the mopper, in time order.

    seed_id: start from this track (manual choice); default = the track that travels farthest.
    """
    s = summarize(tr)
    if not s:
        return []
    seed = seed_id if seed_id is not None else max(s.values(), key=lambda x: x.path_px).track_id
    chain = [seed]
    used = {seed}
    # extend forward
    while True:
        last = s[chain[-1]]
        cands = [c for c in s.values() if c.track_id not in used and 0 <= c.t0 - last.t1 <= max_gap_s
                 and np.linalg.norm(c.start - last.end) <= max_jump_px]
        if not cands:
            break
        nxt = min(cands, key=lambda c: (c.t0 - last.t1) + np.linalg.norm(c.start - last.end) / max_jump_px)
        chain.append(nxt.track_id)
        used.add(nxt.track_id)
    # extend backward
    while True:
        first = s[chain[0]]
        cands = [c for c in s.values() if c.track_id not in used and 0 <= first.t0 - c.t1 <= max_gap_s
                 and np.linalg.norm(first.start - c.end) <= max_jump_px]
        if not cands:
            break
        prv = min(cands, key=lambda c: (first.t0 - c.t1) + np.linalg.norm(first.start - c.end) / max_jump_px)
        chain.insert(0, prv.track_id)
        used.add(prv.track_id)
    return chain


def mopper_boxes(tr: Tracks, chain: list[int], smooth: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame mopper bbox: (t, frame_index, bbox) sorted by time, lightly median-smoothed."""
    m = np.isin(tr.track_id, chain)
    order = np.lexsort((-tr.score[m], tr.t[m]))  # by time, best score first within a frame
    t, fi, bb = tr.t[m][order], tr.frame_index[m][order], tr.bbox[m][order]
    # one box per frame (if two chained ids overlap in a frame keep the higher score)
    _, first = np.unique(t, return_index=True)
    t, fi, bb = t[first], fi[first], bb[first]
    if smooth > 1 and len(bb) > smooth:
        pad = np.pad(bb, ((smooth // 2, smooth // 2), (0, 0)), mode="edge")
        bb = np.stack([np.median(pad[i:i + smooth], axis=0) for i in range(len(bb))])
    return t, fi, bb


def preview(image: np.ndarray, tr: Tracks, chain: list[int]) -> np.ndarray:
    """All tracks' feet paths (grey) with ids; the selected mopper chain in green."""
    out = image.copy()
    s = summarize(tr)
    for tid, ts in sorted(s.items(), key=lambda kv: kv[1].path_px):
        m = tr.track_id == tid
        feet = _feet(tr.bbox[m][np.argsort(tr.t[m])]).astype(np.int32)
        color = (0, 255, 0) if tid in chain else (160, 160, 160)
        cv2.polylines(out, [feet.reshape(-1, 1, 2)], False, color, 3 if tid in chain else 1)
        x, y = feet[len(feet) // 2]
        label = f"{tid}"
        cv2.putText(out, label, (int(x) + 4, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (int(x) + 4, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    txt = f"mopper = tracks {chain}"
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return out


def _feet_series(tr: Tracks, tid: int) -> tuple[np.ndarray, np.ndarray]:
    m = tr.track_id == tid
    order = np.argsort(tr.t[m])
    return tr.t[m][order], _feet(tr.bbox[m][order])


def select_actor_multiview(tracks: dict[str, Tracks], cams: list, top_k: int = 4, max_err_px: float = 60.0,
                           min_overlap: int = 30) -> tuple[dict[str, list[int]], dict]:
    """Choose the actor's track in every camera using the calibrated rig.

    For the most moving candidate tracks of each camera, every combination is checked: the
    bottom-centre of the person box (≈ feet) is triangulated over the shared time and the
    reprojection error measured. The same person seen by all cameras gives a small error;
    different people give errors of hundreds of pixels. Time offsets between the cameras
    (< 0.5 s) only add a little error here. The winning tracks are then extended by
    stitching their fragments (select_mopper).

    Returns {cam: chain of track ids} and diagnostics.
    """
    from itertools import product

    from moppose.triangulate.dlt import triangulate

    names = [c.name for c in cams]
    cands = {}
    for n in names:
        s = summarize(tracks[n])
        # movement x duration: the actor moves, seated people barely do, short false tracks are short
        score = {tid: x.path_px * (x.t1 - x.t0) for tid, x in s.items()}
        cands[n] = sorted(score, key=score.get, reverse=True)[:top_k]
    series = {n: {tid: _feet_series(tracks[n], tid) for tid in cands[n]} for n in names}
    results = []
    for combo in product(*[cands[n] for n in names]):
        t_ref, p_ref = series[names[0]][combo[0]]
        uv = np.full((len(names), len(t_ref), 2), np.nan)
        uv[0] = p_ref
        for c in range(1, len(names)):
            t_c, p_c = series[names[c]][combo[c]]
            j = np.clip(np.searchsorted(t_c, t_ref), 1, len(t_c) - 1)
            j = np.where(np.abs(t_c[j - 1] - t_ref) < np.abs(t_c[j] - t_ref), j - 1, j)
            near = np.abs(t_c[j] - t_ref) < 0.1
            uv[c, near] = p_c[j[near]]
        ok = np.all(np.isfinite(uv[..., 0]), axis=0)
        if ok.sum() < min_overlap:
            continue
        sel = np.flatnonzero(ok)[:: max(1, ok.sum() // 400)]
        _, err = triangulate(cams, uv[:, sel], np.ones((len(names), len(sel))), min_conf=0.5, max_reproj_px=1e9)
        results.append((float(np.nanmedian(err)), int(ok.sum()), combo))
    if not results:
        chains = {n: select_mopper(tracks[n]) for n in names}
        return chains, {"method": "fallback: farthest walker (no overlapping candidates)"}
    results.sort(key=lambda r: (r[0], -r[1]))
    err, overlap, combo = results[0]
    if err > max_err_px:
        chains = {n: select_mopper(tracks[n]) for n in names}
        return chains, {"method": "fallback: farthest walker", "best_multiview_err_px": err}
    chains = {n: select_mopper(tracks[n], seed_id=int(tid)) for n, tid in zip(names, combo)}
    info = {"method": "multiview", "seed_tracks": {n: int(t) for n, t in zip(names, combo)},
            "feet_reproj_err_px": err, "overlap_frames": overlap,
            "runner_up_err_px": results[1][0] if len(results) > 1 else None}
    return chains, info
