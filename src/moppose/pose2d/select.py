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
