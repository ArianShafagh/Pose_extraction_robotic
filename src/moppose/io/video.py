"""Video reading with real per-frame timestamps.

Security cameras / NVR exports are often variable frame rate (VFR) and drop
frames, so frame_index / fps is NOT a reliable clock. Everything downstream
(sync, triangulation) uses the container timestamp of each frame instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    t: float  # seconds from start of file (container PTS)
    image: np.ndarray


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps_nominal: float
    frame_count_header: int
    duration_s: float
    codec: str
    # Filled only when timestamps are scanned:
    frames_scanned: int | None = None
    fps_measured: float | None = None
    dt_median_ms: float | None = None
    dt_min_ms: float | None = None
    dt_max_ms: float | None = None
    gaps: int | None = None  # frame intervals > 1.5x median (dropped frames)
    is_vfr: bool | None = None
    ffprobe: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(
            f"OpenCV cannot open {path}. If it is a proprietary NVR format (.dav etc.) "
            f"convert it first, e.g.: ffmpeg -i in.dav -c copy out.mp4"
        )
    return cap


def _fourcc(cap: cv2.VideoCapture) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00") or "?"


def iter_frames(
    path: str | Path,
    every_s: float = 0.0,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Iterator[Frame]:
    """Yield frames with timestamps. `every_s` > 0 subsamples by time (not index).

    Uses grab() for skipped frames so only kept frames are fully retrieved.
    """
    cap = open_video(path)
    try:
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        next_t = start_s
        while True:
            if not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end_s is not None and t > end_s:
                break
            if t + 1e-6 >= next_t:
                ok, img = cap.retrieve()
                if ok:
                    yield Frame(idx, t, img)
                next_t = t + every_s if every_s > 0 else t
            idx += 1
    finally:
        cap.release()


def read_frame_at(path: str | Path, t_s: float) -> Frame:
    cap = open_video(path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, img = cap.read()
        if not ok:
            raise IOError(f"could not read frame at {t_s:.2f}s from {path}")
        return Frame(idx, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, img)
    finally:
        cap.release()


def _ffprobe(path: Path) -> dict | None:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,profile,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration:"
             "format=format_name,start_time,duration,bit_rate:format_tags",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return json.loads(out.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return None


def probe(path: str | Path, scan: bool = True) -> VideoInfo:
    """Basic metadata; with scan=True also walks every frame to measure real timing."""
    path = Path(path)
    cap = open_video(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = VideoInfo(
        path=str(path), width=w, height=h, fps_nominal=fps, frame_count_header=n,
        duration_s=n / fps if fps > 0 else float("nan"), codec=_fourcc(cap),
    )
    if scan:
        ts = []
        while cap.grab():
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        ts = np.asarray(ts, dtype=np.float64)
        info.frames_scanned = int(len(ts))
        if len(ts) > 2:
            dt = np.diff(ts)
            dt = dt[dt > 0]
            med = float(np.median(dt))
            info.dt_median_ms = med
            info.dt_min_ms = float(dt.min())
            info.dt_max_ms = float(dt.max())
            info.gaps = int(np.sum(dt > 1.5 * med))
            info.duration_s = float((ts[-1] - ts[0]) / 1000.0)
            info.fps_measured = (len(ts) - 1) / info.duration_s if info.duration_s > 0 else None
            # VFR if intervals spread by more than ~10% of the median
            info.is_vfr = bool(np.percentile(dt, 95) - np.percentile(dt, 5) > 0.1 * med)
    cap.release()
    info.ffprobe = _ffprobe(path)
    return info


def contact_sheet(path: str | Path, n: int = 12, cols: int = 4, thumb_w: int = 480) -> np.ndarray:
    """Grid of n frames evenly spaced in time, each labelled with its timestamp."""
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = count / fps if count > 0 else 0.0
    thumbs = []
    for k in range(n):
        t = dur * (k + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        if not ok:
            continue
        real_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        s = thumb_w / img.shape[1]
        img = cv2.resize(img, (thumb_w, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{real_t:7.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, f"{real_t:7.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        thumbs.append(img)
    cap.release()
    if not thumbs:
        raise IOError(f"no frames could be read from {path}")
    th, tw = thumbs[0].shape[:2]
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, img in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = img[:th, :tw]
    return sheet
