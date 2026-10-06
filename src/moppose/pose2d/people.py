"""Person detection + tracking on the raw fisheye frames (YOLO26 + BoT-SORT via ultralytics)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from moppose.io.video import iter_frames, open_video


@dataclass
class Tracks:
    """One row per (frame, person). Frames are identified by their video timestamp."""
    t: np.ndarray            # (N,) seconds
    frame_index: np.ndarray  # (N,)
    track_id: np.ndarray     # (N,) -1 = untracked detection
    bbox: np.ndarray         # (N, 4) x1 y1 x2 y2 raw pixels
    score: np.ndarray        # (N,)
    frame_t: np.ndarray      # (F,) timestamps of ALL processed frames (also frames without people)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> "Tracks":
        z = np.load(path)
        return cls(**{k: z[k] for k in z.files})


def track_people(
    video: Path,
    model: str = "yolo26x.pt",
    imgsz: int = 1280,
    conf: float = 0.25,
    start_s: float = 0.0,
    end_s: float | None = None,
    device: str | None = None,
) -> Tracks:
    from ultralytics import YOLO

    yolo = YOLO(model)
    cap = open_video(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    rows_t, rows_fi, rows_id, rows_bb, rows_sc, frame_t = [], [], [], [], [], []
    for fr in tqdm(iter_frames(video, start_s=start_s, end_s=end_s), total=n or None,
                   desc=f"people {Path(video).name}", unit="f"):
        frame_t.append(fr.t)
        res = yolo.track(fr.image, persist=True, classes=[0], conf=conf, imgsz=imgsz,
                         tracker="botsort.yaml", verbose=False, device=device)[0]
        b = res.boxes
        if b is None or len(b) == 0:
            continue
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else np.full(len(b), -1)
        rows_t.append(np.full(len(b), fr.t))
        rows_fi.append(np.full(len(b), fr.index))
        rows_id.append(ids)
        rows_bb.append(b.xyxy.cpu().numpy())
        rows_sc.append(b.conf.cpu().numpy())
    cat = (lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape))
    return Tracks(
        t=cat(rows_t, (0,)), frame_index=cat(rows_fi, (0,)).astype(int), track_id=cat(rows_id, (0,)).astype(int),
        bbox=cat(rows_bb, (0, 4)), score=cat(rows_sc, (0,)), frame_t=np.asarray(frame_t),
    )
