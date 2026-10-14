"""2D pose tracks of several cameras on one common clock.

Each camera records with its own clock. We use the first camera as the reference
clock and describe every other camera by an offset d_c:

    time in camera c's video  =  reference time + d_c

so the observation of camera c at reference time tau is its track sampled at tau + d_c.
Tracks are sampled at ~15 fps; values in between are linearly interpolated, but never
across a gap in the track.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Track2D:
    cam: str
    t: np.ndarray        # (T,) video timestamps of this camera, s
    kpts: np.ndarray     # (T, J, 2) raw fisheye pixels
    conf: np.ndarray     # (T, J) 0..1
    joint_names: list[str]
    edges: np.ndarray

    @classmethod
    def load(cls, cam: str, path: Path) -> "Track2D":
        z = np.load(path)
        order = np.argsort(z["t"])
        return cls(cam, z["t"][order], z["kpts"][order].astype(np.float64), z["conf"][order].astype(np.float64),
                   [str(n) for n in z["joint_names"]], z["edges"])

    def sample(self, times: np.ndarray, max_gap: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints at arbitrary times (linear interpolation). Returns (N, J, 2), (N, J); NaN / 0 outside data."""
        times = np.asarray(times, np.float64)
        i = np.searchsorted(self.t, times)
        i0 = np.clip(i - 1, 0, len(self.t) - 1)
        i1 = np.clip(i, 0, len(self.t) - 1)
        t0, t1 = self.t[i0], self.t[i1]
        ok = (times >= self.t[0]) & (times <= self.t[-1]) & ((t1 - t0) <= max_gap)
        w = np.where(t1 > t0, (times - t0) / np.where(t1 > t0, t1 - t0, 1), 0.0)[:, None, None]
        k = (1 - w) * self.kpts[i0] + w * self.kpts[i1]
        c = np.minimum(self.conf[i0], self.conf[i1])
        k[~ok] = np.nan
        c[~ok] = 0.0
        return k, c


def load_tracks(pose_dir: Path, videos: dict[str, Path]) -> dict[str, Track2D]:
    """{cam: Track2D} from outputs/<session>/pose2d/<backend>/<video stem>.npz."""
    return {cam: Track2D.load(cam, pose_dir / f"{Path(v).stem}.npz") for cam, v in videos.items()}


def on_common_clock(tracks: dict[str, Track2D], offsets: dict[str, float], times: np.ndarray):
    """Stack all cameras at reference times: kpts (C, N, J, 2), conf (C, N, J)."""
    ks, cs = [], []
    for cam, tr in tracks.items():
        k, c = tr.sample(times + offsets.get(cam, 0.0))
        ks.append(k)
        cs.append(c)
    return np.stack(ks), np.stack(cs)
