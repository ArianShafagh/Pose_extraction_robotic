"""Run 2D pose on the mopper of one camera video, resumable in chunks."""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from moppose.calib.camera import Camera
from moppose.io.video import iter_frames
from moppose.pose2d.backends import PoseBackend
from moppose.pose2d.skeleton import BODY_FEET, EDGE_IDX
from moppose.pose2d.virtual_cam import make_view

CROP_SIZE = (768, 1024)  # Sapiens2 native input (w, h); RTMW resizes it itself


def run_pose(
    cam: Camera,
    video: Path,
    t: np.ndarray,            # (T,) timestamps of frames to process (mopper visible)
    boxes: np.ndarray,        # (T, 4) mopper bbox in raw pixels at those times
    backends: list[PoseBackend],
    out_dir: Path,
    chunk: int = 300,
    batch: int = 4,
) -> dict[str, Path]:
    """Writes out_dir/<backend>/<video stem>/part_XXXXX.npz chunks, then merges them.

    Already finished chunks are skipped, so an interrupted run continues where it stopped.
    """
    out_dir = Path(out_dir)
    parts = [(i, min(i + chunk, len(t))) for i in range(0, len(t), chunk)]
    names = {a: _part_name(t[a:z]) for a, z in parts}
    todo = [(a, z) for a, z in parts if not all(_part_path(out_dir, b.name, video, names[a]).exists() for b in backends)]
    stats = {b.name: 0.0 for b in backends}
    n_done = 0
    for a, z in tqdm(todo, desc=f"pose {video.name}", unit="chunk"):
        want = dict(zip(np.round(t[a:z], 4), range(a, z)))
        pending = [b for b in backends if not _part_path(out_dir, b.name, video, names[a]).exists()]
        res = {b.name: ([], [], []) for b in pending}  # index, raw kpts, scores
        buf_idx, buf_views, buf_crops = [], [], []

        def flush():
            if not buf_idx:
                return
            vboxes = np.stack([v.bbox for v in buf_views])
            for b in pending:
                t0 = time.perf_counter()
                k, sc = b.infer(buf_crops, vboxes)
                stats[b.name] += time.perf_counter() - t0
                res[b.name][0].extend(buf_idx)
                res[b.name][1].extend(v.crop_to_raw(cam, kk) for v, kk in zip(buf_views, k))
                res[b.name][2].extend(sc)
            buf_idx.clear(); buf_views.clear(); buf_crops.clear()

        for fr in iter_frames(video, start_s=max(float(t[a]) - 0.5, 0.0), end_s=float(t[z - 1]) + 0.01):
            i = want.get(round(fr.t, 4))
            if i is None:
                continue
            v = make_view(cam, boxes[i], size=CROP_SIZE)
            buf_idx.append(i); buf_views.append(v); buf_crops.append(v.render(cam, fr.image))
            if len(buf_idx) >= batch:
                flush()
        flush()
        for b in pending:
            idx, kp, sc = res[b.name]
            nj = len(b.native_names)
            p = _part_path(out_dir, b.name, video, names[a])
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                p, t=t[idx], bbox=boxes[idx],
                kpts_native=(np.stack(kp) if kp else np.zeros((0, nj, 2))).astype(np.float32),
                conf_native=(np.stack(sc) if sc else np.zeros((0, nj))).astype(np.float32),
            )
        n_done += len(res[pending[0].name][0]) if pending else 0
    for name, sec in stats.items():
        if n_done:
            print(f"  {name}: {n_done / max(sec, 1e-9):.2f} frames/s")
    return {b.name: merge_parts(out_dir, b, video, [names[a] for a, _ in parts]) for b in backends}


def _part_name(t: np.ndarray) -> str:
    """Chunk file name identifying exactly which frames it holds (safe to resume / reuse)."""
    digest = zlib.crc32(np.round(np.asarray(t, np.float64), 4).tobytes())
    return f"part_{t[0]:010.3f}_{len(t):05d}_{digest:08x}"


def _part_path(out_dir: Path, backend: str, video: Path, name: str) -> Path:
    return out_dir / backend / video.stem / f"{name}.npz"


def merge_parts(out_dir: Path, backend: PoseBackend, video: Path, names: list[str]) -> Path:
    zs = [np.load(_part_path(out_dir, backend.name, video, n)) for n in names]
    t = np.concatenate([z["t"] for z in zs])
    kn = np.concatenate([z["kpts_native"] for z in zs])
    cn = np.concatenate([z["conf_native"] for z in zs])
    bb = np.concatenate([z["bbox"] for z in zs])
    m = backend.to_body_feet
    path = out_dir / backend.name / f"{video.stem}.npz"
    np.savez_compressed(
        path, t=t, bbox=bb,
        kpts=kn[:, m], conf=backend.normalize_conf(cn[:, m]),  # canonical BODY_FEET (23), raw px, conf 0..1
        kpts_native=kn, conf_native=cn,                     # everything the model produced
        joint_names=np.array(BODY_FEET), native_names=np.array(backend.native_names), edges=EDGE_IDX,
    )
    return path
