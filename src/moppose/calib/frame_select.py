"""Choosing a good, diverse subset of calibration views from a long video.

A fisheye calibration is only as good as its coverage: corners must appear near
the image borders/corners, where distortion is strongest. Neighbouring video
frames are nearly identical, so we greedily pick views that cover image cells
that are still under-represented.
"""

from __future__ import annotations

import numpy as np

from moppose.calib.charuco import Detection


def coverage_cells(corners: np.ndarray, size: tuple[int, int], grid: int) -> np.ndarray:
    """Flat indices of the grid cells (grid x grid over the image) hit by the corners."""
    w, h = size
    cx = np.clip((corners[:, 0] / w * grid).astype(int), 0, grid - 1)
    cy = np.clip((corners[:, 1] / h * grid).astype(int), 0, grid - 1)
    return np.unique(cy * grid + cx)


def coverage_map(dets: list[Detection], size: tuple[int, int], grid: int = 12) -> np.ndarray:
    """(grid, grid) count of corners per cell, for the coverage heatmap."""
    m = np.zeros(grid * grid, np.int64)
    w, h = size
    for d in dets:
        cx = np.clip((d.corners[:, 0] / w * grid).astype(int), 0, grid - 1)
        cy = np.clip((d.corners[:, 1] / h * grid).astype(int), 0, grid - 1)
        np.add.at(m, cy * grid + cx, 1)
    return m.reshape(grid, grid)


def select_views(
    dets: list[Detection],
    size: tuple[int, int],
    max_views: int,
    grid: int = 8,
) -> list[Detection]:
    """Greedy diminishing-returns coverage selection.

    Score of a candidate = sum over its cells of 1 / (1 + times_cell_already_used)
    + a small bonus for corner count and sharpness. Each pick reduces the value of
    the cells it covered, so the next pick prefers other image regions.
    """
    if len(dets) <= max_views:
        return list(dets)
    cells = [coverage_cells(d.corners, size, grid) for d in dets]
    n_corners = np.array([len(d.ids) for d in dets], float)
    sharp = np.array([d.sharpness for d in dets], float)
    bonus = 0.25 * n_corners / n_corners.max() + 0.25 * sharp / max(sharp.max(), 1e-9)

    used = np.zeros(grid * grid)
    remaining = set(range(len(dets)))
    chosen: list[int] = []
    # Avoid picking near-duplicate frames: block +-0.5 s around each chosen one.
    times = np.array([d.t for d in dets])
    while remaining and len(chosen) < max_views:
        rem = np.fromiter(remaining, int)
        scores = np.array([np.sum(1.0 / (1.0 + used[cells[i]])) for i in rem]) + bonus[rem]
        best = int(rem[np.argmax(scores)])
        chosen.append(best)
        used[cells[best]] += 1
        near = rem[np.abs(times[rem] - times[best]) < 0.5]
        remaining.difference_update(near.tolist())
        remaining.discard(best)
    return [dets[i] for i in sorted(chosen, key=lambda i: dets[i].t)]
