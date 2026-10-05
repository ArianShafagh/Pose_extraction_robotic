"""Images that let you judge a calibration by eye."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from moppose.calib.camera import Camera  # noqa: E402
from moppose.calib.charuco import Detection  # noqa: E402
from moppose.calib.frame_select import coverage_map  # noqa: E402
from moppose.calib.intrinsics import CalibResult  # noqa: E402


def coverage_image(background: np.ndarray, used: list[Detection], rejected: list[Detection], grid: int = 12) -> np.ndarray:
    """Corner positions of all used (green) / rejected (red) views over a heatmap of cell counts.

    Empty (dark) cells near the image border mean the distortion there is extrapolated:
    record more board views in those areas.
    """
    h, w = background.shape[:2]
    cov = coverage_map(used, (w, h), grid).astype(np.float32)
    heat = cv2.resize((np.clip(cov / max(cov.max(), 1), 0, 1) * 255).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    out = cv2.addWeighted(background, 0.45, heat, 0.55, 0)
    empty = cov == 0
    for (gy, gx) in zip(*np.nonzero(empty)):
        x0, y0 = int(gx * w / grid), int(gy * h / grid)
        x1, y1 = int((gx + 1) * w / grid), int((gy + 1) * h / grid)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 1)
    for d in rejected:
        for x, y in d.corners:
            cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), -1)
    for d in used:
        for x, y in d.corners:
            cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1)
    txt = f"views used {len(used)}  rejected {len(rejected)}  empty cells {int(empty.sum())}/{grid * grid}"
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def per_view_plot(results: dict[str, CalibResult], path: Path) -> None:
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3 * len(results)), squeeze=False)
    for ax, (name, r) in zip(axes[:, 0], results.items()):
        ax.bar([f"{d.t:.1f}" for d in r.views], r.per_view_rms, color="tab:blue")
        ax.axhline(r.rms, color="tab:red", ls="--", label=f"overall RMS {r.rms:.3f}px")
        ax.set_title(f"{name}: per-view reprojection RMS")
        ax.set_ylabel("px")
        ax.set_xlabel("view time (s)")
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def undistort_check(cam: Camera, image: np.ndarray, balances=(0.0, 0.5, 1.0)) -> np.ndarray:
    """Original next to undistorted versions. Straight edges in the world must look straight."""
    tiles = [_label(image, "original")]
    for b in balances:
        tiles.append(_label(cam.undistort_image(image, balance=b), f"undistorted balance={b}"))
    h, w = image.shape[:2]
    s = 640 / w
    tiles = [cv2.resize(t, (640, int(h * s)), interpolation=cv2.INTER_AREA) for t in tiles]
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:4]) if len(tiles) >= 4 else np.zeros_like(top)
    return np.vstack([top, bottom])


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    scale = img.shape[1] / 1000
    cv2.putText(img, text, (10, int(40 * scale) + 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(4 * scale) + 2)
    cv2.putText(img, text, (10, int(40 * scale) + 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), int(2 * scale) + 1)
    return img
