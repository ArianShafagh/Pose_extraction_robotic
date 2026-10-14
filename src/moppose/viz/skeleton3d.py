"""Pictures and videos of the 3D reconstruction."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from moppose.calib.camera import Camera  # noqa: E402

LEFT, RIGHT, MID = "#d62728", "#1f77b4", "#555555"


def _edge_color(names, a, b):
    s = names[a] + names[b]
    return LEFT if "left" in s and "right" not in s else RIGHT if "right" in s and "left" not in s else MID


def _draw(ax, x, edges, names, lw=2.0):
    for a, b in edges:
        if np.all(np.isfinite(x[[a, b]])):
            ax.plot(*x[[a, b]].T, color=_edge_color(names, a, b), lw=lw)


def overview(X: np.ndarray, t: np.ndarray, edges, names, cams: list[Camera], path: Path, n_poses: int = 6) -> None:
    """Top view of the walked path with the cameras, plus a few 3D poses."""
    pel = np.nanmean(X[:, [names.index("left_hip"), names.index("right_hip")]], axis=1)
    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    sc = ax.scatter(pel[:, 0], pel[:, 1], c=t, s=2, cmap="viridis")
    for c in cams:
        p = c.center_world
        ax.plot(p[0], p[1], "k^", ms=10)
        ax.annotate(f"{c.name} ({p[2]:.1f} m high)", p[:2], textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top view: pelvis path over time and camera positions")
    fig.colorbar(sc, ax=ax, label="time [s]")
    ax.grid(alpha=0.3)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    for k, i in enumerate(np.linspace(0, len(t) - 1, n_poses).astype(int)):
        x = X[i] - np.r_[np.nanmean(X[i, :, :2], axis=0), 0] + np.r_[k * 0.9, 0, 0]
        _draw(ax3, x, edges, names)
        ax3.text(k * 0.9, 0, -0.15, f"{t[i]:.0f}s", fontsize=8, ha="center")
    ax3.set_box_aspect((n_poses * 0.9, 1.0, 2.0))
    ax3.set_xlim(-0.5, n_poses * 0.9 - 0.4)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_zlim(0, 2.0)
    ax3.view_init(elev=12, azim=-80)
    ax3.set_title("Skeleton at a few moments (metres, floor at z = 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bone_plot(bl: np.ndarray, t: np.ndarray, edges, names, path: Path) -> None:
    """Bone lengths over time - they should be (nearly) constant."""
    keep = [i for i, (a, b) in enumerate(edges) if not any(k in names[a] + names[b] for k in ("eye", "ear", "nose", "toe"))]
    fig, ax = plt.subplots(figsize=(13, 5))
    for i in keep:
        a, b = edges[i]
        ax.plot(t, bl[:, i] * 100, lw=0.6, label=f"{names[a]}-{names[b]}".replace("left_", "L ").replace("right_", "R "))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("length [cm]")
    ax.set_title("Bone lengths over the session (constant = consistent reconstruction)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def reprojection_grid(cams: list[Camera], videos: dict[str, Path], offsets: dict[str, float], X: np.ndarray,
                      t: np.ndarray, edges, uv2d: np.ndarray | None, times: list[float], path: Path, size: int = 360):
    """For each time (rows) and camera (columns): the 3D skeleton projected into the raw frame
    (green) next to the 2D detections (magenta). They must coincide."""
    from moppose.io.video import read_frame_at

    rows = []
    for tt in times:
        i = int(np.argmin(np.abs(t - tt)))
        tiles = []
        for c, cam in enumerate(cams):
            img = read_frame_at(videos[cam.name], t[i] + offsets.get(cam.name, 0.0)).image
            ok = np.all(np.isfinite(X[i]), axis=1)
            p = np.full((X.shape[1], 2), np.nan)
            p[ok] = cam.project(X[i][ok])
            if uv2d is not None:
                for a, b in edges:
                    q = uv2d[c, i, [a, b]]
                    if np.all(np.isfinite(q)):
                        cv2.line(img, tuple(q[0].astype(int)), tuple(q[1].astype(int)), (255, 0, 255), 1, cv2.LINE_AA)
            for a, b in edges:
                if np.all(np.isfinite(p[[a, b]])):
                    cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
            pts = p[np.all(np.isfinite(p), 1)]
            if len(pts):
                (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                s = max(x1 - x0, y1 - y0) * 1.5 + 40
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                a0, b0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
                crop = img[b0:int(min(cy + s / 2, img.shape[0])), a0:int(min(cx + s / 2, img.shape[1]))]
            else:
                crop = img
            tile = cv2.resize(crop, (size, size))
            cv2.putText(tile, f"{cam.name} {t[i]:.1f}s", (6, 20), 0, 0.55, (0, 255, 255), 2)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows))


def render_video(X: np.ndarray, t: np.ndarray, edges, names, path: Path, t0: float, t1: float,
                 size=(960, 720)) -> None:
    """3D skeleton animation (matplotlib frames -> mp4) between t0 and t1."""
    sel = np.flatnonzero((t >= t0) & (t <= t1))
    if len(sel) == 0:
        return
    fps = 1.0 / np.median(np.diff(t))
    lim = np.nanpercentile(X[sel].reshape(-1, 3), [1, 99], axis=0)
    ctr = lim.mean(0)
    half = max(np.max(lim[1, :2] - lim[0, :2]) / 2 + 0.5, 1.2)
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    pel = np.nanmean(X[:, [names.index("left_hip"), names.index("right_hip")]], axis=1)
    for k, i in enumerate(sel):
        fig.clf()
        ax = fig.add_subplot(projection="3d")
        _draw(ax, X[i], edges, names, lw=2.5)
        ax.plot(*pel[sel[0]:i + 1, :2].T, np.zeros(i + 1 - sel[0]), color="#999999", lw=1)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(0, 2.0)
        ax.set_box_aspect((1, 1, 2.0 / (2 * half)))
        ax.view_init(elev=20, azim=-60 + 0.3 * k)
        ax.set_title(f"t = {t[i]:6.2f} s")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        wr.write(cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_RGB2BGR))
    wr.release()
    plt.close(fig)
