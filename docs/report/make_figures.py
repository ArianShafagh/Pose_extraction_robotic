"""Figures for the progress report (run from the repo root with `uv run python ...`)."""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from moppose.calib.camera import Camera
from moppose.calib.intrinsics import load_detections
from moppose.io.video import read_frame_at
from moppose.pose2d.people import Tracks
from moppose.pose2d.select import mopper_boxes
from moppose.pose2d.virtual_cam import make_view
from moppose.viz.pose_overlay import draw_skeleton

OUT = Path(__file__).parent / "fig"
OUT.mkdir(exist_ok=True)
REPO = Path.cwd()


def save_jpg(img, name, width=1400):
    s = width / img.shape[1]
    if s < 1:
        img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


# 1. pipeline diagram ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.2)
ax.axis("off")
boxes = [
    (0.2, 2.9, "ChArUco videos\n(per camera)", "#dfe7f2"), (2.4, 2.9, "Lens calibration\n(intrinsics)", "#bfe3c0"),
    (0.2, 1.6, "Mopping videos\n(3 cameras)", "#dfe7f2"), (2.4, 1.6, "Person tracking\n+ mopper choice", "#bfe3c0"),
    (4.6, 1.6, "Fisheye crop\n(virtual camera)", "#bfe3c0"), (6.8, 1.6, "2D pose\nSapiens2 / RTMW", "#bfe3c0"),
    (0.2, 0.3, "Reference object\n(seen by all)", "#dfe7f2"), (2.4, 0.3, "Camera poses\n(extrinsics)", "#f6e3b4"),
    (4.6, 0.3, "Time sync\nof recordings", "#f6e3b4"), (6.8, 0.3, "Triangulation\n+ filtering", "#f6e3b4"),
    (9.0, 0.95, "3D landmarks\n+ joint angles\n-> robot", "#f6e3b4"),
]
for x, y, t, c in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 1.9, 0.95, boxstyle="round,pad=0.05", fc=c, ec="#555", lw=1))
    ax.text(x + 0.95, y + 0.47, t, ha="center", va="center", fontsize=9.5)
arrows = [((2.1, 3.37), (2.4, 3.37)), ((2.1, 2.07), (2.4, 2.07)), ((4.3, 2.07), (4.6, 2.07)), ((6.5, 2.07), (6.8, 2.07)),
          ((2.1, 0.77), (2.4, 0.77)), ((4.3, 0.77), (4.6, 0.77)), ((6.5, 0.77), (6.8, 0.77)), ((8.7, 0.77), (9.0, 1.2)),
          ((3.35, 2.9), (5.4, 2.55)), ((7.75, 1.6), (7.75, 1.25))]
for a, b in arrows:
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
ax.text(0.2, 4.05, "green = done      yellow = next", fontsize=9, color="#333")
fig.tight_layout()
fig.savefig(OUT / "pipeline.png", dpi=180)
plt.close(fig)

# 2. board ---------------------------------------------------------------------
b = cv2.imread(str(REPO / "outputs/board.png"))
save_jpg(b, "board.jpg", 900)

# 3. footage contact sheets ----------------------------------------------------
save_jpg(cv2.imread(str(REPO / "outputs/probe/cam1_20260806_160328_sheet.jpg")), "mopping_sheet.jpg")
save_jpg(cv2.imread(str(REPO / "outputs/probe/cam1_sheet.jpg")), "board_sheet.jpg")

# 4. lens curves: first (failed) vs final calibration of cam1 ------------------
cam1 = Camera.load(REPO / "calibration/session1/cam1_intrinsics.yaml")
first = dict(f=940.5, D=[-0.1574, -0.8121, 8.9782, -24.7758], cov=347)
th = np.linspace(0, np.deg2rad(75), 600)


def r_of(f, k, t):
    return f * t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6 + k[3] * t**8)


fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(np.degrees(th), r_of(cam1.K[0, 0], cam1.D, th), lw=2.2, label="final calibration (cam1)")
ax.plot(np.degrees(th), r_of(first["f"], first["D"], th), lw=2, ls="--", label="first attempt (board only in centre)")
ax.plot(np.degrees(th), cam1.K[0, 0] * np.tan(th), lw=1, color="grey", label="ideal pinhole (no distortion)")
ax.axhline(868, color="k", lw=0.8, ls=":")
ax.text(1, 880, "image corner (868 px)", fontsize=8)
ax.axhline(first["cov"], color="tab:orange", lw=0.8, ls=":")
ax.text(46, first["cov"] - 40, "first attempt: board reached 347 px", fontsize=8, color="tab:orange")
ax.axhline(551, color="tab:blue", lw=0.8, ls=":")
ax.text(1, 563, "final: board reached 551 px", fontsize=8, color="tab:blue")
ax.set_ylim(0, 1100)
ax.set_xlabel("angle of the ray from the optical axis  θ  [deg]")
ax.set_ylabel("distance from image centre  r  [px]")
ax.set_title("Fisheye lens model r(θ) = f·θ·(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "lens_curve.png", dpi=180)
plt.close(fig)

# 5. calibration report images -------------------------------------------------
for c in ("cam1", "cam3", "cam4"):
    for kind in ("coverage", "undistort", "work_area"):
        save_jpg(cv2.imread(str(REPO / f"calibration/session1/{c}_{kind}.jpg")), f"{c}_{kind}.jpg", 1100)

# 6. tracking preview ----------------------------------------------------------
save_jpg(cv2.imread(str(REPO / "outputs/session1/people/cam1_tracks.jpg")), "tracks.jpg", 1100)

# 7. virtual-camera crop example ------------------------------------------------
tr = Tracks.load(REPO / "outputs/session1/people/cam1.npz")
chain = json.loads((REPO / "outputs/session1/people/cam1_mopper.json").read_text())["chain"]
t, _, bb = mopper_boxes(tr, chain)
sap = np.load(REPO / "outputs/session1/pose2d/sapiens2/cam1_20260806_160328.npz")
i = len(sap["t"]) // 3
ti = float(sap["t"][i])
j = int(np.argmin(np.abs(t - ti)))
video = REPO / "data/session1/cam1_20260806_160328.mkv"
raw = read_frame_at(video, ti).image
view = make_view(cam1, bb[j])
crop = view.render(cam1, raw)
k_crop = view.raw_to_crop(cam1, sap["kpts"][i])
draw_skeleton(crop, k_crop, sap["conf"][i], sap["edges"], (255, 80, 255))
x1, y1, x2, y2 = view.bbox.astype(int)
cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 220, 0), 2)
# outline of the crop drawn on the raw frame
w, h = view.size
edge = np.concatenate([np.c_[np.linspace(0, w - 1, 60), np.zeros(60)], np.c_[np.full(60, w - 1), np.linspace(0, h - 1, 60)],
                       np.c_[np.linspace(w - 1, 0, 60), np.full(60, h - 1)], np.c_[np.zeros(60), np.linspace(h - 1, 0, 60)]])
poly = view.crop_to_raw(cam1, edge).astype(np.int32)
rawv = raw.copy()
cv2.polylines(rawv, [poly.reshape(-1, 1, 2)], True, (0, 220, 255), 3)
bx = bb[j].astype(int)
cv2.rectangle(rawv, tuple(bx[:2]), tuple(bx[2:]), (0, 220, 0), 2)
crop_s = cv2.resize(crop, (int(crop.shape[1] * rawv.shape[0] / crop.shape[0]), rawv.shape[0]))
save_jpg(np.hstack([rawv, np.full((rawv.shape[0], 12, 3), 255, np.uint8), crop_s]), "virtual_crop.jpg", 1500)

# 8. pose preview + per-joint agreement plot -----------------------------------
save_jpg(cv2.imread(str(REPO / "outputs/session1/pose2d/cam1_preview.jpg")), "pose_preview.jpg", 1300)
rt = np.load(REPO / "outputs/session1/pose2d/rtmw/cam1_20260806_160328.npz")
d = np.median(np.linalg.norm(rt["kpts"] - sap["kpts"], axis=2), axis=0)
names = [str(n) for n in sap["joint_names"]]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar(range(len(names)), d, color="tab:purple")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=70, fontsize=7)
ax.set_ylabel("median distance [px]")
ax.set_title("Agreement of Sapiens2 and RTMW per joint (cam1, 301 frames)", fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "agreement.png", dpi=180)
plt.close(fig)

# 9. confidence / jitter numbers for the text ------------------------------------
def jitter(k):
    return float(np.nanmedian(np.linalg.norm(k[2:] - 2 * k[1:-1] + k[:-2], axis=2)))


stats = {
    "sap_conf": float(sap["conf"].mean()), "rt_conf": float(rt["conf"].mean()),
    "sap_jit": jitter(sap["kpts"]), "rt_jit": jitter(rt["kpts"]), "n": int(len(sap["t"])),
}
(OUT / "stats.json").write_text(json.dumps(stats, indent=1))
print(stats)
print("figures:", sorted(p.name for p in OUT.iterdir()))
