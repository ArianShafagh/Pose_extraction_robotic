# moppose

3D pose extraction of a person **mopping**, filmed by **3 fisheye security cameras**.
The 3D body landmarks produced here will later drive a robot that shadows the motion.
The robot side is a separate project; this repo stops at clean 3D landmarks and joint angles.

```
board videos   ──► 1. lens calibration per camera (ChArUco)                    ✅ done
any recording  ──► 2. find the actor ─► 3. 2D pose per camera                  ✅ done (GPU PC)
                   4. camera rig in metres (self-calibrated from the person)    ✅ done, reused
                   5. time sync of the 3 recordings                             ✅ automatic
                   6. triangulation ─► cleaning ─► smoothing ─► 3D landmarks    ✅ done
                   7. full-length 3D skeleton video + report                    ✅ done
                   8. skeleton fit ─► joint angles for the robot                ⏳ next
```

**Process a new recording** (same cameras, any action): copy `configs/recordings/_template.yaml`, set the
name and the 3 videos, then run one command; everything from the videos to the named outputs and the
full-length 3D video happens automatically. See **[docs/AUTOMATION.md](docs/AUTOMATION.md)**.
```bash
uv run moppose run configs/recordings/<your recording>.yaml
```

---

## The project, step by step

This section tells what was done, why it was done, and what came out, in order.

### Step 1: Repository and basic tools
- Created the Python package `moppose` with one command-line tool, `uv run moppose <command>`. It uses `uv` for the environment.
- Videos (`data/`) and generated results (`outputs/`) stay on the computer and are not in git.
- Wrote **synthetic tests** that need no real footage:
  - A fake fisheye camera (~170°) renders a ChArUco board from many angles. The calibration must recover the fake camera's lens.
  - A fake 3-camera rig projects 3D points. Triangulation must get them back (error < 1 mm).

### Step 2: The calibration board (`configs/board.yaml`)
- The board is `charuco_A3.pdf`: **3×2 squares, 115 mm squares, 86.2 mm markers, dictionary DICT_4X4_50 (marker ids 0–2)**.
  - The values were read from the PDF itself.
  - OpenCV's board generator with these settings reproduces the PDF **pixel for pixel**, so they are certain.
- **Problem: too few points.** A 3×2 board has only **2 inner chessboard corners**, far too few to calibrate from.
  - Fix: we also use the 4 corners of each of the 3 markers, giving **up to 14 points per frame** (`use_marker_corners: true`).
- **Problem: an OpenCV bug.** OpenCV 4.11 returns the chessboard corners shifted by **+0.5 px**, while the marker corners are correct.
  - Mixing the two biased the focal length by about 1.4%.
  - Fix: at start-up, `calib/charuco.py` (`Target`) measures this offset on a synthetic image with known corners and removes it. Focal length then comes out within 1%.
- Synthetic test with this board: about 40 frames give viewing directions accurate to about 0.45°, about 100 frames to about 0.15°. **A small board needs many varied frames.**

### Step 3: Looking at the footage (`moppose probe`)
**Mopping videos** (`data/session1/camX_20260806_160328.mkv`, cameras **cam1, cam3, cam4**):
- 1280×1024 pixels, H.264, about 593 s long.
- The file says 15 fps, but the **real rate is ~30 fps and uneven** (variable frame rate), with a few dropped frames (cam1: 3, cam3: 4, cam4: 23).
- Therefore the whole framework uses **each frame's real timestamp, never frame number ÷ fps**.

**The first board videos had a different resolution** (1920×1080 and 1440×1080). They were exports of the 1280×1024 camera stream, scaled up and padded with black side bars.
- A lens calibration is only valid for one pixel grid, so `calib/frame_map.py` crops the black bars and scales calibration frames back to 1280×1024 before detecting the board.
- This mapping is checked automatically by matching the static room background between the board video and the mopping video. It lined up within **0.4–0.9 px**.

### Step 4: Lens calibration, first attempt (not usable)
- `moppose calib-intrinsics` does the following:
  1. Samples a frame every 0.25 s.
  2. Detects the board.
  3. Drops blurry or degenerate frames.
  4. Picks about 80 frames that cover the image as evenly as possible.
  5. Fits two lens models: **fisheye** (OpenCV equidistant, k1–k4) and **pinhole + rational distortion**.
  6. Repeatedly removes bad frames.
- Result of the first recording: the board had only been held in the **middle of the image**. It reached just 39–46% of the way to the image corners.
  - The error looked low (0.5–0.7 px), but the cam1 lens model **folded back on itself** beyond 480 px from the centre, which is meaningless. The undistorted image came out black.
- Added an automatic **usability check** that reports how far the board reached and whether the distortion curve stays valid, then prints **USABLE / NOT USABLE**.
- The board was filmed again, closer to the cameras and moved around the image.

### Step 5: Lens calibration, second attempt (usable ✅)

| camera | usable board frames | error (RMS) | fx / fy (px) | centre cx, cy (px) | mopping area covered by board |
|---|---|---|---|---|---|
| cam1 | 719 | 0.84 px | 911 / 917 | 596, 490 | **99%** |
| cam3 | 595 | 0.95 px | 932 / 939 | 620, 541 | **98%** |
| cam4 | 783 | 0.76 px | 902 / 907 | 636, 505 | **96%** |

- The fisheye and pinhole fits now agree closely (centre within 5–10 px, focal length within 1%). Straight walls, cabinet edges and window frames come out straight after undistortion.
- The board still did not reach the image corners, which show ceiling and walls. So the usability check now compares the board coverage with **where people actually move** in the mopping video (background subtraction, `io/motion.py`). 96–99% of that area is covered.
- cam3's new board video was again a 1920×1080 export; it was mapped automatically (0.87 px check).
- **Saved in git:** [calibration/session1/](calibration/session1/) holds the calibration files, the detected board corners and the report images. That state is tagged `calib-session1`.

### Step 6: Choosing the 2D pose models (research, Oct 2026)
The goal is accuracy, not real time, because the videos are processed offline.

| model | keypoints | accuracy | role |
|---|---|---|---|
| **Sapiens2-0.8B** (Meta, ICLR 2026) | 308 (body, feet, hands, face) | Sapiens2-1B: 80.4 mAP, 5B: 82.3 mAP on a 308-keypoint in-the-wild test (the 0.8B was not reported separately) | **main model** |
| **RTMW** (rtmlib, ONNX) | 133 COCO-WholeBody | 70.1 AP (COCO-WholeBody) | fast baseline for comparison |
| YOLO26x | person boxes | — | finds and tracks people |
| YOLO26x-pose | 17 body | 71.6 AP (COCO body) | not used: no feet |

The robot needs **body + feet joints now and joint angles later**. Hands are not used, but every model keypoint is stored for the future.

### Step 7: The 2D pose pipeline (`src/moppose/pose2d/`)
1. **Find people** (`people.py`): YOLO26x detection with BoT-SORT tracking on the raw fisheye frames.
2. **Pick the mopper** (`select.py`): the person whose feet travel the farthest, since the other people in the room sit.
   - Broken track pieces of the same person are joined when one ends close in time and place to where the next begins.
   - A preview image shows all tracks. You can force a track with `--track ID`.
   - On cam1 it picked the mopper correctly and ignored 5 seated people.
3. **Straighten the person** (`virtual_cam.py`): pose models are trained on normal photos, so for every frame a **virtual pinhole camera** is aimed at the mopper through the lens calibration.
   - This gives an undistorted, upright 768×1024 crop.
   - Keypoints found in the crop are mapped back to the original fisheye pixels, exact to < 0.05 px (tested).
4. **2D pose** (`backends.py`): Sapiens2 (fp16, flip test) and RTMW run on the same crops. Each is reduced to the same **23 joints**: COCO-17 body plus big toe, small toe and heel on each foot.
5. **Save** (`run.py`): work is saved in chunks, so an interrupted run continues where it stopped.

Results on the first 20 s of cam1:

| | Sapiens2-0.8B | RTMW |
|---|---|---|
| mean confidence | **0.87** | 0.81 |
| frame-to-frame jitter | **0.64 px** | 0.84 px |
| speed on RTX 2070 (8 GB) | 0.83 frames/s | ~30 frames/s |
| agreement between the two | 1–4 px on all 23 joints | |

Problems found and fixed:
- The RTMW-x "cocktail13" release put the **left toes on the head**. We switched to the cocktail14 release, whose feet agree with Sapiens2.
- RTMW's confidences are not on a 0–1 scale, so they are mapped with s/(1+s). The raw values are kept.
- `onnxruntime` (CPU only) silently replaced `onnxruntime-gpu`. It is now excluded, and the GPU build is pinned to CUDA 12 to match PyTorch.

### Step 8: Running the heavy model on another computer
- The GPU computer has a **GTX 1080 (8 GB, Pascal)**. Newer PyTorch CUDA 12.8 builds no longer support it, so the
  project uses the **CUDA 12.6 builds (PyTorch 2.14)**, which run on both the GTX 1080 and the laptop's RTX 2070.
- The GTX 1080 has no tensor cores, so fp16 is slow on it. Sapiens2 automatically runs in **fp32 without the flip
  test** there and in fp16 with flip test on RTX cards. Measured on the RTX 2070:

  | Sapiens2 setting | frames/s | GPU memory | change vs best |
  |---|---|---|---|
  | 0.8B fp16 + flip (RTX) | 0.86 | 2.3 GB | - |
  | **0.8B fp32, no flip (GTX 1080)** | 0.48 | 4.2 GB | 0.75 px median |
  | 0.4B fp32 + flip | 0.45 | 2.6 GB | 1.65 px median |

  Keeping the larger model matters more than the flip test.
- Result chunks are tagged with the model settings, so runs with different settings never mix.
- Expected time on the GTX 1080: ~10 h per camera at 30 fps, ~5 h at 15 fps; both computers can run different
  cameras at the same time. See **[docs/REMOTE_GPU.md](docs/REMOTE_GPU.md)**.

### Step 9: Time sync, camera positions and 3D landmarks
- **No calibration object was needed.** The mopper's own 23 joints, seen by all 3 cameras at the same
  moments, give about 190,000 correspondences per camera pair.
- **Time sync:** the timing of her movements gives the offset between the separately recorded videos.
  It is then refined on the epipolar error: **cam3 −247 ms, cam4 +157 ms** relative to cam1, stable over
  10 minutes.
- **Camera poses:** essential matrices → chained translation scales → sparse bundle adjustment with a
  vectorised fisheye projection. The median reprojection error is **1.4 px**.
- **Floor:** 95% of all heel/toe points lie on one plane; z points up from it.
- **Metric scale:** from the mop handle (140 ± 10 cm). The mop's floor joint triangulates 0.6 cm from the
  floor, which is a consistency check. Cameras come out 2.7–3.0 m high and cam1–cam3 12.2 m apart.
- **Triangulation of all ~8,900 frames:**
  - 99.3% of joint samples reconstructed;
  - reprojection 1.6 / 1.8 / 1.6 px;
  - leg and torso bone lengths constant within 2–3%;
  - the right forearm is the weakest (the hand holding the top of the mop is often hidden).
- **Cleaning:** a left/right swap check per camera, a bone-length outlier check, gap filling up to 0.5 s,
  and a 4 Hz zero-phase low-pass filter.

### Step 10: Automation for any recording
- The calibrated rig is saved in `calibration/rig/rig.yaml` and reused (the cameras are fixed), so a new
  recording only needs its own time sync.
- One config per recording (`configs/recordings/*.yaml`) plus the shared `configs/setup.yaml`; every output
  file name is set in the config.
- `moppose run` runs 8 resumable stages. A changed config section reruns only what depends on it.
- **The actor is found automatically:** the person whose positions in the 3 cameras agree in 3D with the
  rig. On session1: feet consistent within 7.6 px, against 40 px for the next-best combination.
- **A full-length 3D skeleton video** is rendered with OpenCV in about 11 ms per frame.

---

## How to use it

### Setup
Requires [uv](https://docs.astral.sh/uv/) and Git (Python 3.12 is installed by uv).
```bash
uv sync --all-extras     # PyTorch (CUDA 12.6, GTX 10xx..RTX 40xx), ultralytics, rtmlib, onnxruntime-gpu, Sapiens2 dependencies
uv run moppose setup     # Sapiens2 code + weights, RTMW, YOLO26; prints the GPU
```

### Data layout (not in git)
```
data/
  calib/cam1.mkv cam3.mkv cam4.mkv          # board videos, one per camera
  session1/cam1_20260806_160328.mkv  ...     # mopping recordings
```
Files are listed in [configs/session.yaml](configs/session.yaml); the board is in [configs/board.yaml](configs/board.yaml).

### Commands
| command | what it does | output |
|---|---|---|
| `moppose probe <videos>` | resolution, real frame rate, dropped frames, contact sheet | `outputs/probe/` |
| `moppose board-image` | draws the board from board.yaml (compare with the print) | `outputs/board.png` |
| `moppose board-check <video>` | quick board detection test (`--guess` tries all dictionaries) | `outputs/board_check/` |
| `moppose calib-intrinsics --cam cam1` | lens calibration + report + USABLE check | `outputs/calib/` |
| `moppose calib-check --cam cam1 --t 5` | undistort a frame of the scene | `outputs/calib/` |
| `moppose people --cam cam1` | track people, pick the mopper (`--track ID` to override) | `outputs/session1/people/` |
| `moppose pose2d --cam cam1` | 2D pose with Sapiens2 + RTMW (`--end 20` for a test) | `outputs/session1/pose2d/` |
| `moppose pose-preview --cam cam1` | skeleton grid (`--video` for an overlay mp4) | `outputs/session1/pose2d/` |
| `moppose pose-all` | people + pose + previews for every camera | as above |
| **`moppose run <recording.yaml>`** | **everything for one recording (actor, 2D, sync, 3D, exports, 3D video)** | `results/<name>/` |
| `moppose recon3d` | 3D stage of session1 incl. self-calibration and mop scale | `outputs/session1/3d/` |
| `moppose calibrate-rig` | re-create the fixed camera rig (only if a camera moved) | `calibration/rig/rig.yaml` |

### Result files
- `calibration/session1/camX_intrinsics.yaml`: lens model (`K`, `D`, size, RMS, coverage). Load it with `Camera.load(...)`.
- `outputs/session1/pose2d/<model>/<video>.npz`, per camera and model:
  - `t (T,)`: seconds, the real frame time;
  - `kpts (T, 23, 2)`: raw fisheye pixels;
  - `conf (T, 23)`: confidence, 0–1;
  - `joint_names`, `edges`;
  - `kpts_native`, `conf_native`: all keypoints of the model.

## Code map
```
src/moppose/
  cli.py                 all commands
  config.py              board.yaml / session.yaml
  io/video.py            frames with real timestamps, probe, contact sheet
  io/motion.py           where people move (calibration coverage check)
  calib/charuco.py       board, detection, marker corners, OpenCV offset fix
  calib/frame_map.py     crop/scale exported calibration videos to the scene resolution
  calib/frame_select.py  pick diverse, sharp board frames
  calib/intrinsics.py    fisheye / pinhole fitting, outlier rejection, usability check
  calib/camera.py        camera model: undistort, project, rectify, save/load
  pose2d/people.py       YOLO26 + BoT-SORT tracking
  pose2d/select.py       choose and stitch the mopper's track
  pose2d/virtual_cam.py  fisheye -> upright pinhole crop and back
  pose2d/backends.py     Sapiens2 and RTMW behind one interface
  pose2d/skeleton.py     the 23 canonical joints and bones
  pose2d/run.py          chunked, resumable pose extraction
  triangulate/dlt.py     weighted multi-view triangulation with bad-camera rejection
  multiview/data.py      2D tracks of all cameras on one clock
  multiview/sync.py      time offsets (motion correlation + epipolar refinement)
  multiview/selfcalib.py camera poses from the person (essential matrices, bundle adjustment)
  multiview/world.py     floor frame, metric scale (mop)
  multiview/pipeline.py  triangulation, left/right fix, bone outliers, smoothing
  recording.py           recording + setup config, output names
  automation.py          `moppose run`: resumable stages
  viz/                   calibration reports, skeleton drawing, 3D plots, full-length 3D video
tests/                   30 synthetic tests (calibration, triangulation, crops, sync, rig, automation)
calibration/session1/    saved lens calibrations (tag calib-session1)
calibration/rig/         the fixed camera rig in metres
configs/setup.yaml       the fixed lab setup
configs/recordings/      one config per recording (_template.yaml, mopping_session1.yaml)
docs/AUTOMATION.md       processing a new recording
docs/REMOTE_GPU.md       running the pose stage on another computer
```

## Conventions
- Camera: `X_cam = R · X_world + t`; pixels (u, v) with the origin at the top-left.
- All keypoints are stored in **raw fisheye pixels**. Triangulation undistorts them with the calibration.
- World frame: metres, **z up, floor at z = 0**, x from cam1 towards cam3, origin in the middle of
  session1's walked area.
- Time: the first camera (cam1) is the reference clock; for every other camera,
  video time = reference time + offset.

## Next steps
1. **Right wrist:** use the RTMW cross-check where Sapiens2's wrist disagrees with the other cameras.
2. **Joint angles:** fit a skeleton with constant bone lengths and export joint angles for the robot.
3. **More precise scale:** a measured object (e.g. the ArUco sheet on the table) instead of the mop (±7%).

## Tests
```bash
uv run pytest
```
30 tests:
- synthetic fisheye calibration (two boards) and the 0.5 px offset;
- triangulation with missing or bad views;
- the virtual crop round-trip on the real cam1 calibration;
- mopper and actor selection;
- time sync and rig self-calibration;
- floor fit and left/right swap repair;
- recording configs, stage skipping and the 3D video.
