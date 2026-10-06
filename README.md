# moppose

Multi-camera 3D pose extraction of a person mopping, from **3 fisheye security cameras**.
The 3D landmarks produced here are meant to drive a robot that shadows the motion
(the robot side is a separate project; this repo stops at clean 3D landmarks).

```
calibration videos ──► per-camera lens calibration (ChArUco)  ─┐
reference object   ──► camera poses in one world frame        ─┤
mopping videos     ──► time sync ─► 2D pose per camera ────────┴─► triangulation ─► 3D landmarks (.npz)
```

## Status

| Stage | Status |
|---|---|
| Video probe (fps, VFR, dropped frames, contact sheet) | done |
| ChArUco board check / dictionary guess | done |
| Per-camera intrinsics (fisheye + pinhole-rational, view selection, outlier rejection, report) | done |
| Multi-view triangulation (weighted DLT, outlier camera rejection) | done |
| Extrinsics from a shared reference object | next |
| Software time sync of the 3 recordings | next |
| 2D pose: YOLO26-pose / RTMPose backends | next |
| Smoothing, 3D export, visualisation | next |

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is used.

```bash
uv sync
```

Pose backends (later) are optional extras: `uv sync --extra yolo` or `uv sync --extra rtm`.

## Data layout

Videos stay local (`data/` and `outputs/` are gitignored):

```
data/
  calib/cam1.mp4 cam2.mp4 cam3.mp4      # ChArUco videos, one camera at a time
  session1/cam1.mp4 cam2.mp4 cam3.mp4   # mopping recordings
```

Edit [configs/session.yaml](configs/session.yaml) to point at your files and
[configs/board.yaml](configs/board.yaml) with your real board (squares, sizes in meters, dictionary).

## Workflow (current)

1. **Look at the footage**
   ```bash
   uv run moppose probe data/session1/cam1.mp4 data/session1/cam2.mp4 data/session1/cam3.mp4
   ```
   Prints resolution, nominal vs. real frame rate, dropped-frame gaps, VFR, and writes a contact sheet to `outputs/probe/`.

2. **Check the board config** before calibrating
   ```bash
   uv run moppose board-image                       # renders outputs/board.png - compare with your print
   uv run moppose board-check data/calib/cam1.mp4   # add --guess if nothing is detected
   ```

3. **Calibrate each camera's lens** (one at a time)
   ```bash
   uv run moppose calib-intrinsics --cam cam1
   ```
   Writes `outputs/calib/cam1_intrinsics.yaml` plus:
   - `cam1_coverage.jpg` – where board corners were seen. Red-outlined cells had no corners: distortion there is guessed. For fisheye lenses the borders matter most.
   - `cam1_per_view_rms.png` – per-view error; spikes are blurry or mis-detected frames.
   - `cam1_undistort.jpg` – original vs. undistorted; straight lines must come out straight.
   - `cam1_work_area.jpg` – where people move in the mopping video (heatmap) vs. the board-covered area (white).
     The run ends with **USABLE** when ≥95% of that motion lies inside the board coverage.

   Fits both a fisheye and a pinhole-rational model and keeps fisheye unless pinhole is clearly better (`--model fisheye` to force).

4. **Check undistortion on the real scene**
   ```bash
   uv run moppose calib-check --cam cam1 --t 5
   ```

### The board
[configs/board.yaml](configs/board.yaml) is set up for `charuco_A3.pdf`: 3×2 squares of 115 mm,
86.2 mm DICT_4X4_50 markers (ids 0–2). A 3×2 board has only 2 inner chessboard corners, so the
marker corners are used too (`use_marker_corners: true`, up to 14 points per frame). With so few
points per frame, calibration quality comes from **many varied frames** — in synthetic tests ~40
views gave 0.45° ray error, ~100 views 0.15°.

Some OpenCV versions return ChArUco chessboard corners shifted by ~0.5 px; `Target` measures this
offset on a synthetic image at start-up and removes it (otherwise mixing them with marker corners
biases the focal length by >1%).

### Tips for the ChArUco videos
- The board is small: hold it close enough that each marker is at least ~30 px wide in the image.
- Move slowly (security cams have long exposure → motion blur), hold still for a moment in each pose.
- Cover the **whole image, especially corners and edges**, at different distances and tilts (±45°).
- Keep the board flat (glue to a rigid panel); measure the printed square size with a ruler.
- Never change resolution / zoom / lens between calibration and recording.

## Conventions
- Camera: `X_cam = R @ X_world + t`; pixels (u, v) top-left origin.
- Normalized coordinates = undistorted points on the z = 1 plane; triangulation happens there.
- World frame will be defined by the shared reference object, meters, Z up.

## Tests
```bash
uv run pytest
```
Synthetic tests render a ChArUco board through a known ~170° fisheye camera and check that calibration recovers it, and check triangulation on a synthetic 3-camera rig.
