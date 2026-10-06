# Session 1 camera calibrations

Per-camera lens calibrations (intrinsics + fisheye distortion) for the mopping recordings
`data/session1/camX_20260806_160328.mkv` (1280×1024). Saved 2026-10-06.

- Board: `charuco_A3.pdf` – 3×2 squares, 115 mm squares, 86.2 mm DICT_4X4_50 markers, marker corners used (see `configs/board.yaml`).
- Board videos: `data/calib/cam1.mkv`, `cam3.mkv`, `cam4.mkv` (not in git).
  cam3's board video is a 1920×1080 export; picture area x=284, 1351×1080 was cropped and scaled to 1280×1024.
- Lens model: OpenCV fisheye (equidistant, k1..k4) for all three.

| camera | RMS (px) | views | fx | fy | cx | cy | mopping area covered by board |
|---|---|---|---|---|---|---|---|
| cam1 | 0.84 | 78 | 911.4 | 916.7 | 595.7 | 490.4 | 99% |
| cam3 | 0.95 | 79 | 931.6 | 939.2 | 619.9 | 540.5 | 98% |
| cam4 | 0.76 | 77 | 901.9 | 907.4 | 636.0 | 505.3 | 96% |

Files per camera:
- `camX_intrinsics.yaml` – the calibration (load with `moppose.calib.camera.Camera.load`)
- `camX_detections.npz` – detected board corners (re-calibrate without the videos)
- `camX_coverage.jpg`, `camX_work_area.jpg`, `camX_undistort.jpg`, `camX_per_view_rms.png` – report images

The image corners (ceiling/walls) were not covered by the board; the calibration is valid for the
area where people move. To use these files again, copy the yaml files into `outputs/calib/`.
