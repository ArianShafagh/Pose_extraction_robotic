# Processing a new recording - one config, one command

The cameras are fixed, so the lens calibrations (`calibration/session1/`) and the camera rig in metres
(`calibration/rig/rig.yaml`) are reused for every recording. A new recording needs only:

1. the 3 videos, recorded at the same time;
2. a small config file;
3. one command.

## 1. Put the videos in the project
```
data/<recording>/cam1.mkv
data/<recording>/cam3.mkv
data/<recording>/cam4.mkv
```
They must be the cameras' own 1280x1024 recordings (the resolution the lenses were calibrated for).
Names and folders are free; the config says which file is which camera.

## 2. Write the config
Copy the template and edit `name` and `videos`; everything else has sensible defaults:
```bash
copy configs\recordings\_template.yaml configs\recordings\sweeping_2026-10-20.yaml
```
```yaml
name: sweeping_2026-10-20
videos:
  cam1: data/sweeping/cam1.mkv
  cam3: data/sweeping/cam3.mkv
  cam4: data/sweeping/cam4.mkv
```

| section | what it sets |
|---|---|
| `time_range` | process only part of the videos (`start_s`, `end_s`) |
| `actor` | `auto`: the person seen consistently by all cameras; `track_ids` forces a track per camera |
| `pose` | 2D model (`sapiens2` / `rtmw`), model size, `every` (2 = 15 fps), `cross_check` model |
| `sync` | search range; `offsets_s` fixes an offset by hand |
| `reconstruction` | confidence / reprojection thresholds, bone-length check, gap filling, smoothing |
| `video3d` | size, view angle (`azimuth_deg`, `elevation_deg`), `distance_m`, `fixed` or `orbit`, trail length |
| `outputs` | output folder, work folder, and **the name of every output file** (`{name}`, `{cam}` placeholders) |

## 3. Run (on the GPU computer)
```bash
uv run moppose run configs/recordings/sweeping_2026-10-20.yaml
```
Useful options:
- `--dry-run`: shows which stages will run and every output file name, without running anything.
- `--from recon3d`: redoes that stage and everything after it.
- `--only video3d`: runs just one stage.

Stages: `check` → `people` → `actor` → `pose2d` → `sync` → `recon3d` → `export` → `video3d`.
- The run can be stopped (Ctrl+C) and started again: finished stages are skipped, and `pose2d` resumes chunk by chunk.
- Changing a config section reruns only the stages that depend on it. For example, changing `reconstruction` reruns `recon3d`, `export` and `video3d`, not the hours of 2D pose.

Time for a 10-minute recording at 15 fps:

| GPU | 2D pose (Sapiens2-0.8B) per camera | everything else |
|---|---|---|
| GTX 1080 | ~10–12 h | ~25 min (tracking, 3D, video) |
| RTX 2070 | ~3 h | |

## Outputs (`results/<name>/`, names as configured)
| file | content |
|---|---|
| `<name>_landmarks3d.csv` | per frame: `t` + x, y, z (metres) of the 23 joints |
| `<name>_landmarks3d.npz` | the same, plus raw values, filled-in mask, number of views, reprojection errors |
| `<name>_skeleton3d.mp4` | **full-length** 3D skeleton video: floor grid, cameras, walked path |
| `<name>_report.json` | quality numbers, time offsets, bone lengths, warnings |
| `<name>_rig.yaml` | the cameras + this recording's time offsets |
| `<name>_overview.png` | top view of the path + skeleton snapshots |
| `<name>_bone_lengths.png` | bone lengths over time (should be flat) |
| `<name>_reprojection.jpg` | 3D skeleton drawn back into the 3 cameras (must sit on the person) |
| `<name>_<cam>_actor.jpg` | which person was picked in each camera |

World frame: metres, **z up, floor at z = 0**, origin in the middle of the first recording's walked area,
x from cam1 towards cam3.

## Checks after a run
- Open `<name>_<cam>_actor.jpg`. The green path must be the actor. If not, set `actor.track_ids` and run with `--from actor`.
- Read the warnings in `<name>_report.json` (also printed at the end):
  - **"median reprojection ... camera moved?"**: a camera was moved or bumped. Re-create the rig with `moppose calibrate-rig` (needs a recording where a person walks around plus a scale reference, see below).
  - **"heels ... off the floor"**: the rig no longer fits.

## If a camera is moved: `moppose calibrate-rig`
1. Record a few minutes of a person walking around the whole room.
2. Run the 2D pose for it (the `people` + `pose2d` commands of the session workflow).
3. Mark a known length (the mop, see `configs/session.yaml` → `scale`).
4. Run `uv run moppose calibrate-rig`. It writes a new `calibration/rig/rig.yaml`; the old one is kept as `rig.previous.yaml`.
