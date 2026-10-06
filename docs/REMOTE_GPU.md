# Running the pose stage on another (GPU) computer

The 2D pose models are heavy (Sapiens2-0.8B: ~0.85 frames/s on an RTX 2070 with flip test,
~18k frames per camera). Build and test here, run the full videos on a stronger Windows + NVIDIA PC,
and bring back only the small result files.

## What to bring to the GPU computer
1. **The code** as one file (on this laptop, in the repo folder):
   ```powershell
   git bundle create moppose.bundle --all
   ```
2. **The 3 mopping videos**: `data\session1\cam1_20260806_160328.mkv`, `cam3_…`, `cam4_…` (~1.1 GB).

The calibrations are already in the code (`calibration/session1/`).

## One-time setup on the GPU computer
Needs: NVIDIA driver (recent; CUDA 12.8 support), Git, internet.

```powershell
# 1. install uv (Python manager), then open a NEW PowerShell window
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. unpack the code
git clone moppose.bundle moppose
cd moppose

# 3. put the videos in data\session1\  (same file names as on the laptop)

# 4. install everything (Python 3.12, PyTorch CUDA, models' dependencies) - ~5 GB download
uv sync --all-extras

# 5. download Sapiens2 code + weights, RTMW and YOLO26, and check the GPU
uv run moppose setup --sapiens-size 0.8b
```
`setup` prints the GPU name and memory. With **16 GB or more** you may use `--sapiens-size 1b`
(more accurate); 5b needs ~24 GB+.

## Run
Quick test first (first 30 seconds of every camera, a few minutes):
```powershell
uv run moppose pose-all --end 30
```
Check `outputs\session1\people\camX_tracks.jpg` (green path = the mopper) and
`outputs\session1\pose2d\camX_preview.jpg` (skeletons: orange = RTMW, pink = Sapiens2).

If a wrong person was picked in a camera, choose the right track id from the tracks image and run:
```powershell
uv run moppose people --cam cam3 --track 12
```

Full run (hours; it can be stopped and restarted - finished chunks are kept):
```powershell
uv run moppose pose-all
```
Options: `--sapiens-size 1b`, `--backend sapiens2` (only one model).
For speed, `uv run moppose pose2d --no-flip-test` halves Sapiens2 time at a small accuracy cost.

## Bring back
Zip and copy back the folder `outputs\session1\` (people + pose2d results, a few MB without the
`part_*` chunk folders) into the same place on the laptop.
