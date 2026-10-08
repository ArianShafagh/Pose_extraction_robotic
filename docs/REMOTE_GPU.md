# Running the pose stage on the GPU computer (GTX 1080, 8 GB)

Build and test on the laptop, run the long pose extraction on the GPU computer, bring back only
the small result files.

## What the GTX 1080 means
- It is a **Pascal** GPU (compute capability 6.1). The project uses the **CUDA 12.6 PyTorch builds**
  (PyTorch ≤ 2.14), the last ones that support it. The same builds also run on the laptop's RTX 2070.
- It has **no tensor cores**, so half precision (fp16) is very slow on it. Sapiens2 therefore runs in
  **fp32 without the flip test** there (chosen automatically: `precision auto`, flip test on only with fp16).
  Measured effect of dropping the flip test: keypoints move by ~0.75 px (median, crop pixels); this is
  smaller than the effect of switching to the smaller 0.4B model (1.65 px).
- Sapiens2-0.8B in fp32 needs ~4.2 GB of GPU memory, so it fits in 8 GB.

### Expected run time (Sapiens2-0.8B, ~17 800 frames per camera at 30 fps)
| computer | mode | speed | per camera, 30 fps | per camera, 15 fps (`--every 2`) |
|---|---|---|---|---|
| GPU PC, GTX 1080 | fp32, no flip | ~0.5 frames/s (estimate*) | ~10 h | ~5 h |
| laptop, RTX 2070 | fp16 + flip | 0.86 frames/s (measured) | ~6 h | ~3 h |

\* measured 0.46 frames/s with the same settings on the RTX 2070; the GTX 1080 has slightly more fp32
throughput. The quick test below prints the real speed.

YOLO26 tracking and RTMW are fast on both (RTMW ~36 frames/s).

**Tip:** run both computers at the same time, each on different cameras (results are separate files per
camera), e.g. GPU PC: `--cam cam4`, laptop: `--cam cam1,cam3`.

## What to bring to the GPU computer
1. **The code** as one file (on the laptop, in the repo folder):
   ```powershell
   git bundle create moppose.bundle --all
   ```
2. **The 3 mopping videos**: `data\session1\cam1_20260806_160328.mkv`, `cam3_…`, `cam4_…` (~1.1 GB).

The camera calibrations are already in the code (`calibration/session1/`).

## One-time setup on the GPU computer
Needs: NVIDIA driver 560 or newer (CUDA 12.6), Git, internet.

```powershell
# 1. install uv (Python manager), then open a NEW PowerShell window
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. unpack the code
git clone moppose.bundle moppose
cd moppose

# 3. put the videos in data\session1\  (same file names as on the laptop)

# 4. install everything (Python 3.12, PyTorch CUDA 12.6, model dependencies) - ~5 GB download
uv sync --all-extras

# 5. download Sapiens2 code + weights, RTMW and YOLO26, and check the GPU
uv run moppose setup
```
`setup` must print `compute capability 6.1`, `fp32 without flip test` and `CUDA kernels run on this GPU: True`.

## Run
Quick test first (first 30 seconds of every camera, ~10 minutes); it prints the real frames/s:
```powershell
uv run moppose pose-all --end 30
```
Check `outputs\session1\people\camX_tracks.jpg` (green path = the mopper) and
`outputs\session1\pose2d\camX_preview.jpg` (pink = Sapiens2, orange = RTMW).

If a wrong person was picked in a camera, choose the right track id from the tracks image and run:
```powershell
uv run moppose people --cam cam3 --track 12
```

Full run - can be stopped (Ctrl+C) and restarted; finished chunks are kept:
```powershell
uv run moppose pose-all                 # all cameras, 30 fps
uv run moppose pose-all --cam cam4      # one camera only
uv run moppose pose-all --every 2       # 15 fps, half the time
```
If the GPU runs out of memory: `uv run moppose pose2d --batch 1`.

## Bring back
Copy back the folder `outputs\session1\` into the same place on the laptop
(the `part_*` chunk folders can be left out).
