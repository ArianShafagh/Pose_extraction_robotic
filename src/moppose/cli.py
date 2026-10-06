"""Command line entry point:  uv run moppose --help"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from moppose.config import BoardConfig, SessionConfig

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Multi-camera fisheye pose framework")

SESSION = typer.Option(Path("configs/session.yaml"), "--session", "-s", help="session config")
BOARD = typer.Option(Path("configs/board.yaml"), "--board", "-b", help="ChArUco board config")


# ---------------------------------------------------------------------------
@app.command()
def probe(
    videos: list[Path] = typer.Argument(..., help="video file(s)"),
    out: Path = typer.Option(Path("outputs/probe"), help="where contact sheets go"),
    scan: bool = typer.Option(True, help="walk all frames to measure real frame timing (slower)"),
):
    """Resolution, fps, real frame timing (VFR / dropped frames) + a contact sheet per video."""
    from moppose.io.video import contact_sheet, probe as probe_video

    out.mkdir(parents=True, exist_ok=True)
    for v in videos:
        info = probe_video(v, scan=scan)
        typer.secho(f"\n== {v}", bold=True)
        typer.echo(f"  size          {info.width}x{info.height}   codec {info.codec}")
        typer.echo(f"  fps (header)  {info.fps_nominal:.3f}   frames (header) {info.frame_count_header}")
        if info.frames_scanned is not None and info.dt_median_ms is not None:
            typer.echo(f"  fps (real)    {info.fps_measured:.3f}   frames read {info.frames_scanned}   duration {info.duration_s:.2f}s")
            typer.echo(f"  frame dt ms   median {info.dt_median_ms:.2f}  min {info.dt_min_ms:.2f}  max {info.dt_max_ms:.2f}")
            typer.echo(f"  dropped gaps  {info.gaps}   VFR: {info.is_vfr}")
        if info.ffprobe and info.ffprobe.get("format", {}).get("tags"):
            typer.echo(f"  tags          {info.ffprobe['format']['tags']}")
        sheet_path = out / f"{v.stem}_sheet.jpg"
        cv2.imwrite(str(sheet_path), contact_sheet(v))
        (out / f"{v.stem}_probe.json").write_text(json.dumps(info.to_dict(), indent=2))
        typer.echo(f"  contact sheet {sheet_path}")


# ---------------------------------------------------------------------------
@app.command("board-image")
def board_image(
    board: Path = BOARD,
    out: Path = typer.Option(Path("outputs/board.png")),
    px_per_square: int = 120,
):
    """Render the board described by board.yaml. Compare it with your printed board:
    same marker pattern, same orientation, same number of squares."""
    from moppose.calib.charuco import make_board

    cfg = BoardConfig.load(board)
    b = make_board(cfg)
    img = b.generateImage((cfg.squares_x * px_per_square, cfg.squares_y * px_per_square), marginSize=px_per_square // 4)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    typer.echo(f"wrote {out}")


@app.command("board-check")
def board_check(
    video: Path = typer.Argument(..., help="a ChArUco calibration video"),
    board: Path = BOARD,
    n: int = typer.Option(20, help="frames to test, spread over the video"),
    guess: bool = typer.Option(False, help="also try all common ArUco dictionaries"),
    out: Path = typer.Option(Path("outputs/board_check")),
):
    """Quick test that board.yaml matches the board in a video (before a full calibration)."""
    from moppose.calib.charuco import Target, draw_detection, guess_dictionary
    from moppose.io.video import open_video, read_frame_at

    cfg = BoardConfig.load(board)
    target = Target(cfg)
    cap = open_video(video)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    out.mkdir(parents=True, exist_ok=True)
    frames, counts = [], []
    for k in range(n):
        fr = read_frame_at(video, dur * (k + 0.5) / n)
        frames.append(fr.image)
        d = target.detect(fr.image, fr.t, fr.index)
        c = 0 if d is None else len(d.ids)
        counts.append(c)
        img = fr.image if d is None else draw_detection(fr.image, d, target.n_charuco)
        cv2.imwrite(str(out / f"{video.stem}_{fr.t:08.2f}s_{c}corners.jpg"), img)
    extra = f" + {target.n_points - target.n_charuco} marker corners" if target.use_marker_corners else ""
    typer.echo(f"board {cfg.squares_x}x{cfg.squares_y} {cfg.dictionary}: {target.n_charuco} inner corners{extra}"
               f" = {target.n_points} points max")
    typer.echo(f"points detected per test frame: {counts}")
    typer.echo(f"frames with >=8 points: {sum(c >= 8 for c in counts)}/{n}   (annotated frames in {out})")
    if guess or max(counts) == 0:
        typer.echo("\nArUco dictionary scan (dictionary, markers found, max id):")
        for name, found, max_id in guess_dictionary(frames)[:6]:
            typer.echo(f"  {name:22s} {found:5d}  max id {max_id}")
        typer.echo("The board uses markers 0..(squares_x*squares_y//2 - 1); pick the dictionary with most hits.")


# ---------------------------------------------------------------------------
@app.command("calib-intrinsics")
def calib_intrinsics(
    cam: str = typer.Option(..., "--cam", "-c", help="camera name from session.yaml"),
    session: Path = SESSION,
    board: Path = BOARD,
    model: str = typer.Option("", help="override lens model: fisheye | pinhole | auto"),
    redetect: bool = typer.Option(False, help="ignore cached detections"),
):
    """Calibrate one camera's lens (K + distortion) from its ChArUco video(s)."""
    from moppose.calib.camera import Camera
    from moppose.calib.charuco import Target
    from moppose.calib.frame_map import auto_map, check_map
    from moppose.calib.frame_select import select_views
    from moppose.calib.intrinsics import (calibrate, choose_model, collect_detections, load_detections,
                                          save_detections, validity)
    from moppose.io.video import read_frame_at
    from moppose.viz.calib_report import coverage_image, per_view_plot, undistort_check

    ses = SessionConfig.load(session)
    entry = ses.camera(cam)
    cs = ses.calibration
    board_cfg = BoardConfig.load(board)
    target = Target(board_cfg)
    out = ses.calib_dir
    cache = out / f"{cam}_detections.npz"

    for v in entry.calib_videos + entry.videos:
        if not v.exists():
            raise typer.BadParameter(f"video not found: {v}")

    # Calibration exports can differ from the scene recordings (resolution, black bars).
    # Map calibration frames onto the scene pixel grid so K is valid for the scene videos.
    scene = entry.videos[0] if entry.videos else None
    fmaps = [auto_map(v, scene) if scene else None for v in entry.calib_videos]
    for v, fm in zip(entry.calib_videos, fmaps):
        if fm is not None and not fm.is_identity:
            x, y, w, h = fm.crop
            typer.echo(f"{v.name}: using picture area x={x} y={y} {w}x{h} -> scaled to {fm.out_size[0]}x{fm.out_size[1]}")
            chk = check_map(fm, v, scene)
            if chk.get("ok"):
                msg = f"  background check vs {scene.name}: max shift {chk['corner_shift_px']:.2f}px ({chk['inliers']} matches)"
                typer.secho(msg, fg="green" if chk["corner_shift_px"] < 2.0 else "yellow")
                if chk["corner_shift_px"] >= 2.0:
                    typer.secho("  mapping looks off: camera moved/zoomed between recordings?", fg="yellow")
            else:
                typer.secho(f"  background check failed: {chk.get('reason')}", fg="yellow")

    key = repr(([str(v) for v in entry.calib_videos], [fm.to_dict() if fm else None for fm in fmaps],
                vars(board_cfg), cs.sample_every_s, cs.min_corners, cs.min_sharpness))
    cached = load_detections(cache) if cache.exists() and not redetect else None
    if cached is not None and cached[2] == key:
        dets, size, _ = cached
        typer.echo(f"loaded {len(dets)} cached detections from {cache} (use --redetect to redo)")
    else:
        dets, size, stats = collect_detections(target, entry.calib_videos, cs.sample_every_s, cs.min_corners,
                                               cs.min_sharpness, fmaps)
        save_detections(cache, dets, size, key)
        typer.echo(f"detection stats: {stats}")
    typer.echo(f"{len(dets)} usable board views, image size {size[0]}x{size[1]}")

    views = select_views(dets, size, cs.max_views)
    typer.echo(f"selected {len(views)} diverse views for calibration")

    model = model or entry.model
    models = ["fisheye", "pinhole"] if model == "auto" else [model]
    results = {}
    for m in models:
        try:
            r = calibrate(target, views, size, m, outlier_rms_px=cs.outlier_rms_px)
        except (RuntimeError, cv2.error) as e:
            typer.secho(f"  {m}: FAILED - {e}", fg="red")
            continue
        results[m] = r
        typer.echo(f"  {m:8s} RMS {r.rms:.3f}px  views {len(r.views)} (rejected {len(r.rejected)})  "
                   f"fx {r.K[0, 0]:.1f} fy {r.K[1, 1]:.1f} cx {r.K[0, 2]:.1f} cy {r.K[1, 2]:.1f}")
    if not results:
        raise typer.Exit(1)
    best = choose_model(results)
    r = results[best]
    work_pts = work_w = heat = None
    if scene:
        from moppose.io.motion import motion_heatmap, work_area_points
        mcache = out / f"{cam}_motion.npz"
        if mcache.exists() and str(np.load(mcache)["video"]) == str(scene):
            heat = np.load(mcache)["heat"]
        else:
            typer.echo(f"measuring where people move in {scene.name} ...")
            heat = motion_heatmap(scene)
            np.savez_compressed(mcache, heat=heat, video=np.array(str(scene)))
        work_pts, work_w = work_area_points(heat)
    val = validity(best, r.K, r.D, size, dets, work_pts, work_w)
    camera = Camera(
        name=cam, model=best, size=size, K=r.K, D=r.D, rms=r.rms,
        meta={"views_used": len(r.views), "views_rejected": len(r.rejected),
              "rms_by_model": {k: float(v.rms) for k, v in results.items()},
              "calib_videos": [str(p) for p in entry.calib_videos],
              "calib_frame_maps": [fm.to_dict() if fm else None for fm in fmaps],
              "validity": val},
    )
    path = out / f"{cam}_intrinsics.yaml"
    camera.save(path)
    typer.secho(f"chosen model: {best}  ->  {path}", fg="green", bold=True)

    # report images
    bg = read_frame_at(entry.calib_videos[0], r.views[len(r.views) // 2].t).image
    if fmaps[0] is not None:
        bg = fmaps[0].apply(bg)
    cv2.imwrite(str(out / f"{cam}_coverage.jpg"), coverage_image(bg, r.views, r.rejected))
    per_view_plot(results, out / f"{cam}_per_view_rms.png")
    scene_img = read_frame_at(scene, 5.0).image if scene else bg
    cv2.imwrite(str(out / f"{cam}_undistort.jpg"), undistort_check(camera, scene_img))
    if heat is not None:
        from moppose.io.motion import overlay
        hull = cv2.convexHull(np.concatenate([d.corners for d in dets]).astype(np.float32))
        cv2.imwrite(str(out / f"{cam}_work_area.jpg"), overlay(scene_img, heat, hull))
    typer.echo(f"report images in {out}: {cam}_coverage.jpg, {cam}_per_view_rms.png, {cam}_undistort.jpg")
    if r.rms > 1.0:
        typer.secho("RMS > 1px: check board.yaml sizes, blur, and the coverage image.", fg="yellow")
    typer.echo(f"board reached {val['covered_r']:.0f}px from the image centre; image corners are at "
               f"{val['corner_r']:.0f}px ({100 * val['covered_frac']:.0f}%); model valid up to {val['monotonic_r']:.0f}px")
    if "work_inside" in val:
        typer.echo(f"area where people move in the scene video: {100 * val['work_inside']:.0f}% inside the board "
                   f"coverage ({cam}_work_area.jpg)")
    if val["usable"]:
        typer.secho("calibration covers the working area: USABLE", fg="green", bold=True)
    else:
        typer.secho("NOT USABLE: the board did not cover where people move - record board views there "
                    "(see coverage/work_area images), then run again with --redetect", fg="red", bold=True)


@app.command("calib-check")
def calib_check(
    cam: str = typer.Option(..., "--cam", "-c"),
    session: Path = SESSION,
    video: Path = typer.Option(None, help="video to take the frame from (default: first mopping video)"),
    t: float = typer.Option(1.0, help="time (s) of the frame"),
    fov_scale: float = typer.Option(1.0, help=">1 zooms out the rectilinear view (fisheye only)"),
):
    """Undistort a frame of a scene video with the camera's calibration (straight lines must be straight)."""
    from moppose.calib.camera import Camera
    from moppose.io.video import read_frame_at
    from moppose.viz.calib_report import undistort_check

    ses = SessionConfig.load(session)
    entry = ses.camera(cam)
    camera = Camera.load(ses.calib_dir / f"{cam}_intrinsics.yaml")
    video = video or (entry.videos[0] if entry.videos else entry.calib_videos[0])
    img = read_frame_at(video, t).image
    if (img.shape[1], img.shape[0]) != camera.size:
        raise typer.BadParameter(f"frame size {img.shape[1]}x{img.shape[0]} != calibration size {camera.size}")
    p = ses.calib_dir / f"{cam}_check_{Path(video).stem}_{t:.1f}s.jpg"
    cv2.imwrite(str(p), undistort_check(camera, img))
    if fov_scale != 1.0:
        cv2.imwrite(str(p.with_name(p.stem + f"_fov{fov_scale}.jpg")), camera.undistort_image(img, 1.0, fov_scale))
    typer.echo(f"wrote {p}")


# ---------------------------------------------------------------------------
# 2D pose stage
# ---------------------------------------------------------------------------
def _cams(ses: SessionConfig, cam: str) -> list[str]:
    return list(ses.cameras) if cam == "all" else [c.strip() for c in cam.split(",")]


@app.command()
def people(
    cam: str = typer.Option("all", "--cam", "-c", help="camera name, comma list, or all"),
    session: Path = SESSION,
    model: str = typer.Option("yolo26x.pt", help="ultralytics detection model"),
    track: int = typer.Option(None, help="force this track id as the mopper (see the preview image)"),
    start: float = typer.Option(0.0, help="start time (s)"),
    end: float = typer.Option(None, help="end time (s)"),
    redo: bool = typer.Option(False, help="re-run tracking even if results exist"),
):
    """Detect + track people, pick the mopper; writes outputs/<session>/people/ with a preview image."""
    from moppose.io.video import read_frame_at
    from moppose.pose2d.people import Tracks, track_people
    from moppose.pose2d.select import preview, select_mopper, summarize

    ses = SessionConfig.load(session)
    out = ses.session_dir / "people"
    for c in _cams(ses, cam):
        video = ses.camera(c).videos[0]
        path = out / f"{c}.npz"
        if path.exists() and not redo:
            tr = Tracks.load(path)
            typer.echo(f"{c}: loaded tracks from {path}")
        else:
            tr = track_people(video, model=model, start_s=start, end_s=end)
            tr.save(path)
        chain = select_mopper(tr, seed_id=track)
        s = summarize(tr)
        typer.echo(f"{c}: {len(s)} tracks; mopper = {chain} "
                   f"(covers {sum(s[i].n for i in chain)} of {len(tr.frame_t)} frames)")
        (out / f"{c}_mopper.json").write_text(json.dumps({"chain": chain, "seed": track}))
        bg = read_frame_at(video, float(tr.frame_t[len(tr.frame_t) // 2])).image
        img_path = out / f"{c}_tracks.jpg"
        cv2.imwrite(str(img_path), preview(bg, tr, chain))
        typer.echo(f"   check {img_path}: the green path must be the mopper, else rerun with --track ID")


@app.command()
def pose2d(
    cam: str = typer.Option("all", "--cam", "-c", help="camera name, comma list, or all"),
    session: Path = SESSION,
    backend: str = typer.Option("rtmw,sapiens2", help="comma list: rtmw, sapiens2"),
    start: float = typer.Option(0.0, help="start time (s)"),
    end: float = typer.Option(None, help="end time (s)"),
    every: int = typer.Option(1, help="use every N-th mopper frame (1 = all)"),
    sapiens_size: str = typer.Option("0.8b", help="Sapiens2 model: 0.4b | 0.8b | 1b | 5b"),
    flip_test: bool = typer.Option(True, help="Sapiens2 flip test (more accurate, 2x slower)"),
    batch: int = typer.Option(4, help="crops per model call (lower it if GPU memory runs out)"),
):
    """2D body+feet keypoints of the mopper with each backend; writes outputs/<session>/pose2d/."""
    from moppose.calib.camera import Camera
    from moppose.pose2d.backends import make_backend
    from moppose.pose2d.people import Tracks
    from moppose.pose2d.run import run_pose
    from moppose.pose2d.select import mopper_boxes

    ses = SessionConfig.load(session)
    backends = [make_backend(b.strip(), size=sapiens_size, flip_test=flip_test) for b in backend.split(",")]
    for c in _cams(ses, cam):
        video = ses.camera(c).videos[0]
        ppl = ses.session_dir / "people"
        if not (ppl / f"{c}.npz").exists():
            raise typer.BadParameter(f"run `moppose people --cam {c}` first")
        tr = Tracks.load(ppl / f"{c}.npz")
        chain = json.loads((ppl / f"{c}_mopper.json").read_text())["chain"]
        t, _, boxes = mopper_boxes(tr, chain)
        keep = (t >= start) & (t <= (end if end is not None else np.inf))
        t, boxes = t[keep][::every], boxes[keep][::every]
        cam_model = Camera.load(ses.intrinsics_path(c))
        typer.echo(f"{c}: {len(t)} frames with the mopper")
        paths = run_pose(cam_model, video, t, boxes, backends, ses.session_dir / "pose2d", batch=batch)
        for name, p in paths.items():
            z = np.load(p)
            typer.echo(f"   {name}: {p}  ({len(z['t'])} frames, mean conf {np.nanmean(z['conf']):.2f})")


@app.command("pose-preview")
def pose_preview(
    cam: str = typer.Option("all", "--cam", "-c"),
    session: Path = SESSION,
    n: int = typer.Option(12, help="number of sample frames"),
    video_out: bool = typer.Option(False, "--video", help="also write an mp4 with skeletons"),
    start: float = typer.Option(0.0),
    end: float = typer.Option(None),
):
    """Grid of zoomed frames with every backend's skeleton (and optionally an overlay video)."""
    from moppose.io.video import iter_frames, read_frame_at
    from moppose.viz.pose_overlay import COLORS, draw_skeleton, zoom_on

    ses = SessionConfig.load(session)
    pdir = ses.session_dir / "pose2d"
    for c in _cams(ses, cam):
        video = ses.camera(c).videos[0]
        res = {b.name: np.load(b / f"{video.stem}.npz") for b in sorted(pdir.iterdir())
               if b.is_dir() and (b / f"{video.stem}.npz").exists()} if pdir.exists() else {}
        if not res:
            typer.echo(f"{c}: no pose results yet")
            continue
        ref = next(iter(res.values()))
        ts = ref["t"][(ref["t"] >= start) & (ref["t"] <= (end if end is not None else np.inf))]
        tiles = []
        for t in ts[np.linspace(0, len(ts) - 1, min(n, len(ts))).astype(int)]:
            img = read_frame_at(video, float(t)).image
            pts = []
            for name, z in res.items():
                i = int(np.argmin(np.abs(z["t"] - t)))
                if abs(z["t"][i] - t) < 1e-3:
                    draw_skeleton(img, z["kpts"][i], z["conf"][i], z["edges"], COLORS.get(name, (0, 255, 0)), label=name)
                    pts.append(z["kpts"][i])
            tile = zoom_on(img, np.concatenate(pts)) if pts else cv2.resize(img, (480, 480))
            cv2.putText(tile, f"{t:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(tile)
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        p = pdir / f"{c}_preview.jpg"
        cv2.imwrite(str(p), grid)
        typer.echo(f"{c}: {p}")
        if video_out:
            wr = None
            vp = pdir / f"{c}_overlay.mp4"
            for fr in iter_frames(video, start_s=start, end_s=end):
                img = fr.image
                for name, z in res.items():
                    i = int(np.searchsorted(z["t"], fr.t - 1e-4))
                    if i < len(z["t"]) and abs(z["t"][i] - fr.t) < 1e-3:
                        draw_skeleton(img, z["kpts"][i], z["conf"][i], z["edges"], COLORS.get(name, (0, 255, 0)), label=name)
                if wr is None:
                    wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (img.shape[1], img.shape[0]))
                wr.write(img)
            if wr is not None:
                wr.release()
                typer.echo(f"{c}: {vp}")


@app.command("pose-all")
def pose_all(
    session: Path = SESSION,
    backend: str = typer.Option("rtmw,sapiens2"),
    sapiens_size: str = typer.Option("0.8b"),
    end: float = typer.Option(None, help="only the first N seconds (quick test)"),
):
    """Everything for all cameras: people tracking -> mopper -> 2D pose -> previews."""
    people(cam="all", session=session, model="yolo26x.pt", track=None, start=0.0, end=end, redo=False)
    pose2d(cam="all", session=session, backend=backend, start=0.0, end=end, every=1,
           sapiens_size=sapiens_size, flip_test=True, batch=4)
    pose_preview(cam="all", session=session, n=12, video_out=False, start=0.0, end=end)


@app.command()
def setup(
    sapiens_size: str = typer.Option("0.8b", help="Sapiens2 pose model to download: 0.4b | 0.8b | 1b | 5b"),
    skip_sapiens: bool = typer.Option(False, help="only RTMW + YOLO"),
):
    """One-time download of pose models (Sapiens2 code + weights, RTMW, YOLO26) and a GPU check."""
    import torch

    from moppose.pose2d.setup_models import setup_rtmw, setup_sapiens2

    typer.echo(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        typer.echo(f"GPU: {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        typer.secho("No CUDA GPU visible - install/update the NVIDIA driver.", fg="red")
    if not skip_sapiens:
        typer.echo(f"Sapiens2 {sapiens_size}: {setup_sapiens2(sapiens_size)}")
    setup_rtmw()
    typer.echo("RTMW: ready")
    from ultralytics import YOLO

    YOLO("yolo26x.pt")
    typer.echo("YOLO26x: ready")
    typer.secho("setup done", fg="green", bold=True)


if __name__ == "__main__":
    app()
