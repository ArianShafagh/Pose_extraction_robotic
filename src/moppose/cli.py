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


if __name__ == "__main__":
    app()
