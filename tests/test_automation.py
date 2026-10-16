from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from moppose.calib.camera import save_rig
from moppose.pose2d.people import Tracks
from moppose.pose2d.select import select_actor_multiview
from moppose.pose2d.skeleton import BODY_FEET, EDGE_IDX
from moppose.recording import ConfigError, Recording

from test_multiview import _synthetic_tracks, walking_person
from test_triangulate import make_rig


def _project(tmp_path: Path, **over):
    """Minimal repo: pyproject, setup, rig, intrinsics, dummy videos, recording config."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    cams = make_rig()
    (tmp_path / "calib").mkdir()
    for c in cams:
        c.save(tmp_path / "calib" / f"{c.name}_intrinsics.yaml")
    save_rig(cams, tmp_path / "calib" / "rig.yaml")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "setup.yaml").write_text(yaml.safe_dump(
        {"cameras": [c.name for c in cams], "intrinsics_dir": "calib", "rig": "calib/rig.yaml"}))
    (tmp_path / "data").mkdir()
    for c in cams:
        (tmp_path / "data" / f"{c.name}.mkv").write_bytes(b"")
    cfg = {"name": "test_rec", "setup": "configs/setup.yaml",
           "videos": {c.name: f"data/{c.name}.mkv" for c in cams}} | over
    p = tmp_path / "configs" / "rec.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_recording_defaults_and_names(tmp_path):
    rec = Recording.load(_project(tmp_path))
    assert rec.cfg["pose"]["backend"] == "sapiens2" and rec.cfg["reconstruction"]["max_gap_s"] == 0.5
    assert rec.output("video_3d") == tmp_path / "results" / "test_rec" / "test_rec_skeleton3d.mp4"
    assert rec.output("actor_preview", cam="cam2").name == "test_rec_cam2_actor.jpg"
    assert rec.work_dir == tmp_path / "outputs" / "test_rec"


def test_recording_custom_output_names(tmp_path):
    rec = Recording.load(_project(tmp_path, outputs={"dir": "final/{name}", "files": {"video_3d": "{name}-3d.mp4"}}))
    assert rec.output("video_3d") == tmp_path / "final" / "test_rec" / "test_rec-3d.mp4"
    assert rec.output("landmarks_csv").name == "test_rec_landmarks3d.csv"  # untouched defaults stay


@pytest.mark.parametrize("over, msg", [
    ({"name": "bad name!"}, "name"),
    ({"videos": {"cam1": "data/cam1.mkv"}}, "missing"),
    ({"videos": {"cam1": "data/cam1.mkv", "cam2": "data/cam2.mkv", "cam3": "data/nope.mkv"}}, "not found"),
    ({"pose": {"backend": "openpose"}}, "backend"),
])
def test_recording_validation(tmp_path, over, msg):
    with pytest.raises(ConfigError, match=msg):
        Recording.load(_project(tmp_path, **over))


def test_stage_skipping_follows_config_changes(tmp_path):
    from moppose.automation import Runner

    p = _project(tmp_path)
    runner = Runner(Recording.load(p), log=lambda *a: None)
    for s in ("check", "people", "actor", "pose2d", "sync", "recon3d"):
        runner.mark_done(s)
    assert runner.is_done("recon3d")
    cfg = yaml.safe_load(p.read_text())
    cfg["reconstruction"] = {"smooth_cutoff_hz": 3.0}       # only the 3D settings change
    p.write_text(yaml.safe_dump(cfg))
    runner2 = Runner(Recording.load(p), log=lambda *a: None)
    assert runner2.is_done("pose2d") and runner2.is_done("sync")
    assert not runner2.is_done("recon3d")
    cfg["pose"] = {"every": 1}                               # 2D settings change -> everything from pose2d
    p.write_text(yaml.safe_dump(cfg))
    runner3 = Runner(Recording.load(p), log=lambda *a: None)
    assert runner3.is_done("actor") and not runner3.is_done("pose2d") and not runner3.is_done("sync")


def test_video3d_has_one_frame_per_sample(tmp_path):
    from moppose.viz.video3d import render_full_video

    t, X = walking_person(n=45)
    info = render_full_video(X, t, EDGE_IDX, list(BODY_FEET), tmp_path / "v.mp4", cams=make_rig(), title="test",
                             width=320, height=240)
    cap = cv2.VideoCapture(str(tmp_path / "v.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 45 == info["frames"]
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (320, 240)


def _boxes_tracks(cams, people_xyz, ids_per_cam):
    """Tracks with one box per person per frame (box = projected joints' extent)."""
    t = np.arange(people_xyz[0].shape[0]) / 15.0
    out = {}
    for cam, ids in zip(cams, ids_per_cam):
        rows = []
        for X, tid in zip(people_xyz, ids):
            for i, x in enumerate(X):
                uv = cam.project(x)
                rows.append((t[i], tid, [*uv.min(0), *uv.max(0)]))
        tt, tid, bb = zip(*rows)
        tt = np.array(tt)
        out[cam.name] = Tracks(t=tt, frame_index=np.round(tt * 15).astype(int), track_id=np.array(tid),
                               bbox=np.array(bb, float), score=np.ones(len(tt)), frame_t=np.unique(tt))
    return out


def test_actor_is_the_person_consistent_across_cameras():
    cams = make_rig()
    _, walker = walking_person(n=300, seed=0)
    _, other = walking_person(n=300, seed=1)
    other = other * np.array([0.6, 0.6, 1.0]) + np.array([1.2, -0.9, 0])   # a second, also moving person
    _, sitter = walking_person(n=300, seed=2)
    sitter = sitter * np.array([0.05, 0.05, 1.0]) + np.array([-1.0, 1.0, 0])
    # the actor has different track ids in every camera
    tracks = _boxes_tracks(cams, [walker, other, sitter], [(5, 1, 2), (3, 8, 4), (9, 6, 7)])
    chains, info = select_actor_multiview(tracks, cams)
    assert info["method"] == "multiview"
    assert chains == {"cam1": [5], "cam2": [3], "cam3": [9]}


def test_refine_offset_with_known_rig():
    from moppose.multiview.sync import coarse_offset, refine_offset, relative_pose

    cams = make_rig()
    t, X = walking_person(n=900)
    tracks = _synthetic_tracks(cams, t, X, [0.0, 0.173, -0.211], noise_px=0.5)
    times = t[30:-30]
    for b, true in ((1, 0.173), (2, -0.211)):
        d0, _ = coarse_offset(tracks["cam1"], tracks[cams[b].name], max_lag=1.0)
        d, e = refine_offset(tracks["cam1"], tracks[cams[b].name], cams[0], cams[b], *relative_pose(cams[0], cams[b]),
                             times, d0)
        assert abs(d - true) < 0.015 and e < 1.0   # < half a 30 fps frame
