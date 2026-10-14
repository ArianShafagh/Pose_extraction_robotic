import cv2
import numpy as np

from moppose.calib.camera import Camera
from moppose.multiview.data import Track2D
from moppose.multiview.selfcalib import bundle_adjust, normalized, pair_points, relative_pose, triangulate_all
from moppose.multiview.sync import coarse_offset
from moppose.multiview.world import fit_floor
from moppose.pose2d.skeleton import BODY_FEET, EDGE_IDX

from test_triangulate import make_rig


def walking_person(n=600, fps=15.0, seed=0):
    """Synthetic 23-joint person (metres, Z up) walking a curvy path while swinging arms."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    path = np.c_[1.0 * np.sin(0.21 * t), 0.8 * np.sin(0.13 * t + 1)]
    heading = np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0]))
    base = {  # joint offsets in the body frame (x forward, y left, z up), metres
        "nose": (0.08, 0, 1.55), "left_eye": (0.07, 0.03, 1.58), "right_eye": (0.07, -0.03, 1.58),
        "left_ear": (0, 0.07, 1.56), "right_ear": (0, -0.07, 1.56),
        "left_shoulder": (0, 0.18, 1.38), "right_shoulder": (0, -0.18, 1.38),
        "left_elbow": (0.05, 0.22, 1.1), "right_elbow": (0.05, -0.22, 1.1),
        "left_wrist": (0.2, 0.2, 0.9), "right_wrist": (0.2, -0.2, 0.9),
        "left_hip": (0, 0.1, 0.92), "right_hip": (0, -0.1, 0.92),
        "left_knee": (0.03, 0.1, 0.5), "right_knee": (0.03, -0.1, 0.5),
        "left_ankle": (0, 0.1, 0.08), "right_ankle": (0, -0.1, 0.08),
        "left_big_toe": (0.18, 0.12, 0.0), "left_small_toe": (0.16, 0.06, 0.0), "left_heel": (-0.05, 0.1, 0.0),
        "right_big_toe": (0.18, -0.12, 0.0), "right_small_toe": (0.16, -0.06, 0.0), "right_heel": (-0.05, -0.1, 0.0),
    }
    B = np.array([base[nm] for nm in BODY_FEET])
    X = np.zeros((n, len(BODY_FEET), 3))
    for i in range(n):
        c, s = np.cos(heading[i]), np.sin(heading[i])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        b = B.copy()
        swing = 0.25 * np.sin(2 * np.pi * 0.8 * t[i])
        for nm, sgn in (("left_wrist", 1), ("right_wrist", -1), ("left_elbow", 0.5), ("right_elbow", -0.5)):
            b[BODY_FEET.index(nm), 0] += sgn * swing
        X[i] = b @ Rz.T + np.r_[path[i], 0.0]
    return t, X + rng.normal(0, 0.003, X.shape)


def _synthetic_tracks(cams, t, X, offsets, noise_px=0.7, seed=1):
    rng = np.random.default_rng(seed)
    tracks = {}
    for cam, off in zip(cams, offsets):
        # camera c samples the motion at its own clock: video time = reference time + off
        uv = np.stack([cam.project(x) for x in X])
        uv += rng.normal(0, noise_px, uv.shape)
        tracks[cam.name] = Track2D(cam.name, t + off, uv, np.ones(uv.shape[:2]), list(BODY_FEET), EDGE_IDX)
    return tracks


def test_coarse_sync_recovers_offset():
    cams = make_rig()
    t, X = walking_person(n=1500)
    tracks = _synthetic_tracks(cams, t, X, [0.0, 0.2, -0.35])
    for name, true in (("cam2", 0.2), ("cam3", -0.35)):
        d, _ = coarse_offset(tracks["cam1"], tracks[name], max_lag=1.0)
        assert abs(d - true) < 0.02


def test_self_calibration_recovers_rig_up_to_scale():
    true = make_rig()
    t, X = walking_person()
    uv = np.stack([np.stack([c.project(x) for x in X]) for c in true])           # (C, N, J, 2)
    uv += np.random.default_rng(2).normal(0, 0.7, uv.shape)
    conf = np.ones(uv.shape[:3])
    bare = [Camera(c.name, c.model, c.size, c.K, c.D) for c in true]
    xn = normalized(bare, uv)
    rig = [bare[0]]
    for b in (1, 2):
        pa, pb, _ = pair_points(xn, conf, 0, b, 0.5)
        R, tt, _ = relative_pose(pa, pb, thr=2.0 / 420)
        rig.append(Camera(bare[b].name, "fisheye", bare[b].size, bare[b].K, bare[b].D, R=R, t=tt))
    # fix the scale of cam 3 relative to cam 2 by triangulating the same points with both pairs
    U = uv[:, ::3].reshape(3, -1, 2)
    X01 = triangulate_all(rig[:2], U[[0, 1]], np.ones((2, U.shape[1])))
    X02 = triangulate_all([rig[0], rig[2]], U[[0, 2]], np.ones((2, U.shape[1])))
    rig[2].t = rig[2].t * np.median(np.linalg.norm(X01, axis=1) / np.linalg.norm(X02, axis=1))
    U = uv[:, ::4].reshape(3, -1, 2)
    W = np.ones(U.shape[:2])
    rig, _, info = bundle_adjust(rig, U, W, triangulate_all(rig, U, W))
    assert info["median_err_px"] < 1.5
    # compare with the truth in cam1's frame, after removing the unknown scale
    def rel(cams):
        c = [cams[0].R @ k.center_world + cams[0].t for k in cams]
        return np.array(c) / np.linalg.norm(c[1] - c[0])
    np.testing.assert_allclose(rel(rig), rel(true), atol=0.01)
    for est, tru in zip(rig[1:], true[1:]):
        dR = (est.R) @ (tru.R @ true[0].R.T).T
        assert np.degrees(np.linalg.norm(cv2.Rodrigues(dR)[0])) < 0.5


def test_floor_fit_finds_ground_and_up():
    cams = make_rig()
    t, X = walking_person()
    # put the person in an arbitrary rig frame: rotate + shift
    Rr = cv2.Rodrigues(np.array([0.4, -0.2, 0.9]))[0]
    Xr = X @ Rr.T + np.array([0.5, -1.0, 3.0])
    rig_cams = [Camera(c.name, c.model, c.size, c.K, c.D, R=c.R @ Rr.T, t=c.t - c.R @ Rr.T @ np.array([0.5, -1.0, 3.0]))
                for c in cams]
    wt, stats = fit_floor(Xr, list(BODY_FEET), rig_cams)
    Xw = wt.points(Xr)
    feet = Xw[:, [BODY_FEET.index("left_heel"), BODY_FEET.index("right_heel")]]
    assert abs(np.median(feet[..., 2])) < 0.01
    assert np.median(Xw[:, BODY_FEET.index("nose"), 2]) > 1.4
    # cameras must project the world points exactly as before
    wc = wt.camera(rig_cams[0])
    np.testing.assert_allclose(wc.project(Xw[0]), rig_cams[0].project(Xr[0]), atol=1e-6)


def test_left_right_swaps_are_undone():
    from moppose.multiview.pipeline import fix_left_right

    cams = make_rig()
    t, X = walking_person(n=200)
    uv = np.stack([np.stack([c.project(x) for x in X]) for c in cams])
    cf = np.ones(uv.shape[:3])
    bad = uv.copy()
    L = [BODY_FEET.index(f"left_{p}") for p in ("shoulder", "elbow", "wrist")]
    R = [BODY_FEET.index(f"right_{p}") for p in ("shoulder", "elbow", "wrist")]
    frames = np.arange(20, 60)
    for i in frames:  # camera 2 swaps the arms in frames 20-59
        bad[1, i, L], bad[1, i, R] = uv[1, i, R], uv[1, i, L]
    fixed, _, stats = fix_left_right(cams, bad, cf, list(BODY_FEET))
    assert stats["cam2_arm"] >= 35
    assert np.nanmax(np.abs(fixed[1, frames][:, L + R] - uv[1, frames][:, L + R])) < 1e-6
    assert stats["cam1_arm"] == 0 and stats["cam3_arm"] == 0
