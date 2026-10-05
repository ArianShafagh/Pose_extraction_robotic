import numpy as np
import pytest

from moppose.calib.camera import Camera
from moppose.calib.frame_select import select_views
from moppose.calib.intrinsics import calibrate

from synth import A3_BOARD, SIZE, TRUE_CAM, Renderer, random_poses


def _detections(cfg, n, dist, min_points, seed):
    r = Renderer(cfg=cfg, px_per_m=2500.0 if cfg.square_len_m < 0.1 else 1200.0)
    rng = np.random.default_rng(seed)
    dets = []
    for i, (rv, tv) in enumerate(random_poses(n, seed=seed + 2, cfg=cfg, dist=dist)):
        d = r.target.detect(r.render(rv, tv, rng=rng), t=float(i), frame_index=i)
        if d is not None and len(d.ids) >= min_points:
            dets.append(d)
    return r, dets


@pytest.fixture(scope="module")
def detections():
    from synth import BOARD
    return _detections(BOARD, 45, (0.35, 0.7), 12, 1)


@pytest.fixture(scope="module")
def a3_detections():
    # few points per view -> needs many views (same advice holds for the real recordings)
    return _detections(A3_BOARD, 150, (0.4, 1.1), 8, 11)


def _ray_errors_deg(res):
    """Angle between true and estimated viewing rays at every corner used in calibration."""
    est = Camera("est", "fisheye", SIZE, res.K, res.D)
    pts = np.concatenate([d.corners for d in res.views])

    def rays(n):
        v = np.c_[n, np.ones(len(n))]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    cos = np.sum(rays(TRUE_CAM.undistort_points(pts)) * rays(est.undistort_points(pts)), axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _check_recovered(res, max_p95_deg):
    assert res.rms < 0.5
    np.testing.assert_allclose(res.K[0, 0], TRUE_CAM.K[0, 0], rtol=0.01)
    np.testing.assert_allclose(res.K[1, 1], TRUE_CAM.K[1, 1], rtol=0.01)
    np.testing.assert_allclose(res.K[:2, 2], TRUE_CAM.K[:2, 2], atol=3.0)
    # Distortion coefficients are correlated with each other and with f, so judge
    # the actual pixel -> ray mapping over the image area the board covered.
    assert np.percentile(_ray_errors_deg(res), 95) < max_p95_deg


def test_board_detected_in_most_views(detections):
    _, dets = detections
    assert len(dets) >= 25


def test_fisheye_recovers_intrinsics(detections):
    r, dets = detections
    views = select_views(dets, SIZE, max_views=40)
    _check_recovered(calibrate(r.target, views, SIZE, "fisheye"), max_p95_deg=0.15)


def test_a3_board_uses_marker_corners(a3_detections):
    r, dets = a3_detections
    assert r.target.n_charuco == 2 and r.target.n_points == 14
    assert len(dets) >= 80
    assert max(len(d.ids) for d in dets) == 14


def test_a3_board_fisheye_recovers_intrinsics(a3_detections):
    r, dets = a3_detections
    views = select_views(dets, SIZE, max_views=80)
    _check_recovered(calibrate(r.target, views, SIZE, "fisheye"), max_p95_deg=0.4)
