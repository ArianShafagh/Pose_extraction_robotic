import cv2
import numpy as np
import pytest

from moppose.calib.charuco import detect, make_detector
from moppose.calib.frame_select import select_views
from moppose.calib.intrinsics import calibrate

from synth import SIZE, TRUE_CAM, Renderer, random_poses


@pytest.fixture(scope="module")
def detections():
    r = Renderer()
    det = make_detector(r.board)
    rng = np.random.default_rng(1)
    dets = []
    for i, (rv, tv) in enumerate(random_poses(45, seed=3)):
        d = detect(det, r.render(rv, tv, rng=rng), t=float(i), frame_index=i)
        if d is not None and len(d.ids) >= 12:
            dets.append(d)
    return r, dets


def test_board_detected_in_most_views(detections):
    _, dets = detections
    assert len(dets) >= 30


def test_fisheye_recovers_intrinsics(detections):
    r, dets = detections
    views = select_views(dets, SIZE, max_views=40)
    res = calibrate(r.board, views, SIZE, "fisheye")
    assert res.rms < 0.5
    np.testing.assert_allclose(res.K[0, 0], TRUE_CAM.K[0, 0], rtol=0.01)
    np.testing.assert_allclose(res.K[1, 1], TRUE_CAM.K[1, 1], rtol=0.01)
    np.testing.assert_allclose(res.K[:2, 2], TRUE_CAM.K[:2, 2], atol=3.0)
    # The distortion coefficients are correlated, so compare the actual mapping instead:
    from moppose.calib.camera import Camera
    est = Camera("est", "fisheye", SIZE, res.K, res.D)
    w, h = SIZE
    grid = np.stack(np.meshgrid(np.linspace(40, w - 40, 15), np.linspace(40, h - 40, 12)), -1).reshape(-1, 2)
    # Only judge the image area the board actually covered; outside it the model is
    # extrapolating (which is exactly what the coverage report warns about).
    hull = cv2.convexHull(np.concatenate([d.corners for d in res.views]).astype(np.float32))
    grid = grid[[cv2.pointPolygonTest(hull, (float(x), float(y)), False) > 0 for x, y in grid]]
    assert len(grid) > 80
    rays_true = TRUE_CAM.undistort_points(grid)
    reproj = est.project(np.concatenate([rays_true, np.ones((len(grid), 1))], 1))
    assert np.max(np.linalg.norm(reproj - grid, axis=1)) < 3.0
