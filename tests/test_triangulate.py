import numpy as np

from moppose.calib.camera import Camera
from moppose.triangulate.dlt import triangulate


def look_at(cam_pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    """World->camera rotation for a camera at cam_pos looking at target (camera z forward, y down)."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ cam_pos


def make_rig():
    K = [[420.0, 0, 640], [0, 420.0, 480], [0, 0, 1]]
    D = [0.05, -0.02, 0.008, -0.002]
    cams = []
    for i, pos in enumerate([[3, 0, 2.5], [-1.5, 2.6, 2.5], [-1.5, -2.6, 2.5]]):
        R, t = look_at(np.array(pos, float), np.array([0, 0, 1.0]))
        cams.append(Camera(f"cam{i + 1}", "fisheye", (1280, 960), K, D, R=R, t=t))
    return cams


def test_noise_free_triangulation_is_exact():
    cams = make_rig()
    rng = np.random.default_rng(0)
    X = rng.uniform([-0.8, -0.8, 0.0], [0.8, 0.8, 1.8], size=(200, 3))
    uv = np.stack([c.project(X) for c in cams])
    Xh, err = triangulate(cams, uv)
    assert np.max(np.linalg.norm(Xh - X, axis=1)) < 1e-3
    assert np.nanmax(err) < 1e-2


def test_outlier_camera_is_dropped():
    cams = make_rig()
    rng = np.random.default_rng(1)
    X = rng.uniform([-0.5, -0.5, 0.2], [0.5, 0.5, 1.6], size=(50, 3))
    uv = np.stack([c.project(X) for c in cams])
    uv += rng.normal(0, 0.5, uv.shape)
    uv[2, :10] += 80.0  # camera 3 mis-detects the first 10 points
    Xh, err = triangulate(cams, uv, max_reproj_px=15.0)
    d = np.linalg.norm(Xh - X, axis=1)
    assert np.max(d[10:]) < 0.02
    # With only 3 cameras a bad view can be consistent with one of the good ones
    # (shift along an epipolar line) - geometrically ambiguous, so allow one miss.
    assert np.sum(d[:10] < 0.02) >= 9
    assert np.sum(np.isnan(err[2, :10])) >= 9


def test_missing_views():
    cams = make_rig()
    X = np.array([[0.1, 0.2, 1.0], [0.0, 0.0, 0.5]])
    uv = np.stack([c.project(X) for c in cams])
    uv[0, 0] = np.nan          # point 0 seen by 2 cameras -> still solvable
    uv[0, 1] = uv[1, 1] = np.nan  # point 1 seen by 1 camera -> NaN
    Xh, _ = triangulate(cams, uv)
    assert np.linalg.norm(Xh[0] - X[0]) < 1e-3
    assert np.all(np.isnan(Xh[1]))


def test_camera_yaml_roundtrip(tmp_path):
    cam = make_rig()[0]
    cam.save(tmp_path / "c.yaml")
    c2 = Camera.load(tmp_path / "c.yaml")
    np.testing.assert_allclose(c2.K, cam.K)
    np.testing.assert_allclose(c2.R, cam.R)
    pts = np.array([[100.0, 200.0], [640, 480], [1200, 900]])
    np.testing.assert_allclose(c2.undistort_points(pts), cam.undistort_points(pts))
