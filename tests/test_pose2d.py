from pathlib import Path

import numpy as np
import pytest

from moppose.calib.camera import Camera
from moppose.pose2d.people import Tracks
from moppose.pose2d.select import mopper_boxes, select_mopper
from moppose.pose2d.skeleton import BODY_FEET, EDGE_IDX, index_map
from moppose.pose2d.virtual_cam import make_view

from synth import TRUE_CAM

REAL_CAM = Path(__file__).parents[1] / "calibration" / "session1" / "cam1_intrinsics.yaml"


def _cams():
    cams = [TRUE_CAM]
    if REAL_CAM.exists():
        cams.append(Camera.load(REAL_CAM))
    return cams


@pytest.mark.parametrize("bbox", [[560, 380, 700, 700], [40, 300, 200, 760], [1000, 100, 1200, 500]])
def test_virtual_view_round_trip(bbox):
    for cam in _cams():
        v = make_view(cam, np.array(bbox, float))
        rng = np.random.default_rng(0)
        x1, y1, x2, y2 = bbox
        raw = np.c_[rng.uniform(x1, x2, 50), rng.uniform(y1, y2, 50)]
        back = v.crop_to_raw(cam, v.raw_to_crop(cam, raw))
        assert np.max(np.linalg.norm(back - raw, axis=1)) < 0.05


def test_virtual_view_frames_the_person():
    cam = _cams()[-1]
    v = make_view(cam, np.array([40.0, 300, 200, 760]), size=(768, 1024), margin=1.6)
    w, h = v.size
    x1, y1, x2, y2 = v.bbox
    assert 0 < x1 < x2 < w and 0 < y1 < y2 < h
    # the person box (padded by the models' 1.25x) must fit in the crop
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.25, (y2 - y1) * 1.25
    assert cx - bw / 2 >= -1 and cx + bw / 2 <= w + 1 and cy - bh / 2 >= -1 and cy + bh / 2 <= h + 1


def test_crop_pixels_match_rendered_image():
    """A bright dot in the raw image must appear where raw_to_crop says it is."""
    cam = _cams()[-1]
    img = np.zeros((cam.size[1], cam.size[0], 3), np.uint8)
    dot = np.array([150.0, 500.0])
    img[499:502, 149:152] = 255
    v = make_view(cam, np.array([60.0, 350, 260, 700]))
    crop = v.render(cam, img).astype(float).sum(2)
    ys, xs = np.nonzero(crop > 0.5 * crop.max())
    found = np.array([np.average(xs, weights=crop[ys, xs]), np.average(ys, weights=crop[ys, xs])])
    assert np.linalg.norm(found - v.raw_to_crop(cam, dot[None])[0]) < 1.0


def test_skeleton_maps():
    names = ["foo"] + BODY_FEET[::-1]
    m = index_map(names)
    assert [names[i] for i in m] == BODY_FEET
    assert EDGE_IDX.max() < len(BODY_FEET)
    with pytest.raises(KeyError):
        index_map(BODY_FEET[:-1])


def _tracks(rows):
    t, tid, bb = zip(*rows)
    t = np.array(t, float)
    return Tracks(t=t, frame_index=np.round(t * 30).astype(int), track_id=np.array(tid), bbox=np.array(bb, float),
                  score=np.ones(len(t)), frame_t=np.unique(t))


def test_select_mopper_prefers_walker_and_stitches_fragments():
    rows = []
    for k in range(100):
        rows.append((k * 0.1, 1, [500, 400, 560, 600]))                    # sitting person
    for k in range(50):
        x = 100 + 8 * k
        rows.append((k * 0.1, 2, [x, 300, x + 60, 500]))                   # walker, part 1
    for k in range(50):
        x = 100 + 8 * 49 + 8 * (k + 15)
        rows.append((6.5 + k * 0.1, 7, [x, 300, x + 60, 500]))             # walker after occlusion
    tr = _tracks(rows)
    chain = select_mopper(tr)
    assert chain == [2, 7]
    t, _, bb = mopper_boxes(tr, chain)
    assert len(t) == 100 and np.all(np.diff(t) > 0)
