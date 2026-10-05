"""ChArUco board construction and detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from moppose.config import BoardConfig

# Dictionaries tried by `guess_dictionary` (most common printed boards first).
COMMON_DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]


@dataclass
class Detection:
    t: float                # timestamp (s) of the frame
    frame_index: int
    corners: np.ndarray     # (N, 2) float32 ChArUco corner pixels
    ids: np.ndarray         # (N,) int32 ChArUco corner ids
    sharpness: float        # variance of Laplacian over the board region


def make_board(cfg: BoardConfig) -> cv2.aruco.CharucoBoard:
    if cfg.marker_len_m >= cfg.square_len_m:
        raise ValueError("marker_len_m must be smaller than square_len_m")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
    board = cv2.aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y), cfg.square_len_m, cfg.marker_len_m, dictionary
    )
    board.setLegacyPattern(cfg.legacy_pattern)
    return board


def make_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    det_params = cv2.aruco.DetectorParameters()
    # Subpixel refinement of marker corners; ChArUco corners are refined separately.
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Fisheye edges and security-cam compression produce small, soft markers.
    det_params.adaptiveThreshWinSizeMax = 53
    det_params.minMarkerPerimeterRate = 0.01
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.minMarkers = 1  # accept corners adjacent to a single detected marker
    return cv2.aruco.CharucoDetector(board, charuco_params, det_params)


def sharpness(gray: np.ndarray, pts: np.ndarray | None = None) -> float:
    """Variance of Laplacian, restricted to the bounding box of `pts` if given."""
    if pts is not None and len(pts):
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 > 8 and y1 - y0 > 8:
            gray = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class Target:
    """A ChArUco board seen as a set of calibration points with stable integer ids.

    ids 0 .. n_charuco-1            inner chessboard corners
    ids n_charuco + 4*k + j         corner j (clockwise from top-left) of the k-th marker
                                    of the board - only if cfg.use_marker_corners

    Small boards (e.g. 3x2 squares = only 2 inner corners) need the marker corners
    to have enough points per view.
    """

    def __init__(self, cfg: BoardConfig):
        self.cfg = cfg
        self.board = make_board(cfg)
        self.detector = make_detector(self.board)
        chess = np.asarray(self.board.getChessboardCorners(), np.float64).reshape(-1, 3)
        self.n_charuco = len(chess)
        self.use_marker_corners = cfg.use_marker_corners
        self._marker_index = {int(m): k for k, m in enumerate(np.asarray(self.board.getIds()).ravel())}
        if self.use_marker_corners:
            markers = np.concatenate([np.asarray(o, np.float64).reshape(4, 3) for o in self.board.getObjPoints()])
            self.obj = np.vstack([chess, markers])
        else:
            self.obj = chess
        self.charuco_offset = self._measure_charuco_offset()

    def _measure_charuco_offset(self) -> np.ndarray:
        """Systematic pixel offset of the detector's chessboard corners.

        Some OpenCV versions (seen in 4.11) return ChArUco corners shifted by ~+0.5 px
        in x and y (pixel-centre convention mismatch), while marker corners are fine.
        Mixing both point types then biases the calibration (focal length off by
        >1%). We render the board under a known homography, detect it and measure
        the mean offset, so the correction follows whatever OpenCV is installed.
        """
        ppm = 40.0 / self.cfg.square_len_m  # 40 px per square
        margin = 60
        w = int(round(self.cfg.squares_x * self.cfg.square_len_m * ppm))
        h = int(round(self.cfg.squares_y * self.cfg.square_len_m * ppm))
        tex = self.board.generateImage((w + 2 * margin, h + 2 * margin), marginSize=margin)
        H = np.array([[1.35, 0.12, 37.3], [-0.06, 1.28, 41.7], [2.0e-4, 1.5e-4, 1.0]])
        size = (int(tex.shape[1] * 1.8), int(tex.shape[0] * 1.8))
        img = cv2.warpPerspective(tex, H, size, flags=cv2.INTER_LINEAR, borderValue=255)
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
        ch_corners, ch_ids, _, _ = self.detector.detectBoard(img)
        if ch_ids is None or len(ch_ids) == 0:
            return np.zeros(2)
        # texture pixel i covers board coords [i, i+1)/ppm, so its centre is at i + 0.5
        chess = np.asarray(self.board.getChessboardCorners(), np.float64).reshape(-1, 3)
        tex_xy = chess[ch_ids.ravel(), :2] * ppm + margin - 0.5
        true = cv2.perspectiveTransform(tex_xy.reshape(-1, 1, 2), H).reshape(-1, 2)
        off = (ch_corners.reshape(-1, 2) - true).mean(0)
        return off if np.max(np.abs(off)) > 0.1 else np.zeros(2)

    @property
    def n_points(self) -> int:
        return len(self.obj)

    def detect(self, image: np.ndarray, t: float = 0.0, frame_index: int = 0) -> Detection | None:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, m_corners, m_ids = self.detector.detectBoard(gray)
        pts, ids = [], []
        if ch_ids is not None and len(ch_ids):
            pts.append(ch_corners.reshape(-1, 2) - self.charuco_offset)
            ids.append(ch_ids.reshape(-1))
        if self.use_marker_corners and m_ids is not None and len(m_ids):
            for c, m in zip(m_corners, np.asarray(m_ids).ravel()):
                k = self._marker_index.get(int(m))
                if k is None:  # a marker of the same dictionary that is not on this board
                    continue
                pts.append(np.asarray(c).reshape(4, 2))
                ids.append(self.n_charuco + 4 * k + np.arange(4))
        if not pts:
            return None
        corners = np.concatenate(pts).astype(np.float32)
        return Detection(
            t=t, frame_index=frame_index, corners=corners,
            ids=np.concatenate(ids).astype(np.int32), sharpness=sharpness(gray, corners),
        )


def is_degenerate(obj: np.ndarray, min_ratio: float = 0.15) -> bool:
    """True if the detected corners (nearly) lie on one line on the board.

    Collinear points give no homography and break fisheye.calibrate's initialisation.
    Checked in board coordinates via the ratio of the two principal spreads.
    """
    if len(obj) < 6:
        return True
    xy = obj[:, :2] - obj[:, :2].mean(0)
    s = np.linalg.svd(xy, compute_uv=False)
    return s[0] <= 0 or s[1] / s[0] < min_ratio


def draw_detection(image: np.ndarray, det: Detection, n_charuco: int | None = None) -> np.ndarray:
    """Chessboard corners in red, marker corners in green, with their point ids."""
    out = image.copy()
    r = max(3, image.shape[1] // 300)
    for (x, y), i in zip(det.corners, det.ids):
        color = (0, 0, 255) if n_charuco is None or i < n_charuco else (0, 200, 0)
        cv2.circle(out, (int(round(x)), int(round(y))), r, color, -1)
        cv2.putText(out, str(int(i)), (int(x) + r + 2, int(y) - r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * r / 3, color, 1)
    return out


def guess_dictionary(images: list[np.ndarray]) -> list[tuple[str, int, int]]:
    """Try every common ArUco dictionary on the images.

    Returns (dictionary, markers_found, max_marker_id) sorted by markers found.
    Helps when you are not sure how the board was generated.
    """
    grays = [im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    params = cv2.aruco.DetectorParameters()
    results = []
    for name in COMMON_DICTS:
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params)
        found, max_id = 0, -1
        for g in grays:
            _, ids, _ = det.detectMarkers(g)
            if ids is not None:
                found += len(ids)
                max_id = max(max_id, int(ids.max()))
        results.append((name, found, max_id))
    return sorted(results, key=lambda r: -r[1])
