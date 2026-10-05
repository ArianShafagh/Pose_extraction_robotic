"""Loading of the YAML configs in configs/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BoardConfig:
    squares_x: int
    squares_y: int
    square_len_m: float
    marker_len_m: float
    dictionary: str = "DICT_5X5_100"
    legacy_pattern: bool = False
    # Also use the 4 corners of every ArUco marker as calibration points
    # (needed for small boards with few inner chessboard corners).
    use_marker_corners: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "BoardConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


@dataclass
class CalibSettings:
    sample_every_s: float = 0.25
    min_corners: int = 12
    min_sharpness: float = 40.0
    max_views: int = 60
    outlier_rms_px: float = 2.0


@dataclass
class CameraEntry:
    name: str
    calib_videos: list[Path] = field(default_factory=list)
    videos: list[Path] = field(default_factory=list)
    model: str = "auto"  # auto | fisheye | pinhole


@dataclass
class SessionConfig:
    root: Path
    output_dir: Path
    cameras: dict[str, CameraEntry]
    calibration: CalibSettings

    @classmethod
    def load(cls, path: str | Path) -> "SessionConfig":
        path = Path(path)
        # Paths in the session file are relative to the repo root (parent of configs/).
        root = path.resolve().parent.parent
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cams = {}
        for name, c in (raw.get("cameras") or {}).items():
            cams[name] = CameraEntry(
                name=name,
                calib_videos=[root / p for p in c.get("calib_videos", [])],
                videos=[root / p for p in c.get("videos", [])],
                model=c.get("model", "auto"),
            )
        return cls(
            root=root,
            output_dir=root / raw.get("output_dir", "outputs"),
            cameras=cams,
            calibration=CalibSettings(**(raw.get("calibration") or {})),
        )

    def camera(self, name: str) -> CameraEntry:
        if name not in self.cameras:
            raise KeyError(f"camera '{name}' not in session config (have: {list(self.cameras)})")
        return self.cameras[name]

    @property
    def calib_dir(self) -> Path:
        d = self.output_dir / "calib"
        d.mkdir(parents=True, exist_ok=True)
        return d
