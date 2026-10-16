"""Recording configuration: one YAML per recorded action (configs/recordings/*.yaml) plus the
shared lab setup (configs/setup.yaml).

Defaults live in configs/recordings/_template.yaml semantics below; every section is optional
except `name` and `videos`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULTS = {
    "setup": "configs/setup.yaml",
    "time_range": {"start_s": 0.0, "end_s": None},
    "actor": {"mode": "auto", "track_ids": {}},
    "pose": {"backend": "sapiens2", "model_size": "0.8b", "cross_check": "rtmw", "every": 2,
             "flip_test": "auto", "batch": 2},
    "sync": {"max_offset_s": 3.0, "offsets_s": {}},
    "reconstruction": {"min_conf": 0.5, "max_reproj_px": 15, "bone_tolerance": 0.25, "max_gap_s": 0.5,
                       "smooth_cutoff_hz": 4.0},
    "video3d": {"enabled": True, "width": 1280, "height": 720, "view": "fixed", "azimuth_deg": -60,
                "elevation_deg": 25, "distance_m": 6.5, "trail_s": 5, "show_cameras": True},
    "outputs": {
        "dir": "results/{name}",
        "work_dir": "outputs/{name}",
        "files": {
            "landmarks_csv": "{name}_landmarks3d.csv",
            "landmarks_npz": "{name}_landmarks3d.npz",
            "video_3d": "{name}_skeleton3d.mp4",
            "report": "{name}_report.json",
            "rig": "{name}_rig.yaml",
            "overview": "{name}_overview.png",
            "bone_lengths": "{name}_bone_lengths.png",
            "reprojection": "{name}_reprojection.jpg",
            "actor_preview": "{name}_{cam}_actor.jpg",
        },
    },
}
SETUP_DEFAULTS = {"quality": {"max_median_reproj_px": 4.0, "max_floor_offset_m": 0.06}}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class ConfigError(ValueError):
    pass


@dataclass
class Recording:
    root: Path              # repo root (paths are relative to it)
    path: Path              # the recording YAML
    cfg: dict               # recording config with defaults filled in
    setup: dict             # setup config with defaults filled in

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path, root: str | Path | None = None) -> "Recording":
        path = Path(path).resolve()
        root = Path(root).resolve() if root else _find_root(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = _merge(DEFAULTS, raw)
        setup_path = root / cfg["setup"]
        if not setup_path.exists():
            raise ConfigError(f"setup file not found: {setup_path}")
        with open(setup_path, encoding="utf-8") as f:
            setup = _merge(SETUP_DEFAULTS, yaml.safe_load(f) or {})
        rec = cls(root, path, cfg, setup)
        rec.validate()
        return rec

    def validate(self, check_files: bool = True) -> None:
        name = self.cfg.get("name")
        if not name or not re.fullmatch(r"[A-Za-z0-9_\-]+", str(name)):
            raise ConfigError(f"`name` must be letters, digits, _ or - (got {name!r})")
        cams = self.cameras
        vids = self.cfg.get("videos") or {}
        missing = [c for c in cams if c not in vids]
        if missing:
            raise ConfigError(f"`videos` needs an entry for every camera of the setup; missing: {missing}")
        extra = [c for c in vids if c not in cams]
        if extra:
            raise ConfigError(f"`videos` has cameras that are not in the setup: {extra}")
        if check_files:
            for c in cams:
                if not self.video(c).exists():
                    raise ConfigError(f"video for {c} not found: {self.video(c)}")
        if self.cfg["pose"]["backend"] not in ("sapiens2", "rtmw"):
            raise ConfigError("pose.backend must be sapiens2 or rtmw")

    # ---------------------------------------------------------------- accessors
    @property
    def name(self) -> str:
        return str(self.cfg["name"])

    @property
    def cameras(self) -> list[str]:
        return list(self.setup["cameras"])

    def video(self, cam: str) -> Path:
        return self.root / self.cfg["videos"][cam]

    @property
    def videos(self) -> dict[str, Path]:
        return {c: self.video(c) for c in self.cameras}

    def _fmt(self, s: str, **kw) -> str:
        return s.format(name=self.name, **kw)

    @property
    def out_dir(self) -> Path:
        d = self.root / self._fmt(self.cfg["outputs"]["dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def work_dir(self) -> Path:
        d = self.root / self._fmt(self.cfg["outputs"]["work_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def output(self, key: str, **kw) -> Path:
        """Configured output file, e.g. output('video_3d') or output('actor_preview', cam='cam1')."""
        files = self.cfg["outputs"]["files"]
        if key not in files:
            raise ConfigError(f"outputs.files has no entry {key!r}")
        return self.out_dir / self._fmt(files[key], **kw)

    def intrinsics_path(self, cam: str) -> Path:
        return self.root / self.setup["intrinsics_dir"] / f"{cam}_intrinsics.yaml"

    @property
    def rig_path(self) -> Path:
        return self.root / self.setup["rig"]

    def section_hash(self, *sections: str) -> str:
        """Fingerprint of config sections (to know whether a finished stage must be redone)."""
        data = {s: self.cfg.get(s) for s in sections}
        data["videos"] = {c: str(v) for c, v in self.videos.items()}
        return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _find_root(path: Path) -> Path:
    for p in [path.parent, *path.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return path.parent
