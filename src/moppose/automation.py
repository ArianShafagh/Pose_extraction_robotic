"""`moppose run <recording.yaml>`: raw videos -> named 3D outputs, in resumable stages.

Stages (in order) and the config sections they depend on:

    check    videos                      videos open, resolution matches the lens calibration
    people   time_range                  YOLO26 + BoT-SORT person tracks per camera
    actor    actor                       the actor's track in every camera (multi-view check with the rig)
    pose2d   pose, time_range            2D keypoints on fisheye-rectified crops (Sapiens2 / RTMW)
    sync     sync                        time offsets between the cameras
    recon3d  reconstruction              triangulation in metres, cleaning, smoothing
    export   outputs                     named CSV / NPZ / report / images, quality checks
    video3d  video3d, outputs            full-length 3D skeleton video

<work_dir>/state.json stores a fingerprint per finished stage (its config sections + the
fingerprint of the stage before). A stage is skipped when its fingerprint is unchanged, so an
interrupted run continues and changing e.g. `reconstruction` only reruns recon3d, export, video3d.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from moppose.recording import Recording

STAGES = ["check", "people", "actor", "pose2d", "sync", "recon3d", "export", "video3d"]
SECTIONS = {"check": (), "people": ("time_range",), "actor": ("actor",), "pose2d": ("pose", "time_range"),
            "sync": ("sync",), "recon3d": ("reconstruction",), "export": ("outputs",),
            "video3d": ("video3d", "outputs")}


class Runner:
    def __init__(self, rec: Recording, log=print):
        self.rec = rec
        self.log = log
        self.work = rec.work_dir
        self.state_path = self.work / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    # ------------------------------------------------------------------ bookkeeping
    def fingerprint(self, stage: str) -> str:
        i = STAGES.index(stage)
        prev = self.fingerprint(STAGES[i - 1]) if i else ""
        own = self.rec.section_hash(*SECTIONS[stage])
        if stage == "pose2d":  # the rig/intrinsics matter for the crops and the 3D
            own += self._file_hash(*[self.rec.intrinsics_path(c) for c in self.rec.cameras])
        if stage == "sync":
            own += self._file_hash(self.rec.rig_path)
        return hashlib.sha1((prev + own).encode()).hexdigest()[:12]

    @staticmethod
    def _file_hash(*paths: Path) -> str:
        h = hashlib.sha1()
        for p in paths:
            h.update(Path(p).read_bytes() if Path(p).exists() else b"missing")
        return h.hexdigest()[:8]

    def is_done(self, stage: str) -> bool:
        st = self.state.get(stage)
        if st and st.get("fingerprint") == self.fingerprint(stage):
            return True
        # results that already exist (e.g. computed before automation existed) are adopted
        return st is None and self._adoptable(stage)

    def mark_done(self, stage: str, info: dict | None = None) -> None:
        self.state[stage] = {"fingerprint": self.fingerprint(stage), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "info": info or {}}
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def invalidate_from(self, stage: str) -> None:
        for s in STAGES[STAGES.index(stage):]:
            self.state.pop(s, None)
            self.state[s] = {"fingerprint": "invalidated"}
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def _adoptable(self, stage: str) -> bool:
        cams = self.rec.cameras
        if stage == "check":
            return False
        if stage == "people":
            return all((self.work / "people" / f"{c}.npz").exists() for c in cams) and self._tracks_cover()
        if stage == "pose2d":
            return all(self._pose_path(b, c).exists() for b in self._backends() for c in cams)
        return False

    def _tracks_cover(self) -> bool:
        from moppose.pose2d.people import Tracks

        tr0 = self.rec.cfg["time_range"]
        for c in self.rec.cameras:
            ft = Tracks.load(self.work / "people" / f"{c}.npz").frame_t
            end = tr0["end_s"] if tr0["end_s"] is not None else self._duration(c)
            if len(ft) == 0 or ft.min() > tr0["start_s"] + 0.5 or ft.max() < end - 1.0:
                return False
        return True

    def _duration(self, cam: str) -> float:
        import cv2

        from moppose.io.video import open_video

        cap = open_video(self.rec.video(cam))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return n / fps

    def _backends(self) -> list[str]:
        p = self.rec.cfg["pose"]
        return [p["backend"]] + ([p["cross_check"]] if p.get("cross_check") and p["cross_check"] != p["backend"] else [])

    def _pose_path(self, backend: str, cam: str) -> Path:
        return self.work / "pose2d" / backend / f"{self.rec.video(cam).stem}.npz"

    # ------------------------------------------------------------------ driver
    def run(self, start: str | None = None, only: str | None = None, dry_run: bool = False) -> None:
        if start and not dry_run:
            self.invalidate_from(start)
        todo = [only] if only else STAGES
        self.log(f"recording '{self.rec.name}': outputs -> {self.rec.out_dir}   work -> {self.work}")
        for s in todo:
            done = self.is_done(s) and not (only and s == only)
            if dry_run:
                self.log(f"  {s:8s} {'done (skip)' if done else 'RUN'}")
                continue
            if done:
                self.log(f"[{s}] up to date - skipped")
                continue
            self.log(f"[{s}] running ...")
            t0 = time.perf_counter()
            info = getattr(self, f"stage_{s}")() or {}
            self.mark_done(s, info)
            self.log(f"[{s}] done in {time.perf_counter() - t0:.0f} s")
        if dry_run:
            self.log("output files:")
            for k in self.rec.cfg["outputs"]["files"]:
                self.log(f"  {k:15s} {self.rec.output(k, cam='{cam}')}")

    # ------------------------------------------------------------------ stages
    def _intrinsics(self):
        from moppose.calib.camera import Camera

        return {c: Camera.load(self.rec.intrinsics_path(c)) for c in self.rec.cameras}

    def _rig(self):
        from moppose.calib.camera import load_rig

        rig = load_rig(self.rec.rig_path)
        missing = [c for c in self.rec.cameras if c not in rig]
        if missing:
            raise ValueError(f"rig {self.rec.rig_path} has no cameras {missing}")
        return [rig[c] for c in self.rec.cameras]

    def stage_check(self):
        from moppose.io.video import probe

        info = {}
        intr = self._intrinsics()
        for c in self.rec.cameras:
            p = probe(self.rec.video(c), scan=False)
            if (p.width, p.height) != intr[c].size:
                raise ValueError(f"{c}: video is {p.width}x{p.height} but the lens calibration is for "
                                 f"{intr[c].size[0]}x{intr[c].size[1]}")
            info[c] = {"size": [p.width, p.height], "duration_s": round(self._duration(c), 2)}
            self.log(f"  {c}: {p.width}x{p.height}, {info[c]['duration_s']} s  ({self.rec.video(c).name})")
        return info

    def stage_people(self):
        from moppose.pose2d.people import track_people

        tr = self.rec.cfg["time_range"]
        out = self.work / "people"
        for c in self.rec.cameras:
            track_people(self.rec.video(c), start_s=tr["start_s"], end_s=tr["end_s"]).save(out / f"{c}.npz")

    def stage_actor(self):
        from moppose.io.video import read_frame_at
        from moppose.pose2d.people import Tracks
        from moppose.pose2d.select import preview, select_actor_multiview, select_mopper, summarize

        tracks = {c: Tracks.load(self.work / "people" / f"{c}.npz") for c in self.rec.cameras}
        cfg = self.rec.cfg["actor"]
        manual = {c: v for c, v in (cfg.get("track_ids") or {}).items() if v is not None}
        if cfg.get("mode", "auto") == "auto":
            chains, info = select_actor_multiview(tracks, self._rig())
        else:
            chains, info = {c: select_mopper(tracks[c]) for c in self.rec.cameras}, {"method": "farthest walker"}
        for c, tid in manual.items():
            chains[c] = select_mopper(tracks[c], seed_id=int(tid))
            info.setdefault("manual", {})[c] = int(tid)
        self.log(f"  actor: {info.get('method')}  seed tracks {info.get('seed_tracks', manual)}  "
                 f"(multi-view feet error {info.get('feet_reproj_err_px', float('nan')):.1f} px)")
        for c in self.rec.cameras:
            (self.work / "people" / f"{c}_actor.json").write_text(json.dumps({"chain": chains[c]}))
            s = summarize(tracks[c])
            cover = sum(s[i].n for i in chains[c] if i in s)
            self.log(f"  {c}: tracks {chains[c]} cover {cover / max(len(tracks[c].frame_t), 1) * 100:.0f}% of frames")
            ft = tracks[c].frame_t
            bg = read_frame_at(self.rec.video(c), float(ft[len(ft) // 2])).image
            import cv2

            cv2.imwrite(str(self.rec.output("actor_preview", cam=c)), preview(bg, tracks[c], chains[c]))
        return info | {"chains": chains}

    def stage_pose2d(self):
        from moppose.pose2d.backends import make_backend
        from moppose.pose2d.people import Tracks
        from moppose.pose2d.run import run_pose
        from moppose.pose2d.select import mopper_boxes

        p = self.rec.cfg["pose"]
        flip = {"auto": None, True: True, False: False, "on": True, "off": False}.get(p.get("flip_test", "auto"))
        backends = [make_backend(b, size=p.get("model_size", "0.8b"), flip_test=flip) for b in self._backends()]
        tr = self.rec.cfg["time_range"]
        intr = self._intrinsics()
        info = {}
        for c in self.rec.cameras:
            tracks = Tracks.load(self.work / "people" / f"{c}.npz")
            chain = json.loads((self.work / "people" / f"{c}_actor.json").read_text())["chain"]
            t, _, boxes = mopper_boxes(tracks, chain)
            end = tr["end_s"] if tr["end_s"] is not None else np.inf
            keep = (t >= tr["start_s"]) & (t <= end)
            t, boxes = t[keep][::int(p.get("every", 1))], boxes[keep][::int(p.get("every", 1))]
            self.log(f"  {c}: {len(t)} frames")
            run_pose(intr[c], self.rec.video(c), t, boxes, backends, self.work / "pose2d", batch=int(p.get("batch", 2)))
            info[c] = int(len(t))
        return info

    def _tracks2d(self, backend: str | None = None):
        from moppose.multiview.data import Track2D

        b = backend or self.rec.cfg["pose"]["backend"]
        return {c: Track2D.load(c, self._pose_path(b, c)) for c in self.rec.cameras}

    def _times(self, tracks):
        ref = tracks[self.rec.cameras[0]]
        tr = self.rec.cfg["time_range"]
        end = tr["end_s"] if tr["end_s"] is not None else np.inf
        return ref.t[(ref.t >= max(tr["start_s"], ref.t[0]) + 0.5) & (ref.t <= min(end, ref.t[-1]) - 0.5)]

    def stage_sync(self):
        from moppose.multiview.sync import coarse_offset, refine_offset, relative_pose

        cfg = self.rec.cfg["sync"]
        tracks = self._tracks2d()
        rig = self._rig()
        names = self.rec.cameras
        times = self._times(tracks)
        offsets, info = {names[0]: 0.0}, {}
        for b, n in enumerate(names[1:], start=1):
            fixed = (cfg.get("offsets_s") or {}).get(n)
            if fixed is not None:
                offsets[n] = float(fixed)
                info[n] = {"offset_ms": offsets[n] * 1000, "method": "config"}
            else:
                d0, corr = coarse_offset(tracks[names[0]], tracks[n], max_lag=float(cfg.get("max_offset_s", 3.0)))
                d, e = refine_offset(tracks[names[0]], tracks[n], rig[0], rig[b], *relative_pose(rig[0], rig[b]),
                                     times, d0)
                offsets[n] = d
                info[n] = {"offset_ms": d * 1000, "coarse_ms": d0 * 1000, "motion_corr": corr, "epipolar_px": e}
            self.log(f"  {n}: offset {offsets[n] * 1000:+.1f} ms  {info[n]}")
        (self.work / "sync.json").write_text(json.dumps({"offsets_s": offsets, "info": info}, indent=1))
        return info

    def stage_recon3d(self):
        from moppose.multiview.pipeline import bone_lengths, clean_and_smooth, reconstruct, reject_bone_outliers

        rc = self.rec.cfg["reconstruction"]
        tracks = self._tracks2d()
        rig = self._rig()
        offsets = json.loads((self.work / "sync.json").read_text())["offsets_s"]
        times = self._times(tracks)
        jn = tracks[self.rec.cameras[0]].joint_names
        X, err, views, stats = reconstruct(tracks, rig, offsets, times, min_conf=rc["min_conf"],
                                           max_reproj_px=rc["max_reproj_px"])
        Xr, dropped = reject_bone_outliers(X, jn, tol=rc["bone_tolerance"])
        Xs, filled = clean_and_smooth(Xr, times, max_gap_s=rc["max_gap_s"], cutoff_hz=rc["smooth_cutoff_hz"])
        np.savez_compressed(self.work / "recon3d.npz", t=times, xyz=Xs, xyz_raw=X, filled=filled, views=views,
                            reproj_err=err, joint_names=np.array(jn), edges=tracks[self.rec.cameras[0]].edges)
        info = {"frames": int(len(times)), "valid": float(np.isfinite(Xs[..., 0]).mean()), "swaps": stats,
                "bone_outliers": dropped}
        for k, c in enumerate(self.rec.cameras):
            self.log(f"  {c}: reprojection median {np.nanmedian(err[k]):.2f} px")
        bl = bone_lengths(Xs, tracks[self.rec.cameras[0]].edges)
        info["bone_cv_percent"] = float(np.nanmedian(np.nanstd(bl, 0) / np.nanmedian(bl, 0)) * 100)
        return info

    def _recon(self):
        z = np.load(self.work / "recon3d.npz")
        return {k: z[k] for k in z.files}

    def stage_export(self):
        import yaml

        from moppose.calib.camera import save_rig
        from moppose.multiview.data import on_common_clock
        from moppose.multiview.pipeline import bone_lengths
        from moppose.viz.skeleton3d import bone_plot, overview, reprojection_grid

        r = self._recon()
        jn = [str(n) for n in r["joint_names"]]
        edges, t, Xs, err = r["edges"], r["t"], r["xyz"], r["reproj_err"]
        rig = self._rig()
        sync = json.loads((self.work / "sync.json").read_text())
        names = self.rec.cameras
        q = self.rec.setup["quality"]

        np.savez_compressed(self.rec.output("landmarks_npz"), t=t, xyz=Xs.astype(np.float32),
                            xyz_raw=r["xyz_raw"].astype(np.float32), filled=r["filled"],
                            views=r["views"].astype(np.int8), reproj_err=err.astype(np.float32),
                            joint_names=np.array(jn), edges=edges, cameras=np.array(names))
        header = "t," + ",".join(f"{n}_{a}" for n in jn for a in "xyz")
        np.savetxt(self.rec.output("landmarks_csv"), np.c_[t, Xs.reshape(len(t), -1)], delimiter=",", header=header,
                   comments="", fmt="%.4f")
        save_rig(rig, self.rec.output("rig"))
        with open(self.rec.output("rig"), "a", encoding="utf-8") as f:
            yaml.safe_dump({"recording": self.rec.name, "time_offsets_s": sync["offsets_s"],
                            "time_offset_meaning": "video time of a camera = reference (first camera) time + offset",
                            "world": "metres, z up, floor at z = 0"}, f, sort_keys=False)

        bl = bone_lengths(Xs, edges)
        med, cv = np.nanmedian(bl, 0), np.nanstd(bl, 0) / np.nanmedian(bl, 0)
        heels = Xs[:, [jn.index("left_heel"), jn.index("right_heel")], 2]
        floor = float(np.nanpercentile(heels, 10))
        warnings = []
        for k, c in enumerate(names):
            m = float(np.nanmedian(err[k]))
            if m > q["max_median_reproj_px"]:
                warnings.append(f"{c}: median reprojection {m:.1f} px > {q['max_median_reproj_px']} px - "
                                f"camera moved? (run calibrate-rig)")
        if abs(floor) > q["max_floor_offset_m"]:
            warnings.append(f"heels are {floor * 100:.0f} cm off the floor - rig may not fit this recording")
        report = {
            "recording": self.rec.name, "config": str(self.rec.path), "frames": int(len(t)),
            "duration_s": float(t[-1] - t[0]), "fps": float(1 / np.median(np.diff(t))),
            "valid_joint_samples": float(np.isfinite(Xs[..., 0]).mean()),
            "mean_views_per_joint": float(r["views"].mean()),
            "reprojection_px": {c: {"median": float(np.nanmedian(err[k])), "p90": float(np.nanpercentile(err[k], 90))}
                                for k, c in enumerate(names)},
            "time_offsets_ms": {c: v * 1000 for c, v in sync["offsets_s"].items()},
            "heel_height_p10_m": floor,
            "bone_length_cm": {f"{jn[a]}-{jn[b]}": {"median": float(med[i] * 100), "cv_percent": float(cv[i] * 100)}
                               for i, (a, b) in enumerate(edges)},
            "stages": {s: self.state.get(s, {}).get("info") for s in STAGES if s in self.state},
            "warnings": warnings,
        }
        self.rec.output("report").write_text(json.dumps(report, indent=1, default=str))
        overview(Xs, t, edges, jn, rig, self.rec.output("overview"))
        bone_plot(bl, t, edges, jn, self.rec.output("bone_lengths"))
        uv, _ = on_common_clock(self._tracks2d(), sync["offsets_s"], t)
        reprojection_grid(rig, self.rec.videos, sync["offsets_s"], Xs, t, edges, uv,
                          list(np.linspace(t[0] + 0.1 * (t[-1] - t[0]), t[-1] - 0.1 * (t[-1] - t[0]), 4)),
                          self.rec.output("reprojection"))
        for w in warnings:
            self.log(f"  WARNING: {w}")
        return {"warnings": warnings}

    def stage_video3d(self):
        from moppose.viz.video3d import render_full_video

        cfg = self.rec.cfg["video3d"]
        if not cfg.get("enabled", True):
            self.log("  video3d disabled in the config")
            return {"enabled": False}
        r = self._recon()
        return render_full_video(r["xyz"], r["t"], r["edges"], [str(n) for n in r["joint_names"]],
                                 self.rec.output("video_3d"), cams=self._rig(), title=self.rec.name, **cfg)
