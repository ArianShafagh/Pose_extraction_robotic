"""Canonical joint set used downstream (triangulation, robot), and maps from model outputs."""

from __future__ import annotations

import numpy as np

# COCO-17 body + 6 foot points (COCO-WholeBody order).
BODY_FEET = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
]

EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_ankle", "left_big_toe"), ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_heel"), ("right_ankle", "right_big_toe"), ("right_big_toe", "right_small_toe"),
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]
EDGE_IDX = np.array([(BODY_FEET.index(a), BODY_FEET.index(b)) for a, b in EDGES])

# COCO-WholeBody (133): body 0-16 and feet 17-22 are exactly BODY_FEET.
COCO_WHOLEBODY_TO_BODY_FEET = np.arange(23)


def index_map(native_names: list[str]) -> np.ndarray:
    """Indices into a model's native keypoint list for each BODY_FEET joint (matched by name)."""
    lookup = {n: i for i, n in enumerate(native_names)}
    missing = [n for n in BODY_FEET if n not in lookup]
    if missing:
        raise KeyError(f"model keypoints lack {missing}")
    return np.array([lookup[n] for n in BODY_FEET])
