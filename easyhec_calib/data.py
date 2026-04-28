"""Dataset loader for calibration frames.

Layout:
  data_dir/
    meta.json
    frame_*.png         RGB images
    frame_*.qpos.npy    1D joint vector (ordered to match meta.joint_names)
    frame_*.mask.png    optional cached robot silhouette (>=128 = robot)
"""

import glob
import json
import os
from typing import Any

import cv2
import numpy as np


def load_dataset(data_dir: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    frame_paths = sorted(
        p for p in glob.glob(os.path.join(data_dir, "frame_*.png"))
        if not p.endswith(".mask.png")
    )
    frames = []
    for p in frame_paths:
        stem = p[:-4]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        qpos = np.load(stem + ".qpos.npy")
        mask_path = stem + ".mask.png"
        mask = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
        frames.append({"rgb": img, "qpos": qpos, "mask": mask, "path": p, "stem": stem})
    return meta, frames
