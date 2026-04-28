"""Smoke tests that don't require easyhec/torch/cuda."""

import json
import os
import tempfile

import numpy as np


def test_imports():
    import easyhec_calib
    assert hasattr(easyhec_calib, "load_dataset")
    assert hasattr(easyhec_calib, "URDFMeshSource")
    assert hasattr(easyhec_calib, "refine")


def test_invSE3():
    from easyhec_calib.utils import invSE3

    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = rng.standard_normal(3)
    Tinv = invSE3(T)
    np.testing.assert_allclose(T @ Tinv, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(Tinv @ T, np.eye(4), atol=1e-10)


def test_load_dataset_empty_meta():
    import cv2
    from easyhec_calib import load_dataset

    with tempfile.TemporaryDirectory() as d:
        meta = {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "width": 8, "height": 8, "joint_names": ["a", "b"]}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f)
        img = np.zeros((8, 8, 3), np.uint8)
        cv2.imwrite(os.path.join(d, "frame_000.png"), img)
        np.save(os.path.join(d, "frame_000.qpos.npy"), np.zeros(2))
        loaded_meta, frames = load_dataset(d)
        assert loaded_meta["width"] == 8
        assert len(frames) == 1
        assert frames[0]["qpos"].shape == (2,)
        assert frames[0]["mask"] is None


if __name__ == "__main__":
    test_imports()
    test_invSE3()
    test_load_dataset_empty_meta()
    print("All smoke tests passed.")
