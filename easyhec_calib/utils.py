"""SE(3) and rotation helpers."""

import numpy as np


def invSE3(T: np.ndarray) -> np.ndarray:
    """Analytical inverse of a 4x4 SE(3) matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rot_angle_deg(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def pose_errors(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Translation error in mm, rotation error in degrees."""
    dt = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000.0)
    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    return dt, rot_angle_deg(dR)
