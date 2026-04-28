"""Render red semi-transparent URDF overlays on the dataset RGB frames.

Given a URDF, a calibration data directory, and a 4x4 cam-in-world pose, this
re-uses the same nvdiffrast pipeline as `refine` (via easyhec's RBSolver, only
for its renderer/vertex buffers) to rasterize the URDF silhouette at each
frame's joint configuration and alpha-blend it as solid red onto the RGB.
"""

import os
import sys
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from .data import load_dataset
from .urdf_mesh_source import URDFMeshSource
from .utils import invSE3


def _ensure_nvdiffrast_cache_on_path() -> None:
    cache = os.path.expanduser(
        f"~/.cache/torch_extensions/py{sys.version_info.major}{sys.version_info.minor}"
        f"_cu{torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'}"
        "/nvdiffrast_plugin"
    )
    if os.path.isdir(cache) and cache not in sys.path:
        sys.path.insert(0, cache)


def render_urdf_masks(
    *,
    meta: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    urdf_path: str,
    cam_in_world: np.ndarray,
    exclude_link_prefixes: tuple[str, ...] = (),
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Rasterize the URDF silhouette for each frame at the given camera pose.

    Returns:
        (N, H, W) float32 array in [0, 1].
    """
    _ensure_nvdiffrast_cache_on_path()

    import nvdiffrast.torch as dr
    from easyhec.optim.rb_solver import RBSolver, RBSolverConfig
    from easyhec.utils import utils_3d

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    K = np.asarray(meta["K"])
    W, H = int(meta["width"]), int(meta["height"])
    N = len(frames)

    source = URDFMeshSource(urdf_path, exclude_link_prefixes=tuple(exclude_link_prefixes))
    meshes = source.meshes()
    joint_names = meta["joint_names"]
    urdf_joint_idx = [joint_names.index(n) for n in source.actuated_joint_names]
    link_poses_ds = np.stack([
        source.link_world_poses(f["qpos"][urdf_joint_idx]) for f in frames
    ])  # (N, L, 4, 4)

    T_world_cam = invSE3(cam_in_world)  # easyhec uses world->cam

    # Piggyback on RBSolver for its glctx, opencv2blender, and per-link vertex
    # and face buffers. With dof=0 and initial_extrinsic_guess=T_world_cam, the
    # rendered pose is exactly the input cam-in-world.
    cfg = RBSolverConfig(
        camera_width=W, camera_height=H,
        robot_masks=torch.zeros(N, H, W, device=device),
        link_poses_dataset=torch.from_numpy(link_poses_ds).float().to(device),
        meshes=meshes,
        initial_extrinsic_guess=torch.from_numpy(T_world_cam).float().to(device),
    )
    solver = RBSolver(cfg).to(device)

    K_t = torch.from_numpy(K).float().to(device)
    link_poses_t = torch.from_numpy(link_poses_ds).float().to(device)
    proj = utils_3d.K_to_projection(K_t, H, W)
    opencv2blender = solver.renderer.opencv2blender
    glctx = solver.renderer.glctx

    with torch.no_grad():
        Tc_c2b = utils_3d.se3_exp_map(solver.dof[None]).permute(0, 2, 1)[0]  # (4,4)
        mask_sum = torch.zeros(N, H, W, device=device)
        PVM_prefix = proj @ opencv2blender @ Tc_c2b
        for link_idx in range(solver.nlinks):
            verts = getattr(solver, f"vertices_{link_idx}")
            faces = getattr(solver, f"faces_{link_idx}")
            T_link_batch = link_poses_t[:, link_idx]
            PVM = PVM_prefix @ T_link_batch
            verts_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1)
            pos_clip = torch.einsum("nij,vj->nvi", PVM, verts_h).contiguous()
            rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=(H, W))
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=device)
            vtx_color_b = vtx_color[None].expand(N, -1, -1).contiguous()
            color, _ = dr.interpolate(vtx_color_b, rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            link_mask = torch.flip(color[..., 0], dims=[1])
            mask_sum = mask_sum + link_mask
        masks = mask_sum.clamp(max=1)

    return masks.cpu().numpy().astype(np.float32)


def red_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Alpha-blend a solid color over `rgb` using `mask` as the opacity.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        mask: (H, W) float in [0, 1]; 1 = fully covered by the URDF silhouette.
        alpha: peak opacity (at mask=1).
        color: RGB tuple of the overlay color.
    """
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    H, W = mask.shape
    assert rgb.shape[:2] == (H, W)
    a = (alpha * mask).astype(np.float32)[..., None]
    color_layer = np.broadcast_to(np.asarray(color, dtype=np.float32), (H, W, 3))
    out = rgb.astype(np.float32) * (1.0 - a) + color_layer * a
    return np.clip(out, 0, 255).astype(np.uint8)


def render_overlays(
    *,
    data_dir: str,
    urdf_path: str,
    cam_in_world: np.ndarray,
    output_dir: str | None = None,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
    exclude_link_prefixes: tuple[str, ...] = (),
    device: torch.device | str | None = None,
) -> list[str]:
    """Render red URDF overlays for every frame in `data_dir`.

    Writes `<stem>.overlay.png` next to each input frame, or into `output_dir`
    if given. Returns the list of written paths.
    """
    meta, frames = load_dataset(data_dir)
    masks = render_urdf_masks(
        meta=meta,
        frames=frames,
        urdf_path=urdf_path,
        cam_in_world=cam_in_world,
        exclude_link_prefixes=exclude_link_prefixes,
        device=device,
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    for f, m in zip(frames, masks):
        out = red_overlay(f["rgb"], m, alpha=alpha, color=color)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        if output_dir:
            name = os.path.splitext(os.path.basename(f["path"]))[0] + ".overlay.png"
            dst = os.path.join(output_dir, name)
        else:
            dst = f["stem"] + ".overlay.png"
        cv2.imwrite(dst, out_bgr)
        written.append(dst)
    return written
