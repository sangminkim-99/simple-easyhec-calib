"""Core extrinsic refinement.

Differentiable-rendering optimization (silhouette matching) using easyhec's
RBSolver, with all frames rendered per-link in a single rasterize call (batched
across frames instead of the nested frame*link loop in RBSolver.forward).
"""

import os
from typing import Any, Callable

import numpy as np
import torch

from .urdf_mesh_source import URDFMeshSource
from .utils import invSE3, pose_errors


def refine(
    *,
    meta: dict[str, Any],
    frames: list[dict[str, Any]],
    urdf_path: str,
    init_cam_in_world: np.ndarray,
    exclude_link_prefixes: tuple[str, ...] = (),
    iterations: int = 2000,
    lr: float = 3e-3,
    early_stop: int = 200,
    device: torch.device | str | None = None,
    on_step: Callable[[int, float, float, np.ndarray], bool] | None = None,
) -> np.ndarray:
    """Refine camera extrinsic against multi-view robot silhouettes.

    Args:
        meta: dataset meta dict (`K`, `width`, `height`, `joint_names`, optional `gt_cam_in_world`).
        frames: list of dicts with keys `rgb`, `qpos`, `mask` (bool ndarray, required here).
        urdf_path: path to URDF.
        init_cam_in_world: 4x4 cam-in-world initial guess.
        exclude_link_prefixes: skip URDF links whose name starts with these prefixes.
        on_step: optional callback (iter, loss, best_loss, rendered_masks_NHW) -> stop?
            Return True to stop early. Used by the CLI for visualization.

    Returns:
        Refined 4x4 cam-in-world matrix.
    """
    # Torch-extensions cache (nvdiffrast plugin) must be importable before easyhec.
    import sys as _sys
    _nvdr_cache = os.path.expanduser(
        f"~/.cache/torch_extensions/py{_sys.version_info.major}{_sys.version_info.minor}"
        f"_cu{torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'}/nvdiffrast_plugin"
    )
    if os.path.isdir(_nvdr_cache) and _nvdr_cache not in _sys.path:
        _sys.path.insert(0, _nvdr_cache)

    import nvdiffrast.torch as dr
    from easyhec.optim.rb_solver import RBSolver, RBSolverConfig
    from easyhec.utils import utils_3d

    if any(f.get("mask") is None for f in frames):
        raise ValueError("All frames must have a `mask` set before calling refine().")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    K = np.asarray(meta["K"])
    W, H = int(meta["width"]), int(meta["height"])

    robot_masks = np.stack([f["mask"].astype(np.float32) for f in frames])  # (N,H,W)

    source = URDFMeshSource(urdf_path, exclude_link_prefixes=tuple(exclude_link_prefixes))
    meshes = source.meshes()
    joint_names = meta["joint_names"]
    urdf_joint_idx = [joint_names.index(n) for n in source.actuated_joint_names]
    link_poses_per_frame = [
        source.link_world_poses(f["qpos"][urdf_joint_idx]) for f in frames
    ]
    link_poses_ds = np.stack(link_poses_per_frame)  # (N, L, 4, 4)
    print(f"URDF meshes: {len(meshes)}  |  link_poses_dataset: {link_poses_ds.shape}")

    T_cam_world_init = invSE3(init_cam_in_world)  # easyhec uses world->cam

    cfg = RBSolverConfig(
        camera_width=W, camera_height=H,
        robot_masks=torch.from_numpy(robot_masks).float().to(device),
        link_poses_dataset=torch.from_numpy(link_poses_ds).float().to(device),
        meshes=meshes,
        initial_extrinsic_guess=torch.from_numpy(T_cam_world_init).float().to(device),
    )
    solver = RBSolver(cfg).to(device)
    optim_ = torch.optim.Adam(solver.parameters(), lr=lr)
    K_t = torch.from_numpy(K).float().to(device)
    link_poses_t = torch.from_numpy(link_poses_ds).float().to(device)
    mask_t = torch.from_numpy(robot_masks).float().to(device)

    N = link_poses_t.shape[0]
    proj = utils_3d.K_to_projection(K_t, H, W)          # (4,4)
    opencv2blender = solver.renderer.opencv2blender     # (4,4)
    glctx = solver.renderer.glctx

    def _render_masks_batched():
        """Render (N, H, W) mask using current solver.dof, all frames in parallel."""
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
        return mask_sum.clamp(max=1)

    best_loss = float("inf")
    best_dof = solver.dof.detach().clone()
    last_improve = 0
    print("Running optimization.")

    for i in range(iterations):
        rendered_batch = _render_masks_batched()
        loss = torch.sum((rendered_batch - mask_t) ** 2) / N
        optim_.zero_grad()
        loss.backward()
        optim_.step()
        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_dof = solver.dof.detach().clone()
            last_improve = i

        if on_step is not None:
            stop = on_step(i, loss_val, best_loss, rendered_batch.detach().cpu().numpy())
            if stop:
                print(f"on_step requested stop at iter {i}.")
                break

        if i - last_improve >= early_stop:
            print(f"Early stop at iter {i}.")
            break

    solver.dof.data = best_dof
    T_cam_world_refined = solver.get_predicted_extrinsic().cpu().numpy()
    T_cam_in_world_refined = invSE3(T_cam_world_refined)

    print()
    print(f"init  : t = {init_cam_in_world[:3, 3]}")
    print(f"refine: t = {T_cam_in_world_refined[:3, 3]}")
    T_gt = np.asarray(meta["gt_cam_in_world"]) if "gt_cam_in_world" in meta else None
    if T_gt is not None:
        dt_i, dr_i = pose_errors(init_cam_in_world, T_gt)
        dt_r, dr_r = pose_errors(T_cam_in_world_refined, T_gt)
        print(f"gt    : t = {T_gt[:3, 3]}")
        print(f"  init    err: {dt_i:.3f} mm   {dr_i:.4f} deg")
        print(f"  refined err: {dt_r:.3f} mm   {dr_r:.4f} deg")

    return T_cam_in_world_refined
