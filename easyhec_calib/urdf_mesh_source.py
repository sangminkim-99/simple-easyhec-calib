"""URDF-backed mesh/pose source for easyhec.

Same interface in sim and on real hardware:
- `meshes()` returns the list of visual meshes (already scaled)
- `link_world_poses(qpos)` returns one world-frame pose per mesh, computed by
  forward kinematics from joint angles

On real hardware: feed encoder readings into `link_world_poses`.
In sim: feed `data.qpos` from MuJoCo.
"""

import os
from typing import Dict, List, Union

import numpy as np
import trimesh
import urchin


class URDFMeshSource:
    def __init__(
        self,
        urdf_path: str,
        mesh_dir: str = None,
        exclude_link_prefixes: List[str] = None,
    ):
        """Parse URDF, collect per-link visual meshes.

        Args:
            exclude_link_prefixes: drop any link whose name starts with one of
                these prefixes. Useful for calibration where e.g. the fingers
                add little surface area but a lot of silhouette noise.
        """
        self.urdf = urchin.URDF.load(urdf_path, lazy_load_meshes=False)
        self.mesh_dir = mesh_dir or os.path.dirname(os.path.abspath(urdf_path))
        self.actuated_joint_names = [j.name for j in self.urdf.actuated_joints]
        self.exclude_link_prefixes = tuple(exclude_link_prefixes or ())

        # Enumerate (link, mesh, T_link_mesh) for every <visual><mesh>.
        self._entries = []  # list of (link_name, trimesh.Trimesh, T_link_mesh)
        skipped = []
        for link in self.urdf.links:
            if link.name.startswith(self.exclude_link_prefixes):
                skipped.append(link.name)
                continue
            for v in link.visuals:
                if v.geometry.mesh is None:
                    continue
                fn = v.geometry.mesh.filename
                path = fn if os.path.isabs(fn) else os.path.join(self.mesh_dir, fn)
                scale = v.geometry.mesh.scale
                scale = (1.0, 1.0, 1.0) if scale is None else tuple(scale)

                mesh = trimesh.load(path, force="mesh", process=False)
                mesh.vertices = mesh.vertices * np.asarray(scale).reshape(1, 3)

                T = np.eye(4) if v.origin is None else np.asarray(v.origin)
                self._entries.append((link.name, mesh, T))

        if not self._entries:
            raise RuntimeError(f"No visual meshes found in {urdf_path}")
        if skipped:
            print(f"URDFMeshSource: skipped {len(skipped)} link(s) by prefix: {skipped}")

    def meshes(self) -> List[trimesh.Trimesh]:
        return [e[1] for e in self._entries]

    def link_names(self) -> List[str]:
        return [e[0] for e in self._entries]

    def _normalize_cfg(self, qpos: Union[np.ndarray, Dict[str, float], None]):
        if qpos is None:
            return None
        if isinstance(qpos, dict):
            return qpos
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        names = self.actuated_joint_names
        if qpos.size < len(names):
            raise ValueError(
                f"qpos has {qpos.size} values but URDF has {len(names)} actuated joints "
                f"({names})."
            )
        return dict(zip(names, qpos[: len(names)].tolist()))

    def link_world_poses(
        self,
        qpos: Union[np.ndarray, Dict[str, float], None] = None,
    ) -> np.ndarray:
        """Return (L, 4, 4) world poses for each mesh."""
        cfg = self._normalize_cfg(qpos)
        link_fk = self.urdf.link_fk(cfg=cfg)  # {Link: (4,4)}
        # urchin returns a dict keyed by Link objects; map name -> pose.
        by_name = {lk.name: np.asarray(T) for lk, T in link_fk.items()}

        out = []
        for link_name, _, T_link_mesh in self._entries:
            T_world_link = by_name[link_name]
            out.append(T_world_link @ T_link_mesh)
        return np.stack(out)
