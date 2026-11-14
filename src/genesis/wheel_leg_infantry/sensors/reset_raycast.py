"""Exact collision queries for sparse resets, without rebuilding every env's BVH.

Imported after Genesis initializes. Regular sensor frames still use Genesis' BVH.
Reset queries use the same mesh triangles and triangle intersection function.
"""

import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.geom import qd_transform_by_trans_quat
from genesis.utils.raycast_qd import ray_triangle_intersection


@qd.kernel
def _cast_collision_subset(
    env_ids: qd.types.ndarray(ndim=1),
    origins: qd.types.ndarray(ndim=3),
    directions: qd.types.ndarray(ndim=3),
    distances: qd.types.ndarray(ndim=2),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    max_range: float,
    eps: float,
):
    for row, ray, face in qd.ndrange(env_ids.shape[0], origins.shape[1], dyn_info.faces.verts_idx.shape[0]):
        # Torch nonzero/arange use int64; Quadrants field indices require int32.
        env = qd.cast(env_ids[row], qd.i32)
        geom = dyn_info.faces.geom_idx[face]
        link = dyn_info.geoms.link_idx[geom]
        link_index = [link, env] if qd.static(rigid_config.batch_links_info) else link
        if dyn_info.links.geom_start[link_index] <= geom and geom < dyn_info.links.geom_end[link_index]:
            vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
            for corner in qd.static(range(3)):
                vertex = dyn_info.faces.verts_idx[face][corner]
                vertices[:, corner] = qd_transform_by_trans_quat(
                    dyn_info.verts.init_pos[vertex], dyn_state.geoms.pos[geom, env], dyn_state.geoms.quat[geom, env],
                )
            start = qd.math.vec3(origins[row, ray, 0], origins[row, ray, 1], origins[row, ray, 2])
            direction = qd.math.vec3(directions[row, ray, 0], directions[row, ray, 1], directions[row, ray, 2])
            hit = ray_triangle_intersection(start, direction, vertices[:, 0], vertices[:, 1], vertices[:, 2], eps)
            if hit.w > 0 and hit.x >= 0 and hit.x < max_range:
                qd.atomic_min(distances[row, ray], hit.x)


class ResetRaycaster:
    def __init__(self, context, max_range):
        self.entries = tuple(context.bvh_contexts)
        self.max_range = max_range
        # Retain Genesis' refresh path for visual/custom vertex buffers.
        self.supported = bool(self.entries) and all(entry.raycast_mask is None for entry in self.entries)
        self.num_envs = self.entries[0].solver._B if self.entries else 0
        self.warmed = False

    def cast(self, env_ids, origins, directions):
        if not self.supported:
            return None
        sparse = len(env_ids) <= min(512, max(1, self.num_envs // 4))
        if not sparse and self.warmed:
            return None
        # Warm the kernel during the initial full reset rather than at the first
        # episode ending. Large batches continue using the regular BVH path.
        ids = env_ids if sparse else env_ids[:1]
        starts = origins if sparse else origins[:1].contiguous()
        dirs = directions if sparse else directions[:1].contiguous()
        output = torch.full(starts.shape[:2], self.max_range, device=starts.device, dtype=starts.dtype)
        if len(ids):
            for entry in self.entries:
                solver = entry.solver
                _cast_collision_subset(
                    ids, starts, dirs, output, solver.dyn_state, solver.dyn_info, solver.rigid_config,
                    self.max_range, gs.EPS,
                )
            self.warmed = True
        return output if sparse else None
