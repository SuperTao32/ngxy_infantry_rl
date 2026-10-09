"""可复现的二维随机起伏，用分块非凸网格保留凹谷并支持课程切换。"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .base import TerrainGeometry


PRESET = "random_rough"
PARAMETER_KEY = "random_rough"
PARAMETERS = dict(height=0.05, noise_scale=0.50, spawn_flat_radius=0.50, base_thickness=0.10, seed=0)
LEVELS = {
    0: dict(height=0.01, noise_scale=0.80),
    1: dict(height=0.02, noise_scale=0.70),
    2: dict(height=0.03, noise_scale=0.60),
    3: dict(height=0.04, noise_scale=0.50),
    4: dict(height=0.05, noise_scale=0.50),
}


class RandomRough(TerrainGeometry):
    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        self.horizontal_scale = float(horizontal_scale)
        self.height = float(parameters["height"])
        self.noise_scale = float(parameters["noise_scale"])
        self.spawn_flat_radius = float(parameters["spawn_flat_radius"])
        self.base_thickness = float(parameters["base_thickness"])
        if not math.isfinite(self.height) or self.height < 0.0:
            raise ValueError("random_rough.height must be nonnegative and finite")
        if not math.isfinite(self.noise_scale) or self.noise_scale < self.horizontal_scale:
            raise ValueError("random_rough.noise_scale must be finite and at least horizontal_scale")
        if not math.isfinite(self.spawn_flat_radius) or not 0 <= self.spawn_flat_radius < min(self.tile_size) / 2:
            raise ValueError("random_rough.spawn_flat_radius must be nonnegative and smaller than half a tile")
        if not math.isfinite(self.base_thickness) or self.base_thickness <= 0.0:
            raise ValueError("random_rough.base_thickness must be positive and finite")
        self.seed = parameters["seed"]
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("random_rough.seed must be an integer in [0, 2**32)")
        self.heights = self._generate_heights()
        self._height_tensors = {}

    def _generate_heights(self):
        # 局部 RNG 不消耗策略、reset 或其他课程阶段的随机序列。
        rng = np.random.default_rng(self.seed)
        coarse_shape = tuple(max(2, math.ceil(size / self.noise_scale) + 1) for size in self.tile_size)
        shape = tuple(round(size / self.horizontal_scale) + 1 for size in self.tile_size)
        coarse = torch.from_numpy(rng.uniform(-1., 1., coarse_shape))[None, None]
        heights = F.interpolate(coarse, size=shape, mode="bicubic", align_corners=True)[0, 0].numpy()
        heights = np.clip(heights, -1., 1.) * self.height
        if self.spawn_flat_radius > 0:
            x, y = (np.linspace(-size / 2, size / 2, count) for size, count in zip(self.tile_size, shape))
            radius = np.hypot(x[:, None], y[None, :])
            # 留出一个单元对角线，保证出生圆内的整个三角面都为零。
            t = np.clip((radius - self.spawn_flat_radius - math.sqrt(2) * self.horizontal_scale) / self.noise_scale, 0., 1.)
            heights *= t * t * (3. - 2. * t)
        return heights

    def bind(self, *, device, dtype):
        key = (torch.device(device), dtype)
        self._height_tensors[key] = torch.as_tensor(self.heights, device=device, dtype=dtype)

    def height_at(self, world_xy):
        key = (world_xy.device, world_xy.dtype)
        if key not in self._height_tensors:
            self.bind(device=world_xy.device, dtype=world_xy.dtype)
        heights = self._height_tensors[key]
        half = world_xy.new_tensor(self.tile_size) / 2
        grid = (world_xy + half) / self.horizontal_scale
        max_cell = torch.tensor(heights.shape, device=world_xy.device) - 2
        cell = torch.minimum(torch.floor(grid).long().clamp_min(0), max_cell)
        fraction = (grid - cell).clamp(0., 1.)
        i, j = cell.unbind(-1)
        u, v = fraction.unbind(-1)
        h00, h10 = heights[i, j], heights[i + 1, j]
        h01, h11 = heights[i, j + 1], heights[i + 1, j + 1]
        # 与碰撞网格使用相同的 (i+1,j)—(i,j+1) 对角线。
        lower = h00 + u * (h10 - h00) + v * (h01 - h00)
        upper = h11 + (1 - u) * (h01 - h11) + (1 - v) * (h10 - h11)
        height = torch.where(u + v <= 1., lower, upper)
        return torch.where((world_xy.abs() <= half).all(dim=-1), height, 0.)

    def meshes(self):
        """约 2 m 一块的闭合网格；共享边界顶点高度，避免接缝和大型 SDF。"""
        import trimesh

        cells_per_mesh = max(1, round(2.0 / self.horizontal_scale))
        nx, ny = self.heights.shape
        bottom_z = -self.height - self.base_thickness
        for i0 in range(0, nx - 1, cells_per_mesh):
            for j0 in range(0, ny - 1, cells_per_mesh):
                h = self.heights[i0:min(i0 + cells_per_mesh + 1, nx), j0:min(j0 + cells_per_mesh + 1, ny)]
                rows, cols = h.shape
                xx, yy = np.meshgrid(
                    (i0 + np.arange(rows)) * self.horizontal_scale - self.tile_size[0] / 2,
                    (j0 + np.arange(cols)) * self.horizontal_scale - self.tile_size[1] / 2,
                    indexing="ij",
                )
                top = np.column_stack((xx.ravel(), yy.ravel(), h.ravel()))
                i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
                a = (i * cols + j).ravel()
                faces = np.concatenate((np.column_stack((a, a + cols, a + 1)),
                                        np.column_stack((a + 1, a + cols, a + cols + 1))))
                # 从 +z 看，周界按逆时针遍历。
                ring = np.concatenate((np.arange(rows) * cols,
                                       (rows - 1) * cols + np.arange(1, cols),
                                       np.arange(rows - 2, -1, -1) * cols + cols - 1,
                                       np.arange(cols - 2, 0, -1)))
                bottom = top[ring].copy()
                bottom[:, 2] = bottom_z
                center = np.array([[xx.mean(), yy.mean(), bottom_z]])
                vertices = np.concatenate((top, bottom, center))
                b = np.arange(len(ring)) + len(top)
                next_ring, next_b = np.roll(ring, -1), np.roll(b, -1)
                faces = np.concatenate((faces, np.column_stack((ring, b, next_ring)),
                                        np.column_stack((next_ring, b, next_b)),
                                        np.column_stack((next_b, b, np.full(len(ring), len(vertices) - 1)))))
                yield trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def add_to_scene(self, scene):
        import genesis as gs

        # 原生 Terrain 共用一份求解器高度场；课程有多个随机形状时采用独立网格。
        return scene.add_entity(
            gs.morphs.MeshSet(files=tuple(self.meshes()), fixed=True, convexify=False,
                              decimate=False, watertighten=None, batch_fixed_verts=False),
            material=gs.materials.Rigid(sdf_cell_size=min(.02, self.horizontal_scale / 2),
                                       sdf_min_res=16, sdf_max_res=256),
        )


TERRAIN_CLASS = RandomRough
