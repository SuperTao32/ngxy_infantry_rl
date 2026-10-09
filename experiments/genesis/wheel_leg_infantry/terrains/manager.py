"""地形实体创建、出生位置分配和地面高度查询。"""

from __future__ import annotations

import torch

from .config import resolve_terrain_cfg
from .registry import TERRAIN_REGISTRY


def bilinear_height_at(
    height_field: torch.Tensor,
    world_xy: torch.Tensor,
    origin_xy: torch.Tensor,
    horizontal_scale: float,
    vertical_scale: float,
    origin_z: float = 0.0,
) -> torch.Tensor:
    """在 ``[N, 2]`` 世界坐标处双线性插值 Genesis height field。"""
    if world_xy.ndim != 2 or world_xy.shape[-1] != 2:
        raise ValueError("world_xy must have shape [N, 2]")
    if height_field.ndim != 2:
        raise ValueError("height_field must be two-dimensional")

    grid_xy = (world_xy - origin_xy) / horizontal_scale
    max_index = torch.tensor(
        [height_field.shape[0] - 1, height_field.shape[1] - 1],
        dtype=grid_xy.dtype,
        device=grid_xy.device,
    )
    grid_xy = torch.minimum(torch.maximum(grid_xy, torch.zeros_like(grid_xy)), max_index)
    lower = torch.floor(grid_xy).to(dtype=torch.long)
    upper = torch.minimum(lower + 1, max_index.to(dtype=torch.long))
    fraction = grid_xy - lower.to(dtype=grid_xy.dtype)

    h00 = height_field[lower[:, 0], lower[:, 1]]
    h10 = height_field[upper[:, 0], lower[:, 1]]
    h01 = height_field[lower[:, 0], upper[:, 1]]
    h11 = height_field[upper[:, 0], upper[:, 1]]
    hx0 = h00 + (h10 - h00) * fraction[:, 0]
    hx1 = h01 + (h11 - h01) * fraction[:, 0]
    raw_height = hx0 + (hx1 - hx0) * fraction[:, 1]
    return raw_height * vertical_scale + origin_z


class TerrainManager:
    """管理静态地形、并行环境出生 patch 和地面高度查询。"""

    def __init__(self, config: dict | None):
        self.config = resolve_terrain_cfg(config)
        self.preset = self.config["preset"]
        self.is_plane = self.preset == "plane"
        self.is_box_stairs = self.preset == "stairs"
        self.is_platform_ridge = self.preset == "platform_ridge"
        self.tile_types = [[TERRAIN_REGISTRY[self.preset].PARAMETER_KEY or "flat_terrain"]]
        self.n_subterrains = (len(self.tile_types), len(self.tile_types[0]))
        self.tile_size = tuple(self.config["tile_size"])
        self.total_size = (
            self.n_subterrains[0] * self.tile_size[0],
            self.n_subterrains[1] * self.tile_size[1],
        )
        self.origin = (-0.5 * self.total_size[0], -0.5 * self.total_size[1], 0.0)
        self.height_field: torch.Tensor | None = None
        self._dtype = None
        self.max_collision_pairs = 20

        self.flat_tile_types = tuple(value for row in self.tile_types for value in row)

        module = TERRAIN_REGISTRY[self.preset]
        parameters = self.config["subterrain_parameters"].get(module.PARAMETER_KEY, {})
        self.geometry = module.TERRAIN_CLASS(
            parameters, self.tile_size, self.config["horizontal_scale"],
            boundary_margin=self.config["boundary_margin"],
        )

    def create_morph(self):
        return self.geometry.create_morph()

    def add_to_scene(self, scene):
        return self.geometry.add_to_scene(scene)

    def bind_entity(self, entity, *, device, dtype):
        self._dtype = dtype
        self.geometry.bind(device=device, dtype=dtype)

    def adjust_spawn(self, positions, rpy, tiles=None, *, mask=None, offset=None):
        self.geometry.adjust_spawn(positions, rpy, mask=mask, offset=offset)

    # 保留已有诊断/测试读取属性；几何计算只在具体地形模块中执行。
    @property
    def trapezoidal_wave(self):
        return self.geometry if self.preset == "trapezoidal_wave" else None

    @property
    def square_wave(self):
        return self.geometry if self.preset == "square_wave" else None

    @property
    def random_rough(self):
        return self.geometry if self.preset == "random_rough" else None

    @property
    def platform_boxes(self):
        return self.geometry.boxes if self.is_platform_ridge else ()

    def stair_box_bounds(self):
        return self.geometry.boxes if self.is_box_stairs else ()

    @property
    def stair_step_depth(self):
        return getattr(self.geometry, "step_depth", None) if self.is_box_stairs else None

    @property
    def stair_step_height(self):
        return getattr(self.geometry, "step_height", None) if self.is_box_stairs else None

    @property
    def stair_num_steps(self):
        return getattr(self.geometry, "num_steps", None) if self.is_box_stairs else None

    @property
    def stair_start_x(self):
        return getattr(self.geometry, "start_x", None) if self.is_box_stairs else None

    @property
    def stair_base_thickness(self):
        return getattr(self.geometry, "base_thickness", None) if self.is_box_stairs else None

    def reset(self, env_ids: torch.Tensor) -> None:
        """保留环境统一复位接口；当前静态地形无需处理。"""

    def sample_spawn_tiles(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每个环境的 patch 中心 ``[N,2]`` 与扁平 patch 索引 ``[N]``。"""
        # 当前预设各自只有一个 patch，直接使用所选地形，不再按难度筛选。
        tile_indices = torch.zeros_like(env_ids, dtype=torch.long)

        columns = self.n_subterrains[1]
        rows = torch.div(tile_indices, columns, rounding_mode="floor")
        cols = tile_indices % columns
        centers = torch.stack(
            (
                rows.to(dtype=torch.float32) * self.tile_size[0] + 0.5 * self.tile_size[0] + self.origin[0],
                cols.to(dtype=torch.float32) * self.tile_size[1] + 0.5 * self.tile_size[1] + self.origin[1],
            ),
            dim=-1,
        )
        if self._dtype is not None:
            centers = centers.to(dtype=self._dtype)
        return centers, tile_indices

    def height_at(self, world_xy: torch.Tensor) -> torch.Tensor:
        return self.geometry.height_at(world_xy)

    def out_of_bounds(self, world_xy: torch.Tensor, spawn_centers: torch.Tensor) -> torch.Tensor:
        """检查机器人是否接近所属 patch 的接缝。"""
        margin = self.config["boundary_margin"]
        if (self.is_plane and not self.config["bounded"]) or margin is None:
            return torch.zeros((world_xy.shape[0],), dtype=torch.bool, device=world_xy.device)
        half_extent = torch.tensor(
            [0.5 * self.tile_size[0] - margin, 0.5 * self.tile_size[1] - margin],
            dtype=world_xy.dtype,
            device=world_xy.device,
        )
        return torch.any(torch.abs(world_xy - spawn_centers) >= half_extent, dim=-1)

    def tile_type(self, tile_index: int) -> str:
        return self.flat_tile_types[int(tile_index)]


def make_terrain(config):
    """统一创建单一地形或按比例混合的地形集合。"""
    if config and config.get("mixture") is not None:
        from .mixture import MixedTerrain
        return MixedTerrain(config)
    return TerrainManager(config)
