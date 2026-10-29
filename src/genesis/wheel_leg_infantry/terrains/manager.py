"""地形实体创建、出生位置分配和地面高度查询。"""

from __future__ import annotations

import torch

from .config import resolve_terrain_cfg
from .registry import TERRAIN_REGISTRY


class TerrainManager:
    """管理静态地形、并行环境出生 patch 和地面高度查询。"""

    def __init__(self, config: dict | None):
        self.config = resolve_terrain_cfg(config)
        self.preset = self.config["preset"]
        self.is_plane = self.preset == "plane"
        self.tile_size = tuple(self.config["tile_size"])
        self._dtype = None
        self.max_collision_pairs = 20

        self.flat_tile_types = (TERRAIN_REGISTRY[self.preset].PARAMETER_KEY or "flat_terrain",)

        module = TERRAIN_REGISTRY[self.preset]
        parameters = self.config["subterrain_parameters"].get(module.PARAMETER_KEY, {})
        self.geometry = module.TERRAIN_CLASS(
            parameters, self.tile_size, self.config["horizontal_scale"],
            boundary_margin=self.config["boundary_margin"],
        )

    def add_to_scene(self, scene):
        return self.geometry.add_to_scene(scene)

    def bind_entity(self, entity, *, device, dtype):
        self._dtype = dtype
        self.geometry.bind(device=device, dtype=dtype)

    def adjust_spawn(self, positions, rpy, tiles=None, *, mask=None, offset=None):
        self.geometry.adjust_spawn(positions, rpy, mask=mask, offset=offset)

    def reset(self, env_ids: torch.Tensor) -> None:
        """保留环境统一复位接口；当前静态地形无需处理。"""

    def sample_spawn_tiles(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每个环境的 patch 中心 ``[N,2]`` 与扁平 patch 索引 ``[N]``。"""
        tile_indices = torch.zeros_like(env_ids, dtype=torch.long)
        centers = torch.zeros((len(env_ids), 2), device=env_ids.device, dtype=self._dtype or torch.float32)
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
