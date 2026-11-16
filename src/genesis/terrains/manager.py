"""地形实体创建、出生位置分配和地面高度查询。"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .config import resolve_terrain_cfg
from .registry import TERRAIN_REGISTRY
from .curriculum import TerrainDifficultyCurriculum


class TerrainManager:
    """管理静态地形、并行环境出生 patch 和地面高度查询。"""

    def __init__(self, config: dict):
        self.config = resolve_terrain_cfg(config)
        self.preset = self.config["preset"]
        self.is_plane = self.preset == "plane"
        self.is_mixture = self.preset == "mixture"
        self.tile_size = tuple(self.config["tile_size"])
        self.max_collision_pairs = 20
        tiles = self.config["tiles"] if self.is_mixture else [{
            "preset": self.preset, "weight": 1.0, "difficulty": self.config.get("difficulty"),
            "parameters": self.config["subterrain_parameters"].get(TERRAIN_REGISTRY[self.preset].PARAMETER_KEY, {}),
        }]
        self.sampling_types = tuple(tile["preset"] for tile in tiles)
        adaptive = self.config["curriculum"]["enabled"]
        level_sets = {tile["preset"]: tuple(sorted(int(level) for level in tile["resolved_levels"]))
                      for tile in tiles if adaptive and tile["preset"] != "plane"}
        self.curriculum = TerrainDifficultyCurriculum(
            self.config["curriculum"], level_sets,
            {tile["preset"]: tile["difficulty"] for tile in tiles if tile["preset"] in level_sets},
        )
        # 采样先按地形权重选类型，再按该类型的目标等级选预建地块。
        variants, self._family_tiles = [], []
        for tile in tiles:
            levels = ([(int(level), parameters) for level, parameters in sorted(tile["resolved_levels"].items(), key=lambda item: int(item[0]))]
                      if adaptive else [(tile.get("difficulty"), tile["parameters"])])
            lookup = {}
            for level, parameters in levels:
                lookup[level] = len(variants)
                variants.append((tile["preset"], level, parameters))
            self._family_tiles.append(lookup)
        self.tile_types = tuple(name for name, _, _ in variants)
        self.tile_difficulties = tuple(level for _, level, _ in variants)
        # 每块沿 y 排列，保留地形自身的 +x 行进方向；边界复位隔开相邻任务。
        self.centers = tuple((0.0, index * self.tile_size[1]) for index in range(len(variants)))
        self.geometries = tuple(
            TERRAIN_REGISTRY[name].TERRAIN_CLASS(
                {**parameters, **({"finite": True} if self.is_mixture and name == "plane" else {})},
                self.tile_size, self.config["horizontal_scale"], boundary_margin=self.config["boundary_margin"],
            )
            for name, _, parameters in variants
        )
        largest_weight = max(tile["weight"] for tile in tiles)
        self._base_weights = tuple(tile["weight"] / largest_weight for tile in tiles)
        self._set_weights(self._base_weights)

    def _set_weights(self, weights):
        largest = max(weights)
        self.weights = tuple(weight / largest for weight in weights)
        if "_weights" in self.__dict__:
            self._weights.copy_(self._weights.new_tensor(self.weights))

    def set_weights(self, values):
        """课程按类型给出权重；未列出的类型为 0，仅更新后续 reset 的采样。"""
        values = {name: float(weight) for name, weight in values.items()}
        valid = all(math.isfinite(weight) and weight >= 0 for weight in values.values())
        if values.keys() - TERRAIN_REGISTRY.keys() or not valid or not any(values.get(name, 0) > 0 for name in self.sampling_types):
            raise ValueError("terrain_weights requires registered presets, finite non-negative weights and a positive weight for an available type")
        totals = {name: sum(weight for kind, weight in zip(self.sampling_types, self._base_weights) if kind == name)
                  for name in self.sampling_types}
        weights = [values.get(name, 0.0) * (base / totals[name] if totals[name] else 1 / self.sampling_types.count(name))
                   for name, base in zip(self.sampling_types, self._base_weights)]
        self._set_weights(weights)

    def sampling_metrics(self):
        total = sum(self.weights)
        return {f"terrain_sampling/{name}/probability": sum(weight for kind, weight in zip(self.sampling_types, self.weights) if kind == name) / total
                for name in dict.fromkeys(self.sampling_types)}

    def add_to_scene(self, scene):
        entities = []
        for geometry, (x, y) in zip(self.geometries, self.centers):
            entity = geometry.add_to_scene(scene, offset=(x, y, 0.0))
            entities.extend(entity if isinstance(entity, (tuple, list)) else (entity,))
        return tuple(entities)

    def bind_entity(self, entity, *, device, dtype):
        self._centers = torch.tensor(self.centers, device=device, dtype=dtype)
        self._weights = torch.tensor(self.weights, device=device, dtype=dtype)
        self._difficulty_counts = torch.zeros((2, len(self.tile_types)), device=device, dtype=torch.long)
        self._refresh_spawn_tiles()
        for geometry in self.geometries:
            geometry.bind(device=device, dtype=dtype)

    def adjust_spawn(self, positions, rpy, tiles):
        for index, geometry in enumerate(self.geometries):
            geometry.adjust_spawn(positions, rpy, mask=tiles == index, offset=self._centers[index])

    def reset(self, env_ids: torch.Tensor) -> None:
        """保留环境统一复位接口；当前静态地形无需处理。"""

    def sample_spawn_tiles(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每个环境的 patch 中心 ``[N,2]`` 与扁平 patch 索引 ``[N]``。"""
        family_indices = (
            torch.multinomial(self._weights, len(env_ids), replacement=True)
            if self.is_mixture and len(env_ids) else torch.zeros_like(env_ids, dtype=torch.long)
        )
        tile_indices = self._spawn_tiles[family_indices]
        return self._centers[tile_indices], tile_indices

    def _refresh_spawn_tiles(self):
        indices = [lookup[self.curriculum.levels[name]] if name in self.curriculum.levels else next(iter(lookup.values()))
                   for name, lookup in zip(self.sampling_types, self._family_tiles)]
        self._spawn_tiles = torch.tensor(indices, device=self._centers.device, dtype=torch.long)

    def record_results(self, tiles, finished, success):
        if self.curriculum.enabled:
            self._difficulty_counts[0].add_(torch.bincount(tiles[finished], minlength=len(self.tile_types)))
            self._difficulty_counts[1].add_(torch.bincount(tiles[finished & success], minlength=len(self.tile_types)))

    def update_curriculum(self):
        """每轮 PPO 调用一次；多 GPU 汇总结果后使用同一组目标等级。"""
        if not self.curriculum.enabled:
            return {}
        counts = self._difficulty_counts.clone()
        self._difficulty_counts.zero_()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
        completed, successes = counts.cpu().tolist()
        results = [(name, level, completed[index], successes[index])
                   for index, (name, level) in enumerate(zip(self.tile_types, self.tile_difficulties)) if name != "plane"]
        if self.curriculum.update(results):
            self._refresh_spawn_tiles()
        return self.curriculum.metrics()

    def state_dict(self):
        return self.curriculum.state_dict()

    def load_state_dict(self, state):
        self.curriculum.load_state_dict(state)
        self._difficulty_counts.zero_()
        self._refresh_spawn_tiles()

    def height_at(self, world_xy: torch.Tensor) -> torch.Tensor:
        if not self.is_mixture:
            return self.geometries[0].height_at(world_xy)
        height = torch.zeros_like(world_xy[:, 0])
        half_x, half_y = (size / 2 for size in self.tile_size)
        # 按坐标分派，兼容机器人、恢复姿态和展开后的 ToF 光心查询。
        for geometry, center in zip(self.geometries, self._centers):
            local_xy = world_xy - center
            inside = ((local_xy[:, 0] >= -half_x) & (local_xy[:, 0] < half_x)
                      & (local_xy[:, 1] >= -half_y) & (local_xy[:, 1] < half_y))
            height = torch.where(inside, geometry.height_at(local_xy), height)
        return height

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
        return self.tile_types[int(tile_index)]
