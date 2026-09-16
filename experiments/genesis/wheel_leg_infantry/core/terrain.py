"""Genesis 地形配置、生成与高度查询。

地形是任务 MDP 的公共基础设施：任务配置只选择 preset 和难度，环境通过
``TerrainManager`` 创建实体、分配出生 patch，并查询机器人脚下的地面高度。
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch

TERRAIN_PRESETS = (
    "plane",
    "flat",
    "slope",
    "rough",
    "waves",
    "obstacles",
    "stairs",
    "mixed",
)

_PRESET_TYPES = {
    "flat": [["flat_terrain"]],
    "slope": [["sloped_terrain"]],
    "rough": [["random_uniform_terrain"]],
    "waves": [["wave_terrain"]],
    "obstacles": [["discrete_obstacles_terrain"]],
    # 单独的 stairs preset 使用箱体生成真实的水平踏面和垂直立面。
    # mixed 继续使用 Genesis 高度场楼梯，以便和其他 patch 合并为一张地形。
    "stairs": [["box_stairs"]],
    # 从左上到右下大致按难度增加。课程只改变允许出生的 patch，
    # 不在训练中重建静态地形。
    "mixed": [
        ["flat_terrain", "sloped_terrain", "pyramid_sloped_terrain", "random_uniform_terrain"],
        ["wave_terrain", "discrete_obstacles_terrain", "stairs_terrain", "pyramid_stairs_terrain"],
    ],
}

_TYPE_DIFFICULTY = {
    "flat_terrain": 0,
    "sloped_terrain": 1,
    "pyramid_sloped_terrain": 1,
    "random_uniform_terrain": 1,
    "wave_terrain": 2,
    "discrete_obstacles_terrain": 2,
    "stairs_terrain": 3,
    "pyramid_stairs_terrain": 3,
    "box_stairs": 3,
    "stepping_stones_terrain": 3,
    "fractal_terrain": 3,
}


def default_terrain_cfg(preset: str = "plane") -> dict:
    """返回适合当前 0.06 m 轮子的保守地形参数。"""
    return {
        "preset": preset,
        "tile_size": [24.0, 8.0],
        "horizontal_scale": 0.10,
        "vertical_scale": 0.005,
        "randomize": True,
        "assignment": "random",
        "max_difficulty": 3,
        # 机器人接近 patch 接缝时按 timeout 重置，避免把接缝当成任务障碍。
        "boundary_margin": 0.75,
        "subterrain_parameters": {
            "sloped_terrain": {"slope": 0.08},
            "pyramid_sloped_terrain": {"slope": -0.08},
            "random_uniform_terrain": {
                "min_height": -0.025,
                "max_height": 0.025,
                "step": 0.005,
                "downsampled_scale": 0.30,
            },
            "wave_terrain": {"num_waves": 4.0, "amplitude": 0.035},
            "discrete_obstacles_terrain": {
                "max_height": 0.035,
                "min_size": 0.35,
                "max_size": 1.0,
                "num_rects": 30,
            },
            "stairs_terrain": {"step_width": 0.50, "step_height": 0.025},
            "pyramid_stairs_terrain": {"step_width": 0.50, "step_height": -0.025},
            "box_stairs": {
                "step_depth": 0.80,
                "step_height": 0.20,
                "num_steps": 4,
                # 机器人出生在 tile 中心的高平台；第一处下台阶位于其前方 1 m。
                "approach_length": 1.0,
                "base_thickness": 0.10,
            },
        },
    }


def resolve_terrain_cfg(config: dict | None) -> dict:
    """补齐并校验地形配置；旧实验没有该字段时继续使用 Plane。"""
    raw = {} if config is None else deepcopy(config)
    if not isinstance(raw, dict):
        raise TypeError("env_cfg['terrain'] must be a dictionary")

    preset = raw.get("preset", "plane")
    if preset not in TERRAIN_PRESETS:
        raise ValueError(f"unsupported terrain preset {preset!r}; choose from {TERRAIN_PRESETS}")

    resolved = default_terrain_cfg(preset)
    custom_parameters = raw.pop("subterrain_parameters", {})
    resolved.update(raw)
    for terrain_type, parameters in custom_parameters.items():
        resolved["subterrain_parameters"].setdefault(terrain_type, {}).update(parameters)
    tile_size = tuple(float(value) for value in resolved["tile_size"])
    if len(tile_size) != 2 or not all(math.isfinite(value) and value > 0.0 for value in tile_size):
        raise ValueError("terrain.tile_size must contain two positive finite values")
    resolved["tile_size"] = list(tile_size)

    for name in ("horizontal_scale", "vertical_scale"):
        value = float(resolved[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"terrain.{name} must be positive and finite")
        resolved[name] = value
    for size in tile_size:
        cells = size / resolved["horizontal_scale"]
        if not math.isclose(cells, round(cells), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("each terrain.tile_size value must be divisible by horizontal_scale")

    assignment = resolved["assignment"]
    if assignment not in {"random", "cyclic"}:
        raise ValueError("terrain.assignment must be 'random' or 'cyclic'")
    resolved["max_difficulty"] = int(resolved["max_difficulty"])
    if resolved["max_difficulty"] < 0:
        raise ValueError("terrain.max_difficulty cannot be negative")

    boundary_margin = resolved.get("boundary_margin")
    if boundary_margin is not None:
        boundary_margin = float(boundary_margin)
        if boundary_margin < 0.0 or 2.0 * boundary_margin >= min(tile_size):
            raise ValueError("terrain.boundary_margin must be non-negative and smaller than half a tile")
    resolved["boundary_margin"] = boundary_margin
    return resolved


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
        self.tile_types = [["flat_terrain"]] if self.is_plane else deepcopy(_PRESET_TYPES[self.preset])
        self.n_subterrains = (len(self.tile_types), len(self.tile_types[0]))
        self.tile_size = tuple(self.config["tile_size"])
        self.total_size = (
            self.n_subterrains[0] * self.tile_size[0],
            self.n_subterrains[1] * self.tile_size[1],
        )
        self.origin = (-0.5 * self.total_size[0], -0.5 * self.total_size[1], 0.0)
        self.max_difficulty = self.config["max_difficulty"]
        self.height_field: torch.Tensor | None = None
        self._dtype = None

        self.flat_tile_types = tuple(value for row in self.tile_types for value in row)
        self.tile_difficulties = tuple(_TYPE_DIFFICULTY[value] for value in self.flat_tile_types)

        self.stair_step_depth = None
        self.stair_step_height = None
        self.stair_num_steps = None
        self.stair_start_x = None
        self.stair_base_thickness = None
        if self.is_box_stairs:
            self._configure_box_stairs()

    def _configure_box_stairs(self) -> None:
        parameters = self.config["subterrain_parameters"]["box_stairs"]
        self.stair_step_depth = float(parameters["step_depth"])
        self.stair_step_height = float(parameters["step_height"])
        self.stair_num_steps = int(parameters["num_steps"])
        approach_length = float(parameters["approach_length"])
        self.stair_base_thickness = float(parameters["base_thickness"])
        values = (self.stair_step_depth, self.stair_step_height, approach_length, self.stair_base_thickness)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("box_stairs dimensions must be positive and finite")
        if self.stair_num_steps <= 0:
            raise ValueError("box_stairs.num_steps must be positive")

        tile_center_x = self.origin[0] + 0.5 * self.tile_size[0]
        self.stair_start_x = tile_center_x + approach_length
        last_riser_x = self.stair_start_x + (self.stair_num_steps - 1) * self.stair_step_depth
        boundary_margin = self.config["boundary_margin"] or 0.0
        usable_max_x = self.origin[0] + self.tile_size[0] - boundary_margin
        if last_riser_x >= usable_max_x:
            raise ValueError("box_stairs do not fit inside terrain.tile_size and boundary_margin")

    def create_morph(self):
        """创建与配置对应的 Genesis morph。"""
        import genesis as gs

        if self.is_plane:
            return gs.morphs.Plane()
        if self.is_box_stairs:
            raise RuntimeError("box stairs contain multiple morphs; use add_to_scene()")
        return gs.morphs.Terrain(
            pos=self.origin,
            randomize=bool(self.config["randomize"]),
            n_subterrains=self.n_subterrains,
            subterrain_size=self.tile_size,
            horizontal_scale=self.config["horizontal_scale"],
            vertical_scale=self.config["vertical_scale"],
            subterrain_types=self.tile_types,
            subterrain_parameters=self.config["subterrain_parameters"],
        )

    def stair_box_bounds(
        self,
    ) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
        """返回真实下楼梯各固定箱体的 ``(lower, upper)``，便于创建和测试。"""
        if not self.is_box_stairs:
            return ()
        min_x, min_y, _ = self.origin
        max_x = min_x + self.tile_size[0]
        max_y = min_y + self.tile_size[1]
        boxes = [((min_x, min_y, -self.stair_base_thickness), (max_x, max_y, 0.0))]

        top_height = self.stair_num_steps * self.stair_step_height
        boxes.append(((min_x, min_y, 0.0), (self.stair_start_x, max_y, top_height)))

        for step_index in range(self.stair_num_steps - 1):
            lower_x = self.stair_start_x + step_index * self.stair_step_depth
            upper_x = lower_x + self.stair_step_depth
            height = (self.stair_num_steps - step_index - 1) * self.stair_step_height
            boxes.append(((lower_x, min_y, 0.0), (upper_x, max_y, height)))
        return tuple(boxes)

    def add_to_scene(self, scene):
        """向场景添加地形；真实楼梯由多个固定箱体组成。"""
        if not self.is_box_stairs:
            return scene.add_entity(self.create_morph())

        import genesis as gs

        return tuple(
            scene.add_entity(gs.morphs.Box(lower=lower, upper=upper, fixed=True, batch_fixed_verts=False))
            for lower, upper in self.stair_box_bounds()
        )

    def bind_entity(self, entity, *, device, dtype) -> None:
        """保留 Genesis 生成的高度场，供 reset 和每步奖励查询。"""
        self._dtype = dtype
        if self.is_plane or self.is_box_stairs:
            self.height_field = None
            return
        self.height_field = torch.as_tensor(entity.terrain_hf, dtype=dtype, device=device)

    def apply_curriculum(self, values: dict) -> None:
        """限制 reset 可选的最高难度；静态地形本身不重新生成。"""
        unknown = set(values).difference({"max_difficulty"})
        if unknown:
            raise KeyError(f"Unsupported terrain curriculum keys: {sorted(unknown)}")
        max_difficulty = int(values.get("max_difficulty", self.max_difficulty))
        if max_difficulty < 0:
            raise ValueError("terrain max_difficulty cannot be negative")
        self.max_difficulty = max_difficulty
        self.config["max_difficulty"] = max_difficulty

    def sample_spawn_tiles(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每个环境的 patch 中心 ``[N,2]`` 与扁平 patch 索引 ``[N]``。"""
        eligible = [
            index for index, difficulty in enumerate(self.tile_difficulties) if difficulty <= self.max_difficulty
        ]
        if not eligible:
            minimum = min(self.tile_difficulties)
            eligible = [index for index, difficulty in enumerate(self.tile_difficulties) if difficulty == minimum]
        eligible_tensor = torch.tensor(eligible, dtype=torch.long, device=env_ids.device)
        if self.config["assignment"] == "cyclic":
            choice = env_ids.to(dtype=torch.long) % len(eligible)
        else:
            choice = torch.randint(len(eligible), (env_ids.numel(),), device=env_ids.device)
        tile_indices = eligible_tensor[choice]

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
        """查询世界坐标处地面 z；Plane 始终返回 0。"""
        if self.is_plane:
            return torch.zeros((world_xy.shape[0],), dtype=world_xy.dtype, device=world_xy.device)
        if self.is_box_stairs:
            normalized_x = (world_xy[:, 0] - self.stair_start_x) / self.stair_step_depth
            step_index = torch.floor(normalized_x + 1e-6)
            height_index = self.stair_num_steps - step_index - 1.0
            height_index = torch.clamp(height_index, min=0.0, max=float(self.stair_num_steps))
            return height_index * self.stair_step_height + self.origin[2]
        if self.height_field is None:
            raise RuntimeError("terrain entity must be bound before querying height")
        origin_xy = torch.tensor(self.origin[:2], dtype=world_xy.dtype, device=world_xy.device)
        return bilinear_height_at(
            self.height_field,
            world_xy,
            origin_xy,
            self.config["horizontal_scale"],
            self.config["vertical_scale"],
            self.origin[2],
        )

    def out_of_bounds(self, world_xy: torch.Tensor, spawn_centers: torch.Tensor) -> torch.Tensor:
        """检查机器人是否接近所属 patch 的接缝。"""
        margin = self.config["boundary_margin"]
        if self.is_plane or margin is None:
            return torch.zeros((world_xy.shape[0],), dtype=torch.bool, device=world_xy.device)
        half_extent = torch.tensor(
            [0.5 * self.tile_size[0] - margin, 0.5 * self.tile_size[1] - margin],
            dtype=world_xy.dtype,
            device=world_xy.device,
        )
        return torch.any(torch.abs(world_xy - spawn_centers) >= half_extent, dim=-1)

    def tile_type(self, tile_index: int) -> str:
        return self.flat_tile_types[int(tile_index)]
