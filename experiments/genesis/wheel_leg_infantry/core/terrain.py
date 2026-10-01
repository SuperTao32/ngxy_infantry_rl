"""Genesis 地形配置、生成与高度查询。

地形是任务 MDP 的公共基础设施：任务配置直接选择 preset 和几何参数，环境通过
``TerrainManager`` 创建实体、分配出生 patch，并查询机器人脚下的地面高度。
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch

from .loose_spheres import LooseSpheres
from .trapezoidal_wave import TrapezoidalWave

TERRAIN_PRESETS = (
    "plane",
    "stairs",
    "loose_spheres",
    "platform_ridge",
    "trapezoidal_wave",
)

_PRESET_TYPES = {
    "stairs": [["box_stairs"]],
    "platform_ridge": [["platform_ridge"]],
    "trapezoidal_wave": [["trapezoidal_wave"]],
    "loose_spheres": [["loose_spheres"]],
}


def default_terrain_cfg(preset: str = "plane") -> dict:
    """返回适合当前 0.06 m 轮子的保守地形参数。"""
    return {
        "preset": preset,
        "bounded": preset == "flat",  # 兼容旧 flat；plane 默认不启用边界复位
        "tile_size": [24.0, 8.0],
        "horizontal_scale": 0.10,
        "vertical_scale": 0.005,
        "randomize": True,
        "assignment": "random",
        # 机器人接近 patch 接缝时按 timeout 重置，避免把接缝当成任务障碍。
        "boundary_margin": 0.75,
        "subterrain_parameters": {
            "trapezoidal_wave": {
                "height": 0.20,
                "platform_length": 1.50,  # 高、低平台各自的水平长度
                "slope_angle_deg": 23.0,
                "base_thickness": 0.10,
            },
            "box_stairs": {
                "step_depth": 2.00,
                "step_height": 0.20,
                "num_steps": 4,
                # 机器人出生在 tile 中心的高平台；第一处下台阶位于其前方 1 m。
                "approach_length": 1.0,
                "base_thickness": 0.10,
            },
            "platform_ridge": {
                # 高度均相对地面；长度/宽度均沿前进方向 +x。
                "first_height": 0.20,
                "first_length": 0.80,
                "ridge_height": 0.35,
                "ridge_width": 0.15,
                "second_height": 0.30,
                "second_length": 0.80,  # 也可设为 None 延伸到 tile 边界
                "approach_length": 1.0,  # tile 中心到一级平台前沿的距离
                "base_thickness": 0.10,
            },
            "loose_spheres": {
                "diameter": 0.017,  # m，17 mm 为直径
                "count": 256,
                "scatter_size": [4.0, 2.0],  # m，以出生 patch 中心为中心
                "spawn_clearance": 0.5,  # 中央无球正方形的半边长，避免出生穿插
                "mass": 0.0032,  # kg，单球 3.2 g；由质量和直径推导密度
                "shore_a": 90.0,  # 材料信息；当前刚体模型不据此改变接触刚度或摩擦
                "friction": 0.3,
                "ground_friction": 0.8,
            },
        },
    }


def resolve_terrain_cfg(config: dict | None) -> dict:
    """补齐并校验地形配置；旧实验没有该字段时继续使用 Plane。"""
    raw = {} if config is None else deepcopy(config)
    if not isinstance(raw, dict):
        raise TypeError("env_cfg['terrain'] must be a dictionary")

    # 兼容旧实验；难度编号不再参与地形选择。
    raw.pop("max_difficulty", None)
    preset = raw.get("preset", "plane")
    if preset == "flat":
        # 旧 flat 的表面也是 z=0，迁移到 Plane 并保留边界 timeout。
        preset = raw["preset"] = "plane"
        raw.setdefault("bounded", True)
    if preset in {"slope", "mixed"}:
        raise ValueError(f"terrain preset {preset!r} was removed; select a supported terrain with --terrain: {TERRAIN_PRESETS}")
    if preset not in TERRAIN_PRESETS:
        raise ValueError(f"unsupported terrain preset {preset!r}; choose from {TERRAIN_PRESETS}")

    resolved = default_terrain_cfg(preset)
    custom_parameters = raw.pop("subterrain_parameters", {})
    resolved.update(raw)
    for terrain_type, parameters in custom_parameters.items():
        # 旧实验可能携带已移除类型的参数，仅合并当前支持的地形。
        if terrain_type not in resolved["subterrain_parameters"]:
            continue
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

    if not isinstance(resolved["bounded"], bool):
        raise TypeError("terrain.bounded must be a boolean")

    assignment = resolved["assignment"]
    if assignment not in {"random", "cyclic"}:
        raise ValueError("terrain.assignment must be 'random' or 'cyclic'")

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
    """管理静态/动态地形、并行环境出生 patch 和地面高度查询。"""

    def __init__(self, config: dict | None):
        self.config = resolve_terrain_cfg(config)
        self.preset = self.config["preset"]
        self.is_plane = self.preset == "plane"
        self.is_box_stairs = self.preset == "stairs"
        self.is_platform_ridge = self.preset == "platform_ridge"
        self.tile_types = [["flat_terrain"]] if self.is_plane else deepcopy(_PRESET_TYPES[self.preset])
        self.n_subterrains = (len(self.tile_types), len(self.tile_types[0]))
        self.tile_size = tuple(self.config["tile_size"])
        self.total_size = (
            self.n_subterrains[0] * self.tile_size[0],
            self.n_subterrains[1] * self.tile_size[1],
        )
        self.origin = (-0.5 * self.total_size[0], -0.5 * self.total_size[1], 0.0)
        self.height_field: torch.Tensor | None = None
        self._dtype = None
        self.loose_spheres = (
            LooseSpheres(self.config["subterrain_parameters"]["loose_spheres"], self.tile_size) if self.preset == "loose_spheres" else None
        )
        self.max_collision_pairs = 20 if self.loose_spheres is None else 20 + 8 * self.loose_spheres.count

        self.flat_tile_types = tuple(value for row in self.tile_types for value in row)

        self.trapezoidal_wave = (
            TrapezoidalWave(self.config["subterrain_parameters"]["trapezoidal_wave"], self.tile_size) if self.preset == "trapezoidal_wave" else None
        )
        self.platform_boxes = self._configure_platform_ridge() if self.is_platform_ridge else ()

        self.stair_step_depth = None
        self.stair_step_height = None
        self.stair_num_steps = None
        self.stair_start_x = None
        self.stair_base_thickness = None
        if self.is_box_stairs:
            self._configure_box_stairs()

    def _configure_platform_ridge(self):
        """三个连续、无重叠的固定箱体，避免高度场把 15 cm 窄凸台插值成斜坡。"""
        parameters = self.config["subterrain_parameters"]["platform_ridge"]
        values = {}
        for name in ("first_height", "first_length", "ridge_height", "ridge_width", "second_height", "base_thickness"):
            value = float(parameters[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"platform_ridge.{name} must be positive and finite")
            values[name] = value
        approach = float(parameters["approach_length"])
        if not math.isfinite(approach) or approach < 0:
            raise ValueError("platform_ridge.approach_length must be nonnegative and finite")
        if values["ridge_height"] <= max(values["first_height"], values["second_height"]):
            raise ValueError("platform_ridge.ridge_height must exceed both platform heights")
        min_x, min_y, base_z = self.origin
        max_x, max_y = min_x + self.tile_size[0], min_y + self.tile_size[1]
        start_x = min_x + self.tile_size[0] / 2 + approach
        ridge_x = start_x + values["first_length"]
        second_x = ridge_x + values["ridge_width"]
        margin = self.config["boundary_margin"] or 0.0
        if second_x >= max_x - margin:
            raise ValueError("platform_ridge must leave a second platform inside tile_size and boundary_margin")
        second_length = parameters["second_length"]
        end_x = max_x
        if second_length is not None:
            second_length = float(second_length)
            if not math.isfinite(second_length) or second_length <= 0:
                raise ValueError("platform_ridge.second_length must be positive and finite or None")
            end_x = second_x + second_length
            if end_x > max_x:
                raise ValueError("platform_ridge.second_length extends beyond terrain.tile_size")
        return (
            ((min_x, min_y, base_z - values["base_thickness"]), (max_x, max_y, base_z)),
            ((start_x, min_y, base_z), (ridge_x, max_y, base_z + values["first_height"])),
            ((ridge_x, min_y, base_z), (second_x, max_y, base_z + values["ridge_height"])),
            ((second_x, min_y, base_z), (end_x, max_y, base_z + values["second_height"])),
        )

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
        if self.loose_spheres is not None:
            raise RuntimeError("loose spheres contain multiple morphs; use add_to_scene()")
        if self.is_box_stairs or self.is_platform_ridge or self.trapezoidal_wave is not None:
            raise RuntimeError("segmented terrain contains multiple morphs; use add_to_scene()")
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
        """添加高度场、固定楼梯箱体或可滚动的散落球。"""
        if self.trapezoidal_wave is not None:
            return self.trapezoidal_wave.add_to_scene(scene)
        if self.loose_spheres is not None:
            return self.loose_spheres.add_to_scene(scene)
        if not (self.is_box_stairs or self.is_platform_ridge):
            return scene.add_entity(self.create_morph())

        import genesis as gs

        return tuple(
            scene.add_entity(gs.morphs.Box(lower=lower, upper=upper, fixed=True, batch_fixed_verts=False))
            for lower, upper in (self.platform_boxes if self.is_platform_ridge else self.stair_box_bounds())
        )

    def bind_entity(self, entity, *, device, dtype) -> None:
        """保留 Genesis 生成的高度场，供 reset 和每步奖励查询。"""
        self._dtype = dtype
        if self.is_plane or self.is_box_stairs or self.is_platform_ridge or self.loose_spheres is not None or self.trapezoidal_wave is not None:
            self.height_field = None
            return
        self.height_field = torch.as_tensor(entity.terrain_hf, dtype=dtype, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """仅重置选中环境的动态地形；静态地形无需处理。"""
        if self.loose_spheres is not None:
            self.loose_spheres.reset(env_ids, dtype=self._dtype or torch.float32)

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
        """查询世界坐标处地面 z；Plane 始终返回 0。"""
        # 散落球是可移动物体；奖励和出生高度以承托球体的平地为基准。
        if self.is_plane or self.loose_spheres is not None:
            return torch.zeros((world_xy.shape[0],), dtype=world_xy.dtype, device=world_xy.device)
        if self.trapezoidal_wave is not None:
            return self.trapezoidal_wave.height_at(world_xy)
        if self.is_platform_ridge:
            height = torch.zeros_like(world_xy[:, 0])
            for lower, upper in self.platform_boxes[1:]:
                inside = (world_xy[:, 0] >= lower[0]) & (world_xy[:, 0] < upper[0]) & (world_xy[:, 1] >= lower[1]) & (world_xy[:, 1] < upper[1])
                height = torch.where(inside, upper[2], height)
            return height
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
