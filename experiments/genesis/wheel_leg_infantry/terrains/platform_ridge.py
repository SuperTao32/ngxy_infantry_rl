"""平台凸台：保留原 downstairs 课程，从第二平台朝 -x 出发，逐级增加凸台高差。"""

import math

import torch

from .base import TerrainGeometry


PRESET = "platform_ridge"
PARAMETER_KEY = "platform_ridge"
PARAMETERS = dict(first_height=0.20, first_length=0.80, ridge_height=0.35,
                  ridge_width=0.15, second_height=0.30, second_length=None,
                  approach_length=2.0, base_thickness=0.10)
LEVELS = {
    0: dict(second_height=0.34, approach_length=2.0),
    1: dict(second_height=0.33, approach_length=2.0),
    2: dict(second_height=0.32, approach_length=1.8),
    3: dict(second_height=0.31, approach_length=1.6),
    4: dict(second_height=0.30, approach_length=1.4),
}


class PlatformRidge(TerrainGeometry):
    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        self.boxes = self._box_bounds(parameters)

    def _box_bounds(self, parameters):
        """三个连续、无重叠的固定箱体，避免高度场把 15 cm 窄凸台插值成斜坡。"""
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
        margin = self.boundary_margin or 0.0
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

    def add_to_scene(self, scene):
        import genesis as gs
        return tuple(
            scene.add_entity(gs.morphs.Box(lower=lower, upper=upper, fixed=True, batch_fixed_verts=False))
            for lower, upper in self.boxes
        )

    def height_at(self, local_xy):
        height = torch.zeros_like(local_xy[:, 0])
        for lower, upper in self.boxes[1:]:
            inside = ((local_xy[:, 0] >= lower[0]) & (local_xy[:, 0] < upper[0])
                      & (local_xy[:, 1] >= lower[1]) & (local_xy[:, 1] < upper[1]))
            height = torch.where(inside, upper[2], height)
        return height

    def adjust_spawn(self, positions, rpy, *, mask=None, offset=None):
        # 下台阶训练从第二平台中央朝 -x 出发，避免出生在窄凸台上。
        lower, upper = self.boxes[-1]
        if mask is None:
            mask = torch.ones_like(positions[:, 0], dtype=torch.bool)
        if offset is None:
            offset = positions.new_zeros(2)
        positions[:, 0] = torch.where(mask, .5 * (lower[0] + upper[0]) + offset[0], positions[:, 0])
        positions[:, 1] = torch.where(mask, .5 * (lower[1] + upper[1]) + offset[1], positions[:, 1])
        rpy[:, 2] = torch.where(mask, 180.0, rpy[:, 2])


TERRAIN_CLASS = PlatformRidge
