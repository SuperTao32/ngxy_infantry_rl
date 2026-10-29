"""下行平台凸台：以出生点为原点，沿 +x 越过凸台、第一平台后降到地面。"""

import math

import torch

from .base import TerrainGeometry


PRESET = "platform_ridge_down"
PARAMETER_KEY = "platform_ridge_down"
PARAMETERS = dict(first_height=0.20, first_length=0.80, ridge_height=0.35,
                  ridge_width=0.15, second_height=0.30, second_length=3.0,
                  approach_length=1.0, base_thickness=0.10)
LEVELS = {
    0: dict(second_height=0.34),
    1: dict(second_height=0.33),
    2: dict(second_height=0.32),
    3: dict(second_height=0.31),
    4: dict(second_height=0.30),
}


class PlatformRidgeDown(TerrainGeometry):
    preset = PRESET
    ascending = False

    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        self.boxes = self._box_bounds(parameters)
        self.travel_direction = 1.0
        # 上行越过凸台后落到第二平台；下行经过第一平台后落到地面。
        self.traversal_end = self.boxes[2 if self.ascending else 1][1][0]

    def _box_bounds(self, parameters):
        """三个连续、无重叠的固定箱体，避免高度场把 15 cm 窄凸台插值成斜坡。"""
        values = {}
        for name in ("first_height", "first_length", "ridge_height", "ridge_width", "second_height", "base_thickness"):
            value = float(parameters[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{self.preset}.{name} must be positive and finite")
            values[name] = value
        approach = float(parameters["approach_length"])
        if not math.isfinite(approach) or approach < 0:
            raise ValueError(f"{self.preset}.approach_length must be nonnegative and finite")
        if values["ridge_height"] <= max(values["first_height"], values["second_height"]):
            raise ValueError(f"{self.preset}.ridge_height must exceed both platform heights")
        min_x, min_y, base_z = self.origin
        max_x, max_y = min_x + self.tile_size[0], min_y + self.tile_size[1]
        second_length = parameters["second_length"]
        if second_length is not None:
            second_length = float(second_length)
            if not math.isfinite(second_length) or second_length <= 0:
                raise ValueError(f"{self.preset}.second_length must be positive and finite or None")
        margin = self.boundary_margin or 0.0
        if self.ascending:
            first_start = approach
            first_end = ridge_start = first_start + values["first_length"]
            ridge_end = second_start = ridge_start + values["ridge_width"]
            second_end = max_x if second_length is None else second_start + second_length
            if second_start >= max_x - margin:
                raise ValueError(f"{self.preset} must leave a second platform inside tile_size and boundary_margin")
        else:
            # approach_length 是出生点到凸台的前向距离；第二平台向出生点后方延伸。
            second_end = ridge_start = approach
            second_start = min_x if second_length is None else second_end - second_length
            ridge_end = first_start = ridge_start + values["ridge_width"]
            first_end = first_start + values["first_length"]
            if not second_start <= 0 < second_end:
                raise ValueError(f"{self.preset}.second_length must cover the spawn point at x=0")
            if first_end >= max_x - margin:
                raise ValueError(f"{self.preset} must leave ground after the first platform inside tile_size and boundary_margin")
        if second_start < min_x or second_end > max_x:
            raise ValueError(f"{self.preset}.second_length extends beyond terrain.tile_size")
        return (
            ((min_x, min_y, base_z - values["base_thickness"]), (max_x, max_y, base_z)),
            ((first_start, min_y, base_z), (first_end, max_y, base_z + values["first_height"])),
            ((ridge_start, min_y, base_z), (ridge_end, max_y, base_z + values["ridge_height"])),
            ((second_start, min_y, base_z), (second_end, max_y, base_z + values["second_height"])),
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


TERRAIN_CLASS = PlatformRidgeDown
