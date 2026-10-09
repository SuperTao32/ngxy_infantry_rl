"""下楼梯：高度单位 m；LEVELS 中未写的字段继承 PARAMETERS。"""

import math

import torch

from .base import TerrainGeometry


PRESET = "stairs"
PARAMETER_KEY = "box_stairs"
PARAMETERS = dict(step_depth=2.0, step_height=0.20, num_steps=4, approach_length=1.0, base_thickness=0.10)
LEVELS = {
    0: {"step_height": 0.04},
    1: {"step_height": 0.08},
    2: {"step_height": 0.12},
    3: {"step_height": 0.16}, 
    4: {"step_height": 0.20},
}


class Stairs(TerrainGeometry):
    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        self._configure(parameters)
        self.boxes = self._box_bounds()

    def _configure(self, parameters):
        self.step_depth = float(parameters["step_depth"])
        self.step_height = float(parameters["step_height"])
        self.num_steps = int(parameters["num_steps"])
        approach_length = float(parameters["approach_length"])
        self.base_thickness = float(parameters["base_thickness"])
        values = (self.step_depth, self.step_height, approach_length, self.base_thickness)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("box_stairs dimensions must be positive and finite")
        if self.num_steps <= 0:
            raise ValueError("box_stairs.num_steps must be positive")

        tile_center_x = self.origin[0] + 0.5 * self.tile_size[0]
        self.start_x = tile_center_x + approach_length
        last_riser_x = self.start_x + (self.num_steps - 1) * self.step_depth
        boundary_margin = self.boundary_margin or 0.0
        usable_max_x = self.origin[0] + self.tile_size[0] - boundary_margin
        if last_riser_x >= usable_max_x:
            raise ValueError("box_stairs do not fit inside terrain.tile_size and boundary_margin")

    def _box_bounds(
        self,
    ) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
        """返回真实下楼梯各固定箱体的 ``(lower, upper)``，便于创建和测试。"""
        min_x, min_y, _ = self.origin
        max_x = min_x + self.tile_size[0]
        max_y = min_y + self.tile_size[1]
        boxes = [((min_x, min_y, -self.base_thickness), (max_x, max_y, 0.0))]

        top_height = self.num_steps * self.step_height
        boxes.append(((min_x, min_y, 0.0), (self.start_x, max_y, top_height)))

        for step_index in range(self.num_steps - 1):
            lower_x = self.start_x + step_index * self.step_depth
            upper_x = lower_x + self.step_depth
            height = (self.num_steps - step_index - 1) * self.step_height
            boxes.append(((lower_x, min_y, 0.0), (upper_x, max_y, height)))
        return tuple(boxes)

    def add_to_scene(self, scene):
        import genesis as gs
        return tuple(
            scene.add_entity(gs.morphs.Box(lower=lower, upper=upper, fixed=True, batch_fixed_verts=False))
            for lower, upper in self.boxes
        )

    def height_at(self, local_xy):
        normalized_x = (local_xy[:, 0] - self.start_x) / self.step_depth
        step_index = torch.floor(normalized_x + 1e-6)
        height_index = self.num_steps - step_index - 1.0
        height_index = torch.clamp(height_index, min=0.0, max=float(self.num_steps))
        return height_index * self.step_height + self.origin[2]


TERRAIN_CLASS = Stairs
