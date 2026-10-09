"""沿 x 方向交替的高低平台，保留真实垂直台阶。"""

import math

import torch

from .base import TerrainGeometry


PRESET = "square_wave"
PARAMETER_KEY = "square_wave"
PARAMETERS = dict(height=0.05, high_length=1.50, low_length=1.50, base_thickness=0.10)
LEVELS = {
    0: {"height": 0.02},
    1: {"height": 0.03},
    2: {"height": 0.04},
    3: {"height": 0.045},
    4: {"height": 0.05},
}


class SquareWave(TerrainGeometry):
    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        for name in ("height", "high_length", "low_length", "base_thickness"):
            value = float(parameters[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"square_wave.{name} must be positive and finite")
            setattr(self, name, value)
        self.period = self.high_length + self.low_length
        # x=0 位于高平台中央，沿 +x 首先下台阶。
        half_x, half_y = (size / 2 for size in self.tile_size)
        boxes = [((-half_x, -half_y, -self.base_thickness), (half_x, half_y, 0.))]
        first = math.floor((-half_x + self.high_length / 2) / self.period)
        last = math.floor((half_x + self.high_length / 2) / self.period)
        for index in range(first, last + 1):
            start = index * self.period - self.high_length / 2
            x0, x1 = max(-half_x, start), min(half_x, start + self.high_length)
            if x1 > x0:
                boxes.append(((x0, -half_y, 0.), (x1, half_y, self.height)))
        self.boxes = tuple(boxes)

    def height_at(self, world_xy):
        phase = torch.remainder(world_xy[:, 0] + self.high_length / 2, self.period)
        high = phase < self.high_length
        inside = (world_xy[:, 0].abs() <= self.tile_size[0] / 2) & (world_xy[:, 1].abs() <= self.tile_size[1] / 2)
        return (inside & high).to(world_xy.dtype) * self.height

    def add_to_scene(self, scene):
        import genesis as gs

        return tuple(
            scene.add_entity(gs.morphs.Box(lower=lower, upper=upper, fixed=True, batch_fixed_verts=False))
            for lower, upper in self.boxes
        )


TERRAIN_CLASS = SquareWave
