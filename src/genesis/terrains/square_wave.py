"""单级上台阶：保留 square_wave 入口，沿 +x 从低平地登上高平台。"""

from .base import TerrainGeometry


PRESET = "square_wave"
PARAMETER_KEY = "square_wave"
PARAMETERS = dict(height=0.05, approach_length=0.8, base_thickness=0.10)
LEVELS = {
    0: {"height": 0.020},
    1: {"height": 0.025},
    2: {"height": 0.030},
    3: {"height": 0.035},
    4: {"height": 0.040},
    5: {"height": 0.045},
    6: {"height": 0.050},
}


class SquareWave(TerrainGeometry):
    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        for name in ("height", "approach_length", "base_thickness"):
            value = float(parameters[name])
            setattr(self, name, value)
        self.start_x = self.approach_length
        self.traversal_end = self.start_x
        half_x, half_y = (size / 2 for size in self.tile_size)
        if self.start_x >= half_x - (self.boundary_margin or 0.0):
            raise ValueError("square_wave step does not fit inside terrain.tile_size and boundary_margin")
        # 出生点 x=0 位于低平地，台阶之后的高平台延伸至地块末端。
        self.boxes = (
            ((-half_x, -half_y, -self.base_thickness), (half_x, half_y, 0.)),
            ((self.start_x, -half_y, 0.), (half_x, half_y, self.height)),
        )

    def height_at(self, world_xy):
        high = world_xy[:, 0] >= self.start_x
        inside = (world_xy[:, 0].abs() <= self.tile_size[0] / 2) & (world_xy[:, 1].abs() <= self.tile_size[1] / 2)
        return (inside & high).to(world_xy.dtype) * self.height

    def add_to_scene(self, scene, *, offset=(0.0, 0.0, 0.0)):
        return tuple(
            self.add_box(scene, lower, upper, offset)
            for lower, upper in self.boxes
        )


TERRAIN_CLASS = SquareWave
