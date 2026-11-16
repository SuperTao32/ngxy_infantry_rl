"""平地只有 0 级。尺寸等公共配置仍在 terrains/config.py。"""

import torch

from .base import TerrainGeometry


PRESET = "plane"
PARAMETER_KEY = None
PARAMETERS = {}
LEVELS = {0: {}}


class Plane(TerrainGeometry):
    def __init__(self, parameters, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        self.finite = parameters.get("finite", False)

    def create_morph(self):
        import genesis as gs
        return gs.morphs.Plane()

    def add_to_scene(self, scene, *, offset=(0.0, 0.0, 0.0)):
        if self.finite:
            half_x, half_y = (size / 2 for size in self.tile_size)
            return self.add_box(scene, (-half_x, -half_y, -0.1), (half_x, half_y, 0.0), offset)
        return scene.add_entity(self.create_morph())

    def height_at(self, local_xy):
        return torch.zeros_like(local_xy[:, 0])


TERRAIN_CLASS = Plane
