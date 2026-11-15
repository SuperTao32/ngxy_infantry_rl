"""平地只有 0 级。尺寸等公共配置仍在 terrains/config.py。"""

import torch

from .base import TerrainGeometry


PRESET = "plane"
PARAMETER_KEY = None
PARAMETERS = {}
LEVELS = {0: {}}


class Plane(TerrainGeometry):
    def create_morph(self):
        import genesis as gs
        return gs.morphs.Plane()

    def add_to_scene(self, scene):
        return scene.add_entity(self.create_morph())

    def height_at(self, local_xy):
        return torch.zeros_like(local_xy[:, 0])


TERRAIN_CLASS = Plane
