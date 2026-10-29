"""独立上坡"""
import math

from .base import TerrainGeometry

PRESET = "slope_up"
PARAMETER_KEY = "slope_up"
PARAMETERS = dict(height=0.25, slope_angle_deg=23.0, approach_length=1.0, base_thickness=0.10)

LEVELS = {
    0: dict(slope_angle_deg=10.0),
    1: dict(slope_angle_deg=15.0),
    2: dict(slope_angle_deg=20.0),
    3: dict(slope_angle_deg=23.0),
    4: dict(slope_angle_deg=25.0),
}


class SlopeUp(TerrainGeometry):
    descending = False

    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        for name in ("height", "slope_angle_deg", "approach_length", "base_thickness"):
            value = float(parameters[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"slope.{name} must be positive and finite")
            setattr(self, name, value)
        if self.slope_angle_deg >= 60:
            raise ValueError("slope angle must be below 60 degrees")
        self.start_x = self.approach_length
        self.ramp_length = self.height / math.tan(math.radians(self.slope_angle_deg))
        self.traversal_end = self.start_x + self.ramp_length
        if self.traversal_end >= self.tile_size[0] / 2 - (boundary_margin or 0):
            raise ValueError("slope must fit inside tile_size and boundary_margin")
        z0, z1 = (self.height, 0.0) if self.descending else (0.0, self.height)
        self.segments = (
            (-self.tile_size[0] / 2, self.start_x, z0, z0),
            (self.start_x, self.traversal_end, z0, z1),
            (self.traversal_end, self.tile_size[0] / 2, z1, z1),
        )

    def height_at(self, local_xy):
        height = ((local_xy[:, 0] - self.start_x) / self.ramp_length).clamp(0, 1) * self.height
        return self.height - height if self.descending else height

    def add_to_scene(self, scene):
        import genesis as gs
        import numpy as np
        import trimesh

        half_y = self.tile_size[1] / 2
        entities = []
        for x0, x1, z0, z1 in self.segments:
            if z0 == z1:
                morph = gs.morphs.Box(lower=(x0, -half_y, -self.base_thickness), upper=(x1, half_y, z0),
                                      fixed=True, batch_fixed_verts=False)
            else:
                vertices = np.array([(x, y, z) for x, top in ((x0, z0), (x1, z1))
                                     for y in (-half_y, half_y) for z in (-self.base_thickness, top)])
                mesh = trimesh.convex.convex_hull(vertices)
                morph = gs.morphs.MeshSet(files=(mesh,), fixed=True, convexify=True,
                                          decimate=False, batch_fixed_verts=False)
            entities.append(scene.add_entity(morph))
        return tuple(entities)


TERRAIN_CLASS = SlopeUp
