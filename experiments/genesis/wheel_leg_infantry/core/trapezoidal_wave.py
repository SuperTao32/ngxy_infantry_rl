"""沿 x 周期排列的梯形波：精确凸棱柱坡面与一致的解析高度查询。"""

import math

import torch


class TrapezoidalWave:
    def __init__(self, parameters, tile_size):
        for name in ("height", "platform_length", "base_thickness"):
            value = float(parameters[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"trapezoidal_wave.{name} must be positive and finite")
            setattr(self, name, value)
        angle = float(parameters["slope_angle_deg"])
        if not math.isfinite(angle) or not 0.0 < angle < 90.0:
            raise ValueError("trapezoidal_wave.slope_angle_deg must be between 0 and 90 degrees")
        self.ramp_length = self.height / math.tan(math.radians(angle))
        self.period = 2.0 * (self.platform_length + self.ramp_length)
        self.tile_size = tuple(tile_size)
        # x=0 位于高平台中央，+x 方向依次经过下坡、低平台、上坡。
        self.segments = self._build_segments()

    def _build_segments(self):
        half_x = self.tile_size[0] / 2.0
        length, ramp = self.platform_length, self.ramp_length
        knots = ((0.0, self.height), (length, self.height), (length + ramp, 0.0),
                 (2.0 * length + ramp, 0.0), (self.period, self.height))
        first = math.floor((-half_x + length / 2.0) / self.period)
        last = math.floor((half_x + length / 2.0) / self.period)
        segments = []
        for index in range(first, last + 1):
            offset = index * self.period - length / 2.0
            for (a, ha), (b, hb) in zip(knots, knots[1:]):
                x0, x1 = max(-half_x, offset + a), min(half_x, offset + b)
                if x1 <= x0:
                    continue
                slope = (hb - ha) / (b - a)
                z0 = ha + (x0 - offset - a) * slope
                z1 = ha + (x1 - offset - a) * slope
                segments.append((x0, x1, z0, z1))
        return tuple(segments)

    def height_at(self, world_xy):
        phase = torch.remainder(world_xy[:, 0] + self.platform_length / 2.0, self.period)
        length, ramp = self.platform_length, self.ramp_length
        down = self.height * (1.0 - (phase - length) / ramp)
        up = self.height * (phase - (2.0 * length + ramp)) / ramp
        height = torch.where(phase < length + ramp, down, up).clamp(0.0, self.height)
        inside = (world_xy[:, 0].abs() <= self.tile_size[0] / 2.0) & (world_xy[:, 1].abs() <= self.tile_size[1] / 2.0)
        return torch.where(inside, height, 0.0)

    def meshes(self):
        """每段独立凸棱柱，避免整体凸包填平低谷或高度场网格改变坡角。"""
        import numpy as np
        import trimesh

        half_y = self.tile_size[1] / 2.0
        for x0, x1, z0, z1 in self.segments:
            vertices = np.array([
                (x, y, z)
                for x, top in ((x0, z0), (x1, z1))
                for y in (-half_y, half_y)
                for z in (-self.base_thickness, top)
            ])
            yield trimesh.convex.convex_hull(vertices)

    def add_to_scene(self, scene):
        import genesis as gs

        return tuple(
            scene.add_entity(gs.morphs.MeshSet(files=(mesh,), fixed=True, convexify=True, decimate=False, batch_fixed_verts=False))
            for mesh in self.meshes()
        )
