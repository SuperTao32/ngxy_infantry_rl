"""特殊地形的默认姿态复位：只采样障碍距离和 yaw，初速度归零。"""

import math

import torch

from ...sensors.tof import tof_direction
from ...utils.tensor_utils import as_range_tensors, sample_uniform


class TerrainResetSampler:
    def __init__(self, env, config, names, tile_groups):
        self.env = env
        self.names, self.tile_groups = names, tile_groups
        self.tof_cfg = env.tof_cfg
        forward = [sensor for sensor in self.tof_cfg.get("sensors", ())
                   if 0.0 < sensor["downward_angle_deg"] < 90.0]
        self.tof_positions = env.commands.new_tensor([sensor["pos_offset"] for sensor in forward])
        self.tof_directions = env.commands.new_tensor([tof_direction(sensor) for sensor in forward])
        self.ranges = {}
        self.starts = env.commands.new_tensor([
            geometry.start_x if name != "plane" else 0.0
            for name, geometry in zip(env.terrain.tile_types, env.terrain.geometries)
        ])
        for name in names:
            self.configure(name, config[name]["reset"])

    def configure(self, name, config):
        env = self.env
        lower, upper = as_range_tensors(
            [config["distance_range_m"], config["yaw_range_deg"]], 2,
            f"terrain_cross.{name}.reset", device=env.device, dtype=env.commands.dtype,
        )
        height = float(config["base_height_m"])
        margin = env.terrain.config["boundary_margin"] or 0.0
        limit = env.terrain.tile_size[0] / 2 - margin
        starts = [geometry.start_x for kind, geometry in zip(env.terrain.tile_types, env.terrain.geometries) if kind == name]
        near, far = config["distance_range_m"]
        tof_margin = float(config["standing_tof_margin_m"])
        if (near <= 0 or not math.isfinite(height) or height <= 0
                or not math.isfinite(tof_margin) or tof_margin <= 0
                or any(start - far <= -limit or start - near >= limit for start in starts)):
            raise ValueError(f"{name}: reset requires positive distance/height/ToF margin and spawn positions inside the tile")
        self.ranges[name] = lower, upper, height, tof_margin

    def _standing_distances(self, yaw_deg, height, margin):
        """光心在边缘前，假想平地落点在边缘后，使前向射线遇到真实地形变化。"""
        if not self.tof_positions.numel():
            raise ValueError("standing ToF placement requires a forward downward-facing ray")
        yaw = torch.deg2rad(yaw_deg)[:, None]
        origin_x = self.tof_positions[:, 0] * yaw.cos() - self.tof_positions[:, 1] * yaw.sin()
        direction_x = self.tof_directions[:, 0] * yaw.cos()
        flat_distance = (height + self.tof_positions[:, 2]) / -self.tof_directions[:, 2]
        near = (origin_x + margin).amax(dim=1)
        far = (origin_x + direction_x * flat_distance - margin).amin(dim=1)
        if (torch.any(near >= far) or torch.any(flat_distance <= self.tof_cfg["min_range_m"])
                or torch.any(flat_distance >= self.tof_cfg["max_range_m"])):
            raise ValueError("standing ToF placement is infeasible; check reset height/yaw, ToF mounts/range and margin")
        return sample_uniform(near, far, ())

    def apply(self, positions, rpy, lin_vel, ang_vel, tiles, parking):
        groups = self.tile_groups[tiles]
        for index, name in enumerate(self.names):
            selected = groups == index
            count = int(selected.sum())
            if not count:
                continue
            lower, upper, height, tof_margin = self.ranges[name]
            samples = sample_uniform(lower, upper, (count,))
            standing = parking[selected]
            if self.tof_cfg["enabled"] and standing.any():
                samples[standing, 0] = self._standing_distances(samples[standing, 1], height, tof_margin)
            positions[selected, 0] = self.starts[tiles[selected]] - samples[:, 0]
            positions[selected, 1] = 0.0
            positions[selected, 2] = height
            rpy[selected] = 0.0
            rpy[selected, 2] = samples[:, 1]
            lin_vel[selected] = 0.0
            ang_vel[selected] = 0.0
