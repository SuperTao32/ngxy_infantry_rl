"""每种地形独立的奖励权重、当步值及回合累计。"""

import math

import torch

from ...reward_buffer import RewardBuffer


class TerrainCrossRewards(RewardBuffer):
    def __init__(self, env, buffers, scales, *, wheel_wall_config=None):
        self.env, self.buffers = env, buffers
        self.configure_wheel_wall_force(wheel_wall_config or {"normal_x_min": 0.5, "surface_tolerance_m": 0.01})
        self.square_wave_walls = env.commands.new_tensor([
            [geometry.start_x, geometry.height] if name == "square_wave" else [0., 0.]
            for name, geometry in zip(env.terrain.tile_types, env.terrain.geometries)
        ])
        super().__init__(
            env.num_envs, env.device, env.dt,
            lambda name: getattr(self, "_reward_" + name, None) or getattr(env, "_reward_" + name),
            scales, event_rewards={"progress", "success", "failure", "stairs_up_leg_retraction"},
        )

    def configure_wheel_wall_force(self, config):
        normal_x = float(config["normal_x_min"])
        tolerance = float(config["surface_tolerance_m"])
        if not (math.isfinite(normal_x) and 0 < normal_x <= 1 and math.isfinite(tolerance) and tolerance > 0):
            raise ValueError("wheel_wall_force requires 0 < normal_x_min <= 1 and positive surface_tolerance_m")
        self.wall_normal_x_min, self.wall_surface_tolerance = normal_x, tolerance

    def _reward_square_wave_wheel_wall_force(self):
        """逐点累计轮子撞入口立面的法向力；失败当拍仍计入，单位 N。"""
        env = self.env
        contacts = env.robot.get_contacts(exclude_self_contact=True)
        links = torch.as_tensor(env.wheel_links_idx, device=env.device) + env.robot.link_start
        is_a = (contacts["link_a"][..., None] == links).any(dim=-1)
        is_b = (contacts["link_b"][..., None] == links).any(dim=-1)
        force = torch.where(is_a[..., None], contacts["force_a"], contacts["force_b"])
        normal = contacts["normal"]
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        position = contacts["position"]
        local_x = position[..., 0] - env.terrain_spawn_centers[:, :1]
        local_y = position[..., 1] - env.terrain_spawn_centers[:, 1:2]
        walls = self.square_wave_walls[env.terrain_tile_index]
        tolerance = self.wall_surface_tolerance
        on_wall = (
            self.buffers.assigned[:, None] & contacts["valid_mask"] & (is_a | is_b)
            & (normal[..., 0].abs() >= self.wall_normal_x_min)
            & ((local_x - walls[:, :1]).abs() <= tolerance)
            & (local_y.abs() <= env.terrain.tile_size[1] / 2)
            & (position[..., 2] >= -tolerance)
            & (position[..., 2] <= walls[:, 1:2] + tolerance)
        )
        magnitude = (force * normal).sum(dim=-1).abs()
        return torch.where(on_wall, magnitude, 0.0).sum(dim=1)

    def compute(self, mask, *, exclude=()):
        # 通关后仅落地窗口继续计分，其他越障奖励仍使用原任务 mask。
        overrides = {"stairs_down_landing_stability": self.buffers.landing.gate} if self.buffers.landing is not None else None
        return super().compute(mask, exclude=exclude, mask_overrides=overrides)

    def _reward_stairs_down_landing_stability(self):
        """落地窗口内的竖直与 roll/pitch 振荡，配负权重并按 dt 积分。"""
        vertical_velocity = (self.env.base_lin_vel * self.env.projected_gravity).sum(dim=1)
        oscillation = vertical_velocity.square() + self.buffers.landing.angular_weight * self.env.base_ang_vel[:, :2].square().sum(dim=1)
        return self.buffers.landing.gate * oscillation

    def _reward_stairs_up_leg_retraction(self):
        """缩腿新纪录的增量，单位 m；事件奖励不乘 dt，伸缩或退回台阶不重新计奖。"""
        return self.buffers.retraction.delta

    def _reward_progress(self):
        return self.buffers.progress_delta

    def _reward_success(self):
        return self.buffers.success_event

    def _reward_failure(self):
        return self.buffers.failure_event

    def _reward_task_lateral(self):
        delta = self.env.base_pos[:, :2] - self.buffers.path_origin_xy
        forward = self.buffers.path_forward_xy
        return (delta[:, 1] * forward[:, 0] - delta[:, 0] * forward[:, 1]).square()

    def _reward_task_heading(self):
        yaw = torch.deg2rad(self.env.base_euler[:, 2])
        forward = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        return 1.0 - (forward * self.buffers.path_forward_xy).sum(dim=-1).clamp(-1.0, 1.0)
