"""一种地形的逐环境越障状态；部分 reset 只清理对应行。"""

import math

import torch


class StairsUpRetraction:
    """入台时固定平均腿长，只累计新的最大缩腿量，整回合最多一个台阶高度。"""

    def __init__(self, env):
        self.steps = env.commands.new_tensor([
            [geometry.start_x, geometry.step_height] if name == "stairs_up" else [0., 0.]
            for name, geometry in zip(env.terrain.tile_types, env.terrain.geometries)
        ])
        self.started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.entry_length = env.commands.new_zeros(env.num_envs)
        self.max_retraction = torch.zeros_like(self.entry_length)
        self.delta = torch.zeros_like(self.entry_length)

    def reset(self, mask):
        for value in (self.started, self.entry_length, self.max_retraction, self.delta):
            value[mask] = 0

    def update(self, env, buffers, in_lane):
        moving = ~env._standing_command_mask()
        valid = buffers.assigned & buffers.was_active & in_lane & ~env.terminated_buf & moving
        steps = self.steps[env.terrain_tile_index]
        local_x = env.base_pos[:, 0] - env.terrain_spawn_centers[:, 0]
        above_step = (local_x >= steps[:, 0]) & (env.base_height > 0.0)
        entered = valid & above_step & ~self.started
        length = env.leg_length.mean(dim=1)
        self.entry_length.copy_(torch.where(entered, length, self.entry_length))
        self.started.logical_or_(entered)
        retraction = torch.minimum((self.entry_length - length).clamp_min(0.0), steps[:, 1])
        self.delta.copy_(torch.where(valid & self.started, (retraction - self.max_retraction).clamp_min(0.0), 0.0))
        self.max_retraction.add_(self.delta)


class StairsDownLanding:
    """首次接触最后一级下方地面后，独立计满落地振荡窗口。"""

    def __init__(self, env, config):
        self.dt = env.dt
        self.configure(config)
        self.started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.steps_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.gate = torch.zeros_like(self.started)

    def configure(self, config):
        duration = float(config["duration_s"])
        angular_weight = float(config["ang_vel_weight"])
        tolerance = float(config["surface_tolerance_m"])
        if (not all(math.isfinite(value) for value in (duration, angular_weight, tolerance))
                or duration <= 0 or angular_weight < 0 or tolerance <= 0):
            raise ValueError("stairs landing_stability requires positive duration/tolerance and nonnegative ang_vel_weight")
        self.window_steps = max(1, math.ceil(duration / self.dt - 1e-9))
        self.angular_weight, self.surface_tolerance = angular_weight, tolerance

    def reset(self, mask):
        self.started[mask] = False
        self.steps_left[mask] = 0
        self.gate[mask] = False

    def update(self, env, buffers, wheels, edge_x):
        wheel_height = env.terrain.height_at(wheels[..., :2].reshape(-1, 2)).reshape(wheels.shape[:2])
        beyond_edge = wheels[..., 0] - env.terrain_spawn_centers[:, :1] > edge_x[:, None]
        on_surface = (wheels[..., 2] - env.wheel_radius - wheel_height).abs() <= self.surface_tolerance
        contact = (beyond_edge & on_surface & (env.wheel_contact > 0.5)).any(dim=1)
        landed = buffers.assigned & buffers.started & (buffers.elapsed_steps > 1) & contact & ~self.started
        self.steps_left.sub_(1).clamp_min_(0)
        self.steps_left.copy_(torch.where(landed, self.window_steps, self.steps_left))
        self.started.logical_or_(landed)
        self.gate.copy_(buffers.assigned & (self.steps_left > 0))


class TerrainCrossBuffers:
    def __init__(self, env, config, *, landing_config=None, stairs_up_retraction=False):
        self.landing = StairsDownLanding(env, landing_config) if landing_config is not None else None
        self.retraction = StairsUpRetraction(env) if stairs_up_retraction else None
        self.hold_steps = max(1, math.ceil(config["success_hold_s"] / env.dt - 1e-9))
        self.min_upright_cos = math.cos(math.radians(config["max_tilt_deg"]))
        bool_names = ("assigned", "started", "active", "was_active", "success", "failure", "started_event")
        float_names = ("start_x", "goal_x", "max_progress", "progress_delta", "success_event", "failure_event")
        for names, dtype in ((bool_names, torch.bool), (float_names, env.commands.dtype),
                             (("elapsed_steps", "stable_steps"), torch.long)):
            for name in names:
                setattr(self, name, torch.zeros(env.num_envs, device=env.device, dtype=dtype))
        self.path_origin_xy = env.commands.new_zeros((env.num_envs, 2))
        self.path_forward_xy = torch.zeros_like(self.path_origin_xy)
        self._episode_buffers = tuple(getattr(self, name) for name in (
            *bool_names, *float_names, "elapsed_steps", "stable_steps", "path_origin_xy", "path_forward_xy",
        ))

    def reset(self, env, mask, assigned, goals):
        if self.landing is not None:
            self.landing.reset(mask)
        if self.retraction is not None:
            self.retraction.reset(mask)
        for value in self._episode_buffers:
            value[mask] = 0
        self.assigned[mask] = assigned[mask]
        self.goal_x[mask] = goals[mask]
        self.path_origin_xy[mask] = env.base_pos[mask, :2]
        yaw = torch.deg2rad(env.base_euler[mask, 2])
        self.path_forward_xy[mask] = torch.stack((yaw.cos(), yaw.sin()), dim=-1)

    def update(self, env, rear_x, in_lane):
        # 特殊地形从默认姿态起步；出生在终点之后的行不算一次有效越障。
        self.started_event.copy_(self.assigned & ~self.started & ~env.reset_buf & (rear_x < self.goal_x))
        self.started.logical_or_(self.started_event)
        self.active.logical_or_(self.started_event)
        self.start_x.copy_(torch.where(self.started_event, rear_x, self.start_x))
        self.was_active.copy_(self.active)
        self.elapsed_steps.add_(self.was_active)

        progress = ((rear_x - self.start_x) / (self.goal_x - self.start_x).clamp_min(1e-6)).clamp(0.0, 1.0)
        valid = self.was_active & in_lane & ~env.terminated_buf
        # 只奖励新的最远进度，往返移动不能重复刷分；物理失败当步不发进度奖励。
        self.progress_delta.copy_(torch.where(valid, (progress - self.max_progress).clamp_min(0.0), 0.0))
        self.max_progress.add_(self.progress_delta)
        supported = ((env.wheel_contact > 0.5).all(dim=1) & (env.base_contact < 0.5)
                     & (-env.projected_gravity[:, 2] >= self.min_upright_cos))
        stable = valid & (rear_x >= self.goal_x) & supported
        self.stable_steps.copy_(torch.where(stable, self.stable_steps + 1, 0))
        succeeded = self.was_active & (self.stable_steps >= self.hold_steps)
        failed = self.active & env.reset_buf & ~succeeded
        self.success_event.copy_(succeeded)
        self.failure_event.copy_(failed)
        self.success.logical_or_(succeeded)
        self.failure.logical_or_(failed)
        # 结算后保持结果，直到下一次物理 reset；后续行动不重复结算这次越障。
        self.active.logical_and_(~(succeeded | failed))
