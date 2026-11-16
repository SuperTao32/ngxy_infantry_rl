"""一种地形的逐环境越障状态；部分 reset 只清理对应行。"""

import math

import torch


class TerrainCrossBuffers:
    def __init__(self, env, config):
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
