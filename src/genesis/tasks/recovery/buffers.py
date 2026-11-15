"""恢复组件独占的连续站稳、耗时和完成事件 buffer。"""

import math

import torch


class RecoveryBuffers:
    def __init__(self, config, num_envs, dt, device):
        self.config = dict(config)
        self.hold_steps = max(1, math.ceil(self.config["hold_time_s"] / dt))
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.was_active = self.active.clone()
        self.just_completed = torch.zeros_like(self.active)
        self.stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.elapsed_steps = torch.zeros_like(self.stable_steps)

    def reset(self, mask=None, *, active=True):
        selected = slice(None) if mask is None else mask
        self.active[selected] = active
        self.was_active[selected] = active
        self.just_completed[selected] = False
        self.stable_steps[selected] = 0
        self.elapsed_steps[selected] = 0

    def enforce_commands(self, commands, mask=None):
        # 只覆盖恢复中的环境；完成后由 env 重新采样 locomotion 命令。
        selected = self.active if mask is None else mask
        commands[selected, :2] = 0.0
        commands[selected, 2] = self.config["height"]

    def update(self, env):
        """成功必须连续满足姿态、接触和速度条件，每回合只触发一次。"""
        self.was_active.copy_(self.active)
        self.just_completed.zero_()
        if not self.active.any():
            self.stable_steps.zero_()
            return
        cfg = self.config
        stable = (
            ((env.base_height - cfg["height"]).abs() <= cfg["height_tolerance"])
            & (env.base_euler[:, :2].abs().amax(dim=1) <= cfg["max_tilt_deg"])
            & (env.leg_angle.abs().amax(dim=1) <= math.radians(cfg["max_leg_angle_deg"]))
            & (torch.linalg.vector_norm(env.base_lin_vel, dim=1) <= cfg["max_lin_vel"])
            & (torch.linalg.vector_norm(env.base_ang_vel, dim=1) <= cfg["max_ang_vel"])
            & (env.wheel_contact > 0.5).all(dim=1)
            & (env.base_contact < 0.5)
        )
        self.elapsed_steps.add_(self.active)
        self.stable_steps.copy_(torch.where(self.active & stable, self.stable_steps + 1, 0))
        self.just_completed.copy_(self.active & (self.stable_steps >= self.hold_steps))
        self.active.logical_and_(~self.just_completed)
