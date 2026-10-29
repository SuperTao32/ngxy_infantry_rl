"""独立起身任务的连续站稳判定与每回合成功状态。"""

import math

import torch


class StandUpPhase:
    def __init__(self, config, num_envs, dt, device):
        self.config = dict(config)
        for name in (
            "height", "height_tolerance", "max_tilt_deg", "max_leg_angle_deg",
            "max_lin_vel", "max_ang_vel", "hold_time_s",
        ):
            value = float(self.config[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"stand_up.{name} must be finite and positive")
            self.config[name] = value
        self.hold_steps = max(1, math.ceil(self.config["hold_time_s"] / dt))
        self.active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.was_active = self.active.clone()
        self.just_completed = torch.zeros_like(self.active)
        self.stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.elapsed_steps = torch.zeros_like(self.stable_steps)

    def reset(self, mask=None):
        selected = slice(None) if mask is None else mask
        self.active[selected] = True
        self.was_active[selected] = True
        self.just_completed[selected] = False
        self.stable_steps[selected] = 0
        self.elapsed_steps[selected] = 0

    def enforce_commands(self, commands):
        commands[self.active, :2] = 0.0
        commands[self.active, 2] = self.config["height"]

    def update(self, env):
        """成功必须连续满足姿态、接触和速度条件，每回合只触发一次。"""
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
        self.was_active.copy_(self.active)
        self.elapsed_steps.add_(self.active)
        self.stable_steps.copy_(torch.where(self.active & stable, self.stable_steps + 1, 0))
        self.just_completed.copy_(self.active & (self.stable_steps >= self.hold_steps))
        self.active.logical_and_(~self.just_completed)
