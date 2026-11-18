"""命令输入接口：支持重采样触发和评估时的即时跳跃。"""

from copy import deepcopy
import math

import torch

from .buffers import JumpBuffers
from .config import MODE_NAMES
from .height_reference import resolve_height_reference, resolve_retract_duration, sample_height_reference
from .phase import resolve_input_durations, validate_phase_durations
from .rewards import JumpRewards
from ..metrics import TaskMetrics


class Jump:
    def __init__(self, env, config):
        self.env = env
        self.config = deepcopy(config)
        self.metrics = TaskMetrics(env.device, MODE_NAMES)
        self.apply_curriculum({"probability": self.config["probability"]})
        self.phase_durations = validate_phase_durations(self.config["jump_phase_durations_s"])
        inputs = resolve_input_durations(self.config)
        self.input_steps = torch.tensor([math.ceil(value / env.dt) for value in inputs], device=env.device)
        self.duration_steps = torch.maximum(self.input_steps, self.input_steps.new_full((), math.ceil(sum(self.phase_durations) / env.dt)))
        self.mode_probabilities = torch.tensor(self.config["mode_probabilities"], device=env.device)
        self.min_height, self.max_height = resolve_height_reference(self.config)
        self.retract_duration = resolve_retract_duration(self.config)
        self.buffers = JumpBuffers(env, self.config, self.phase_durations)
        self.rewards = JumpRewards(env, self.buffers, self.config, self.phase_durations)

    def apply_curriculum(self, values):
        """概率只影响后续命令重采样，当前跳跃继续执行。"""
        probability = float(values["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("jump.probability must be in [0, 1]")
        self.config["probability"] = probability

    def reset(self, mask):
        self.buffers.reset(mask)
        self.rewards.reset(mask)

    def request(self, mask, mode=0):
        """排队显式跳跃请求，下一次命令重采样消费；active 中的请求留待下一轮。"""
        if type(mode) is not int or not 0 <= mode < len(MODE_NAMES):
            raise ValueError(f"jump mode must be an integer in [0, {len(MODE_NAMES)})")
        self.buffers.pending_mode[mask] = mode

    def trigger(self, mask, mode=0):
        """立即跳跃；手动 eval 可在特殊地形触发，自动模式仍由地形接管。"""
        env = self.env
        terrain_available = ~env.terrain_cross.assigned | getattr(env, "manual_control", False)
        selected = mask & ~env.recovery.buffers.active & ~self.buffers.active & terrain_available
        self.request(selected, mode)
        self.on_resample(env, selected, env.commands[:, 0].clone())
        return selected

    def on_resample(self, env, mask, previous_vx):
        """缓存已采样的新命令，沿用重采样前的 vx 插入 jump。"""
        if self.config["probability"] == 0.0 and not (self.buffers.pending_mode[mask] >= 0).any():
            return
        # mask 由 env 排除 recovery、已激活 jump 及本拍结束的回合。
        buffers = self.buffers
        selected = mask & ((buffers.pending_mode >= 0) | (torch.rand_like(env.base_height) < self.config["probability"]))
        ids = selected.nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return
        sampled = torch.multinomial(self.mode_probabilities, ids.numel(), replacement=True)
        modes = torch.where(buffers.pending_mode[ids] >= 0, buffers.pending_mode[ids], sampled)
        buffers.start(env, selected, modes, previous_vx[ids])
        self.metrics.start(selected, buffers.mode)
        self._update_reference(env.dt)
        env.commands[ids, 0] = buffers.forward_velocity[ids]
        env.commands[ids, 1] = 0.0
        env.commands[ids, 2] = buffers.height_reference[ids]

    def update(self, env):
        self.buffers.update(env, self.duration_steps)
        if self.buffers.was_active.any():
            self._update_reference(env.dt)

    def _update_reference(self, dt):
        self.buffers.height_reference.copy_(sample_height_reference(
            self.buffers.elapsed_steps * dt, takeoff_s=self.phase_durations[0],
            retract_duration_s=self.retract_duration, min_distance_m=self.min_height, max_distance_m=self.max_height,
        ))

    def apply_commands(self, commands):
        buffers = self.buffers
        commands[buffers.just_completed] = buffers.locomotion_commands[buffers.just_completed]
        commands[buffers.active, 0] = buffers.forward_velocity[buffers.active]
        commands[buffers.active, 1] = 0.0
        commands[buffers.active, 2] = buffers.height_reference[buffers.active]

    def compute_rewards(self):
        self.buffers.settle_rewards(self.env)
        buffers = self.buffers
        finished = buffers.was_active & (buffers.just_completed | self.env.reset_buf)
        self.metrics.finish(finished, buffers.success_seen, invalid=buffers.jump_invalid > 0.5, modes=buffers.mode)
        return self.rewards.compute(self.buffers.was_active)

    def observations(self):
        return self.buffers.observations(self.input_steps)
