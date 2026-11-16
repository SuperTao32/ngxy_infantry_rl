"""恢复接口：reset 时组合姿态模板与 env 范围，再接管起身命令。"""

from copy import deepcopy
import math

from .buffers import RecoveryBuffers
from .rewards import RecoveryRewards
from .reset_sampling import RecoveryResetSampler
from ..metrics import TaskMetrics


class Recovery:
    def __init__(self, env, config):
        self.env = env
        self.config = deepcopy(config)
        self.metrics = TaskMetrics(env.device)
        duration = float(self.config["max_duration_s"])
        self.max_steps = math.ceil(duration / env.dt)
        self.buffers = RecoveryBuffers(self.config["stand_up"], env.num_envs, env.dt, env.device)
        self.rewards = RecoveryRewards(env, self.buffers, self.config)
        reset_pos = self.config.get("reset_pos")
        self.reset_sampler = RecoveryResetSampler(env, reset_pos) if reset_pos is not None else None

    def apply_curriculum(self, values):
        """概率只影响后续 reset，当前恢复继续执行。"""
        if self.reset_sampler is not None:
            self.reset_sampler.apply_curriculum(values)
            self.config["reset_pos"]["probability"] = self.reset_sampler.probability

    def sample_reset_pose(self, env, env_ids, positions, *, eligible):
        """只对 eligible 平地环境按概率使用快照，其余保持默认模板。"""
        if self.reset_sampler is not None:
            return self.reset_sampler.sample(env, env_ids, positions, eligible=eligible)
        return env.init_base_quat.expand(len(env_ids), -1), env.init_dof_pos.expand(len(env_ids), -1)

    def ensure_reset_clearance(self, env, env_ids, state):
        if self.reset_sampler is not None:
            self.reset_sampler.ensure_clearance(env, env_ids, state)

    def reset(self, env, mask):
        """只接管本次从姿态池复位的行；默认姿态直接进入 locomotion。"""
        active = self.reset_sampler.pose_ids[mask] >= 0 if self.reset_sampler is not None else False
        self.buffers.reset(mask, active=active)
        self.metrics.start(mask & self.buffers.active)
        self.buffers.enforce_commands(env.commands)
        self.rewards.reset(mask)

    def update(self, env):
        self.buffers.update(env)

    def apply_commands(self, commands):
        self.buffers.enforce_commands(commands)

    def termination(self):
        return self.buffers.active & (self.buffers.elapsed_steps >= self.max_steps)

    def compute_rewards(self):
        finished = self.buffers.was_active & (self.buffers.just_completed | self.env.reset_buf)
        self.metrics.finish(finished, self.buffers.just_completed & ~self.env.terminated_buf)
        return self.rewards.compute(self.buffers.was_active)
