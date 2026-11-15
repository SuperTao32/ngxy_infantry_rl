"""旧版起身奖励适配；仅在恢复阶段及完成当步结算。"""

import math

import torch

from ...reward_buffer import RewardBuffer


class RecoveryRewards(RewardBuffer):
    def __init__(self, env, buffers, config):
        self.env, self.buffers, self.config = env, buffers, config
        super().__init__(
            env.num_envs, env.device, env.dt,
            lambda name: getattr(self, "_reward_" + name, None) or getattr(env, "_reward_" + name),
            config["reward_scales"], event_rewards={"success", "speed", "failure"},
        )

    def _near_upright(self):
        """沿用旧版 30° 满门、80° 关门的姿态过渡。"""
        cosine = -self.env.projected_gravity[:, 2]
        low, high = math.cos(math.radians(80.0)), math.cos(math.radians(30.0))
        return ((cosine - low) / (high - low)).clamp(0.0, 1.0)

    def _reward_height(self):
        env = self.env
        upright = self._near_upright()
        reference = self.config["stand_up_height_reference_m"]
        target_error = (env.base_height - env.commands[:, 2]).square()
        low_height_error = (reference - env.base_height).clamp_min(0.0) / 15.0
        return upright * target_error + (1.0 - upright) * low_height_error

    def _reward_balance(self):
        return self.env.projected_gravity[:, :2].square().sum(dim=1)

    def _reward_leg_angle(self):
        tolerance = math.radians(self.config["leg_angle_tolerance_deg"])
        excess = (self.env.leg_angle.abs() - tolerance).clamp_min(0.0)
        return torch.exp(-excess.square().mean(dim=-1) / self.config["leg_angle_sigma"])

    def _reward_success(self):
        return (self.buffers.just_completed & ~self.env.terminated_buf).float()

    def _reward_speed(self):
        elapsed = self.buffers.elapsed_steps * self.env.dt
        return self._reward_success() * (1.0 - elapsed / self.config["max_duration_s"]).clamp(0.0, 1.0)

    def _reward_failure(self):
        # 起身未完成时的回合截断也按失败结算；站稳完成后由 locomotion 接手。
        return (self.env.terminated_buf | (self.env.reset_buf & self.buffers.active)).float()

    def _reward_wheel_ground_distance(self):
        env = self.env
        wheels = env.robot.get_links_pos(env.wheel_links_idx)
        ground = env.terrain.height_at(wheels[..., :2].reshape(-1, 2)).reshape(wheels.shape[:2])
        clearance = (wheels[..., 2] - env.wheel_radius - ground).clamp_min(0.0)
        return (1.0 - self._near_upright()) * clearance.sum(dim=1)

    def _reward_stand_up_wheel_push(self):
        env = self.env
        upward_force = -(env.wheel_normal_force * env.projected_gravity[:, None, :]).sum(dim=-1)
        support = upward_force.clamp_min(0.0).sum(dim=1)
        reference = self.config["stand_up_wheel_push_force_reference_n"]
        return (1.0 - self._near_upright()) * (support / reference).clamp(0.0, 1.0)
