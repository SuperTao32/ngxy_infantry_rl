import math

import torch


class RecoveryRewards:
    def _near_upright(self):
        """倾角不超过 30° 时为 1，达到 80° 时为 0，按姿态余弦过渡。"""
        cosine = -self.projected_gravity[:, 2]
        low, high = math.cos(math.radians(80.0)), math.cos(math.radians(30.0))
        return ((cosine - low) / (high - low)).clamp(0.0, 1.0)

    def _reward_base_height(self):
        """直立时跟踪高度，倾倒时只惩罚低于参考高度的部分。"""
        active = self._near_upright()
        reference = self.reward_cfg.get("stand_up_height_reference_m", self.env_cfg["stand_up"]["height"])
        target_error = (self.base_height - self.commands[:, 2]).square()
        low_height_error = (reference - self.base_height).clamp_min(0.0) / 15.0
        return active * target_error + (1.0 - active) * low_height_error

    def _reward_death(self):
        # reset_buf 已包含本拍超时；extras["time_outs"] 在计算奖励后才更新。
        # 成功免罚，成功与物理失败同拍时仍按失败处理；并发原因只扣一次。
        failed = self.terminated_buf | (self.reset_buf & ~self._task_success())
        return failed.to(self.reward_buf.dtype)

    def _reward_wheel_airborne(self):
        penalty = super()._reward_wheel_airborne()
        sampler = self.__dict__.get("reset_sampler")
        if sampler is not None:
            # 动态释放到首次落地期间不惩罚必然的离地；策略始终保持控制。
            penalty = penalty * (~sampler.is_dynamic | sampler.landed)
        return penalty

    def _reward_stand_up_success(self):
        return self.stand_up.just_completed.to(self.reward_buf.dtype)

    def _reward_stand_up_speed(self):
        """成功时按剩余回合时间发放一次奖励，越早站稳越高。"""
        elapsed_s = self.stand_up.elapsed_steps.to(self.reward_buf.dtype) * self.dt
        remaining = (1.0 - elapsed_s / self.env_cfg["episode_length_s"]).clamp(0.0, 1.0)
        succeeded = self.stand_up.just_completed & ~self.terminated_buf
        return succeeded.to(self.reward_buf.dtype) * remaining

    def _reward_stand_up_leg_angle(self):
        """双腿在向下方向的容差内满奖励，超出部分按均方误差衰减。"""
        tolerance = math.radians(self.reward_cfg.get("stand_up_leg_angle_tolerance_deg", 10.0))
        excess = (self.leg_angle.abs() - tolerance).clamp_min(0.0)
        error = excess.square().mean(dim=-1)
        return torch.exp(-error / self.reward_cfg["stand_up_leg_angle_sigma"])
