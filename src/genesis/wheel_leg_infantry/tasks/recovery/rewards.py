import torch


class RecoveryRewards:
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
        """奖励双腿沿机身 -Z；leg_angle 为弧度，0 表示相对机身向下。"""
        error = self.leg_angle.square().mean(dim=-1)
        return torch.exp(-error / self.reward_cfg["stand_up_leg_angle_sigma"])
