import torch


class RecoveryRewards:
    def _reward_stand_up_success(self):
        return self.stand_up.just_completed.to(self.reward_buf.dtype)

    def _reward_stand_up_leg_length(self):
        """起身时惩罚超出收腿目标的长度；低于目标不再鼓励缩短。"""
        if self.stand_up is None:
            return torch.zeros_like(self.reward_buf)
        excess = (self.leg_length - self.reward_cfg["stand_up_leg_length_target"]).clamp_min(0.0)
        error = excess / self.reward_cfg["stand_up_leg_length_sigma"]
        return self.stand_up.was_active * error.square().mean(dim=1)

    def _reward_stand_up_leg_vertical(self):
        """起身时惩罚左右虚拟腿相对机身向下方向的摆角（rad）。"""
        if self.stand_up is None:
            return torch.zeros_like(self.reward_buf)
        error = self.leg_angle / self.reward_cfg["stand_up_leg_angle_sigma"]
        return self.stand_up.was_active * error.square().mean(dim=1)
