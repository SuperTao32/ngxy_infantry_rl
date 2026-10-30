import torch


class RecoveryRewards:
    def _reward_wheel_airborne(self):
        penalty = super()._reward_wheel_airborne()
        sampler = self.__dict__.get("reset_sampler")
        if sampler is not None:
            # 动态释放到首次落地期间不惩罚必然的离地；策略始终保持控制。
            penalty = penalty * (~sampler.is_dynamic | sampler.landed)
        return penalty

    def _reward_stand_up_success(self):
        return self.stand_up.just_completed.to(self.reward_buf.dtype)
