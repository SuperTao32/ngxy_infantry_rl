"""奖励结算与回合累计；每个所有者创建自己的实例。"""

import torch


class RewardBuffer:
    def __init__(self, num_envs, device, dt, resolve, scales, *, event_rewards=()):
        self.dt = dt
        self.resolve = resolve
        self.event_rewards = set(event_rewards)
        self.value = torch.zeros(num_envs, device=device)
        self.scales = {}
        self.functions = {}
        self.episode_sums = {}
        self.set_scales(scales)

    def set_scales(self, scales):
        for name, scale in scales.items():
            scale = torch.as_tensor(scale, dtype=self.value.dtype, device=self.value.device)
            self.functions[name] = self.resolve(name)
            self.scales[name] = float(scale) if scale.ndim == 0 else scale
            if name not in self.episode_sums:
                self.episode_sums[name] = torch.zeros_like(self.value)

    def _get_scale(self, name):
        return self.scales[name]

    def compute(self, mask, *, scale_overrides=None, exclude=(), mask_overrides=None):
        """按所有者 mask 结算，允许个别奖励使用独立的生效窗口。"""
        self.value.zero_()
        if not mask.any() and not mask_overrides:
            return self.value
        for name in self.scales:
            if name in exclude:
                continue
            scale = scale_overrides[name] if scale_overrides is not None and name in scale_overrides else self._get_scale(name)
            if not isinstance(scale, torch.Tensor) and scale == 0.0:
                continue
            weight = scale * (1.0 if name in self.event_rewards else self.dt)
            reward_mask = mask_overrides.get(name, mask) if mask_overrides else mask
            value = torch.where(reward_mask, self.functions[name]() * weight, 0.0)
            self.value.add_(value)
            self.episode_sums[name].add_(value)
        return self.value

    def metrics(self, mask, prefix):
        return {f"{prefix}/reward_{name}": total[mask].clone() for name, total in self.episode_sums.items()}

    def reset(self, mask):
        self.value[mask] = 0.0
        for total in self.episode_sums.values():
            total[mask] = 0.0

    def state_dict(self):
        # 物理回合在加载后重新开始，只保存可随课程变化的权重。
        return {"reward_scales": {
            name: scale.tolist() if isinstance(scale, torch.Tensor) else scale
            for name, scale in self.scales.items()
        }}

    def load_state_dict(self, state):
        self.set_scales(state["reward_scales"])
