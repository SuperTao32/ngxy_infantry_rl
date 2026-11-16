"""每种地形独立的奖励权重、当步值及回合累计。"""

import torch

from ...reward_buffer import RewardBuffer


class TerrainCrossRewards(RewardBuffer):
    def __init__(self, env, buffers, scales):
        self.env, self.buffers = env, buffers
        super().__init__(
            env.num_envs, env.device, env.dt,
            lambda name: getattr(self, "_reward_" + name, None) or getattr(env, "_reward_" + name),
            scales, event_rewards={"progress", "success", "failure"},
        )

    def _reward_progress(self):
        return self.buffers.progress_delta

    def _reward_success(self):
        return self.buffers.success_event

    def _reward_failure(self):
        return self.buffers.failure_event

    def _reward_task_lateral(self):
        delta = self.env.base_pos[:, :2] - self.buffers.path_origin_xy
        forward = self.buffers.path_forward_xy
        return (delta[:, 1] * forward[:, 0] - delta[:, 0] * forward[:, 1]).square()

    def _reward_task_heading(self):
        yaw = torch.deg2rad(self.env.base_euler[:, 2])
        forward = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        return 1.0 - (forward * self.buffers.path_forward_xy).sum(dim=-1).clamp(-1.0, 1.0)
