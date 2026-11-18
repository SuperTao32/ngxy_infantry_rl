"""手动 eval：保留地形和组件，命令由键盘驱动，只有显式请求才复位。"""

import torch

from ..env import WheelLegInfantryEnv


class InteractiveEnv(WheelLegInfantryEnv):
    manual_control = True

    def __init__(self, num_envs, **kwargs):
        if num_envs != 1 or not kwargs.get("evaluation"):
            raise ValueError("InteractiveEnv requires one evaluation robot")
        super().__init__(num_envs, **kwargs)
        zeros = [[0.0, 0.0] for _ in range(3)]
        self._apply_reset_ranges({
            "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
            "base_init_rpy_offset_range_deg": zeros,
            "base_init_lin_vel_range": zeros,
            "base_init_ang_vel_range": zeros,
        })

    def reset(self, *, recovery=False):
        return super().reset(recovery=recovery)

    def _resample_commands(self, mask, *, allow_jump=True):
        # reset 时从静止开始；周期重采样不能覆盖用户命令或随机触发跳跃。
        if not allow_jump:
            self.commands[mask, :2] = 0.0
            low, high = self.command_cfg["base_height_range"]
            self.commands[mask, 2] = min(max(0.22, low), high)
        self.recovery.apply_commands(self.commands)

    def return_to_drive(self):
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump.reset(mask)
        self.recovery.buffers.active.zero_()

    def _update_termination(self):
        if self.scene.rigid_solver.get_error_envs_mask().any():
            raise RuntimeError("Physics solver error; continuous evaluation stopped")
        self.terminated_buf.zero_()
        self.reset_buf.zero_()
        return torch.zeros_like(self.reset_buf)
