"""恢复任务仅管理起身状态；仿真与动作执行由统一环境负责。"""

from ..base import Task
from .rewards import RecoveryRewards
from .buffers import RecoveryBuffers


class RecoveryTask(RecoveryRewards, Task):
    def _initialize_task_buffers(self):
        self.stand_up = RecoveryBuffers(self.env_cfg["stand_up"], self.num_envs, self.dt, self.device)
        self.stand_up.active &= self.mask

    def _reset_task_buffers(self, mask):
        self.stand_up.reset(mask)

    def _resample_commands(self, mask):
        super()._resample_commands(mask)
        self.stand_up.enforce_commands(self.commands)

    def _update_task_state(self):
        self.stand_up.update(self)
        self.stand_up.enforce_commands(self.commands)

    def _task_success(self):
        return self.stand_up.just_completed

    def episode_metrics(self, mask):
        return {"success": (~self.stand_up.active[mask]).float(),
                "recovery_duration_s": self.stand_up.elapsed_steps[mask] * self.dt}
