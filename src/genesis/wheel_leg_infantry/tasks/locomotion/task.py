"""平地速度、转向与站立跟踪。"""

from ..base import Task


class LocomotionTask(Task):
    EPISODE_METRICS = ("height_gate", "attitude_gate", "tracking_gate", "tracking_gate_fully_open", "base_contact")

    def _episode_steps(self):
        return self.episode_length_buf - self.env.task_manager.locomotion_start_steps
