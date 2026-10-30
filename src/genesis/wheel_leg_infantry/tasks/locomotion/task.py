"""平地速度、转向与站立跟踪。"""

from ..base import Task


class LocomotionTask(Task):
    EPISODE_METRICS = ("height_gate", "attitude_gate", "tracking_gate", "tracking_gate_fully_open", "base_contact")
