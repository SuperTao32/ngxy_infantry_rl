"""Locomotion 奖励项。

这里只描述“测量什么”，正负号和相对权重全部留在 ``config.py``。拆成
mixin 后，环境文件可以专注于仿真状态、动作和 reset 数据流。
"""

import torch


class LocomotionRewards:
    def _reward_tracking_lin_vel(self):
        return torch.square(self.base_lin_vel[:, 0] - self.commands[:, 0])

    def _reward_tracking_ang_vel(self):
        return torch.square(self.base_ang_vel[:, 2] - self.commands[:, 1])

    def _reward_gated_tracking_lin_vel(self):
        error = torch.square(self.base_lin_vel[:, 0] - self.commands[:, 0])
        return self.tracking_gate * torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_gated_tracking_ang_vel(self):
        error = torch.square(self.base_ang_vel[:, 2] - self.commands[:, 1])
        return self.tracking_gate * torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_base_balance(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_leg_symmetry(self):
        return torch.square(self.leg_angle[:, 0] - self.leg_angle[:, 1])

    def _reward_leg_symmetry_bonus(self):
        error = torch.square(self.leg_angle[:, 0] - self.leg_angle[:, 1])
        return torch.exp(-error / 0.04)

    def _reward_base_height(self):
        # base_height 是机身世界 z 减去当前位置地面 z；在坡面和起伏路面上
        # 不应把地形自身的海拔变化算成跟踪误差。
        return torch.square(self.base_height - self.commands[:, 2])

    def _reward_height_gate(self):
        # 直接奖励高度门的开启程度，不依赖姿态门，避免高度还没学会时奖励消失。
        return self.height_gate

    def _reward_joint_vel(self):
        return torch.sum(torch.square(self.joint_vel), dim=1)

    def _reward_leg_action_rate(self):
        """惩罚相邻控制拍的腿部动作变化，抑制逐拍反向振荡。"""
        delta = self.actions[:, :self.num_joints] - self.last_actions[:, :self.num_joints]
        return torch.sum(delta.square(), dim=1)

    def _reward_wheel_action_rate(self):
        wheel = slice(self.num_joints, self.num_joints + self.num_wheels)
        delta = self.actions[:, wheel] - self.last_actions[:, wheel]
        return torch.sum(delta.square(), dim=1)

    def _reward_landing_base_oscillation(self):
        # 两个向量都在机体系；点积对应沿世界重力方向的速度。
        vertical_velocity = torch.sum(self.base_lin_vel * self.projected_gravity, dim=1)
        roll_pitch_ang_vel = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        oscillation = (
            torch.square(vertical_velocity)
            + self.reward_cfg["landing_base_ang_vel_weight"] * roll_pitch_ang_vel
        )
        return self.landing_penalty_gate * oscillation

    def _reward_landing_joint_vel(self):
        return self.landing_penalty_gate * torch.sum(torch.square(self.joint_vel), dim=1)

    def _reward_base_contact(self):
        return self.base_contact

    def _reward_alive(self):
        return self.alive_gate

    def _reward_death(self):
        return self.terminated_buf.to(dtype=self.reward_buf.dtype)
