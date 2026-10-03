"""Locomotion 奖励项。

这里只描述“测量什么”，正负号和相对权重全部留在各 ``config_*.py`` 配置中。拆成
mixin 后，环境文件可以专注于仿真状态、动作和 reset 数据流。
"""

import torch


class LocomotionRewards:
    def _standing_command_mask(self):
        # 只由命令决定，避免实际漂移后退出静止约束；高度命令不参与判定。
        return (
            (self.commands[:, 0].abs() < self.reward_cfg.get("standing_lin_vel_threshold", 0.02))
            & (self.commands[:, 1].abs() < self.reward_cfg.get("standing_ang_vel_threshold", 0.02))
        )

    def _tracking_sigma(self):
        moving_sigma = self.reward_cfg["tracking_sigma"]
        # 旧 checkpoint 缺少此配置时保持原来的跟踪容差。
        return torch.where(
            self._standing_command_mask(),
            self.reward_cfg.get("standing_tracking_sigma", moving_sigma),
            moving_sigma,
        )

    def _reward_tracking_lin_vel(self):
        return torch.square(self.base_lin_vel[:, 0] - self.commands[:, 0])

    def _reward_tracking_ang_vel(self):
        return torch.square(self.base_ang_vel[:, 2] - self.commands[:, 1])

    def _reward_gated_tracking_lin_vel(self):
        error = torch.square(self.base_lin_vel[:, 0] - self.commands[:, 0])
        return self.tracking_gate * torch.exp(-error / self._tracking_sigma())

    def _reward_gated_tracking_ang_vel(self):
        error = torch.square(self.base_ang_vel[:, 2] - self.commands[:, 1])
        return self.tracking_gate * torch.exp(-error / self._tracking_sigma())

    def _reward_standing_drift(self):
        """静止命令下的水平 L1 漂移；保留平衡死区，不乘姿态/高度 gate。"""
        excess_speed = torch.clamp_min(
            self.base_lin_vel[:, :2].abs() - self.reward_cfg.get("standing_drift_deadband", 0.02), 0.0
        )
        return self._standing_command_mask() * excess_speed.sum(dim=1)

    def _reward_base_balance(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_leg_symmetry(self):
        return torch.square(self.leg_angle[:, 0] - self.leg_angle[:, 1])

    def _reward_leg_angle_limits(self):
        """惩罚实际虚拟腿摆角越界，范围内（含边界）不惩罚，单位为 rad²。"""
        below = torch.clamp_min(self.leg_angle_lower - self.leg_angle, 0.0)
        above = torch.clamp_min(self.leg_angle - self.leg_angle_upper, 0.0)
        return torch.sum(below.square() + above.square(), dim=1)

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

    def _reward_wheel_airborne(self):
        """按未接触地面的轮子数量惩罚离地：双轮接触为 0，单轮/双轮离地为 1/2。"""
        return torch.sum(1.0 - self.wheel_contact, dim=1)

    def _reward_base_contact(self):
        return self.base_contact

    def _reward_alive(self):
        return self.alive_gate

    def _reward_death(self):
        return self.terminated_buf.to(dtype=self.reward_buf.dtype)

    def _reward_stand_up_posture(self):
        """起身姿态误差代价；越接近目标越小，完成后归零，避免拖延起身刷分。"""
        error = (self.base_height - self.stand_up.config["height"]) / self.reward_cfg["stand_up_height_sigma"]
        upright = (-self.projected_gravity[:, 2]).clamp(0.0, 1.0)
        return self.stand_up.was_active * (1.0 - torch.exp(-error.square()) * upright.square())

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
