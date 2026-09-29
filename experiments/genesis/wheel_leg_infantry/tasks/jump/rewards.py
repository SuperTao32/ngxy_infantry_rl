"""按起跳、腾空和落地分阶段门控的跳跃奖励。"""

import torch

from .geometry import target_wheel_support


class JumpRewards:
    def _jump_tilt_error(self):
        # 正立 gz=-1，倒立 gz=+1；小角度下约等于倾角平方。
        return 2.0 * torch.clamp(1.0 + self.projected_gravity[:, 2], 0.0, 2.0)

    def _reward_base_balance(self):
        return self._jump_tilt_error()

    def _reward_jump_invalid(self):
        return self.jump_invalid

    def _reward_landing_airborne(self):
        return self.landing_airborne_gate

    def _reward_base_to_wheel_bottom_distance_tracking(self):
        """落地双轮有效支撑时跟踪 base 到轮底的距离参考。"""
        error = torch.clamp(
            torch.abs(self.base_to_wheel_bottom_distance - self.commands[:, 2])
            - self.reward_cfg["height_reference_tolerance_m"],
            min=0.0,
        )
        return self.jump_landing_gate * torch.exp(-torch.square(error) / self.reward_cfg["height_reference_sigma"])

    def _reward_tracking_lin_vel(self):
        """全程只跟踪进入 jump 时锁存的前向线速度，不约束侧向速度。"""
        error = torch.square(self.base_lin_vel[:, 0] - self.commands[:, 0])
        return torch.exp(-error / self.reward_cfg["reference_forward_velocity_sigma"])

    def _reward_tracking_ang_vel(self):
        """全程跟踪零 yaw 角速度。"""
        error = torch.square(self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["zero_yaw_rate_sigma"])

    def _reward_takeoff_event(self):
        """连续腾空达到阈值时才给一次奖励，物理起跳事件仍单独记录。"""
        return self.jump_reward_state.takeoff_event

    def _reward_flight_height_shortfall(self):
        target = self.base_height_target
        shortfall = torch.clamp(1.0 - self.jump_reward_state.peak_clearance / target, 0.0, 1.0)
        # 晚于触地的失稳需补齐全额缺高惩罚，与此前的结算合计为 1。
        correction = torch.clamp(self.jump_reward_state.revoked_settled_clearance / target, 0.0, 1.0)
        return self.jump_reward_state.height_settlement_event * shortfall + correction

    def _revoked_height_progress(self):
        return torch.clamp(
            self.jump_reward_state.revoked_peak_clearance / self.base_height_target, min=0.0
        )

    def _reward_flight_peak_height(self):
        """首次双轮腾空按新增峰值给分，超过目标仍线性增长；失稳全额撤回。"""
        progress = torch.clamp(
            self.jump_reward_state.peak_clearance / self.base_height_target,
            min=0.0,
        )
        previous = torch.clamp(
            self.jump_reward_state.previous_peak_clearance / self.base_height_target, min=0.0
        )
        return self.flight_gate * torch.clamp(progress - previous, min=0.0) - self._revoked_height_progress()

    def _target_takeoff_velocity(self):
        """按最后一个支撑拍距绝对目标高度的差值计算弹道起跳速度。"""
        remaining_height = torch.clamp(self.base_height_target - self.takeoff_contact_base_height, min=0.0)
        return torch.sqrt(2.0 * self.gravity_magnitude * remaining_height)

    def _reward_takeoff_upward_velocity(self):
        """支撑期间向上速度奖励无上界；下落惩罚仍保底为 -1。"""
        # 归一化速度至少 1 m/s，避免剩余高度趋零时放大到百万量级。
        target = self._target_takeoff_velocity().clamp_min(1.0)
        progress = torch.clamp(self.world_vertical_velocity / target, min=-1.0)
        return self.takeoff_gate * progress

    def _reward_takeoff_vertical_velocity(self):
        """离地时按最后一个轮地接触拍的 base-link vz 结算一次。"""
        progress = torch.clamp(
            self.takeoff_contact_vertical_velocity / self._target_takeoff_velocity().clamp_min(1.0), min=0.0
        )
        return self.jump_takeoff_event * (1.0 - self.jump_invalid) * progress

    def _reward_flight_airtime(self):
        """实际腾空每步给分；框架统一乘 dt，累计奖励 = 权重 × 腾空秒数。"""
        return self.flight_gate

    def _reward_flight_height_progress(self):
        target = self.base_height_target
        progress = torch.clamp(self.jump_base_height / target, min=0.0)
        return self.flight_gate * progress

    def _reward_flight_wheel_clearance(self):
        """奖励较低轮底高度；仅收腿在 base 下 18 cm 饱和，整体跳高仍增分。"""
        ceiling = torch.clamp(self.jump_base_height - 0.18, min=0.0)
        clearance = torch.minimum(torch.clamp(self.wheel_clearance, min=0.0), ceiling)
        return self.flight_gate * clearance

    def _reward_flight_balance(self):
        tilt_error = self._jump_tilt_error()
        return self.flight_gate * torch.exp(-tilt_error / self.reward_cfg["flight_balance_sigma"])

    def _reward_flight_leg_vertical(self):
        """在腾空期间，奖励腿部竖直。"""
        error = torch.mean(torch.square(self.leg_angle * self.obs_scales["leg_angle"]), dim=1)
        return self.flight_gate * torch.exp(-error)

    def _reward_flight_height_tracking(self):
        error = torch.square(self.jump_base_height - self.base_height_target)
        return self.flight_gate * torch.exp(-error / self.reward_cfg["flight_height_sigma"])

    def _reward_flight_tuck(self):
        target = self.reward_cfg["short_leg_length_target"]
        error = torch.mean(torch.square(self.leg_length - target), dim=1)
        return self.flight_gate * torch.exp(-error / self.reward_cfg["leg_length_sigma"])

    def _reward_soft_landing(self):
        tilt_error = self._jump_tilt_error()
        impact_quality = torch.exp(
            -torch.square(self.impact_vertical_speed) / self.reward_cfg["landing_velocity_sigma"]
        )
        attitude_quality = torch.exp(-tilt_error / self.reward_cfg["landing_tilt_sigma"])
        support = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.jump_mode, self.task_cfg, self.wheel_radius, margin=0.0,
        ).any(dim=1)
        return self.jump_landing_event * impact_quality * attitude_quality * support * (self.jump_invalid < 0.5)

    def _reward_landing_stability(self):
        tilt_error = self._jump_tilt_error()
        # yaw 可以是 warmup 锁存的非零目标，这里只抑制 roll/pitch 角速度。
        angular_error = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        velocity_error = torch.square(self.world_vertical_velocity)
        quality = torch.exp(
            -velocity_error / self.reward_cfg["landing_velocity_sigma"]
            -tilt_error / self.reward_cfg["landing_tilt_sigma"]
            -angular_error / self.reward_cfg["landing_ang_vel_sigma"]
        )
        return self.jump_landing_gate * quality

    def _reward_leg_extension_at_limit(self):
        """实测腿长到限位后，惩罚实际下发目标中继续伸腿的电机角差（rad²）。"""
        if not self.collect_jump_data:
            return torch.zeros_like(self.leg_length[:, 0])
        actual_separation = (
            self.joint_pos[:, self.leg_front_joint_indices] - self.joint_pos[:, self.leg_rear_joint_indices]
        )
        target_separation = (
            self.target_joint_pos[:, self.leg_front_joint_indices]
            - self.target_joint_pos[:, self.leg_rear_joint_indices]
        )
        # sin 的符号对应虚拟腿长增加的方向；目标误差不 wrap，与 PD 一致。
        # 不能比较两者绝对值：跨过零点的收腿目标可能具有更大的绝对角差。
        extension_error = torch.clamp(
            torch.sign(torch.sin(actual_separation)) * (target_separation - actual_separation), min=0.0,
        )
        at_limit = self.leg_length >= self.max_leg_length
        return torch.mean(at_limit * torch.square(extension_error), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_target_landing(self):
        return self.jump_landing_gate * self._reward_landing_stability()

    def _reward_task_success(self):
        return self.task_success_event

    def _reward_task_failure(self):
        return self.task_failure_event
