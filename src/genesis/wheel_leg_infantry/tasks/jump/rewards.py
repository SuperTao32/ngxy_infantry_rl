"""按起跳、腾空和落地分阶段门控的跳跃奖励。"""

import math

import torch

from .geometry import target_wheel_support


class JumpRewards:
    def _landing_height_reached(self):
        """仅认可首次合格腾空的目标高度，落地站高不能补足资格。"""
        state = self.jump_reward_state
        return state.takeoff_rewarded & (state.peak_clearance >= self.base_height_target)

    def _jump_tilt_error(self):
        # 正立 gz=-1，倒立 gz=+1；小角度下约等于倾角平方。
        return 2.0 * torch.clamp(1.0 + self.projected_gravity[:, 2], 0.0, 2.0)

    def _reward_base_balance(self):
        return self._jump_tilt_error()

    def _reward_jump_invalid(self):
        return self.jump_invalid

    def _reward_landing_airborne(self):
        """首次触地后，单轮支撑和双轮离地均持续惩罚。"""
        return self.landing_airborne_gate

    def _reward_airborne_wheel_vel(self):
        """逐轮惩罚实际离地后的关节角速度平方，接地轮不参与。"""
        return torch.where(self.wheel_airborne, self.wheel_vel.square(), 0.0).sum(dim=1)

    def _reward_base_to_wheel_bottom_distance_tracking(self):
        """落地双轮有效支撑时跟踪 base 到轮底的距离参考。"""
        error = torch.clamp(
            torch.abs(self.base_to_wheel_bottom_distance - self.commands[:, 2])
            - self.reward_cfg["height_reference_tolerance_m"],
            min=0.0,
        )
        return (self.jump_landing_gate * self._landing_height_reached()
                * torch.exp(-torch.square(error) / self.reward_cfg["height_reference_sigma"]))

    def _reward_landing_base_height_tracking(self):
        """首次触地后留出配置的缓冲时间，再奖励 base_link 高度追踪。"""
        # 实测高度沿用起跳平地的世界 z 基准；落地站姿目标随台面抬高。
        # 该目标用于落地站姿跟踪，不改变跳高成绩、缺高或失败惩罚的目标。
        target = self.landing_surface_height + self.commands[:, 2]
        error = (
            (self.jump_base_height - target).abs() - self.reward_cfg["height_reference_tolerance_m"]
        ).clamp_min(0.0)
        quality = torch.exp(-error.square() / self.reward_cfg["height_reference_sigma"])
        support = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.terrain_tile_index, self.task_cfg, self.wheel_radius, margin=0.0,
        ).to(quality.dtype).mean(dim=1)
        # 缓冲结束后，单轮支撑给半额、双轮给满额；反弹、机身撑地和无效跳跃不发奖励。
        ready = self.jump_steps - self.landing_step >= self.landing_base_height_tracking_delay_steps
        valid = (self.has_landed & ready & self._landing_height_reached()
                 & (self.base_contact < 0.5) & (self.jump_invalid < 0.5))
        return torch.where(valid, support * quality, 0.0)

    def _reward_tracking_lin_vel(self):
        """平地跳和台阶跳全程跟踪指令前向速度。"""
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

    def _reward_flight_height_excess(self):
        """平地和台阶首次合格腾空只扣新增超高；下降不重复扣罚，失稳不退回已扣部分。"""
        state = self.jump_reward_state
        buffer = self.task_cfg.get("ground_base_height_excess_buffer_m", 0.03)
        excess = torch.zeros_like(state.peak_clearance)
        for mode, success_height in enumerate(self.task_cfg["ground_success_base_height_m"]):
            ceiling = success_height + buffer
            current = (state.peak_clearance - ceiling).clamp_min(0.0)
            previous = (state.previous_peak_clearance - ceiling).clamp_min(0.0)
            excess = torch.where(self.jump_mode == mode, (current - previous).clamp_min(0.0), excess)
        return self.flight_gate * excess / self.base_height_target

    def _target_takeoff_velocity(self):
        """按最后一个支撑拍距绝对目标高度的差值计算弹道起跳速度。"""
        remaining_height = torch.clamp(self.base_height_target - self.takeoff_contact_base_height, min=0.0)
        return torch.sqrt(2.0 * self.gravity_magnitude * remaining_height)

    def _reward_takeoff_upward_velocity(self):
        """支撑期间向上速度奖励无上界；下落惩罚仍保底为 -1。"""
        target = self._target_takeoff_velocity().clamp_min(1.0)
        progress = torch.clamp(self.world_vertical_velocity / target, min=-1.0)
        return self.takeoff_gate * progress

    def _reward_takeoff_wheel_push(self):
        """计划起跳期奖励接触轮向上的法向支撑力，按参考力线性归一化。"""
        reference = float(self.reward_cfg.get("takeoff_wheel_push_force_reference_n", 200.0))
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("takeoff_wheel_push_force_reference_n must be finite and positive")
        # 力和重力均在机身坐标系；与反重力点积得到世界竖直向上的分量。
        upward_force = -(self.wheel_normal_force * self.projected_gravity[:, None, :]).sum(dim=-1)
        support = torch.where(self.wheel_contact > 0.5, upward_force.clamp_min(0.0), 0.0).sum(dim=1)
        return self.takeoff_gate * support / reference

    def _reward_takeoff_vertical_velocity(self):
        """连续双轮腾空合格时，按最后一个轮地接触拍的 base-link vz 结算一次。"""
        progress = torch.clamp(
            self.takeoff_contact_vertical_velocity / self._target_takeoff_velocity().clamp_min(1.0), min=0.0
        )
        return self.jump_reward_state.takeoff_event * (1.0 - self.jump_invalid) * progress

    def _reward_flight_airtime(self):
        """首次有效腾空按秒给分，累计最多奖励配置的 flight 时长。"""
        # 非整控制步的时长只奖励最后一拍的剩余比例，累计秒数不向上取整。
        previous_steps = (self.flight_airtime_steps - 1).clamp_min(0)
        remaining_steps = self.phase_durations[1] / self.dt - previous_steps.to(self.flight_gate.dtype)
        return self.flight_gate * remaining_steps.clamp(0.0, 1.0)

    def _bounded_flight_height_progress(self, height):
        ceiling = height.new_tensor(self.task_cfg["ground_success_base_height_m"])[self.jump_mode]
        span = (ceiling - self.height_reference_min_m).clamp_min(1e-6)
        return ((height - self.height_reference_min_m) / span).clamp(0.0, 1.0)

    def _reward_flight_height_progress(self):
        """首次合格腾空只发新增高度进度，各模式成功高度处封顶；失稳撤回已发进度。"""
        state = self.jump_reward_state
        progress = self._bounded_flight_height_progress(state.peak_clearance)
        previous = self._bounded_flight_height_progress(state.previous_peak_clearance)
        revoked = self._bounded_flight_height_progress(state.revoked_peak_clearance)
        return self.flight_gate * (progress - previous).clamp_min(0.0) - revoked

    def _reward_flight_wheel_clearance(self):
        ceiling = torch.clamp(self.jump_base_height - 0.20, min=0.0)
        clearance = torch.minimum(torch.clamp(self.wheel_clearance, min=0.0), ceiling)
        return self.flight_gate * clearance

    def _reward_flight_balance(self):
        tilt_error = self._jump_tilt_error()
        return self.flight_gate * torch.exp(-tilt_error / self.reward_cfg["flight_balance_sigma"])

    def _reward_flight_height_tracking(self):
        error = torch.square(self.jump_base_height - self.base_height_target)
        return self.flight_gate * torch.exp(-error / self.reward_cfg["flight_height_sigma"])

    def _reward_flight_tuck(self):
        target = self.reward_cfg["short_leg_length_target"]
        error = torch.mean(torch.square(self.leg_length - target), dim=1)
        return self.flight_gate * torch.exp(-error / self.reward_cfg["leg_length_sigma"])

    def _reward_flight_leg_angle(self):
        """腾空时分别惩罚左右虚拟腿摆角超过容差的部分，单位 rad²。"""
        excess = (self.leg_angle.abs() - self.flight_leg_angle_limit_rad).clamp_min(0.0)
        return self.flight_gate * excess.square().mean(dim=1)

    def _reward_landing_base_acceleration(self):
        """首次触地当拍及之后惩罚 base_link 三轴运动加速度平方。"""
        # 不依赖双轮有效落台或姿态门，单轮/机身先触地和失稳不能免除冲击惩罚。
        return self.has_landed * self.base_link_acceleration.square().sum(dim=1)

    def _reward_landing_vertical_velocity(self):
        """首次着陆结算触地前 base_link 世界竖直速度平方，配合负权重惩罚。"""
        # 取最后腾空拍，避免接触冲量改变触地当拍速度；向上/向下均惩罚。
        # 任意轮或机身首次触地均结算，失稳或错误落台也不能免除惩罚。
        return self.jump_landing_event * torch.square(self.last_airborne_vertical_velocity)

    def _reward_soft_landing(self):
        tilt_error = self._jump_tilt_error()
        impact_quality = torch.exp(
            -torch.square(self.impact_vertical_speed) / self.reward_cfg["landing_velocity_sigma"]
        )
        attitude_quality = torch.exp(-tilt_error / self.reward_cfg["landing_tilt_sigma"])
        support = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.terrain_tile_index, self.task_cfg, self.wheel_radius, margin=0.0,
        ).any(dim=1)
        return (self.jump_landing_event * self._landing_height_reached()
                * impact_quality * attitude_quality * support * (self.jump_invalid < 0.5))

    def _reward_landing_stability(self):
        tilt_error = self._jump_tilt_error()
        # 只抑制 roll/pitch 角速度，保留命令指定的转向。
        angular_error = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        velocity_error = torch.square(self.world_vertical_velocity)
        quality = torch.exp(
            -velocity_error / self.reward_cfg["landing_velocity_sigma"]
            -tilt_error / self.reward_cfg["landing_tilt_sigma"]
            -angular_error / self.reward_cfg["landing_ang_vel_sigma"]
        )
        return self.jump_landing_gate * self._landing_height_reached() * quality

    def _reward_leg_extension_at_limit(self):
        """实测腿长到限位后，惩罚实际下发目标中继续伸腿的电机角差（rad²）。"""
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

    def _reward_task_success(self):
        # 通用死亡判定晚于任务状态更新；同拍死亡不能领取成功奖励。
        return self.task_success_event * self._landing_height_reached() * ~self.terminated_buf

    def _reward_task_failure(self):
        """按首次合格腾空的高度缺额惩罚，落地后站高不能减少失败惩罚。"""
        shortfall = torch.clamp(1.0 - self.jump_reward_state.peak_clearance / self.base_height_target, 0.0, 1.0)
        return self.task_failure_event * shortfall
