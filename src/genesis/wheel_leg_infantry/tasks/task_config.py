"""环境与任务共享的命令、重置采样和配置更新方法。"""

import math
import genesis as gs
import torch
from genesis.utils.geom import inv_quat, transform_by_quat, transform_quat_by_quat, xyz_to_quat

from ..config import validate_high_base_height_velocity_limits
from ..utils.tensor_utils import as_range_tensors, sample_uniform
from ..utils.tracking_gate import merge_tracking_gate_config, smooth_gate


class TaskConfiguration:
    def _sample_base_reset_state(self, env_ids):
        """为本次 reset 的每个子环境独立采样 base pose 和速度。"""
        num_resets = env_ids.numel()
        reset_base_pos = sample_uniform(
            self.base_init_pos_lower,
            self.base_init_pos_upper,
            (num_resets,),
        )
        terrain_centers, terrain_tile_index = self.terrain.sample_spawn_tiles(env_ids)
        # 配置中的 x/y 是相对 patch 中心的扰动，z 是相对当地地面的初始高度。
        reset_base_pos[:, :2] += terrain_centers
        rpy_offset_deg = sample_uniform(
            self.base_init_rpy_lower,
            self.base_init_rpy_upper,
            (num_resets,),
        )
        self.terrain.adjust_spawn(reset_base_pos, rpy_offset_deg, terrain_tile_index)
        terrain_height = self.terrain.height_at(reset_base_pos[:, :2])
        reset_base_pos[:, 2] += terrain_height
        quat_offset = xyz_to_quat(rpy_offset_deg, rpy=True, degrees=True)
        nominal_quat = self.init_base_quat.expand(num_resets, -1)
        reset_base_quat = transform_quat_by_quat(quat_offset, nominal_quat)

        reset_base_lin_vel = sample_uniform(
            self.base_init_lin_vel_lower,
            self.base_init_lin_vel_upper,
            (num_resets,),
        )
        reset_base_ang_vel = sample_uniform(
            self.base_init_ang_vel_lower,
            self.base_init_ang_vel_upper,
            (num_resets,),
        )
        reset_projected_gravity = transform_by_quat(
            self.global_gravity_dir.expand(num_resets, -1),
            inv_quat(reset_base_quat),
        )
        reset_qpos = torch.concatenate(
            (
                reset_base_pos,
                reset_base_quat,
                self.init_dof_pos.expand(num_resets, -1),
            ),
            dim=-1,
        )
        return (
            reset_qpos,
            reset_base_pos,
            reset_base_quat,
            reset_base_lin_vel,
            reset_base_ang_vel,
            reset_projected_gravity,
            terrain_height,
            terrain_centers,
            terrain_tile_index,
        )

    def _sample_commands(self, command_cfg, limits):
        """按给定配置采样，不写回，供混合任务各自使用命令分布。"""
        commands = sample_uniform(*limits, (self.num_envs,))
        lower, upper = limits
        angular_lower = lower[1].expand(self.num_envs)
        angular_upper = upper[1].expand(self.num_envs)
        angular_resample = torch.zeros(self.num_envs, dtype=torch.bool, device=commands.device)
        height_velocity_limits_cfg = command_cfg.get("high_base_height_velocity_limits")
        if height_velocity_limits_cfg is not None:
            high_height = commands[:, 2] > height_velocity_limits_cfg["base_height_threshold"]
            limit = height_velocity_limits_cfg["max_abs_lin_vel"]
            # 先收紧采样区间；固定速度或不相交区间退到最近的限速边界。
            high_height_lin_vel = sample_uniform(lower[0].clamp(-limit, limit), upper[0].clamp(-limit, limit), (self.num_envs,))
            commands[:, 0] = torch.where(high_height, high_height_lin_vel, commands[:, 0])
            angular_resample |= high_height
        high_speed_cfg = command_cfg.get("high_speed_ang_vel")
        if high_speed_cfg is not None:
            # 按最终 |vx| 分档，两项限制同时触发时取更严格的角速度区间。
            high_speed = commands[:, 0].abs() > high_speed_cfg["lin_vel_threshold"]
            limit = high_speed_cfg["max_abs_ang_vel"]
            angular_lower = torch.where(high_speed, angular_lower.clamp(min=-limit), angular_lower)
            angular_upper = torch.where(high_speed, angular_upper.clamp(max=limit), angular_upper)
            angular_resample |= high_speed
        if height_velocity_limits_cfg is not None:
            limit = height_velocity_limits_cfg["max_abs_ang_vel"]
            angular_lower = torch.where(high_height, angular_lower.clamp(-limit, limit), angular_lower)
            angular_upper = torch.where(high_height, angular_upper.clamp(-limit, limit), angular_upper)
        if height_velocity_limits_cfg is not None or high_speed_cfg is not None:
            angular = sample_uniform(angular_lower, angular_upper, ())
            commands[:, 1] = torch.where(angular_resample, angular, commands[:, 1])
        standing_probability = command_cfg.get("standing_probability", 0.0)
        if standing_probability > 0.0:
            # 静止样本只清零 vx、wz，仍跟踪当前课程采样的机身高度。
            standing = torch.rand_like(commands[:, 0]) < standing_probability
            commands[:, :2].masked_fill_(standing[:, None], 0.0)
        return commands

    def _update_tracking_gate(self):
        """根据高度目标和机身倾角更新平滑 AND 门控。"""
        height_error = torch.abs(self.base_height - self.commands[:, 2])
        self.height_gate = smooth_gate(
            height_error,
            self.tracking_gate_cfg["height_full_error"],
            self.tracking_gate_cfg["height_zero_error"],
        )

        # projected_gravity 的水平分量模长等于 sin(tilt)，可连续表示 roll/pitch 合成倾角。
        sin_tilt = torch.linalg.vector_norm(self.projected_gravity[:, :2], dim=1).clamp(0.0, 1.0)
        tilt = torch.asin(sin_tilt)
        self.attitude_gate = smooth_gate(
            tilt,
            math.radians(self.tracking_gate_cfg["attitude_full_angle_deg"]),
            math.radians(self.tracking_gate_cfg["attitude_zero_angle_deg"]),
        )

        self.tracking_gate_raw = self.height_gate * self.attitude_gate
        gate_floor = self.tracking_gate_cfg["floor"]
        self.tracking_gate = gate_floor + (1.0 - gate_floor) * self.tracking_gate_raw

    def _build_commands_scale(self):
        """构造 locomotion 的 [vx, wz, base_height] 命令缩放。"""
        return torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["base_height"]],
            dtype=gs.tc_float,
            device=self.device,
        )

    def _build_command_limits(self, command_cfg=None):
        """构造 [vx, wz, base_height] 的采样上下限。"""
        command_cfg = self.command_cfg if command_cfg is None else command_cfg
        validate_high_base_height_velocity_limits(command_cfg)
        standing_probability = float(command_cfg.get("standing_probability", 0.0))
        if not math.isfinite(standing_probability) or not 0.0 <= standing_probability <= 1.0:
            raise ValueError("standing_probability must be finite and in [0, 1]")
        high_speed_cfg = command_cfg.get("high_speed_ang_vel")
        if high_speed_cfg is not None:
            threshold = float(high_speed_cfg["lin_vel_threshold"])
            limit = float(high_speed_cfg["max_abs_ang_vel"])
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError("high_speed_ang_vel.lin_vel_threshold must be finite and nonnegative")
            if not math.isfinite(limit) or limit < 0.0:
                raise ValueError("high_speed_ang_vel.max_abs_ang_vel must be finite and nonnegative")
            lower, upper = command_cfg["ang_vel_range"]
            if max(lower, -limit) > min(upper, limit):
                raise ValueError("ang_vel_range must overlap the high-speed angular velocity limits")
        return tuple(
            torch.tensor(items, dtype=gs.tc_float, device=gs.device)
            for items in zip(
                command_cfg["lin_vel_range"],
                command_cfg["ang_vel_range"],
                command_cfg["base_height_range"],
            )
        )

    def _apply_command_ranges(self, values):
        """应用课程指令范围和静止采样率；在下一次命令重采样时生效。"""
        allowed = {"lin_vel_range", "ang_vel_range", "base_height_range", "standing_probability"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported command range curriculum keys: {sorted(unknown)}")
        normalized = {}
        for name, limits in values.items():
            if name == "standing_probability":
                probability = float(limits)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError("standing_probability must be finite and in [0, 1]")
                normalized[name] = probability
                continue
            if len(limits) != 2 or limits[0] > limits[1]:
                raise ValueError(f"{name} must be [lower, upper], got {limits}")
            normalized[name] = list(limits)
        self.command_cfg.update(normalized)
        self.commands_limit = self._build_command_limits()

    def _apply_standing_reward(self, values):
        """随课程收紧静止跟踪容差/漂移死区；惩罚权重仍由 reward_scales 设置。"""
        allowed = {"tracking_sigma", "drift_deadband"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported standing reward curriculum keys: {sorted(unknown)}")
        normalized = {}
        for name, value in values.items():
            value = float(value)
            if not math.isfinite(value) or value < 0.0 or (name == "tracking_sigma" and value == 0.0):
                raise ValueError(f"standing_reward.{name} must be finite and {'positive' if name == 'tracking_sigma' else 'nonnegative'}")
            normalized["standing_" + name] = value
        self.reward_cfg.update(normalized)

    def _apply_tracking_gate(self, values):
        """应用速度跟踪门控参数。"""
        self.tracking_gate_cfg = merge_tracking_gate_config(self.tracking_gate_cfg, values)
        self.reward_cfg["tracking_gate"] = dict(self.tracking_gate_cfg)

    def _apply_termination_limits(self, values):
        """应用姿态/接触终止条件，并同步刷新由持续时间换算出的步数。"""
        angle_keys = {"termination_if_roll_greater_than", "termination_if_pitch_greater_than"}
        duration_keys = {"tilt_termination_duration_s", "base_contact_termination_duration_s"}
        allowed = angle_keys | duration_keys
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported termination curriculum keys: {sorted(unknown)}")

        normalized = {}
        for name in angle_keys.intersection(values):
            angle = float(values[name])
            if not math.isfinite(angle) or angle <= 0.0:
                raise ValueError(f"{name} must be a positive finite angle")
            normalized[name] = angle

        tilt_duration = self.tilt_termination_duration_s
        if "tilt_termination_duration_s" in values:
            tilt_duration = float(values["tilt_termination_duration_s"])
            if not math.isfinite(tilt_duration) or tilt_duration <= 0.0:
                raise ValueError("tilt_termination_duration_s must be positive and finite")
            normalized["tilt_termination_duration_s"] = tilt_duration

        base_contact_duration = self.base_contact_termination_duration_s
        if "base_contact_termination_duration_s" in values:
            raw_duration = values["base_contact_termination_duration_s"]
            base_contact_duration = None if raw_duration is None else float(raw_duration)
            if base_contact_duration is not None and (not math.isfinite(base_contact_duration) or base_contact_duration <= 0.0):
                raise ValueError("base_contact_termination_duration_s must be positive and finite, or None")
            normalized["base_contact_termination_duration_s"] = base_contact_duration

        self.env_cfg.update(normalized)
        self.tilt_termination_duration_s = tilt_duration
        self.base_contact_termination_duration_s = base_contact_duration
        self.tilt_termination_steps = max(1, math.ceil(tilt_duration / self.dt))
        self.base_contact_termination_steps = None if base_contact_duration is None else max(1, math.ceil(base_contact_duration / self.dt))

    def _apply_reset_ranges(self, values):
        """应用课程阶段的 base 初始位置、姿态和速度随机化范围。"""
        allowed = {
            "base_init_pos_range",
            "base_init_rpy_offset_range_deg",
            "base_init_lin_vel_range",
            "base_init_ang_vel_range",
        }
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported reset range curriculum keys: {sorted(unknown)}")

        tensor_attributes = {
            "base_init_pos_range": ("base_init_pos_lower", "base_init_pos_upper"),
            "base_init_rpy_offset_range_deg": ("base_init_rpy_lower", "base_init_rpy_upper"),
            "base_init_lin_vel_range": ("base_init_lin_vel_lower", "base_init_lin_vel_upper"),
            "base_init_ang_vel_range": ("base_init_ang_vel_lower", "base_init_ang_vel_upper"),
        }
        for name, limits in values.items():
            lower, upper = as_range_tensors(limits, 3, name, device=self.device)
            lower_attr, upper_attr = tensor_attributes[name]
            setattr(self, lower_attr, lower)
            setattr(self, upper_attr, upper)
            # 同步运行时配置，便于日志和调试输出反映当前课程阶段。
            self.env_cfg[name] = [list(axis_limits) for axis_limits in limits]

    def _base_contact_failure(self):
        """子任务可豁免合法的机身支撑接触。"""
        return self.base_contact > 0.5

    def _tilt_failure(self):
        return (self.base_euler[:, 0].abs() > self.env_cfg["termination_if_roll_greater_than"]) | (
            self.base_euler[:, 1].abs() > self.env_cfg["termination_if_pitch_greater_than"]
        )
