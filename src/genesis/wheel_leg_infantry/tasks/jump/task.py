"""条件跳跃任务；直接进入统一 PPO rollout，无 teacher 或独立 runner。"""

import math
from copy import deepcopy
from typing import Mapping
import genesis as gs
import torch

from ...utils.kinematics import compute_mean_base_to_wheel_bottom_distance
from .approach import JumpApproach
from .terrain import JumpTerrain
from ..base import Task
from .buffers import JumpBuffers
from .config import (
    MODE_NAMES, REWARD_TERRAIN_NAMES, get_approach_cfg, validate_configs, validate_mode_probabilities,
    validate_platform_enabled, validate_platform_probabilities,
)
from .geometry import target_wheel_support, terrain_tile_heights, terrain_tile_modes
from .height_reference import resolve_height_reference, sample_height_reference
from .phase import phase_encoding, resolve_input_durations, validate_phase_durations
from .rewards import JumpRewards


class JumpTask(JumpApproach, JumpRewards, JumpBuffers, Task):
    def __init__(self, env, name, config, env_ids):
        validate_configs(config["env_cfg"], config["curriculum_cfg"])
        config = self._resolve_reward_scales(self._merge_legacy_height_rewards(config))
        self.approach_cfg = {**get_approach_cfg(), **config["env_cfg"].get("jump_approach", {})}
        self.approach_cfg.pop("reward_scales", None)  # 兼容旧存档，助跑奖励配置不再使用。
        self.task_cfg = config["env_cfg"]["jump_modes"]
        self.phase_durations = validate_phase_durations(config["env_cfg"]["jump_phase_durations_s"])
        self.phase_cycle_s = sum(self.phase_durations)
        input_durations = resolve_input_durations(config["env_cfg"])
        super().__init__(env, name, config, env_ids)
        self.phase_input_duration_s = torch.tensor(input_durations, dtype=gs.tc_float, device=self.device)
        self.phase_input_steps = torch.tensor(
            [math.ceil(duration / self.dt - 1e-9) for duration in input_durations],
            dtype=torch.long, device=self.device,
        )
        # 超时必须覆盖完整输入窗口；输入较短时保留阶段总时长供落稳。
        episode_lengths = [math.ceil(max(self.phase_cycle_s, duration) / self.dt - 1e-9) for duration in input_durations]
        self.mode_episode_lengths = torch.tensor(episode_lengths, dtype=torch.long, device=self.device)
        self.max_episode_length = max(episode_lengths)

    def _initialize_task_buffers(self):
        super()._initialize_task_buffers()
        self._initialize_approach_buffers()

    def create_terrain(self):
        # Keep task and terrain configuration identical after the defensive copy.
        self.task_cfg = self.env_cfg["jump_modes"]
        return JumpTerrain(
            self.task_cfg, self.approach_cfg,
            height_curriculum=self.curriculum_cfg.get("height_curriculum"),
            curriculum_active=self.curriculum_cfg.get("enabled", False) and not self.evaluation,
        )

    def _update_task_state(self):
        """父类 step 钩子：先更新物理阶段，再判断目标面支撑与任务成败。"""
        self.jump_steps.add_((self.mask & self.jump_active).long())
        self._update_approach_state()
        self._update_jump_state()
        cfg = self.task_cfg
        # 支撑只用于本拍落台判定，无需跨步保存。
        target_support = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.terrain_tile_index, cfg, self.wheel_radius,
        ).all(dim=1)
        # 首次单轮触地可以开始正常缓冲，但侧壁/台阶下方接触不是有效落台。
        top_contact = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.terrain_tile_index, cfg, self.wheel_radius, margin=0.0,
        ).any(dim=1)
        # 机身先触地不算错误落台；只有实际轮接触落在目标面之外才判错。
        bad_landing = (self.jump_active & (self.landing_surface_height > 0) & self.has_taken_off
                       & (self.wheel_contact > 0.5).any(dim=1) & ~top_contact)
        # 起跳前轮子低于台面并到达立面，属于撞沿，不能通过贴墙爬升刷跳跃奖励。
        wheel_x = self.wheel_center_pos[:, :, 0]
        wheel_bottom = self.wheel_center_pos[:, :, 2] - self.wheel_radius
        riser_hit = (self.jump_active & (self.landing_surface_height > 0) & ~self.has_landed
                     & torch.any((wheel_x + self.wheel_radius >= 0)
                                 & (wheel_x < 0)
                                 & (wheel_bottom < self.landing_surface_height[:, None] - cfg["landing_height_tolerance_m"])
                                 & (self.wheel_contact > 0.5), dim=1))
        self.jump_invalid.copy_(torch.maximum(self.jump_invalid, (bad_landing | riser_hit).to(gs.tc_float)))
        self.jump_landing_gate.mul_(target_support)
        valid = self.jump_invalid < 0.5
        # 过倾、撞沿和错误落台全部确定后再结算，保证终止当拍也能追回高度奖励。
        self.takeoff_gate.mul_(valid)
        self.flight_gate.mul_(valid)
        # 首次腾空的累计有效拍数不因局部间隙短暂低于阈值而清零。
        self.flight_airtime_steps.add_((self.flight_gate > 0.5).long())
        # 成绩只取首次有效双轮腾空；站高、单轮支撑和落地当拍都不能补成绩。
        self.max_wheel_clearance.copy_(torch.where(
            self.flight_gate > 0.5, torch.maximum(self.max_wheel_clearance, self.wheel_clearance), self.max_wheel_clearance,
        ))
        self.jump_reward_state.update(
            self.flight_gate > 0.5, self.jump_base_height, self.jump_landing_event, self.jump_steps,
            takeoff_eligible=(self.takeoff_contact_vertical_velocity > 0.0) & valid,
            valid_jump=valid,
            horizon=self.mode_episode_lengths[self.jump_mode],
        )
        tilt_ok = -self.projected_gravity[:, 2] >= math.cos(math.radians(cfg["landing_max_tilt_deg"]))
        # 台阶落稳允许短暂机身接触；通用接触计时器仍独立判定持续碰撞死亡。
        contact_ok = (self.base_contact < 0.5) | (self.landing_surface_height > 0.0)
        stable = (self.jump_active & self._landing_height_reached() & target_support & valid & tilt_ok & contact_ok
                  & (self.world_vertical_velocity.abs() <= cfg["landing_max_vertical_speed_m_s"]))
        self.stable_landing_steps.copy_(torch.where(stable, self.stable_landing_steps + 1, 0))
        # 达到目标高度后，平地还须满足轮底/机身最低门槛，台阶须在目标面落稳。
        height_ok = self.landing_surface_height > 0
        for mode, (wheel_min, base_min) in enumerate(zip(
            cfg["ground_success_wheel_clearance_m"], cfg["ground_success_base_height_m"],
        )):
            height_ok |= ((self.jump_mode == mode)
                          & (self.max_wheel_clearance > wheel_min)
                          & (self.jump_reward_state.peak_clearance > base_min))
        self.task_success.copy_((self.stable_landing_steps >= self.required_stable_steps) & height_ok)
        self.task_success_event.copy_(self.task_success & ~self.success_seen)
        self.success_seen.logical_or_(self.task_success)
        failed = ~valid | self.approach_timed_out | (self._episode_timeout() & ~self.task_success)
        self.task_failure_event.copy_(failed & ~self.failure_seen)
        self.failure_seen.logical_or_(failed)

    def _update_jump_state(self):
        """按物理先后更新运动、起跳/落地事件、失败历史与奖励门控。"""
        elapsed_s = self.jump_steps.to(dtype=gs.tc_float) * self.dt
        scheduled_stage = torch.bucketize(
            self.jump_steps.contiguous(),
            self.phase_step_boundaries, right=True,
        )
        # 模式与相位由同一独立窗口控制，在同一整数控制步边界关闭。
        phase_active = self.jump_active & (self.jump_steps < self.phase_input_steps[self.jump_mode])
        self.jump_mode_one_hot.masked_fill_(~phase_active[:, None], 0.0)
        # 逐环境归一化，避免每步把 GPU 时长张量拷回 CPU 校验。
        progress = elapsed_s / self.phase_input_duration_s[self.jump_mode]
        self.phase_features.copy_(torch.where(
            phase_active[:, None], phase_encoding(progress, 1.0), 0.0,
        ))
        current_velocity = self.robot.get_vel()
        # 使用 base_link 原点的世界系真实速度，避免机体系旋转、IMU 安装点和噪声干扰。
        self.base_link_acceleration.copy_((current_velocity - self.last_base_link_velocity) / self.dt)
        self.last_base_link_velocity.copy_(current_velocity)
        current_vertical_velocity = current_velocity[:, 2]
        self.world_vertical_velocity.copy_(current_vertical_velocity)

        both_wheels_contact = torch.all(self.wheel_contact > 0.5, dim=1)
        safe_wheel_contact = both_wheels_contact & (self.base_contact < 0.5)
        self._update_jump_geometry()
        # 接触信号来自力阈值，卸载时可能为零；还须检查两轮底部确实离开局部地面。
        surface = self.terrain.height_at(self.wheel_center_pos[:, :, :2].reshape(-1, 2)).reshape(self.num_envs, 2)
        wheel_gap = self.wheel_center_pos[:, :, 2] - self.wheel_radius - surface
        wheel_touching = (self.wheel_contact > 0.5) | (wheel_gap <= 0.0)
        self.wheel_airborne.copy_(
            (self.wheel_contact <= 0.5) & (wheel_gap > self.env_cfg.get("takeoff_min_wheel_clearance_m", 0.01))
        )
        wheels_airborne = self.wheel_airborne.all(dim=1)

        # 起跳接触期不依赖计划阶段：首次双轮完全离地前，只要至少一轮仍在
        # 安全接触，就同步锁存最后一个接触拍的 base-link vz 和高度。
        wheel_support = torch.any(self.wheel_contact > 0.5, dim=1) & (self.base_contact < 0.5)
        accumulate_takeoff = self.jump_active & wheel_support & ~self.has_taken_off
        self.takeoff_gate.copy_(accumulate_takeoff.to(dtype=gs.tc_float))
        # 稠密起跳奖励只在计划起跳时段生效，接触拍速度仍按物理状态锁存。
        self.takeoff_gate.mul_((scheduled_stage == 0).to(dtype=gs.tc_float))
        self.takeoff_contact_vertical_velocity.copy_(
            torch.where(
                accumulate_takeoff,
                current_vertical_velocity,
                self.takeoff_contact_vertical_velocity,
            )
        )
        self.takeoff_contact_base_height.copy_(
            torch.where(accumulate_takeoff, self.jump_base_height, self.takeoff_contact_base_height)
        )
        takeoff_now = (
            self.jump_active & wheels_airborne
            & ~self.has_taken_off
            & (self.base_contact < 0.5)
            & (self.takeoff_contact_vertical_velocity > 0.0)
        )
        self.jump_takeoff_event.copy_(takeoff_now.to(dtype=gs.tc_float))
        self.has_taken_off.logical_or_(takeoff_now)

        # 机身先触地也必须冻结触地前速度，不能被本拍接触冲量覆盖。
        in_flight_before_update = (
            self.has_taken_off & ~self.has_landed & ~wheel_touching.any(dim=1) & (self.base_contact < 0.5)
        )
        self.last_airborne_vertical_velocity.copy_(
            torch.where(
                in_flight_before_update,
                current_vertical_velocity,
                self.last_airborne_vertical_velocity,
            )
        )
        # 第一次任意轮/机身触地就关闭跳跃，单轮落地后弹起不能接着累计高度。
        landing_now = (wheel_touching.any(dim=1) | (self.base_contact > 0.5)) & self.has_taken_off & ~self.has_landed
        self.jump_landing_event.copy_(landing_now.to(dtype=gs.tc_float))
        self.landing_step.copy_(torch.where(landing_now, self.jump_steps, self.landing_step))
        self.has_landed.logical_or_(landing_now)
        self.impact_vertical_speed.zero_()
        self.impact_vertical_speed.copy_(
            torch.where(
                landing_now,
                torch.clamp(-self.last_airborne_vertical_velocity, min=0.0),
                self.impact_vertical_speed,
            )
        )

        # 失稳在本回合锁存，恢复姿态也不能重新取得翻转产生的高度成绩。
        invalid_now = self.jump_active & (-self.projected_gravity[:, 2] < self.jump_min_upright_cos)
        self.jump_invalid.copy_(torch.maximum(self.jump_invalid, invalid_now.to(dtype=gs.tc_float)))
        valid_jump = self.jump_invalid < 0.5
        self.takeoff_gate.mul_(valid_jump)
        airborne = self.has_taken_off & ~self.has_landed & wheels_airborne & (self.base_contact < 0.5) & valid_jump
        # 首次触地后任一轮仍未着地就持续惩罚，不能靠单轮支撑拖延双轮落稳。
        # 不使用 1 cm 起跳离地阈值，避免另一轮悬停在更小间隙内绕过惩罚。
        incomplete_landing = self.jump_active & self.has_landed & ~wheel_touching.all(dim=1)
        self.landing_airborne_gate.copy_(incomplete_landing.to(dtype=gs.tc_float))
        # 反弹历史仍只记录再次双轮离地，单轮支撑不记为反弹，不单独判失败。
        rebound = self.jump_active & self.has_landed & wheels_airborne & (self.base_contact < 0.5)
        self.rebound_seen.copy_(torch.maximum(self.rebound_seen, rebound.to(dtype=gs.tc_float)))
        self.flight_gate.copy_(airborne.to(dtype=gs.tc_float))
        self.jump_landing_gate.copy_(
            (self.has_landed & safe_wheel_contact).to(dtype=gs.tc_float)
        )
        self._update_jump_commands()

    def _update_jump_geometry(self):
        """分别更新机身跳高、腿部相对距离和轮底几何。"""
        # 所有通道都从 z=0 平地起跳；台阶上方也不扣台面高度或出生时的机身高度。
        # 跳高奖励、峰值、缺高和失败惩罚统一使用这一固定基准。
        # 不复用 locomotion 的 base_height，因为它相对当前位置地表计算。
        self.jump_base_height.copy_(self.base_pos[:, 2])
        self.max_jump_base_height.copy_(torch.where(
            self.jump_active, torch.maximum(self.max_jump_base_height, self.jump_base_height), self.max_jump_base_height,
        ))
        self.wheel_center_pos.copy_(self.robot.get_links_pos(self.wheel_links_idx))
        wheel_bottom_height = self.wheel_center_pos[:, :, 2] - self.wheel_radius
        # 两侧距离沿世界 z 方向测量后取平均，单独用于伸腿/收腿轨迹跟踪。
        # 与 base 离地高度独立，整体腾空或地形高度变化不影响该相对距离。
        self.base_to_wheel_bottom_distance.copy_(
            compute_mean_base_to_wheel_bottom_distance(
                self.base_pos,
                self.wheel_center_pos,
                self.wheel_radius,
            )
        )
        # 两种模式均从 z=0 的平地起跳；用较低一侧轮底衡量过障间隙。
        self.wheel_clearance.copy_(torch.min(wheel_bottom_height, dim=1).values)

    def _update_jump_commands(self):
        """第三列为共享的 base 到轮底距离参考，按 takeoff / flight 时长生成。"""
        self.commands[self.mask, 1] = 0.0
        self.commands[self.mask & ~self.jump_active, 2] = self.command_cfg["base_height_range"][0]
        elapsed = self.jump_steps.to(gs.tc_float) * self.dt
        selected = self.mask & self.jump_active
        self.commands[selected, 2] = sample_height_reference(
            elapsed[selected], takeoff_s=self.phase_durations[0], retract_duration_s=self.retract_duration_s,
            min_distance_m=self.height_reference_min_m, max_distance_m=self.height_reference_max_m,
        )

    def _task_termination(self):
        """返回 jump 特有的终止原因，并区分正常任务失败与 solver 异常。"""
        # 失败仍可提前终止；只有成功终止需要等输入窗口结束。
        # 二次腾空仅惩罚；旧存档的 on_rebound 字段不再生效。
        cfg = self.env_cfg.get("jump_termination", {})
        failed = self.mask & (((self.jump_invalid > 0.5) & cfg.get("on_invalid", True)) | self.approach_timed_out)
        # solver error 始终需要 runner 抛错，即使同拍也发生正常任务失败。
        solver_error = self.scene.rigid_solver.get_error_envs_mask().bool()
        self.extras["jump_rebound_termination"] = torch.zeros_like(failed)
        self.extras["jump_task_termination"] = failed & ~solver_error
        return failed

    @classmethod
    def _normalize_reward_scale(cls, name, value, *, allow_partial=False):
        """权重支持统一标量、逐模式列表，或按实际平地/台阶分别配置。"""
        if isinstance(value, Mapping):
            if (not value or set(value) - set(REWARD_TERRAIN_NAMES)
                    or (not allow_partial and set(value) != set(REWARD_TERRAIN_NAMES))):
                raise ValueError(f"reward scale {name} must specify without_platform and with_platform")
            if any(isinstance(weight, Mapping) for weight in value.values()):
                raise ValueError(f"reward scale {name} terrain weights must be scalars or per-mode lists")
            return {terrain: cls._normalize_reward_scale(name, weight) for terrain, weight in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) != len(MODE_NAMES) or any(isinstance(weight, (list, tuple, Mapping)) for weight in value):
                raise ValueError(f"reward scale {name} must contain {len(MODE_NAMES)} finite values")
            return [cls._normalize_reward_scale(name, weight) for weight in value]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"reward scale {name} must be finite")
        return float(value)

    @staticmethod
    def _merge_reward_scale(previous, override):
        """课程可只覆盖一种地形，另一种沿用此前标量、逐模式或地形权重。"""
        if not isinstance(override, Mapping):
            return deepcopy(override)
        weights = (deepcopy(previous) if isinstance(previous, Mapping)
                   else {terrain: deepcopy(previous) for terrain in REWARD_TERRAIN_NAMES})
        weights.update(deepcopy(override))
        return weights

    @classmethod
    def _resolve_reward_scales(cls, config):
        """预先展开课程局部覆盖，恢复或跳过阶段时仍保留完整的累计权重。"""
        config = deepcopy(config)
        weights = {
            name: cls._normalize_reward_scale(name, value)
            for name, value in config["reward_cfg"]["reward_scales"].items()
        }
        config["reward_cfg"]["reward_scales"] = deepcopy(weights)
        for stage in config["curriculum_cfg"].get("stages", []):
            overrides = stage.get("targets", {}).get("reward_scales", {})
            for name, value in overrides.items():
                cls._normalize_reward_scale(name, value, allow_partial=True)
                if isinstance(value, Mapping) and name in weights:
                    value = cls._merge_reward_scale(weights[name], value)
                weights[name] = cls._normalize_reward_scale(name, value)
                overrides[name] = deepcopy(weights[name])
        return config

    @classmethod
    def _merge_legacy_height_rewards(cls, config):
        """旧存档的峰值/进度权重合并为成功高度处的总预算，课程按累计覆盖转换。"""
        stages = config["curriculum_cfg"].get("stages", [])
        scales = config["reward_cfg"]["reward_scales"]
        if "flight_peak_height" not in scales and not any("flight_peak_height" in stage.get("targets", {}).get("reward_scales", {}) for stage in stages):
            return config

        config = deepcopy(config)
        modes = config["env_cfg"]["jump_modes"]
        minimum, _ = resolve_height_reference(config["env_cfg"])

        def merged_weights(peak_weight, progress_weight):
            if isinstance(peak_weight, Mapping) or isinstance(progress_weight, Mapping):
                return {
                    terrain: merged_weights(
                        peak_weight[terrain] if isinstance(peak_weight, Mapping) else peak_weight,
                        progress_weight[terrain] if isinstance(progress_weight, Mapping) else progress_weight,
                    ) for terrain in REWARD_TERRAIN_NAMES
                }
            weights = []
            for mode, (ceiling, target) in enumerate(zip(modes["ground_success_base_height_m"], modes["clearance_targets_m"])):
                peak = peak_weight[mode] if isinstance(peak_weight, (list, tuple)) else peak_weight
                progress = progress_weight[mode] if isinstance(progress_weight, (list, tuple)) else progress_weight
                span = max(ceiling - minimum, 0.0)
                weights.append(peak * span / target + progress * min(span / max(target - minimum, 1e-6), 1.0))
            return weights

        scales = config["reward_cfg"]["reward_scales"]
        peak_weight = scales.pop("flight_peak_height", 0.0)
        progress_weight = scales.get("flight_height_progress", 0.0)
        scales["flight_height_progress"] = merged_weights(peak_weight, progress_weight)
        for stage in config["curriculum_cfg"].get("stages", []):
            overrides = stage.get("targets", {}).get("reward_scales", {})
            if "flight_peak_height" in overrides or "flight_height_progress" in overrides:
                peak_weight = cls._merge_reward_scale(peak_weight, overrides.pop("flight_peak_height", peak_weight))
                progress_weight = cls._merge_reward_scale(progress_weight, overrides.get("flight_height_progress", progress_weight))
                overrides["flight_height_progress"] = merged_weights(peak_weight, progress_weight)
        return config

    def _get_reward_scale(self, name):
        scale = self.reward_scales[name]
        # 使用回合锁存的实际台面高度；课程移除台阶、平地采样和 reset 均自动切换。
        if isinstance(scale, torch.Tensor):
            scale = (scale[(self.landing_surface_height > 0).long(), self.jump_mode]
                     if scale.ndim == 2 else scale[self.jump_mode])
        # 助跑仅推理，不发奖励；旧配置中的助跑专用项也不参与 jump。
        if name in {"base_height", "wheel_airborne"}:
            return scale * torch.zeros_like(self.jump_active)
        return scale * self.jump_active

    def compute_rewards(self):
        active = self.mask & self.jump_active
        if not active.any():
            return
        for name, function in self.reward_functions.items():
            values = torch.where(active, function() * self._get_reward_scale(name), 0.)
            self.env.reward_buf.add_(values)
            self.episode_sums[name].add_(values)

    def _apply_reward_scales(self, values):
        """兼容统一/逐模式权重；地形字典分别设置平地和台阶，可由课程局部覆盖。"""
        normalized = {}
        for name, value in values.items():
            if isinstance(value, Mapping) and name in self.raw_reward_scales:
                value = self._merge_reward_scale(self.raw_reward_scales[name], value)
            normalized[name] = self._normalize_reward_scale(name, value)

        def first_weight(value):
            value = value[REWARD_TERRAIN_NAMES[0]] if isinstance(value, Mapping) else value
            return value[0] if isinstance(value, list) else value

        super()._apply_reward_scales({
            name: first_weight(value) for name, value in normalized.items()
        })
        for name, scales in normalized.items():
            if not isinstance(scales, (Mapping, list)):
                continue
            self.raw_reward_scales[name] = deepcopy(scales)
            self.reward_cfg["reward_scales"][name] = deepcopy(scales)
            if isinstance(scales, Mapping):
                scales = [scales[terrain] if isinstance(scales[terrain], list)
                          else [scales[terrain]] * len(MODE_NAMES) for terrain in REWARD_TERRAIN_NAMES]
            scale = torch.tensor(scales, dtype=gs.tc_float, device=self.device)
            self.reward_scales[name] = scale if name == "death" else scale * self.dt

    def _apply_terrain_curriculum(self, values):
        if "mode_probabilities" in values:
            validate_mode_probabilities(values["mode_probabilities"])
            self.task_cfg["mode_probabilities"] = list(values["mode_probabilities"])
        if "platform_enabled" in values:
            validate_platform_enabled(values["platform_enabled"])
            self.terrain.set_platform_enabled(values["platform_enabled"])
        if "platform_probabilities" in values:
            validate_platform_probabilities(values["platform_probabilities"])
            self.task_cfg["platform_probabilities"] = list(values["platform_probabilities"])

    def _sample_base_reset_state(self, env_ids):
        state = super()._sample_base_reset_state(env_ids)
        qpos, positions, quat, velocity, angular, gravity, height, centers, tiles = state
        modes = tiles.new_tensor(terrain_tile_modes(self.task_cfg))[tiles]
        self.height_assignment_epoch[env_ids] = tiles.new_tensor(self.terrain.controller.epochs)[modes]
        commands = self._sample_commands(self.command_cfg, self.commands_limit)[env_ids]
        minimum = commands.new_tensor(self.task_cfg["min_forward_speeds_m_s"])[modes]
        has_step = commands.new_tensor(terrain_tile_heights(self.task_cfg))[tiles] > 0.0
        commands[:, 0] = torch.where(has_step, torch.maximum(commands[:, 0], minimum), commands[:, 0])
        # 从静止实际加速；目标速度只写命令，不写入机器人速度。
        velocity.zero_()
        angular.zero_()
        self.commands[env_ids] = commands
        return state

    def _resample_commands(self, mask):
        # Sampled together with the reset state; only the height reference evolves.
        self._update_jump_commands()

    def _reset_task_buffers(self, mask):
        self._reset_jump_buffers(mask)
        for buffer in (self.jump_active, self.jump_steps, self.approach_ready_steps,
                       self.approach_duration_steps, self.approach_timed_out):
            buffer[mask] = 0
        self.jump_mode_one_hot[mask] = 0
        self.phase_features[mask] = 0
        self._update_jump_commands()

    def _episode_timeout(self):
        return self.jump_active & (self.jump_steps >= self.mode_episode_lengths[self.jump_mode])

    def remaining_episode_steps(self):
        return self.mode_episode_lengths[self.jump_mode] - self.jump_steps

    def _task_success(self):
        # 落稳成功可提前记录、奖励，但不能截短模式和相位输入。
        return self.task_success & self.jump_active & (self.jump_steps >= self.phase_input_steps[self.jump_mode])

    def _record_task_outcomes(self):
        if self.evaluation or not self.terrain.curriculum_active:
            return
        ids = (self.mask & self.reset_buf & (self.landing_surface_height > 0)).nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return
        # 曾经落稳、随后失稳或终止的回合仍算失败；助跑超时也进入分母。
        passed = self.task_success & ~self.terminated_buf & ~self.failure_seen
        rows = torch.stack((self.terrain_tile_index[ids], self.height_assignment_epoch[ids], passed[ids].long()), dim=1).cpu().tolist()
        groups = {}
        for tile, epoch, success in rows:
            mode, height = self.terrain.variants[tile]
            level = self.terrain.controller.heights[mode].index(height)
            key = mode, level, epoch
            wins, count = groups.get(key, (0, 0))
            groups[key] = wins + success, count + 1
        for (mode, level, epoch), (wins, count) in groups.items():
            self.terrain.controller.observe(mode, level, epoch, wins, count)

    def state_dict(self):
        return {"height_curriculum": self.terrain.controller.state_dict()}

    def load_state_dict(self, state):
        if "height_curriculum" in state:
            self.terrain.controller.load_state_dict(state["height_curriculum"])

    def episode_metrics(self, mask):
        episode = {
            "success": (self.success_seen & ~self.terminated_buf & ~self.failure_seen)[mask].float(),
            "peak_base_height_m": self.max_jump_base_height[mask],
            "peak_wheel_clearance_m": self.max_wheel_clearance[mask],
            "jump_started": self.jump_active[mask].float(),
            "approach_duration_s": torch.where(self.jump_active, self.approach_duration_steps, self.episode_length_buf)[mask] * self.dt,
        }
        metrics = dict(episode)
        # 用回合模式分组；输入窗口结束后 one-hot 会归零，但 jump_mode 保持不变。
        modes = self.jump_mode[mask]
        for mode, mode_name in enumerate(MODE_NAMES):
            selected = modes == mode
            if selected.any():
                metrics.update({f"mode/{mode_name}/{key}": value[selected] for key, value in episode.items()})
                steps = selected & (self.landing_surface_height[mask] > 0)
                if steps.any():
                    metrics[f"mode/{mode_name}/platform_success"] = (
                        self.task_success & ~self.terminated_buf & ~self.failure_seen
                    )[mask][steps].float()
                    metrics[f"mode/{mode_name}/platform_height_m"] = self.landing_surface_height[mask][steps]
                if self.terrain.curriculum_active:
                    controller = self.terrain.controller
                    prefix = f"mode/{mode_name}/height_curriculum/"
                    metrics[prefix + "level"] = controller.levels[mode] + 1
                    metrics[prefix + "height_m"] = controller.heights[mode][controller.levels[mode]]
                    metrics[prefix + "success_rate"] = controller.rates[mode]
                    metrics[prefix + "window_episodes"] = controller.counts[mode]
        return metrics
