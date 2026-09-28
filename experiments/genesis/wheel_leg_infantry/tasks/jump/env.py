"""平地 / 20 cm / 40 cm 条件跳跃；env 指定起跳距离。"""

from __future__ import annotations

import math

import genesis as gs
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from ...core.kinematics import compute_mean_base_to_wheel_bottom_distance
from ...core.tensor_utils import as_gain_tensor
from ..locomotion.env import LocomotionEnv
from .config import MODE_NAMES, PHASE_NAMES, validate_configs, validate_mode_probabilities, validate_platform_enabled
from .height_reference import sample_height_reference
from .geometry import active_step_heights, target_wheel_support, trigger_distance
from .terrain import JumpTerrain
from .observation import JUMP_POLICY_LAYOUT, LOCOMOTION_POLICY_LAYOUT, layout_dim
from .phase import phase_encoding, validate_phase_durations
from .reward_state import JumpRewardState
from .rewards import JumpRewards


class JumpEnv(JumpRewards, LocomotionEnv):
    """共享动作契约，以三维 one-hot 控制目标任务。"""

    policy_observation_layout = JUMP_POLICY_LAYOUT

    # ============ 初始化与状态定义 ============

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        curriculum_cfg=None,
        steps_per_iteration=24,
        show_viewer=False,
    ):
        # LocomotionEnv.__init__ 会调用子类 hook，因此这些属性必须提前存在。
        validate_configs(env_cfg, obs_cfg, curriculum_cfg)
        self.task_cfg = env_cfg["jump_modes"]
        self._viewer_enabled = show_viewer
        self._focused_jump = False
        # 批次级开关：False 为 teacher 预热，True 为 jump 采集；决定电机参数和训练时钟。
        self.collect_jump_data = False
        self.phase_durations = validate_phase_durations(env_cfg["jump_phase_durations_s"])
        self.phase_cycle_s = sum(self.phase_durations)
        if not math.isclose(self.phase_cycle_s, float(env_cfg["episode_length_s"]), abs_tol=1e-6):
            raise ValueError("jump phase durations must sum to env_cfg['episode_length_s']")
        if command_cfg.get("num_commands") != 3:
            raise ValueError("jump must retain locomotion [vx, wz, base_height] commands")

        super().__init__(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            curriculum_cfg=curriculum_cfg,
            steps_per_iteration=steps_per_iteration,
            show_viewer=show_viewer,
        )
        self._initialize_jump_motor_params()

    def _add_terrain(self):
        """父类构造钩子：创建平地与两条固定高度的平台通道。"""
        self.terrain = JumpTerrain(self.task_cfg)
        self.terrain_entity = self.terrain.add_to_scene(self.scene)
        self.friction_terrain_entities = self.terrain_entity

    def _constrain_joint_targets(self, target_joint_pos):
        # jump 允许越过几何目标限位，保留伸腿末段的 PD 位置误差和力矩。
        # teacher 预热仍遵循 locomotion 的动作契约。
        if self.collect_jump_data:
            return target_joint_pos
        return super()._constrain_joint_targets(target_joint_pos)

    def _requires_batched_motor_params(self):
        return self.env_cfg.get("jump_step_40cm_joint_kd") is not None

    def _initialize_jump_motor_params(self):
        self._jump_motor_params = {}
        self._locomotion_motor_params = {}
        counts = {
            "joint_kp": self.num_joints,
            "joint_kd": self.num_joints,
            "wheel_kd": self.num_wheels,
            "joint_force_limit": self.num_joints,
            "wheel_force_limit": self.num_wheels,
        }
        for name, value in self.env_cfg.get("jump_motor_params", {}).items():
            if name not in counts:
                raise ValueError(f"unknown jump_motor_params key: {name}")
            if value is None:
                continue
            tensor = as_gain_tensor(
                value,
                counts[name],
                f"jump_motor_params.{name}",
                device=self.device,
            )
            if tensor.shape != (counts[name],):
                raise ValueError(f"jump_motor_params.{name} must be a scalar or a flat list")
            if not torch.all(torch.isfinite(tensor)) or torch.any(tensor < 0):
                raise ValueError(f"jump_motor_params.{name} must be finite and nonnegative")
            if name.endswith("force_limit") and torch.any(tensor <= 0):
                raise ValueError(f"jump_motor_params.{name} must be positive")
            self._jump_motor_params[name] = tensor
            self._locomotion_motor_params[name] = self.domain_rand.nominal_motor_params[name].clone()

        self._step_40cm_joint_kd = None
        value = self.env_cfg.get("jump_step_40cm_joint_kd")
        if value is not None:
            tensor = as_gain_tensor(value, self.num_joints, "jump_step_40cm_joint_kd", device=self.device)
            if tensor.shape != (self.num_joints,) or not torch.all(torch.isfinite(tensor)) or torch.any(tensor < 0):
                raise ValueError("jump_step_40cm_joint_kd must be a finite nonnegative scalar or joint list")
            self._step_40cm_joint_kd = tensor
            # 即使公共 jump kd 为 None，其他任务及 teacher 也必须能恢复继承值。
            nominal = self.domain_rand.nominal_motor_params["joint_kd"]
            self._jump_motor_params.setdefault("joint_kd", nominal.clone())
            self._locomotion_motor_params.setdefault("joint_kd", nominal.clone())

    def _initialize_task_buffers(self):
        """集中声明 jump 状态；父类创建基础物理 buffer 后、首次 reset 前调用。

        bool 锁存量记录本回合历史，long 保存模式/计数，float 事件和门控供奖励相乘。
        除模式配置外，可重置的逐环境状态统一列于 _jump_episode_buffers()。
        """
        # 固定配置：只在构造时计算，不随 reset 清零。
        tolerance = float(self.reward_cfg["height_reference_tolerance_m"])
        sigma = float(self.reward_cfg["height_reference_sigma"])
        if not math.isfinite(tolerance) or tolerance < 0.0 or not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("height reference tolerance must be non-negative and sigma positive, both finite")
        max_tilt = float(self.env_cfg.get("jump_max_tilt_deg", 45.0))
        if not math.isfinite(max_tilt) or not 0.0 < max_tilt < 90.0:
            raise ValueError("jump_max_tilt_deg must be between 0 and 90 degrees")
        self.jump_min_upright_cos = math.cos(math.radians(max_tilt))
        self.required_stable_steps = math.ceil(self.task_cfg["landing_stable_time_s"] / self.dt)
        # 整数控制步避免阶段边界因浮点误差推迟一拍；仅调度起跳奖励和时间编码。
        self.phase_step_boundaries = torch.tensor(
            [math.ceil(sum(self.phase_durations[:i]) / self.dt - 1e-9) for i in range(1, len(PHASE_NAMES))],
            dtype=torch.long,
            device=self.device,
        )
        self.mode_height_references = [
            tuple(torch.tensor(values, dtype=gs.tc_float, device=self.device) for values in zip(*points))
            for points in self.task_cfg["height_references"]
        ]

        # 本轮任务：reset 时从地形通道确定；jump 期间冻结，不因提前终止而重新采样。
        self.jump_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.jump_mode_one_hot = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.base_height_target = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        self.landing_surface_height = torch.zeros_like(self.base_height_target)
        self.trigger_distance_m = torch.zeros_like(self.base_height_target)  # 开始 jump 时记录的起跳距离。

        # 实测几何与运动：每步覆盖；max_jump_base_height 为未经有效性筛选的原始峰值。
        # 奖励认可的峰值单独由 jump_reward_state 保存，失稳时可撤销，不能与原始峰值合并。
        self.wheel_center_pos = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=self.device)
        self.base_to_wheel_bottom_distance = torch.zeros_like(self.base_height_target)
        self.wheel_clearance = torch.zeros_like(self.base_height_target)
        self.max_wheel_clearance = torch.zeros_like(self.base_height_target)
        self.jump_base_height = torch.zeros_like(self.base_height_target)
        self.max_jump_base_height = torch.zeros_like(self.base_height_target)
        self.world_vertical_velocity = torch.zeros_like(self.base_height_target)
        self.world_vertical_acceleration = torch.zeros_like(self.base_height_target)
        self.previous_world_vertical_velocity = torch.zeros_like(self.base_height_target)  # 计算加速度。
        self.takeoff_contact_vertical_velocity = torch.zeros_like(self.base_height_target)  # 最后一拍支撑速度。
        self.takeoff_contact_base_height = torch.zeros_like(self.base_height_target)  # 与支撑速度同拍的 base 高度。
        self.last_airborne_vertical_velocity = torch.zeros_like(self.base_height_target)  # 触地前一拍速度。
        self.impact_vertical_speed = torch.zeros_like(self.base_height_target)  # 仅落地事件拍非零。

        # 物理阶段历史：首次起跳/落地后保持 True，直到下一次 reset；jump_stage 由二者推导。
        # has_taken_off 为物理离地，不等于奖励要求的“连续腾空达到最短时间”。
        self.has_taken_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_landed = torch.zeros_like(self.has_taken_off)
        self.phase_features = torch.empty((self.num_envs, 6), dtype=gs.tc_float, device=self.device)  # actor 时间编码。

        # 本拍奖励门控：每步重新计算，条件消失即归零，不记录历史。
        self.takeoff_gate = torch.zeros_like(self.base_height_target)  # 计划起跳期且尚有安全轮支撑。
        self.flight_gate = torch.zeros_like(self.base_height_target)  # 首次起跳后、首次触地前的有效腾空。
        self.jump_landing_gate = torch.zeros_like(self.base_height_target)  # 已落地且双轮有效支撑目标面。
        self.landing_airborne_gate = torch.zeros_like(self.base_height_target)  # 已落地后再次双轮离地。

        # 单拍事件：只在边沿发生时置 1，下一步重算；用于一次性奖励。
        self.jump_takeoff_event = torch.zeros_like(self.base_height_target)  # 首次物理离地。
        self.jump_landing_event = torch.zeros_like(self.base_height_target)  # 起跳后首次任意轮/机身触地。
        self.task_success_event = torch.zeros_like(self.base_height_target)  # 本回合第一次满足成功条件。
        self.task_failure_event = torch.zeros_like(self.base_height_target)  # 本回合第一次失败。

        # 失败历史：float 0/1 锁存，兼容奖励与日志；恢复姿态/接触不能抹掉记录。
        self.jump_invalid = torch.zeros_like(self.base_height_target)  # 过倾、机身触地、撞沿或错误落台。
        self.rebound_seen = torch.zeros_like(self.base_height_target)  # 曾在落地后再次双轮离地。

        # 成功判定与事件去重：task_success 是当前状态，会随失稳变 False；seen 保留回合历史。
        self.stable_landing_steps = torch.zeros_like(self.jump_mode)  # 连续落稳拍数，中断即清零。
        self.task_success = torch.zeros_like(self.has_taken_off)
        self.success_seen = torch.zeros_like(self.has_taken_off)  # 防止重新站稳后重复发成功奖励。
        self.failure_seen = torch.zeros_like(self.has_taken_off)  # 防止终止后的补齐步重复发失败奖励。

        # 采集终点：首次终止时锁存统计，补齐步仍仿真，但不能覆盖最终成绩。
        self.finished_jump = torch.zeros_like(self.has_taken_off)
        self.terminal_jump_stats = {}

        # 跨步奖励结算：管理连续腾空、合格起跳事件、有效峰值及失稳后的奖励追回。
        self.jump_reward_state = JumpRewardState(
            self.num_envs,
            dt=self.dt,
            min_airborne_time_s=float(self.env_cfg.get("takeoff_min_airborne_time_s", 0.10)),
            horizon=round(self.phase_cycle_s / self.dt),
            device=self.device,
            dtype=gs.tc_float,
        )

    @property
    def jump_stage(self):
        """物理阶段 0/1/2：起跳前、已起跳、已落地；不另存可与历史标志失同步的 buffer。"""
        return torch.where(self.has_landed, 2, self.has_taken_off.long())

    # ============ 每轮生命周期：teacher 预热 → jump 采集 → 统计结算 ============

    def prepare_locomotion_warmup(self, command_ranges=None):
        """重置物理环境并采样固定 teacher command；返回 32D teacher 观测。"""
        # teacher step 不推进 global_step，因此用已完成的 jump 步数在 warmup 前
        # 切换课程，保证新阶段 command 当轮生效，而不是到 jump 第一拍才更新。
        iteration = self.global_step // self.steps_per_iteration
        self.training_iteration = iteration
        if self.curriculum.update(iteration):
            print(
                f"[curriculum] stage={self.curriculum.current_stage_name} "
                f"iteration={iteration} step={self.global_step}"
            )
        self.set_collect_jump_data(False)
        self.reset()
        self.sample_locomotion_commands(command_ranges)
        return self.get_locomotion_observations()

    def sample_locomotion_commands(self, command_ranges=None):
        self._sample_locomotion_commands(command_ranges)
        ranges = command_ranges or self.env_cfg["locomotion_warmup"]["command_ranges"]
        lower, upper = map(float, ranges["lin_vel_range"])
        if lower < 0 or ranges["ang_vel_range"] != [0.0, 0.0]:
            raise ValueError("jump_modes currently supports forward approach with zero yaw command")
        minimum = self.commands.new_tensor(self.task_cfg["min_forward_speeds_m_s"])[self.jump_mode]
        minimum = minimum.clamp_min(lower)
        if torch.any(minimum > upper):
            raise ValueError("command range does not cover the minimum speed for selected step modes")
        self.commands[:, 0] = minimum + torch.rand_like(minimum) * (upper - minimum)
        if lower == 0:
            stationary = ((self.jump_mode == 0)
                          & (torch.rand_like(minimum) < self.task_cfg["flat_stationary_probability"]))
            self.commands[stationary, 0] = 0
        # 预热区足够远，不能在 teacher 准备完成前撞到固定台阶。
        warmup_travel = upper * self.env_cfg["locomotion_warmup"]["max_steps"] * self.dt
        if warmup_travel + 2.0 >= self.task_cfg["warmup_distance_m"]:
            raise ValueError("increase jump_modes.warmup_distance_m for this speed/warmup duration")
        trigger_distance(self.commands[:, 0], self.jump_mode, self.task_cfg["distance_tables"])
        return self.commands

    def _sample_locomotion_commands(self, command_ranges=None):
        """为每个并行环境采样一次 [vx, wz, base_height]，随后保持不变。"""
        if command_ranges is None:
            command_ranges = self.env_cfg["locomotion_warmup"]["command_ranges"]
        names = ("lin_vel_range", "ang_vel_range", "base_height_range")
        lower = []
        upper = []
        for name in names:
            limits = command_ranges[name]
            if len(limits) != 2 or not all(math.isfinite(float(value)) for value in limits):
                raise ValueError(f"{name} must contain two finite values, got {limits}")
            if limits[0] > limits[1]:
                raise ValueError(f"{name} lower bound must not exceed upper bound, got {limits}")
            lower.append(float(limits[0]))
            upper.append(float(limits[1]))
        lower_tensor = torch.tensor(lower, dtype=gs.tc_float, device=self.device)
        upper_tensor = torch.tensor(upper, dtype=gs.tc_float, device=self.device)
        self.commands.copy_(
            lower_tensor + torch.rand_like(self.commands) * (upper_tensor - lower_tensor)
        )
        return self.commands

    def get_locomotion_stability(self, warmup_cfg):
        """返回每个环境是否已跟稳固定 locomotion command 及其诊断量。"""
        forward_error = torch.abs(self.base_lin_vel[:, 0] - self.commands[:, 0])
        yaw_error = torch.abs(self.base_ang_vel[:, 2] - self.commands[:, 1])
        height_error = torch.abs(self.base_height - self.commands[:, 2])
        vertical_speed = torch.abs(self.world_vertical_velocity)
        sin_tilt = torch.linalg.vector_norm(self.projected_gravity[:, :2], dim=1).clamp(0.0, 1.0)
        tilt_deg = torch.rad2deg(torch.asin(sin_tilt))
        stable = (
            (forward_error <= float(warmup_cfg["forward_velocity_tolerance"]))
            & (yaw_error <= float(warmup_cfg["yaw_rate_tolerance"]))
            & (height_error <= float(warmup_cfg["base_height_tolerance"]))
            & (vertical_speed <= float(warmup_cfg["vertical_velocity_tolerance"]))
            & (tilt_deg <= float(warmup_cfg["tilt_tolerance_deg"]))
            & (self.base_contact < 0.5)
        )
        if warmup_cfg.get("require_both_wheels_contact", True):
            stable &= torch.all(self.wheel_contact > 0.5, dim=1)
        diagnostics = {
            "forward_error": forward_error,
            "yaw_error": yaw_error,
            "height_error": height_error,
            "vertical_speed": vertical_speed,
            "tilt_deg": tilt_deg,
        }
        return stable, diagnostics

    def begin_jump_rollout(self):
        """将稳速后的机器人平移到指定起跳点，保持运动连续，再开始 jump 采集。"""
        distance = trigger_distance(self.commands[:, 0], self.jump_mode, self.task_cfg["distance_tables"])
        jitter = (2 * torch.rand_like(distance) - 1) * self.task_cfg["distance_jitter_m"]
        distance = torch.where(self.jump_mode > 0, distance + jitter, 0.0)
        position = self.robot.get_pos().clone()
        position[:, 0] = -distance
        position[:, 1] = self.jump_mode * self.task_cfg["lane_spacing_m"]
        # 只平移 x/y，保留稳速后的高度、姿态、全部 DOF 速度和 action history。
        self.robot.set_pos(position, zero_velocity=False)
        self.base_pos = self.robot.get_pos()
        self.terrain_height.copy_(self.terrain.height_at(self.base_pos[:, :2]))
        self.base_height.copy_(self.base_pos[:, 2] - self.terrain_height)
        wheels = self.robot.get_links_pos(self.wheel_links_idx)
        overlaps = (self.landing_surface_height > 0) & torch.any(wheels[:, :, 0] + self.wheel_radius >= 0, dim=1)
        if torch.any(overlaps):
            raise ValueError("trigger distance puts a wheel inside the riser; increase distance table values")
        observations = self._start_jump_episode()
        self.trigger_distance_m.copy_(distance)
        if self._viewer_enabled and not self._focused_jump:
            self.focus_viewer()
            self._focused_jump = True
        return observations

    def _start_jump_episode(self):
        """不重置机器人，只在 locomotion -> jump 切换点开启新 PPO episode。"""
        self.set_collect_jump_data(False)
        self.episode_length_buf.zero_()
        self.reward_buf.zero_()
        self.reset_buf.zero_()
        self.terminated_buf.zero_()
        self.tilt_out_steps.zero_()
        self.base_contact_steps.zero_()
        self.landing_event.zero_()
        self.landing_penalty_steps_left.zero_()
        self.landing_penalty_gate.zero_()
        self.extras.clear()
        for value in self.episode_sums.values():
            value.zero_()
        for value in self.episode_metric_sums.values():
            value.zero_()

        # 保留 actions/last_actions，使开启动作延迟时第一拍仍连续。
        self._reset_task_buffers(None)
        current_world_velocity = self.robot.get_vel()
        self.world_vertical_velocity.copy_(current_world_velocity[:, 2])
        self.previous_world_vertical_velocity.copy_(current_world_velocity[:, 2])
        self.world_vertical_acceleration.zero_()
        self._update_jump_geometry()
        self.max_wheel_clearance.copy_(self.wheel_clearance.clamp_min(0.0))
        self.max_jump_base_height.copy_(self.jump_base_height.clamp_min(0.0))

        self.previous_both_wheels_contact.copy_(torch.all(self.wheel_contact > 0.5, dim=1))
        self.set_collect_jump_data(True)
        # 第一次 actor 观测就使用 t=0 参考；不把实测初始高度或接触状态写入指令。
        self._update_jump_commands()
        self._update_observations()
        return self.get_observations()

    def finish_jump_rollout(self, warmup_steps=None):
        """结算完整 jump 周期；提前终止的环境使用终止拍统计，然后切回 teacher 模式。"""
        episode = self._jump_episode_stats()
        if warmup_steps is not None:
            episode["warmup_steps"] = torch.full_like(self.reward_buf, float(warmup_steps))
        self.extras = {
            "episode": episode,
            # 这是技能自然终点，不是 timeout，因此不加 critic bootstrap。
            "time_outs": torch.zeros_like(self.reward_buf),
        }
        for value in self.episode_sums.values():
            value.zero_()
        for value in self.episode_metric_sums.values():
            value.zero_()
        self.set_collect_jump_data(False)
        return self.extras

    def set_collect_jump_data(self, enabled: bool):
        """切换训练边界；False 时 step 只运行 teacher，不推进 PPO 课程时钟。"""
        if bool(enabled) != self.collect_jump_data:
            profile = "_jump_motor_params" if enabled else "_locomotion_motor_params"
            params = dict(getattr(self, profile, {}))
            if enabled and getattr(self, "_step_40cm_joint_kd", None) is not None:
                step_40cm = self.jump_mode == MODE_NAMES.index("step_40cm")
                params["joint_kd"] = torch.where(
                    step_40cm[:, None], self._step_40cm_joint_kd, params["joint_kd"],
                )
            self._apply_motor_params(params)
        self.collect_jump_data = bool(enabled)

    # ============ 逐步更新：物理事件 → 任务判定 → 终止与奖励门控 ============

    def _update_task_state(self):
        """父类 step 钩子：先更新物理阶段，再判断目标面支撑与任务成败。"""
        self._update_jump_state()
        if not self.collect_jump_data:
            return
        cfg = self.task_cfg
        # 支撑只用于本拍落台判定，无需跨步保存。
        target_support = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.jump_mode, cfg, self.wheel_radius,
        ).all(dim=1)
        # 首次单轮触地可以开始正常缓冲，但侧壁/台阶下方接触不是有效落台。
        top_contact = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.jump_mode, cfg, self.wheel_radius, margin=0.0,
        ).any(dim=1)
        bad_landing = (self.landing_surface_height > 0) & (self.jump_landing_event > 0.5) & ~top_contact
        # 起跳前轮子低于台面并到达立面，属于撞沿，不能通过贴墙爬升刷跳跃奖励。
        wheel_x = self.wheel_center_pos[:, :, 0]
        wheel_bottom = self.wheel_center_pos[:, :, 2] - self.wheel_radius
        riser_hit = ((self.landing_surface_height > 0) & ~self.has_landed
                     & torch.any((wheel_x + self.wheel_radius >= 0)
                                 & (wheel_x < 0)
                                 & (wheel_bottom < self.landing_surface_height[:, None] - cfg["landing_height_tolerance_m"])
                                 & (self.wheel_contact > 0.5), dim=1))
        self.jump_invalid.copy_(torch.maximum(self.jump_invalid, (bad_landing | riser_hit).to(gs.tc_float)))
        self.jump_landing_gate.mul_(target_support)
        valid = (self.jump_invalid < 0.5) & (self.rebound_seen < 0.5)
        tilt_ok = -self.projected_gravity[:, 2] >= math.cos(math.radians(cfg["landing_max_tilt_deg"]))
        stable = (self.has_taken_off & target_support & valid & tilt_ok
                  & (self.world_vertical_velocity.abs() <= cfg["landing_max_vertical_speed_m_s"]))
        self.stable_landing_steps.copy_(torch.where(stable, self.stable_landing_steps + 1, 0))
        height_ok = ((self.landing_surface_height > 0)
                     | (self.jump_reward_state.peak_clearance >= self.base_height_target))
        self.task_success.copy_((self.stable_landing_steps >= self.required_stable_steps) & height_ok)
        self.task_success_event.copy_(self.task_success & ~self.success_seen)
        self.success_seen.logical_or_(self.task_success)
        failed = ~valid | ((self.episode_length_buf >= self.max_episode_length) & ~self.task_success)
        self.task_failure_event.copy_(failed & ~self.failure_seen)
        self.failure_seen.logical_or_(failed)

    def _update_jump_state(self):
        """按物理先后更新运动、起跳/落地事件、失败历史与奖励门控。"""
        elapsed_s = self.episode_length_buf.to(dtype=gs.tc_float) * self.dt
        scheduled_stage = torch.bucketize(
            self.episode_length_buf.to(dtype=torch.long).contiguous(),
            self.phase_step_boundaries, right=True,
        )
        self.phase_features.copy_(phase_encoding(elapsed_s, self.phase_cycle_s))
        current_vertical_velocity = self.robot.get_vel()[:, 2]
        self.world_vertical_acceleration.copy_(
            (current_vertical_velocity - self.previous_world_vertical_velocity) / self.dt
        )
        self.world_vertical_velocity.copy_(current_vertical_velocity)

        both_wheels_contact = torch.all(self.wheel_contact > 0.5, dim=1)
        no_wheels_contact = torch.all(self.wheel_contact <= 0.5, dim=1)
        safe_wheel_contact = both_wheels_contact & (self.base_contact < 0.5)
        self._update_jump_geometry()

        # 起跳接触期不依赖计划阶段：首次双轮完全离地前，只要至少一轮仍在
        # 安全接触，就同步锁存最后一个接触拍的 base-link vz 和高度。
        wheel_support = torch.any(self.wheel_contact > 0.5, dim=1) & (self.base_contact < 0.5)
        accumulate_takeoff = wheel_support & ~self.has_taken_off
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
            no_wheels_contact
            & ~self.has_taken_off
            & (self.base_contact < 0.5)
            & (self.takeoff_contact_vertical_velocity > 0.0)
        )
        self.jump_takeoff_event.copy_(takeoff_now.to(dtype=gs.tc_float))
        self.has_taken_off.logical_or_(takeoff_now)

        in_flight_before_update = self.has_taken_off & ~self.has_landed & no_wheels_contact
        self.last_airborne_vertical_velocity.copy_(
            torch.where(
                in_flight_before_update,
                current_vertical_velocity,
                self.last_airborne_vertical_velocity,
            )
        )
        # 第一次任意轮/机身触地就关闭跳跃，单轮落地后弹起不能接着累计高度。
        landing_now = (~no_wheels_contact | (self.base_contact > 0.5)) & self.has_taken_off & ~self.has_landed
        self.jump_landing_event.copy_(landing_now.to(dtype=gs.tc_float))
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
        invalid_now = (-self.projected_gravity[:, 2] < self.jump_min_upright_cos) | (self.base_contact > 0.5)
        self.jump_invalid.copy_(torch.maximum(self.jump_invalid, invalid_now.to(dtype=gs.tc_float)))
        valid_jump = self.jump_invalid < 0.5
        self.takeoff_gate.mul_(valid_jump)
        airborne = self.has_taken_off & ~self.has_landed & no_wheels_contact & (self.base_contact < 0.5) & valid_jump
        # 首次触地后再次双轮离地，立即判定二次起跳，不设高度容差。
        rebound = self.has_landed & no_wheels_contact
        self.landing_airborne_gate.copy_(rebound.to(dtype=gs.tc_float))
        self.rebound_seen.copy_(torch.maximum(self.rebound_seen, self.landing_airborne_gate))
        self.jump_reward_state.update(
            airborne,
            self.jump_base_height,
            self.jump_landing_event,
            self.episode_length_buf,
            takeoff_eligible=(self.takeoff_contact_vertical_velocity > 0.0) & valid_jump,
            valid_jump=valid_jump,
        )
        self.flight_gate.copy_(airborne.to(dtype=gs.tc_float))
        self.jump_landing_gate.copy_(
            (self.has_landed & safe_wheel_contact).to(dtype=gs.tc_float)
        )
        self._update_jump_commands()
        self.previous_world_vertical_velocity.copy_(current_vertical_velocity)

    def _update_jump_geometry(self):
        """分别更新机身跳高、腿部相对距离和轮底几何。"""
        # 所有通道都从 z=0 平地起跳；台阶上方也不扣台面高度。
        # 不复用 locomotion 的 base_height，因为它相对当前位置地表计算。
        self.jump_base_height.copy_(self.base_pos[:, 2])
        self.max_jump_base_height.copy_(torch.maximum(self.max_jump_base_height, self.jump_base_height))
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
        # 三种模式均从 z=0 的平地起跳；用较低一侧轮底衡量过障间隙。
        self.wheel_clearance.copy_(torch.min(wheel_bottom_height, dim=1).values)
        self.max_wheel_clearance.copy_(torch.maximum(self.max_wheel_clearance, self.wheel_clearance))

    def _update_jump_commands(self):
        """第三列为目标 base 到轮底平均竖直距离；参考仅由模式和时间决定。"""
        if not self.collect_jump_data:
            return
        self.commands[:, 1].zero_()
        elapsed = self.episode_length_buf.to(gs.tc_float) * self.dt
        for mode, (times, heights) in enumerate(self.mode_height_references):
            selected = self.jump_mode == mode
            self.commands[selected, 2] = sample_height_reference(elapsed[selected], times, heights)

    def _task_termination(self):
        """返回 jump 特有的终止原因，并区分正常任务失败与 solver 异常。"""
        rebound = (self.rebound_seen > 0.5) & self.collect_jump_data
        failed = (self.jump_invalid > 0.5) & self.collect_jump_data
        # solver error 始终需要 runner 抛错，即使同拍也发生正常任务失败。
        solver_error = self.scene.rigid_solver.get_error_envs_mask().bool()
        self.extras["jump_rebound_termination"] = rebound & ~solver_error
        self.extras["jump_task_termination"] = failed & ~solver_error
        return rebound | failed

    def _update_tracking_gate(self):
        # 跳跃奖励独立门控，不让 locomotion 的姿态/高度门抑制速度保持奖励。
        self.height_gate.fill_(1.0)
        self.attitude_gate.fill_(1.0)
        self.tracking_gate_raw.fill_(1.0)
        self.tracking_gate.fill_(1.0)

    def _landing_penalty_window_enabled(self):
        return False

    # ============ 重置与终止快照：保留终止拍，隔离后续补齐步 ============

    def _reset_idx(self, env_idx=None):
        if self.collect_jump_data and env_idx is not None:
            # 分阶段 runner 屏蔽终止后的 transition；保留终止状态和模式直到全轮结算。
            # 避免在同一 rollout 中换任务、清掉失败统计或把新回合混进 GAE。
            newly_finished = env_idx & ~self.finished_jump
            if torch.any(newly_finished):
                for name, value in self._current_jump_episode_stats().items():
                    snapshot = self.terminal_jump_stats.setdefault(name, torch.zeros_like(value))
                    snapshot[newly_finished] = value[newly_finished]
                self.finished_jump.logical_or_(newly_finished)
            self.extras.pop("episode", None)
            return
        super()._reset_idx(env_idx)

    def _reset_task_buffers(self, env_idx):
        """重置全部或布尔掩码选中的环境；既用于物理 reset，也用于预热到 jump 的切换。"""
        selected = slice(None) if env_idx is None else env_idx
        self.jump_reward_state.reset(env_idx)
        for buffer in self._jump_episode_buffers():
            buffer[selected] = 0
        # t=0 的六维编码并非全零；广播到选中的行，避免为局部 reset 同步 GPU 计数。
        self.phase_features[selected] = phase_encoding(
            torch.zeros(1, dtype=gs.tc_float, device=self.device), self.phase_cycle_s
        )

        # 本轮模式由地形通道给定；只在这里刷新配置派生量。
        self.jump_mode[selected] = self.terrain_tile_index[selected]
        self.jump_mode_one_hot[selected] = F.one_hot(self.jump_mode[selected], 3).to(gs.tc_float)
        targets = self.commands.new_tensor(self.task_cfg["clearance_targets_m"])
        heights = self.commands.new_tensor(active_step_heights(self.task_cfg))
        self.base_height_target[selected] = targets[self.jump_mode[selected]]
        self.landing_surface_height[selected] = heights[self.jump_mode[selected]]
        if env_idx is None:
            self.terminal_jump_stats.clear()
        else:
            for value in self.terminal_jump_stats.values():
                value[selected] = 0

    def _jump_episode_buffers(self):
        """需要在回合边界清零的逐环境状态；全量/局部 reset 共用这一份清单。"""
        return (
            # 实测量及用于差分/事件结算的历史量。
            self.wheel_center_pos, self.base_to_wheel_bottom_distance,
            self.wheel_clearance, self.max_wheel_clearance,
            self.jump_base_height, self.max_jump_base_height,
            self.world_vertical_velocity, self.world_vertical_acceleration,
            self.previous_world_vertical_velocity, self.takeoff_contact_vertical_velocity,
            self.takeoff_contact_base_height,
            self.last_airborne_vertical_velocity, self.impact_vertical_speed,
            # 物理阶段历史。
            self.has_taken_off, self.has_landed,
            # 本拍奖励门控。
            self.takeoff_gate, self.flight_gate, self.jump_landing_gate, self.landing_airborne_gate,
            # 单拍事件。
            self.jump_takeoff_event, self.jump_landing_event, self.task_success_event, self.task_failure_event,
            # 失败历史、成功判定与事件去重。
            self.jump_invalid, self.rebound_seen, self.stable_landing_steps,
            self.task_success, self.success_seen, self.failure_seen,
            # 起跳位置与采集终点。
            self.trigger_distance_m, self.finished_jump,
        )

    # ============ 课程与指令采样钩子 ============

    def _should_advance_training_clock(self):
        return self.collect_jump_data

    def _get_reward_scale(self, name):
        scale = self.reward_scales[name]
        # 每次结算按当前模式索引，reset 重新分配任务后立即使用对应权重。
        return scale[self.jump_mode] if isinstance(scale, torch.Tensor) else scale

    def _apply_reward_scales(self, values):
        """标量对所有任务生效；三元素列表按平地、20 cm、40 cm 分别设置。"""
        per_mode = {}
        for name, value in values.items():
            if isinstance(value, (list, tuple)):
                if len(value) != len(MODE_NAMES) or any(not math.isfinite(v) for v in value):
                    raise ValueError(f"reward scale {name} must contain three finite values")
                per_mode[name] = [float(v) for v in value]
        super()._apply_reward_scales({
            name: per_mode[name][0] if name in per_mode else value
            for name, value in values.items()
        })
        for name, scales in per_mode.items():
            self.raw_reward_scales[name] = scales
            self.reward_cfg["reward_scales"][name] = list(scales)
            scale = torch.tensor(scales, dtype=gs.tc_float, device=self.device)
            self.reward_scales[name] = scale if name == "death" else scale * self.dt

    def _apply_command_ranges(self, values):
        """课程只更新 locomotion command；jump 高度参考保持配置中的时间表。"""
        super()._apply_command_ranges(values)
        warmup_ranges = self.env_cfg["locomotion_warmup"]["command_ranges"]
        for name, limits in values.items():
            warmup_ranges[name] = list(limits)

    def _apply_terrain_curriculum(self, values):
        """更新下一次 reset 的模式抽样权重；已经开始的 rollout 保持原模式。"""
        if "platform_enabled" in values:
            validate_platform_enabled(values["platform_enabled"])
        if "mode_probabilities" in values:
            probabilities = values["mode_probabilities"]
            validate_mode_probabilities(probabilities)
            # 第 0 阶段在场景创建前应用；JumpTerrain 随后持有同一个 task_cfg。
            self.task_cfg["mode_probabilities"] = list(probabilities)
        if "platform_enabled" in values:
            if isinstance(getattr(self, "terrain", None), JumpTerrain):
                self.terrain.set_platform_enabled(values["platform_enabled"])
            else:
                # 父类先创建普通 TerrainManager；第 0 阶段只写配置，
                # 随后的 _add_terrain 才创建 JumpTerrain 和台阶实体。
                self.task_cfg["platform_enabled"] = list(values["platform_enabled"])
        terrain_values = {
            name: value for name, value in values.items()
            if name not in ("mode_probabilities", "platform_enabled")
        }
        if terrain_values:
            super()._apply_terrain_curriculum(terrain_values)

    def _resample_commands(self, envs_idx):
        # 一次 warmup 和 jump rollout 内都冻结 command；只允许显式全量采样。
        if self.collect_jump_data or envs_idx is not None:
            return
        super()._resample_commands(None)

    # ============ 观测：teacher 公共前缀、actor 输入与 critic 真值 ============

    def get_locomotion_observations(self):
        """返回 jump actor 的 32 维 locomotion 公共前缀。"""
        self._update_observations()
        names = tuple(name for name, _ in LOCOMOTION_POLICY_LAYOUT)
        if tuple(self.obs_components)[:len(names)] != names:
            raise RuntimeError("jump observation no longer starts with the locomotion checkpoint contract")
        width = layout_dim(LOCOMOTION_POLICY_LAYOUT)
        return TensorDict({"policy": self.obs_buf[:, :width]}, batch_size=[self.num_envs])

    def _update_observations(self):
        super()._update_observations()
        expected_names = tuple(name for name, _ in self.policy_observation_layout)
        if tuple(self.obs_components) != expected_names:
            raise RuntimeError(
                "jump actor observation order no longer matches the warm-start/deployment contract: "
                f"got={tuple(self.obs_components)}, expected={expected_names}"
            )
        expected_policy = int(self.obs_cfg["num_policy_obs"])
        expected_critic = int(self.obs_cfg["num_critic_obs"])
        if self.obs_buf.shape != (self.num_envs, expected_policy):
            raise RuntimeError(
                f"jump policy observation contract mismatch: got {tuple(self.obs_buf.shape)}, "
                f"expected {(self.num_envs, expected_policy)}"
            )
        if self.critic_obs_buf.shape != (self.num_envs, expected_critic):
            raise RuntimeError(
                f"jump critic observation contract mismatch: got {tuple(self.critic_obs_buf.shape)}, "
                f"expected {(self.num_envs, expected_critic)}"
            )

    def _get_task_observation_components(self):
        return {
            "last_actions": self.last_actions,  # 6
            "jump_phase": self.phase_features,  # 6
            "jump_mode": self.jump_mode_one_hot,  # 3
        }

    def _get_landing_privileged_observation_components(self):
        # Critic 读取本回合是否已落地的锁存状态；actor 不读取接触真值。
        return {
            "privileged_jump_landed": self.has_landed.unsqueeze(-1).to(dtype=gs.tc_float),
        }

    def _get_task_privileged_observation_components(self):
        distance = torch.where(self.jump_mode > 0, -self.base_pos[:, 0], 0.0).clamp(-3.0, 3.0)
        return {
            **self._get_jump_privileged_components(),
            "privileged_edge_distance": distance.unsqueeze(-1),
            "privileged_stable_landing": (self.stable_landing_steps / self.required_stable_steps).clamp(0, 1).unsqueeze(-1),
        }

    def _get_jump_privileged_components(self):
        return {
            "privileged_jump_base_height": (
                self.jump_base_height.unsqueeze(-1) * self.obs_scales["jump_base_height"]
            ),  # 1
            "privileged_world_vertical_velocity": (
                self.world_vertical_velocity.unsqueeze(-1) * self.obs_scales["vertical_velocity"]
            ),  # 1
            "privileged_world_vertical_acceleration": (
                self.world_vertical_acceleration.unsqueeze(-1) * self.obs_scales["vertical_acceleration"]
            ),  # 1
            "privileged_max_jump_base_height": (
                self.max_jump_base_height.unsqueeze(-1) * self.obs_scales["jump_base_height"]
            ),  # 1
            "privileged_jump_stage": F.one_hot(self.jump_stage, num_classes=len(PHASE_NAMES)).to(
                dtype=gs.tc_float
            ),  # 3
        }

    # ============ 回合统计与诊断 ============

    def _current_jump_episode_stats(self):
        """读取当前拍的回合统计；终止快照在 _jump_episode_stats 中覆盖。"""
        episode = {
            "reward_" + key: value.clone() for key, value in self.episode_sums.items()
        }
        if self.curriculum.enabled:
            episode["curriculum_stage"] = torch.full_like(
                self.reward_buf, float(self.curriculum.current_stage_index)
            )
        # 只保留可直接解释的回合级结果。
        episode["jump_invalid"] = self.jump_invalid.clone()
        episode["rebound_seen"] = self.rebound_seen.clone()
        episode["qualified_takeoff"] = self.jump_reward_state.takeoff_rewarded.to(dtype=gs.tc_float)
        episode["peak_jump_base_height_m"] = self.jump_reward_state.peak_clearance.clone()
        episode["max_airborne_duration_s"] = self.jump_reward_state.max_airborne_steps * self.dt
        episode["takeoff_contact_vertical_velocity_m_s"] = (
            self.takeoff_contact_vertical_velocity.clone()
        )
        episode["height_target_reached"] = (
            self.jump_reward_state.peak_clearance >= self.base_height_target
        ).to(dtype=gs.tc_float)
        episode["task_success"] = self.task_success.to(gs.tc_float)
        episode["trigger_distance_m"] = self.trigger_distance_m.clone()
        return episode

    def _jump_episode_stats(self):
        stats = self._current_jump_episode_stats()
        # 终止后的补齐步不参与 PPO，也不能污染回合奖励/峰值等诊断。
        for name, snapshot in self.terminal_jump_stats.items():
            stats[name] = torch.where(self.finished_jump, snapshot, stats[name])
        for mode, name in enumerate(MODE_NAMES):
            selected = self.jump_mode == mode
            # 每个子任务单独统计，不能用未被采样的环境充零稀释成功率。
            if torch.any(selected):
                stats[f"success_{name}"] = stats["task_success"][selected]
        return stats

    def get_jump_diagnostics(self, env_idx=0):
        """返回相位、高度参考、实际离地高度和事件；command_base_height 保留兼容键名。"""
        return {
            "phase_name": PHASE_NAMES[int(self.jump_stage[env_idx].item())],
            "base_height_target": self.base_height_target[env_idx].detach(),
            "mode": MODE_NAMES[int(self.jump_mode[env_idx])],
            "trigger_distance_m": self.trigger_distance_m[env_idx].detach(),
            "task_success": bool(self.task_success[env_idx]),
            "command_vx": self.commands[env_idx, 0].detach(),
            "command_wz": self.commands[env_idx, 1].detach(),
            "command_base_height": self.commands[env_idx, 2].detach(),
            "base_height": self.base_height[env_idx].detach(),
            "jump_base_height": self.jump_base_height[env_idx].detach(),
            "max_jump_base_height": self.max_jump_base_height[env_idx].detach(),
            "terrain_height": self.terrain_height[env_idx].detach(),
            "base_to_wheel_bottom_distance": self.base_to_wheel_bottom_distance[env_idx].detach(),
            "wheel_clearance": self.wheel_clearance[env_idx].detach(),
            "max_wheel_clearance": self.max_wheel_clearance[env_idx].detach(),
            "world_vertical_velocity": self.world_vertical_velocity[env_idx].detach(),
            "world_vertical_acceleration": self.world_vertical_acceleration[env_idx].detach(),
            "takeoff_contact_vertical_velocity": (
                self.takeoff_contact_vertical_velocity[env_idx].detach()
            ),
            "has_taken_off": bool(self.has_taken_off[env_idx].item()),
            "has_landed": bool(self.has_landed[env_idx].item()),
        }
