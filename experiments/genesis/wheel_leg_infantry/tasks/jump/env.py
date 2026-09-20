"""在 LocomotionEnv 上扩展的单次原地跳跃环境。"""

from __future__ import annotations

import math

import genesis as gs
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from ...core.kinematics import compute_mean_base_to_wheel_bottom_distance
from ...core.tensor_utils import as_gain_tensor
from ..locomotion.env import LocomotionEnv
from .config_25cm import PHASE_NAMES
from .height_reference import sample_height_reference, validate_height_reference
from .observation import JUMP_ESTIMATOR_LAYOUT, LOCOMOTION_ESTIMATOR_LAYOUT, layout_dim
from .phase import phase_encoding, validate_phase_durations
from .reward_state import JumpRewardState
from .rewards import JumpRewards


class JumpEnv(JumpRewards, LocomotionEnv):
    """复用 locomotion 动作/动力学契约，只增加跳跃任务状态与 MDP。"""

    # ============ 初始化：先准备父类 hook 依赖，再建立跳跃专用状态 ============

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
        self.collect_jump_data = False
        self.phase_durations = validate_phase_durations(env_cfg["jump_phase_durations_s"])
        self.phase_cycle_s = sum(self.phase_durations)
        if not math.isclose(self.phase_cycle_s, float(env_cfg["episode_length_s"]), abs_tol=1e-6):
            raise ValueError("jump phase durations must sum to env_cfg['episode_length_s']")
        if command_cfg.get("num_commands") != 3:
            raise ValueError("jump must retain locomotion [vx, wz, base_height] commands")
        self.wheel_clearance_target = float(env_cfg["wheel_clearance_target_m"])
        if not math.isfinite(self.wheel_clearance_target) or self.wheel_clearance_target <= 0.0:
            raise ValueError("wheel_clearance_target_m must be finite and positive")

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
        self.locomotion_handoff = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._teacher_commands = self.commands.clone()
        self._handoff_episode_stats = {}

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

    def _initialize_task_buffers(self):
        """创建 LocomotionEnv 不具备的跳跃参考、状态、事件和门控 buffer。"""
        # 高度指令只由连续时间参考轨迹生成，不依赖接触状态或物理阶段。
        times, heights = validate_height_reference(self.env_cfg["height_reference"], self.phase_cycle_s)
        self.height_reference_times = torch.tensor(times, dtype=gs.tc_float, device=self.device)
        self.height_reference_values = torch.tensor(heights, dtype=gs.tc_float, device=self.device)
        tolerance = float(self.reward_cfg["height_reference_tolerance_m"])
        sigma = float(self.reward_cfg["height_reference_sigma"])
        if not math.isfinite(tolerance) or tolerance < 0.0 or not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("height reference tolerance must be non-negative and sigma positive, both finite")
        # 阶段调度与跨步奖励状态。使用整数控制步，避免边界因浮点误差推迟一拍。
        self.phase_step_boundaries = torch.tensor(
            [math.ceil(sum(self.phase_durations[:i]) / self.dt - 1e-9) for i in (1, 2, 3)],
            dtype=torch.long,
            device=self.device,
        )
        self.jump_reward_state = JumpRewardState(
            self.num_envs,
            dt=self.dt,
            min_airborne_time_s=float(self.env_cfg.get("takeoff_min_airborne_time_s", 0.10)),
            horizon=round(self.phase_cycle_s / self.dt),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.takeoff_contact_vertical_velocity = torch.zeros(
            (self.num_envs,), dtype=gs.tc_float, device=self.device
        )
        self.phase_features = torch.empty((self.num_envs, 6), dtype=gs.tc_float, device=self.device)
        self.jump_stage = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.has_taken_off = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.has_landed = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # 世界系运动状态与轮地间隙。
        self.base_world_lin_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.world_vertical_velocity = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device)
        self.previous_world_vertical_velocity = torch.zeros_like(self.world_vertical_velocity)
        self.world_vertical_acceleration = torch.zeros_like(self.world_vertical_velocity)
        self.last_airborne_vertical_velocity = torch.zeros_like(self.world_vertical_velocity)
        self.wheel_center_pos = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=self.device)
        self.wheel_clearance = torch.zeros_like(self.world_vertical_velocity)
        self.max_wheel_clearance = torch.zeros_like(self.world_vertical_velocity)
        # 平地为 0；后续跳台阶时由地形/目标落脚点模块写入。
        self.takeoff_surface_height = torch.zeros_like(self.world_vertical_velocity)
        self.landing_surface_height = torch.zeros_like(self.world_vertical_velocity)
        self.impact_vertical_speed = torch.zeros_like(self.world_vertical_velocity)

        self.base_to_wheel_bottom_distance = torch.zeros_like(self.world_vertical_velocity)

        # 单拍事件、阶段奖励门控，以及需要跨步锁存的失败状态。
        self.crouch_gate = torch.zeros_like(self.world_vertical_velocity)
        self.crouch_airborne_gate = torch.zeros_like(self.world_vertical_velocity)
        self.takeoff_gate = torch.zeros_like(self.world_vertical_velocity)
        self.flight_gate = torch.zeros_like(self.world_vertical_velocity)
        self.jump_landing_gate = torch.zeros_like(self.world_vertical_velocity)
        self.jump_takeoff_event = torch.zeros_like(self.world_vertical_velocity)
        self.jump_landing_event = torch.zeros_like(self.world_vertical_velocity)
        self.jump_invalid = torch.zeros_like(self.world_vertical_velocity)
        self.landing_airborne_gate = torch.zeros_like(self.world_vertical_velocity)
        self.rebound_seen = torch.zeros_like(self.world_vertical_velocity)

        # 倾斜超过该阈值后，本回合永久标记为无效跳跃。
        max_tilt = float(self.env_cfg.get("jump_max_tilt_deg", 45.0))
        if not math.isfinite(max_tilt) or not 0.0 < max_tilt < 90.0:
            raise ValueError("jump_max_tilt_deg must be between 0 and 90 degrees")
        self.jump_min_upright_cos = math.cos(math.radians(max_tilt))

    # ============ 运行时 hook：随 step/reset 更新跳跃状态、观测和终止条件 ============

    def _update_wheel_clearance(self):
        """更新轮底离地间隙，以及机身到两侧轮底的平均距离。"""
        self.wheel_center_pos.copy_(self.robot.get_links_pos(self.wheel_links_idx))
        wheel_bottom_height = self.wheel_center_pos[:, :, 2] - self.wheel_radius
        # 两侧距离先各自以世界 z 方向测量，再取平均；它描述机体相对轮底的高度，
        # 与左右虚拟腿长不是同一个物理量。
        self.base_to_wheel_bottom_distance.copy_(
            compute_mean_base_to_wheel_bottom_distance(
                self.base_pos,
                self.wheel_center_pos,
                self.wheel_radius,
            )
        )
        # 用较低一侧轮子作为过障间隙，避免倾斜时只有一个轮子到达目标。
        self.wheel_clearance.copy_(
            torch.min(wheel_bottom_height - self.takeoff_surface_height[:, None], dim=1).values
        )
        self.max_wheel_clearance.copy_(torch.maximum(self.max_wheel_clearance, self.wheel_clearance))

    def _update_task_state(self):
        elapsed_s = self.episode_length_buf.to(dtype=gs.tc_float) * self.dt
        scheduled_stage = torch.bucketize(
            self.episode_length_buf.to(dtype=torch.long).contiguous(),
            self.phase_step_boundaries, right=True,
        )
        self.phase_features.copy_(phase_encoding(elapsed_s, self.phase_cycle_s))
        self.base_world_lin_vel.copy_(self.robot.get_vel())
        current_vertical_velocity = self.base_world_lin_vel[:, 2]
        self.world_vertical_acceleration.copy_(
            (current_vertical_velocity - self.previous_world_vertical_velocity) / self.dt
        )
        self.world_vertical_velocity.copy_(current_vertical_velocity)

        both_wheels_contact = torch.all(self.wheel_contact > 0.5, dim=1)
        no_wheels_contact = torch.all(self.wheel_contact <= 0.5, dim=1)
        safe_wheel_contact = both_wheels_contact & (self.base_contact < 0.5)
        self._update_wheel_clearance()

        # 起跳接触期不依赖计划阶段：首次双轮完全离地前，只要至少一轮仍在
        # 安全接触，就锁存最后一个接触拍的 base-link vz。
        wheel_support = torch.any(self.wheel_contact > 0.5, dim=1) & (self.base_contact < 0.5)
        accumulate_takeoff = wheel_support & ~self.has_taken_off
        self.takeoff_gate.copy_(accumulate_takeoff.to(dtype=gs.tc_float))
        # 下蹲跟踪需要向下运动，不能同时用起跳奖励惩罚负 vz。
        # 接触拍速度锁存仍按物理状态更新，只限制稠密奖励的生效时间。
        self.takeoff_gate.mul_((scheduled_stage == 1).to(dtype=gs.tc_float))
        self.takeoff_contact_vertical_velocity.copy_(
            torch.where(
                accumulate_takeoff,
                current_vertical_velocity,
                self.takeoff_contact_vertical_velocity,
            )
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
            self.wheel_clearance,
            self.jump_landing_event,
            self.episode_length_buf,
            takeoff_eligible=(self.takeoff_contact_vertical_velocity > 0.0) & valid_jump,
            valid_jump=valid_jump,
        )
        # policy 只读取连续时间编码；该物理阶段只供奖励门控、critic 与诊断。
        pre_takeoff_stage = torch.minimum(scheduled_stage, torch.ones_like(scheduled_stage))
        physical_stage = torch.where(self.has_taken_off, torch.full_like(scheduled_stage, 2), pre_takeoff_stage)
        physical_stage = torch.where(self.has_landed, torch.full_like(scheduled_stage, 3), physical_stage)
        self.jump_stage.copy_(physical_stage)

        scheduled_crouch = scheduled_stage == 0
        self.crouch_gate.copy_((scheduled_crouch & safe_wheel_contact).to(dtype=gs.tc_float))
        # 离地惩罚按计划 crouch 时段门控，不受物理起跳或机身接触影响。
        self.crouch_airborne_gate.copy_((scheduled_crouch & no_wheels_contact).to(dtype=gs.tc_float))
        self.flight_gate.copy_(airborne.to(dtype=gs.tc_float))
        self.jump_landing_gate.copy_(
            (self.has_landed & safe_wheel_contact).to(dtype=gs.tc_float)
        )
        self._update_jump_commands()
        self.previous_world_vertical_velocity.copy_(current_vertical_velocity)

    def _update_jump_commands(self):
        """冻结 vx、保持 wz=0；新配置仅按计时器生成距离参考。"""
        if not self.collect_jump_data:
            return
        self.commands[:, 1].zero_()
        elapsed_s = self.episode_length_buf.to(dtype=self.commands.dtype) * self.dt
        self.commands[:, 2].copy_(
            sample_height_reference(
                elapsed_s,
                self.height_reference_times,
                self.height_reference_values,
            )
        )
        if hasattr(self, "locomotion_handoff"):
            self.commands[self.locomotion_handoff] = self._teacher_commands[self.locomotion_handoff]

    def _reset_task_buffers(self, env_idx):
        # 跳过 crouch 时，初始状态直接从 takeoff 开始。
        initial_stage = int(self.phase_durations[0] == 0.0)
        self.jump_reward_state.reset(env_idx)
        if env_idx is None:
            self.takeoff_contact_vertical_velocity.zero_()
        else:
            self.takeoff_contact_vertical_velocity.masked_fill_(env_idx, 0.0)
        if env_idx is None:
            self.phase_features.copy_(
                phase_encoding(torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device), self.phase_cycle_s)
            )
            self.jump_stage.fill_(initial_stage)
            self.has_taken_off.zero_()
            self.has_landed.zero_()
            self.base_world_lin_vel.zero_()
            self.world_vertical_velocity.zero_()
            self.previous_world_vertical_velocity.zero_()
            self.world_vertical_acceleration.zero_()
            self.last_airborne_vertical_velocity.zero_()
            self.wheel_center_pos.zero_()
            self.wheel_clearance.zero_()
            self.max_wheel_clearance.zero_()
            self.takeoff_surface_height.zero_()
            self.landing_surface_height.zero_()
            self.impact_vertical_speed.zero_()
            self.base_to_wheel_bottom_distance.zero_()
            for gate in self._jump_event_and_gate_buffers():
                gate.zero_()
            return

        self.phase_features[env_idx] = phase_encoding(
            torch.zeros((int(env_idx.sum().item()),), dtype=gs.tc_float, device=self.device), self.phase_cycle_s
        )
        self.jump_stage.masked_fill_(env_idx, initial_stage)
        self.has_taken_off.masked_fill_(env_idx, False)
        self.has_landed.masked_fill_(env_idx, False)
        self.base_world_lin_vel.masked_fill_(env_idx[:, None], 0.0)
        self.world_vertical_velocity.masked_fill_(env_idx, 0.0)
        self.previous_world_vertical_velocity.masked_fill_(env_idx, 0.0)
        self.world_vertical_acceleration.masked_fill_(env_idx, 0.0)
        self.last_airborne_vertical_velocity.masked_fill_(env_idx, 0.0)
        self.wheel_center_pos.masked_fill_(env_idx[:, None, None], 0.0)
        self.wheel_clearance.masked_fill_(env_idx, 0.0)
        self.max_wheel_clearance.masked_fill_(env_idx, 0.0)
        self.takeoff_surface_height.masked_fill_(env_idx, 0.0)
        self.landing_surface_height.masked_fill_(env_idx, 0.0)
        self.impact_vertical_speed.masked_fill_(env_idx, 0.0)
        self.base_to_wheel_bottom_distance.masked_fill_(env_idx, 0.0)
        for gate in self._jump_event_and_gate_buffers():
            gate.masked_fill_(env_idx, 0.0)

    def _jump_event_and_gate_buffers(self):
        return (
            self.crouch_gate,
            self.crouch_airborne_gate,
            self.takeoff_gate,
            self.flight_gate,
            self.jump_landing_gate,
            self.jump_takeoff_event,
            self.jump_landing_event,
            self.jump_invalid,
            self.landing_airborne_gate,
            self.rebound_seen,
        )

    def _task_termination(self):
        terminated = (self.rebound_seen > 0.5) & self.collect_jump_data
        if hasattr(self, "locomotion_handoff"):
            # 已交接给 teacher 的环境不再适用 jump 的二次起跳死亡规则。
            terminated = terminated & ~self.locomotion_handoff
        # reset 会清空任务状态，runner 需要保留这一拍的死亡原因。
        self.extras["jump_rebound_termination"] = terminated.clone()
        return terminated

    def _update_tracking_gate(self):
        # 跳跃奖励独立门控，不让 locomotion 的姿态/高度门抑制速度保持奖励。
        self.height_gate.fill_(1.0)
        self.attitude_gate.fill_(1.0)
        self.tracking_gate_raw.fill_(1.0)
        self.tracking_gate.fill_(1.0)

    def _get_task_observation_components(self):
        return {
            "last_actions": self.last_actions,  # 6
            "jump_phase": self.phase_features,  # 6
        }

    def _landing_penalty_window_enabled(self):
        return False

    def _get_landing_privileged_observation_components(self):
        # Critic 读取本回合是否已落地的锁存状态；actor 不读取接触真值。
        return {
            "privileged_jump_landed": self.has_landed.unsqueeze(-1).to(dtype=gs.tc_float),
        }

    def _get_task_privileged_observation_components(self):
        return {
            "privileged_wheel_clearance": (
                self.wheel_clearance.unsqueeze(-1) * self.obs_scales["wheel_clearance"]
            ),  # 1
            "privileged_world_vertical_velocity": (
                self.world_vertical_velocity.unsqueeze(-1) * self.obs_scales["vertical_velocity"]
            ),  # 1
            "privileged_world_vertical_acceleration": (
                self.world_vertical_acceleration.unsqueeze(-1) * self.obs_scales["vertical_acceleration"]
            ),  # 1
            "privileged_max_wheel_clearance": (
                self.max_wheel_clearance.unsqueeze(-1) * self.obs_scales["wheel_clearance"]
            ),  # 1
            "privileged_jump_stage": F.one_hot(self.jump_stage, num_classes=len(PHASE_NAMES)).to(
                dtype=gs.tc_float
            ),  # 4
        }

    def _update_observations(self):
        super()._update_observations()
        expected_names = tuple(name for name, _ in JUMP_ESTIMATOR_LAYOUT)
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

    def _apply_command_ranges(self, values):
        """课程只更新 locomotion command；jump 高度参考保持配置中的时间表。"""
        super()._apply_command_ranges(values)
        warmup_ranges = self.env_cfg["locomotion_warmup"]["command_ranges"]
        for name, limits in values.items():
            warmup_ranges[name] = list(limits)

    def _resample_commands(self, envs_idx):
        # 一次 warmup 和 jump rollout 内都冻结 command；只允许显式全量采样。
        if self.collect_jump_data or envs_idx is not None:
            return
        super()._resample_commands(None)

    def _should_advance_training_clock(self):
        return self.collect_jump_data

    # ============ 分阶段训练：locomotion warmup、jump rollout 与落地交接 ============

    def set_collect_jump_data(self, enabled: bool):
        """切换训练边界；False 时 step 只运行 teacher，不推进 PPO 课程时钟。"""
        if bool(enabled) != self.collect_jump_data:
            profile = "_jump_motor_params" if enabled else "_locomotion_motor_params"
            self._apply_motor_params(getattr(self, profile, {}))
        self.collect_jump_data = bool(enabled)

    def sample_locomotion_commands(self, command_ranges=None):
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

    def prepare_locomotion_warmup(self, command_ranges=None):
        """重置物理环境并采样固定 teacher command；返回 33D teacher 观测。"""
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

    def get_locomotion_observations(self):
        """返回 jump actor 的 33 维 locomotion 公共前缀。"""
        self._update_observations()
        names = tuple(name for name, _ in LOCOMOTION_ESTIMATOR_LAYOUT)
        if tuple(self.obs_components)[:len(names)] != names:
            raise RuntimeError("jump observation no longer starts with the locomotion checkpoint contract")
        width = layout_dim(LOCOMOTION_ESTIMATOR_LAYOUT)
        return TensorDict({"policy": self.obs_buf[:, :width]}, batch_size=[self.num_envs])

    def get_locomotion_stability(self, warmup_cfg):
        """返回每个环境是否已跟稳固定 locomotion command 及其诊断量。"""
        forward_error = torch.abs(self.base_lin_vel[:, 0] - self.commands[:, 0])
        yaw_error = torch.abs(self.base_ang_vel[:, 2] - self.commands[:, 1])
        height_error = torch.abs(self.base_pos[:, 2] - self.commands[:, 2])
        vertical_speed = torch.abs(self.base_world_lin_vel[:, 2])
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
        """不重置机器人，只在 locomotion -> jump 切换点开启新 PPO episode。"""
        self.set_collect_jump_data(False)
        if hasattr(self, "locomotion_handoff"):
            self.locomotion_handoff.zero_()
            self._teacher_commands.copy_(self.commands)
            self._handoff_episode_stats.clear()
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
        self.base_world_lin_vel.copy_(current_world_velocity)
        self.world_vertical_velocity.copy_(current_world_velocity[:, 2])
        self.previous_world_vertical_velocity.copy_(current_world_velocity[:, 2])
        self.world_vertical_acceleration.zero_()
        self._update_wheel_clearance()
        self.max_wheel_clearance.copy_(self.wheel_clearance.clamp_min(0.0))

        self.previous_both_wheels_contact.copy_(torch.all(self.wheel_contact > 0.5, dim=1))
        self.set_collect_jump_data(True)
        # 第一次 actor 观测就使用 t=0 参考；不把实测初始高度或接触状态写入指令。
        self._update_jump_commands()
        self._update_observations()
        return self.get_observations()

    def ready_for_locomotion(self):
        """返回已经起跳并重新双轮着地、可以交还 teacher 的环境。"""
        return self.has_taken_off & torch.all(self.wheel_contact > 0.5, dim=1)

    def handoff_landed_environments(self):
        """逐环境交接；不重置物理状态，下一拍开始运行 teacher。"""
        landed = self.ready_for_locomotion() & ~self.locomotion_handoff
        indices = landed.nonzero(as_tuple=False).flatten()
        if indices.numel():
            for name, value in self._jump_episode_stats().items():
                snapshot = self._handoff_episode_stats.setdefault(name, torch.zeros_like(value))
                snapshot[landed] = value[landed]
            self._apply_motor_params(self._locomotion_motor_params, envs_idx=indices)
            self.locomotion_handoff.logical_or_(landed)
            self.commands[landed] = self._teacher_commands[landed]
        return landed

    def _jump_episode_stats(self):
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
        episode["peak_wheel_clearance_m"] = self.jump_reward_state.peak_clearance.clone()
        episode["max_airborne_duration_s"] = self.jump_reward_state.max_airborne_steps * self.dt
        episode["takeoff_contact_vertical_velocity_m_s"] = (
            self.takeoff_contact_vertical_velocity.clone()
        )
        episode["height_target_reached"] = (
            self.jump_reward_state.peak_clearance >= self.wheel_clearance_target
        ).to(dtype=gs.tc_float)
        return episode

    def finish_jump_rollout(self, warmup_steps=None):
        """结算技能终点；已交接环境使用落地拍的统计，不混入 teacher 后续行为。"""
        episode = self._jump_episode_stats()
        for name, value in getattr(self, "_handoff_episode_stats", {}).items():
            episode[name] = torch.where(self.locomotion_handoff, value, episode[name])
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

    # ============ 诊断接口：只读取状态，不推进环境 ============

    def get_jump_diagnostics(self, env_idx=0):
        """返回 eval overlay/终端需要的相位、目标高度和事件状态。"""
        return {
            "phase_name": PHASE_NAMES[int(self.jump_stage[env_idx].item())],
            "wheel_clearance_target": self.wheel_clearance_target,
            "command_vx": self.commands[env_idx, 0].detach(),
            "command_wz": self.commands[env_idx, 1].detach(),
            "command_base_height": self.commands[env_idx, 2].detach(),
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
