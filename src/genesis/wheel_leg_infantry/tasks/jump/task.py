"""条件跳跃任务；直接进入统一 PPO rollout，无 teacher 或独立 runner。"""

import math
import genesis as gs
import torch

from ...utils.kinematics import compute_mean_base_to_wheel_bottom_distance
from .approach import JumpApproach
from .terrain import JumpTerrain
from ..base import Task
from .buffers import JumpBuffers
from .config import MODE_NAMES, get_approach_cfg, validate_mode_probabilities, validate_platform_enabled
from .geometry import target_wheel_support
from .height_reference import sample_height_reference
from .phase import phase_encoding, validate_phase_durations
from .rewards import JumpRewards


class JumpTask(JumpApproach, JumpRewards, JumpBuffers, Task):
    def __init__(self, env, name, config, env_ids):
        self.approach_cfg = {**get_approach_cfg(), **config["env_cfg"].get("jump_approach", {})}
        self.approach_cfg.pop("reward_scales", None)  # 兼容旧存档，助跑奖励配置不再使用。
        self.task_cfg = config["env_cfg"]["jump_modes"]
        self.phase_durations = validate_phase_durations(config["env_cfg"]["jump_phase_durations_s"])
        self.phase_cycle_s = sum(self.phase_durations)
        super().__init__(env, name, config, env_ids)

    def _initialize_task_buffers(self):
        super()._initialize_task_buffers()
        self._initialize_approach_buffers()

    def create_terrain(self):
        # Keep task and terrain configuration identical after the defensive copy.
        self.task_cfg = self.env_cfg["jump_modes"]
        return JumpTerrain(self.task_cfg, self.approach_cfg)

    def _update_task_state(self):
        """父类 step 钩子：先更新物理阶段，再判断目标面支撑与任务成败。"""
        self.jump_steps.add_((self.mask & self.jump_active).long())
        self._update_approach_state()
        self._update_jump_state()
        cfg = self.task_cfg
        # 支撑只用于本拍落台判定，无需跨步保存。
        target_support = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.jump_mode, cfg, self.wheel_radius,
        ).all(dim=1)
        # 首次单轮触地可以开始正常缓冲，但侧壁/台阶下方接触不是有效落台。
        top_contact = target_wheel_support(
            self.wheel_center_pos, self.wheel_contact, self.jump_mode, cfg, self.wheel_radius, margin=0.0,
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
        self.jump_reward_state.update(
            self.flight_gate > 0.5, self.jump_base_height, self.jump_landing_event, self.jump_steps,
            takeoff_eligible=(self.takeoff_contact_vertical_velocity > 0.0) & valid,
            valid_jump=valid,
        )
        tilt_ok = -self.projected_gravity[:, 2] >= math.cos(math.radians(cfg["landing_max_tilt_deg"]))
        stable = (self.jump_active & self.has_taken_off & target_support & valid & tilt_ok & (self.base_contact < 0.5)
                  & (self.world_vertical_velocity.abs() <= cfg["landing_max_vertical_speed_m_s"]))
        self.stable_landing_steps.copy_(torch.where(stable, self.stable_landing_steps + 1, 0))
        # 无台阶模式的成功高度独立于奖励 target；有台阶时以实际落稳为准。
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
        self.phase_features.copy_(torch.where(self.jump_active[:, None], phase_encoding(elapsed_s, self.phase_cycle_s), 0.0))
        current_vertical_velocity = self.robot.get_vel()[:, 2]
        self.world_vertical_velocity.copy_(current_vertical_velocity)

        both_wheels_contact = torch.all(self.wheel_contact > 0.5, dim=1)
        no_wheels_contact = torch.all(self.wheel_contact <= 0.5, dim=1)
        safe_wheel_contact = both_wheels_contact & (self.base_contact < 0.5)
        self._update_jump_geometry()

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
            self.jump_active & no_wheels_contact
            & ~self.has_taken_off
            & (self.base_contact < 0.5)
            & (self.takeoff_contact_vertical_velocity > 0.0)
        )
        self.jump_takeoff_event.copy_(takeoff_now.to(dtype=gs.tc_float))
        self.has_taken_off.logical_or_(takeoff_now)

        # 机身先触地也必须冻结触地前速度，不能被本拍接触冲量覆盖。
        in_flight_before_update = (
            self.has_taken_off & ~self.has_landed & no_wheels_contact & (self.base_contact < 0.5)
        )
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
        invalid_now = self.jump_active & (-self.projected_gravity[:, 2] < self.jump_min_upright_cos)
        self.jump_invalid.copy_(torch.maximum(self.jump_invalid, invalid_now.to(dtype=gs.tc_float)))
        valid_jump = self.jump_invalid < 0.5
        self.takeoff_gate.mul_(valid_jump)
        airborne = self.has_taken_off & ~self.has_landed & no_wheels_contact & (self.base_contact < 0.5) & valid_jump
        # 首次触地后再次双轮离地，只记录并施加逐步惩罚，不判失败。
        rebound = self.has_landed & no_wheels_contact
        self.landing_airborne_gate.copy_(rebound.to(dtype=gs.tc_float))
        self.rebound_seen.copy_(torch.maximum(self.rebound_seen, self.landing_airborne_gate))
        self.flight_gate.copy_(airborne.to(dtype=gs.tc_float))
        self.jump_landing_gate.copy_(
            (self.has_landed & safe_wheel_contact).to(dtype=gs.tc_float)
        )
        self._update_jump_commands()

    def _update_jump_geometry(self):
        """分别更新机身跳高、腿部相对距离和轮底几何。"""
        # 所有通道都从 z=0 平地起跳；台阶上方也不扣台面高度。
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
        # 三种模式均从 z=0 的平地起跳；用较低一侧轮底衡量过障间隙。
        self.wheel_clearance.copy_(torch.min(wheel_bottom_height, dim=1).values)
        # 首次触地后冻结轮底成绩，二次弹跳不能补足首次跳跃高度。
        self.max_wheel_clearance.copy_(torch.where(
            self.jump_active & ~self.has_landed, torch.maximum(self.max_wheel_clearance, self.wheel_clearance), self.max_wheel_clearance,
        ))

    def _update_jump_commands(self):
        """第三列为目标 base 到轮底平均竖直距离；参考仅由模式和时间决定。"""
        self.commands[self.mask, 1] = 0.0
        self.commands[self.mask & ~self.jump_active, 2] = self.command_cfg["base_height_range"][0]
        elapsed = self.jump_steps.to(gs.tc_float) * self.dt
        for mode, (times, heights) in enumerate(self.mode_height_references):
            selected = self.mask & self.jump_active & (self.jump_mode == mode)
            self.commands[selected, 2] = sample_height_reference(elapsed[selected], times, heights)

    def _task_termination(self):
        """返回 jump 特有的终止原因，并区分正常任务失败与 solver 异常。"""
        # 只控制提前终止；失败历史及奖励仍由 _update_task_state 保留。
        # 二次腾空仅惩罚；旧存档的 on_rebound 字段不再生效。
        cfg = self.env_cfg.get("jump_termination", {})
        failed = self.mask & (((self.jump_invalid > 0.5) & cfg.get("on_invalid", True)) | self.approach_timed_out)
        # solver error 始终需要 runner 抛错，即使同拍也发生正常任务失败。
        solver_error = self.scene.rigid_solver.get_error_envs_mask().bool()
        self.extras["jump_rebound_termination"] = torch.zeros_like(failed)
        self.extras["jump_task_termination"] = failed & ~solver_error
        return failed

    def _get_reward_scale(self, name):
        scale = self.reward_scales[name]
        # 每次结算按当前模式索引，reset 重新分配任务后立即使用对应权重。
        scale = scale[self.jump_mode] if isinstance(scale, torch.Tensor) else scale
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

    def _apply_terrain_curriculum(self, values):
        if "mode_probabilities" in values:
            validate_mode_probabilities(values["mode_probabilities"])
            self.task_cfg["mode_probabilities"] = list(values["mode_probabilities"])
        if "platform_enabled" in values:
            validate_platform_enabled(values["platform_enabled"])
            self.terrain.set_platform_enabled(values["platform_enabled"])

    def _sample_base_reset_state(self, env_ids):
        state = super()._sample_base_reset_state(env_ids)
        qpos, positions, quat, velocity, angular, gravity, height, centers, modes = state
        commands = self._sample_commands(self.command_cfg, self.commands_limit)[env_ids]
        minimum = commands.new_tensor(self.task_cfg["min_forward_speeds_m_s"])[modes]
        commands[:, 0] = torch.maximum(commands[:, 0], minimum)
        stationary = (modes == 0) & (torch.rand(len(env_ids), device=self.device) < self.task_cfg["flat_stationary_probability"])
        commands[stationary, 0] = 0.0
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
        return self.jump_active & (self.jump_steps >= self.max_episode_length)

    def _task_success(self):
        return self.task_success

    def episode_metrics(self, mask):
        return {"success": self.success_seen[mask].float(),
                "peak_base_height_m": self.max_jump_base_height[mask],
                "peak_wheel_clearance_m": self.max_wheel_clearance[mask],
                "jump_started": self.jump_active[mask].float(),
                "approach_duration_s": torch.where(self.jump_active, self.approach_duration_steps, self.episode_length_buf)[mask] * self.dt}
