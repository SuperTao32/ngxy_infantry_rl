"""跳跃独占的命令、物理事件、峰值资格和奖励结算状态。"""

import math

import torch

from .config import MODE_NAMES
from .phase import phase_encoding
from .reward_state import JumpRewardState


class JumpBuffers:
    def __init__(self, env, config, phase_durations):
        self.config = config
        self.dt = env.dt
        self.takeoff_steps = math.ceil(phase_durations[0] / env.dt - 1e-9)
        self.min_upright_cos = math.cos(math.radians(config["jump_max_tilt_deg"]))
        self.mode_targets = env.commands.new_tensor(config["clearance_targets_m"])
        self.mode_success_heights = env.commands.new_tensor(config["ground_success_base_height_m"])
        self.mode_success_clearances = env.commands.new_tensor(config["ground_success_wheel_clearance_m"])

        bool_names = (
            "active", "was_active", "just_completed", "has_taken_off", "has_landed",
            "success_seen", "failure_seen", "landing_support", "rebound_seen",
        )
        long_names = ("elapsed_steps", "mode", "flight_airtime_steps")
        float_names = (
            "height_reference", "forward_velocity", "ground_height", "base_height_target",
            "jump_base_height", "wheel_clearance", "max_wheel_clearance",
            "takeoff_contact_vertical_velocity", "takeoff_contact_base_height",
            "last_airborne_vertical_velocity", "base_link_acceleration_sq", "takeoff_gate", "flight_gate",
            "jump_landing_gate", "landing_airborne_gate", "jump_landing_event", "jump_invalid",
            "task_success_event", "task_failure_event",
        )
        for names, dtype in ((bool_names, torch.bool), (long_names, torch.long), (float_names, env.commands.dtype)):
            for name in names:
                setattr(self, name, torch.zeros(env.num_envs, dtype=dtype, device=env.device))
        self.pending_mode = torch.full_like(self.mode, -1)
        self.locomotion_commands = torch.zeros_like(env.commands)
        self.wheel_airborne = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self.base_link_velocity = env.commands.new_zeros((env.num_envs, 3))
        self._episode_buffers = tuple(getattr(self, name) for name in (*bool_names, *long_names, *float_names)) + (
            self.locomotion_commands, self.wheel_airborne, self.base_link_velocity,
        )
        self.jump_reward_state = JumpRewardState(
            env.num_envs, dt=env.dt, min_airborne_time_s=config["takeoff_min_airborne_time_s"],
            device=env.device, dtype=env.commands.dtype,
        )

    def reset(self, mask):
        for value in self._episode_buffers:
            value[mask] = 0
        self.pending_mode[mask] = -1
        self.jump_reward_state.reset(mask)

    def start(self, env, mask, modes, previous_vx):
        """缓存本次新命令，并锁存起跳地面和上一条命令的 vx。"""
        self.reset(mask)
        self.mode[mask] = modes
        self.active[mask] = True
        self.locomotion_commands[mask] = env.commands[mask]
        self.forward_velocity[mask] = previous_vx
        self.ground_height[mask] = env.terrain_height[mask]
        self.base_height_target[mask] = self.mode_targets[modes]
        velocity = env.robot.get_vel()[mask]
        self.base_link_velocity[mask] = velocity
        self.takeoff_contact_base_height[mask] = env.base_pos[mask, 2] - self.ground_height[mask]
        support = (env.wheel_contact[mask] > 0.5).any(dim=1) & (env.base_contact[mask] < 0.5)
        self.takeoff_contact_vertical_velocity[mask] = torch.where(support, velocity[:, 2], 0.0)

    def update(self, env, duration_steps):
        self.was_active.copy_(self.active)
        for event in (
            self.just_completed, self.jump_landing_event, self.task_success_event, self.task_failure_event,
            self.takeoff_gate, self.flight_gate, self.jump_landing_gate, self.landing_airborne_gate,
        ):
            event.zero_()
        if not self.active.any():
            return
        self.elapsed_steps.add_(self.active)
        self.just_completed.copy_(self.active & (self.elapsed_steps >= duration_steps[self.mode]))
        velocity = env.robot.get_vel()
        self.base_link_acceleration_sq.copy_(((velocity - self.base_link_velocity) / self.dt).square().sum(dim=1))
        self.base_link_velocity.copy_(velocity)
        self.jump_base_height.copy_(env.base_pos[:, 2] - self.ground_height)
        wheel_center_pos = env.robot.get_links_pos(env.wheel_links_idx)
        wheel_bottom = wheel_center_pos[..., 2] - env.wheel_radius
        ground = env.terrain.height_at(wheel_center_pos[..., :2].reshape(-1, 2)).reshape(wheel_bottom.shape)
        gap = wheel_bottom - ground
        self.wheel_airborne.copy_((env.wheel_contact <= 0.5) & (gap > self.config["takeoff_min_wheel_clearance_m"]))
        wheel_touching = (env.wheel_contact > 0.5) | (gap <= 0.0)
        airborne = self.wheel_airborne.all(dim=1)
        self.wheel_clearance.copy_((wheel_bottom - self.ground_height[:, None]).amin(dim=1))
        # 仅平地支撑：轮底回到本次起跳地面高度，不接入旧版台阶通道/落台几何。
        target_support = (
            (env.wheel_contact > 0.5)
            & ((wheel_bottom - self.ground_height[:, None]).abs() <= self.config["landing_height_tolerance_m"])
        )
        self.landing_support.copy_(target_support.any(dim=1))
        body_clear = env.base_contact < 0.5
        supporting = self.active & (env.wheel_contact > 0.5).any(dim=1) & body_clear & ~self.has_taken_off
        self.takeoff_contact_vertical_velocity.copy_(torch.where(supporting, velocity[:, 2], self.takeoff_contact_vertical_velocity))
        self.takeoff_contact_base_height.copy_(torch.where(supporting, self.jump_base_height, self.takeoff_contact_base_height))
        self.takeoff_gate.copy_(supporting & (self.elapsed_steps < self.takeoff_steps))
        takeoff = self.active & airborne & body_clear & ~self.has_taken_off & (self.takeoff_contact_vertical_velocity > 0.0)
        self.has_taken_off.logical_or_(takeoff)
        in_flight = self.active & self.has_taken_off & ~self.has_landed & ~wheel_touching.any(dim=1) & body_clear
        self.last_airborne_vertical_velocity.copy_(torch.where(in_flight, velocity[:, 2], self.last_airborne_vertical_velocity))
        landing = self.active & self.has_taken_off & ~self.has_landed & (wheel_touching.any(dim=1) | ~body_clear)
        self.jump_landing_event.copy_(landing)
        self.has_landed.logical_or_(landing)
        invalid = self.active & (-env.projected_gravity[:, 2] < self.min_upright_cos)
        self.jump_invalid.copy_(torch.maximum(self.jump_invalid, invalid.to(self.jump_invalid.dtype)))
        valid = self.jump_invalid < 0.5
        self.takeoff_gate.mul_(valid)
        self.flight_gate.copy_(self.active & self.has_taken_off & ~self.has_landed & airborne & body_clear & valid)
        self.jump_landing_gate.copy_(self.active & self.has_landed & target_support.all(dim=1) & body_clear)
        self.landing_airborne_gate.copy_(self.active & self.has_landed & ~wheel_touching.all(dim=1))
        # 二次起跳历史供奖励使用，首次落地后再次双轮离地即锁存，不能当作首次腾空补成绩。
        self.rebound_seen.logical_or_(self.active & self.has_landed & airborne & body_clear)
        self.flight_airtime_steps.add_((self.flight_gate > 0.5).long())
        self.max_wheel_clearance.copy_(torch.where(
            self.flight_gate > 0.5, torch.maximum(self.max_wheel_clearance, self.wheel_clearance), self.max_wheel_clearance,
        ))
        self.active.logical_and_(~self.just_completed)

    def settle_rewards(self, env):
        """env 已完成本拍终止判定；再确认峰值资格、失败追回及一次性结果。"""
        if not self.was_active.any():
            self.jump_reward_state.clear_events()
            return
        terminated = self.was_active & env.terminated_buf
        self.jump_invalid.copy_(torch.maximum(self.jump_invalid, terminated.to(self.jump_invalid.dtype)))
        valid = self.jump_invalid < 0.5
        self.takeoff_gate.mul_(valid)
        self.flight_gate.mul_(valid)
        self.jump_reward_state.update(
            self.flight_gate > 0.5, self.jump_base_height, self.jump_landing_event,
            takeoff_eligible=(self.takeoff_contact_vertical_velocity > 0.0) & valid,
            active=self.was_active, ended=env.reset_buf | self.just_completed, valid_jump=valid,
        )
        height_reached = self.jump_reward_state.takeoff_rewarded & (self.jump_reward_state.peak_clearance >= self.base_height_target)
        height_ok = (
            (self.max_wheel_clearance > self.mode_success_clearances[self.mode])
            & (self.jump_reward_state.peak_clearance > self.mode_success_heights[self.mode])
        )
        # 合格腾空达到 base/轮底高度门槛即可成功，不等待落地或连续站稳。
        succeeded = self.was_active & height_reached & height_ok & valid & ~env.terminated_buf
        revoked = self.was_active & self.success_seen & ~valid
        self.task_success_event.copy_(torch.where(revoked, -1.0, (succeeded & ~self.success_seen).float()))
        self.success_seen.logical_or_(succeeded)
        self.success_seen.logical_and_(~revoked)
        failed = self.was_active & (~valid | env.reset_buf | self.just_completed) & ~succeeded
        self.task_failure_event.copy_(failed & ~self.failure_seen)
        self.failure_seen.logical_or_(failed)

    def observations(self, input_steps):
        """模式和相位由当前状态直接生成，不保存重复的观测 buffer。"""
        visible = self.active & (self.elapsed_steps < input_steps[self.mode])
        mode_one_hot = torch.nn.functional.one_hot(self.mode, len(MODE_NAMES)).to(self.height_reference.dtype)
        mode_one_hot.masked_fill_(~visible[:, None], 0.0)
        # 每行按自己的模式归一化，避免逐模式将 GPU 标量转回 CPU。
        progress = self.elapsed_steps * self.dt / (input_steps[self.mode] * self.dt)
        phase_features = torch.where(visible[:, None], phase_encoding(progress, 1.0), 0.0)
        return mode_one_hot, phase_features
