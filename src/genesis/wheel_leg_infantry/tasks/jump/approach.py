"""同一策略先在空旷跑道加速，达速后保留物理状态平移到起跳位置。"""

import math

import torch

from .geometry import trigger_distance


class JumpApproach:
    def _initialize_approach_buffers(self):
        cfg = self.approach_cfg
        for name, value in cfg.items():
            if not math.isfinite(value) or (name not in {"spawn_x_m", "spawn_y_m"} and value <= 0):
                raise ValueError(f"jump_approach.{name} must be finite and positive (spawn coordinates may be negative)")
        if cfg["stable_time_s"] >= cfg["timeout_s"]:
            raise ValueError("jump approach stable_time_s must be smaller than timeout_s")
        self.approach_stable_steps = max(1, math.ceil(cfg["stable_time_s"] / self.dt - 1e-9))
        self.approach_timeout_steps = math.ceil(cfg["timeout_s"] / self.dt)
        self.jump_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.approach_ready_steps = torch.zeros_like(self.jump_steps)
        self.approach_duration_steps = torch.zeros_like(self.jump_steps)
        self.approach_timed_out = torch.zeros_like(self.jump_active)

    def _update_approach_state(self):
        cfg = self.approach_cfg
        approaching = self.mask & ~self.jump_active
        ready = (
            approaching
            & ((self.base_lin_vel[:, 0] - self.commands[:, 0]).abs() <= cfg["speed_tolerance_m_s"])
            & (self.base_lin_vel[:, 1].abs() <= cfg["lateral_speed_tolerance_m_s"])
            & (self.robot.get_vel()[:, 2].abs() <= cfg["vertical_speed_tolerance_m_s"])
            & ((self.base_height - self.commands[:, 2]).abs() <= cfg["height_tolerance_m"])
            & (-self.projected_gravity[:, 2] >= math.cos(math.radians(cfg["max_tilt_deg"])))
            & (self.base_euler[:, 2].abs() <= cfg["max_yaw_deg"])
            & (self.wheel_contact > 0.5).all(dim=1)
            & (self.base_contact < 0.5)
        )
        self.approach_ready_steps.copy_(torch.where(ready, self.approach_ready_steps + 1, 0))
        self.approach_timed_out.logical_or_(approaching & (self.episode_length_buf >= self.approach_timeout_steps))

    def before_observations(self):
        # 加速仅运行策略推理；移动后发布带 t=0 jump 编码的新观测。
        # 不制造物理 done，保留 GRU 隐状态；从下一拍动作开始采集 PPO 样本。
        ready = (self.mask & ~self.jump_active & ~self.reset_buf & ~self.approach_timed_out
                 & (self.approach_ready_steps >= self.approach_stable_steps))
        if ready.any():
            self._start_jump(ready)

    def _start_jump(self, mask):
        ids = mask.nonzero(as_tuple=False).flatten()
        modes = self.jump_mode[ids]
        distance = trigger_distance(self.commands[ids, 0], modes, self.task_cfg["distance_tables"])
        distance += (torch.rand_like(distance) * 2 - 1) * self.task_cfg["distance_jitter_m"]
        positions = self.base_pos[ids].clone()
        positions[:, 0] = -distance
        positions[:, 1] = modes * self.task_cfg["lane_spacing_m"]
        # 两处地面均为 z=0；只移动 x/y，保留高度、姿态、全部关节和实际速度。
        self.robot.set_pos(positions, envs_idx=ids, zero_velocity=False)
        self.env.base_pos[ids] = positions
        self.env.terrain_height[ids] = self.terrain.height_at(positions[:, :2])
        self.env.base_height[ids] = positions[:, 2] - self.env.terrain_height[ids]
        self.env.terrain_spawn_centers[ids, 0] = 0.0
        self.env.terrain_spawn_centers[ids, 1] = positions[:, 1]
        self.jump_active[ids] = True
        self.jump_steps[ids] = 0
        self.approach_duration_steps[ids] = self.episode_length_buf[ids].to(self.approach_duration_steps.dtype)
        self._reset_jump_buffers(mask)
        self.trigger_distance_m[ids] = distance
        # t=0 的接触拍作为起跳历史起点，下一物理步即可识别离地。
        self.takeoff_contact_vertical_velocity[ids] = self.robot.get_vel()[ids, 2]
        self.takeoff_contact_base_height[ids] = positions[:, 2]
        self._update_jump_commands()
        if self.tof is not None:
            # 立即替换空旷跑道的测距历史，不重采样随机化参数或推进物理时钟。
            self.tof_distances[ids] = self.tof.read(after_reset=True, env_ids=ids, strength=self.sensor_noise.strength)
            self.tof_clock.reset(ids)
            if self.tof_history is not None:
                self.tof_history.reset(self.tof_distances, ids)
                self.true_tof_distances[ids] = self.tof.read_ground_truth(after_reset=True, env_ids=ids)
                self.true_tof_history.reset(self.true_tof_distances, ids)
