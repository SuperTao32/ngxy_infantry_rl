"""跳跃任务专用 buffer 及回合复位。"""

import math
import genesis as gs
import torch
import torch.nn.functional as F

from .config import PHASE_NAMES
from .geometry import active_step_heights
from .phase import phase_encoding
from .reward_state import JumpRewardState


class JumpBuffers:
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
        self.takeoff_contact_vertical_velocity = torch.zeros_like(self.base_height_target)  # 最后一拍支撑速度。
        self.takeoff_contact_base_height = torch.zeros_like(self.base_height_target)  # 与支撑速度同拍的 base 高度。
        self.last_airborne_vertical_velocity = torch.zeros_like(self.base_height_target)  # 触地前一拍速度。
        self.impact_vertical_speed = torch.zeros_like(self.base_height_target)  # 仅落地事件拍非零。

        # 物理阶段历史：首次起跳/落地后保持 True，直到下一次 reset。
        # has_taken_off 为物理离地，不等于奖励要求的“连续腾空达到最短时间”。
        self.has_taken_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_landed = torch.zeros_like(self.has_taken_off)
        self.phase_features = torch.zeros((self.num_envs, 6), dtype=gs.tc_float, device=self.device)  # actor 时间编码。

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

        # 历史：无效任务不能恢复；二次腾空只记录日志和惩罚，不作为失败。
        self.jump_invalid = torch.zeros_like(self.base_height_target)  # 过倾、撞沿或错误落台。
        self.rebound_seen = torch.zeros_like(self.base_height_target)  # 曾在落地后再次双轮离地。

        # 成功判定与事件去重：task_success 是当前状态，会随失稳变 False；seen 保留回合历史。
        self.stable_landing_steps = torch.zeros_like(self.jump_mode)  # 连续落稳拍数，中断即清零。
        self.task_success = torch.zeros_like(self.has_taken_off)
        self.success_seen = torch.zeros_like(self.has_taken_off)  # 防止重新站稳后重复发成功奖励。
        self.failure_seen = torch.zeros_like(self.has_taken_off)  # 防止终止后的补齐步重复发失败奖励。

        # 采集终点：首次终止时锁存统计，补齐步仍仿真，但不能覆盖最终成绩。

        # 跨步奖励结算：管理连续腾空、合格起跳事件、有效峰值及失稳后的奖励追回。
        self.jump_reward_state = JumpRewardState(
            self.num_envs,
            dt=self.dt,
            min_airborne_time_s=float(self.env_cfg.get("takeoff_min_airborne_time_s", 0.10)),
            horizon=round(self.phase_cycle_s / self.dt),
            device=self.device,
            dtype=gs.tc_float,
        )

    def _reset_jump_buffers(self, env_idx):
        """清零跳跃结算状态；物理 reset 和达速触发共用，不改加速历史。"""
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

    def _jump_episode_buffers(self):
        """需要在回合边界清零的逐环境状态；全量/局部 reset 共用这一份清单。"""
        return (
            # 实测量及用于差分/事件结算的历史量。
            self.wheel_center_pos, self.base_to_wheel_bottom_distance,
            self.wheel_clearance, self.max_wheel_clearance,
            self.jump_base_height, self.max_jump_base_height,
            self.world_vertical_velocity, self.takeoff_contact_vertical_velocity,
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
            self.trigger_distance_m,
        )
