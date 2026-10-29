"""各任务专用 buffer 集中定义；同一任务的创建和回合重置放在同一个类中。

所有张量保留 num_envs 维度，任务 mask 隔离用途；部分 reset 只清空选中的环境。
通用进度、终止、奖励与命令查表仍由 env.py 管理。
"""

from dataclasses import dataclass
import math

import torch
from genesis.utils.geom import inv_quat, transform_by_quat


@dataclass
class StairsDownBuffers:
    """下台阶任务：当前回合任务 mask、每难度参数表和每回合落地窗口状态。"""

    # 任务 mask 在 reset 时随分组更新；难度参数表保持不变。
    task_mask: torch.Tensor
    landing_window_steps: torch.Tensor
    landing_ang_vel_weight: torch.Tensor

    # 回合状态；全量及部分 reset 使用同一入口。
    landing_started: torch.Tensor
    landing_steps_left: torch.Tensor
    landing_gate: torch.Tensor

    @classmethod
    def create(cls, env):
        task_mask = env.mask & torch.tensor([name == "stairs_down" for name in env.tasks], device=env.device)[env.task_ids]
        options = [stage.get("landing_stability", {}) for stage in env.terrain.stages]
        return cls(
            task_mask=task_mask,
            landing_window_steps=torch.tensor(
                [max(1, math.ceil(cfg.get("duration_s", 0.3) / env.dt)) for cfg in options],
                device=env.device, dtype=torch.long,
            ),
            landing_ang_vel_weight=env.base_pos.new_tensor([cfg.get("ang_vel_weight", 0.25) for cfg in options]),
            landing_started=torch.zeros_like(task_mask),
            landing_steps_left=torch.zeros_like(env.episode_length_buf),
            landing_gate=torch.zeros_like(env.base_height),
        )

    def reset(self, env_idx=None):
        ids = slice(None) if env_idx is None else env_idx
        self.landing_started[ids] = False
        self.landing_steps_left[ids] = 0
        self.landing_gate[ids] = 0


@dataclass
class StairsUpBuffers:
    """上台阶：触台收腿进度、首次轮子落点、双轮落台及站立确认窗口。"""

    task_mask: torch.Tensor
    tables: dict
    support_steps: torch.Tensor
    success_steps: torch.Tensor
    retract_start_length: torch.Tensor
    retract_peak: torch.Tensor
    support_awarded: torch.Tensor
    retract_awarded: torch.Tensor
    support_event: torch.Tensor
    lower_ground_seen: torch.Tensor
    takeoff_awarded: torch.Tensor
    takeoff_event: torch.Tensor
    retract_progress: torch.Tensor
    wheels_landed: torch.Tensor
    first_landing_seen: torch.Tensor
    first_landing_distance: torch.Tensor
    landing_distance_event: torch.Tensor
    wheel_wall_force: torch.Tensor
    fallen_off: torch.Tensor
    approach_start_height: torch.Tensor
    approach_peak_height: torch.Tensor
    approach_lift_progress: torch.Tensor
    base_collision_vertices: torch.Tensor
    top_contact_seen: torch.Tensor
    backswing_peak: torch.Tensor
    backswing_progress: torch.Tensor
    landing_started: torch.Tensor
    forward_start_angle: torch.Tensor
    forward_peak: torch.Tensor
    forward_progress: torch.Tensor
    extend_start_length: torch.Tensor
    extend_peak: torch.Tensor
    extend_progress: torch.Tensor
    wheel_support_event: torch.Tensor
    wheel_support_awarded: torch.Tensor

    @classmethod
    def create(cls, env):
        mask = env.mask & torch.tensor([name == "stairs_up" for name in env.tasks], device=env.device)[env.task_ids]
        # 兼容未保存 ascent 参数的旧运行配置；新配置在 stairs_up task 中集中设置。
        defaults = dict(
            approach_distance_m=0.50, approach_clearance_m=0.01,
            backswing_angle_deg=20.0,
            forward_swing_angle_deg=20.0, landing_extension_m=0.05,
            support_duration_s=0.10, support_force_n=5.0, support_normal_z=0.9,
            support_surface_tolerance_m=0.015, support_margin_m=0.005,
            retract_shortening_m=0.02, wheel_edge_margin_m=0.01, wheel_bottom_tolerance_m=0.005,
            fall_height_tolerance_m=0.03,
            wall_normal_z_max=0.3, wall_surface_tolerance_m=0.015,
            success_duration_s=0.20, standing_height_m=0.20,
            max_vertical_speed_m_s=0.3, max_roll_pitch_speed_rad_s=1.5,
        )
        tables = {key: env.base_pos.new_tensor([stage.get("ascent", {}).get(key, value)
                  for stage in env.terrain.stages]) for key, value in defaults.items()}
        # 旧存档未配置这些奖励时保留原有收腿阶段，避免续训隐式改变课程行为。
        tables["landing_enabled"] = torch.tensor([
            any(stage["rewards"].get(key, 0) > 0 for key in
                ("stairs_up_forward_swing", "stairs_up_extend", "stairs_up_wheel_support"))
            for stage in env.terrain.stages], device=env.device)
        for name in ("support", "success"):
            tables[name + "_steps"] = (tables[name + "_duration_s"] / env.dt).ceil().long().clamp_min(1)
        tables["top_height"] = env.base_pos.new_tensor([
            track.geometry.origin[2] + track.geometry.num_steps * track.geometry.step_height
            if track.kind == "stairs_up" else 0.0 for track in env.terrain.tracks])
        tables["approach_edge"] = env.base_pos.new_tensor([
            track.geometry.start_x if track.kind == "stairs_up" else 0.0 for track in env.terrain.tracks])
        tables["lower_height"] = env.base_pos.new_tensor([
            track.geometry.origin[2] if track.kind == "stairs_up" else 0.0 for track in env.terrain.tracks])
        tables["approach_top_height"] = env.base_pos.new_tensor([
            track.geometry.origin[2] + track.geometry.step_height
            if track.kind == "stairs_up" else 0.0 for track in env.terrain.tracks])
        vertices = env.base_pos.new_zeros((1, 3))
        if "stairs_up" in env.tasks:
            # 缓存实际碰撞体在机体系中的顶点；排除视觉网格，并包含碰撞体自身偏移/旋转。
            link = env.robot.get_link(env.env_cfg.get("base_link_name", "base_link"))
            world_vertices = link.get_verts()[0]
            vertices = transform_by_quat(
                world_vertices - env.robot.get_pos()[0], inv_quat(env.robot.get_quat()[0]))
        return cls(
            task_mask=mask, tables=tables,
            support_steps=torch.zeros_like(env.episode_length_buf),
            success_steps=torch.zeros_like(env.episode_length_buf),
            retract_start_length=env.base_pos.new_zeros((env.num_envs, 2)),
            retract_peak=env.base_pos.new_zeros((env.num_envs, 2)),
            support_awarded=torch.zeros_like(mask), retract_awarded=torch.zeros_like(mask),
            support_event=torch.zeros_like(mask),
            lower_ground_seen=torch.zeros_like(mask),
            takeoff_awarded=torch.zeros_like(mask), takeoff_event=torch.zeros_like(mask),
            retract_progress=torch.zeros_like(env.base_height),
            wheels_landed=torch.zeros_like(mask), fallen_off=torch.zeros_like(mask),
            first_landing_seen=torch.zeros_like(mask),
            first_landing_distance=torch.zeros_like(env.base_height),
            landing_distance_event=torch.zeros_like(env.base_height),
            wheel_wall_force=torch.zeros_like(env.base_height),
            approach_start_height=torch.zeros_like(env.base_height),
            approach_peak_height=torch.zeros_like(env.base_height),
            approach_lift_progress=torch.zeros_like(env.base_height),
            base_collision_vertices=vertices,
            top_contact_seen=torch.zeros_like(mask),
            backswing_peak=env.base_pos.new_zeros((env.num_envs, 2)),
            backswing_progress=torch.zeros_like(env.base_height),
            landing_started=torch.zeros_like(mask),
            forward_start_angle=env.base_pos.new_zeros((env.num_envs, 2)),
            forward_peak=env.base_pos.new_zeros((env.num_envs, 2)),
            forward_progress=torch.zeros_like(env.base_height),
            extend_start_length=env.base_pos.new_zeros((env.num_envs, 2)),
            extend_peak=env.base_pos.new_zeros((env.num_envs, 2)),
            extend_progress=torch.zeros_like(env.base_height),
            wheel_support_event=torch.zeros_like(mask),
            wheel_support_awarded=torch.zeros_like(mask),
        )

    def bottom_height(self, env):
        """真实底盘碰撞体最低点的世界高度，随 roll/pitch 改变；reset 不依赖延迟的 FK。"""
        if "stairs_up" not in env.tasks:
            return env.base_pos[:, 2]
        rotated = transform_by_quat(self.base_collision_vertices[None, :, :], env.base_quat[:, None, :])
        return env.base_pos[:, 2] + rotated[..., 2].amin(dim=1)

    def reset(self, env_idx=None):
        ids = slice(None) if env_idx is None else env_idx
        self.support_steps[ids] = 0
        self.success_steps[ids] = 0
        self.retract_start_length[ids] = 0
        self.retract_peak[ids] = 0
        self.support_awarded[ids] = False
        self.retract_awarded[ids] = False
        self.support_event[ids] = False
        self.lower_ground_seen[ids] = False
        self.takeoff_awarded[ids] = False
        self.takeoff_event[ids] = False
        self.retract_progress[ids] = 0
        self.wheels_landed[ids] = False
        self.first_landing_seen[ids] = False
        self.first_landing_distance[ids] = 0
        self.landing_distance_event[ids] = 0
        self.wheel_wall_force[ids] = 0
        self.fallen_off[ids] = False
        self.approach_start_height[ids] = 0
        self.approach_peak_height[ids] = 0
        self.approach_lift_progress[ids] = 0
        self.top_contact_seen[ids] = False
        self.backswing_peak[ids] = 0
        self.backswing_progress[ids] = 0
        self.landing_started[ids] = False
        self.forward_start_angle[ids] = 0
        self.forward_peak[ids] = 0
        self.forward_progress[ids] = 0
        self.extend_start_length[ids] = 0
        self.extend_peak[ids] = 0
        self.extend_progress[ids] = 0
        self.wheel_support_event[ids] = False
        self.wheel_support_awarded[ids] = False


from .intermittent import IntermittentTraversal
from .parking import ParkingCurriculum


class MultiTerrainBuffers:
    def _initialize_task_buffers(self):
        # 所有任务共用：分组、进度、计时、回合结果。
        # 局部 task_ids 仅用于地形分组和任务状态路由。
        self.task_ids = self.terrain.fixed_types
        self.traversal = self.mask.clone()
        self.task_origin_xy = self.base_pos.new_zeros((self.num_envs, 2))
        self.task_forward_xy = self.base_pos.new_zeros((self.num_envs, 2))
        self.task_forward_xy[:, 0] = 1.0
        self.task_goal_distance = torch.zeros_like(self.base_height)
        self.episode_start_x = torch.zeros_like(self.base_height)
        self.progress = torch.zeros_like(self.base_height)
        self.farthest_x = torch.zeros_like(self.base_height)
        self.progress_anchor = torch.zeros_like(self.base_height)
        self.stall_steps = torch.zeros_like(self.episode_length_buf)
        self.task_steps = torch.zeros_like(self.episode_length_buf)
        self.success = torch.zeros_like(self.traversal)
        self.outcome_success = torch.zeros_like(self.traversal)
        # 按任务集中创建；每类内部定义其全部状态、专用参数表与 reset 行为。
        self.stairs_down_buffers = StairsDownBuffers.create(self)
        self.stairs_up_buffers = StairsUpBuffers.create(self)

        # 所有任务共用：按当前地形/难度查表的奖励、命令及终止参数。
        self.reward_tables = {name: self.base_pos.new_tensor([
            stage["rewards"].get(name, 0.0) * (1.0 if name in self.EVENT_REWARDS else self.dt)
            for stage in self.terrain.stages]) for name in self.reward_functions}
        self._apply_task_command_ranges(getattr(self, "_task_command_ranges", {}))
        self.tables = {}
        for name, default in (("corridor_half_width", 1.0), ("stall_timeout_s", 5.0), ("progress_epsilon", 0.01),
                              ("episode_length_s", 20.0), ("termination_if_roll_greater_than", 10.0),
                              ("termination_if_pitch_greater_than", 10.0), ("tilt_termination_duration_s", 0.3),
                              ("base_contact_termination_duration_s", 0.25)):
            self.tables[name] = self.base_pos.new_tensor([s["termination"].get(name, default) for s in self.terrain.stages])
        for name, default in (("max_roll_deg", 12.0), ("max_pitch_deg", 15.0),
                              ("wheel_clearance_m", 0.15), ("wheel_surface_tolerance_m", 0.08)):
            self.tables[name] = self.base_pos.new_tensor([s.get("success", {}).get(name, default) for s in self.terrain.stages])
        self.tables["direction"] = self.base_pos.new_tensor([t.direction for t in self.terrain.tracks])
        self.tables["goal_x"] = self.base_pos.new_tensor([t.goal_x for t in self.terrain.tracks])
        self.tables["landing_length"] = self.base_pos.new_tensor([
            s.get("traversal", {}).get("landing_length", 0.0) for s in self.terrain.stages])
        self.tables["obstacle_end"] = self.base_pos.new_tensor([t.obstacle_end for t in self.terrain.tracks])
        self.tables["resample_steps"] = self.base_pos.new_tensor([
            max(1, math.ceil(s["resampling_time_s"] / self.dt)) for s in self.terrain.stages])
        self.tables["min_x"] = self.base_pos.new_tensor([-t.tile_size[0] / 2 + t.boundary_margin for t in self.terrain.tracks])
        self.tables["max_x"] = self.base_pos.new_tensor([t.tile_size[0] / 2 - t.boundary_margin for t in self.terrain.tracks])
        self.tables["half_y"] = self.base_pos.new_tensor([t.tile_size[1] / 2 - t.boundary_margin for t in self.terrain.tracks])
        # 基类每拍回调，具体重采样周期由每项任务决定。
        self.intermittent = IntermittentTraversal(self)
        self.parking = ParkingCurriculum(self, evaluation=(getattr(self, "evaluation", False)
                                                           and not getattr(self, "parking_evaluation", False)))
        if self.parking.enabled:
            for i, (track, stage) in enumerate(zip(self.terrain.tracks, self.terrain.stages)):
                approach = self.intermittent.tables["approach_speed_m_s"][i].item()
                if min(approach, stage["commands"]["lin_vel_range"][1]) <= self.parking.config["required_speed_m_s"]:
                    raise ValueError(f"{track.kind}: parking approach command must exceed required_speed_m_s")


    def _reset_task_buffers(self, env_idx):
        # 通用回合状态。
        ids = slice(None) if env_idx is None else env_idx
        self.stairs_down_buffers.task_mask[ids] = self.mask[ids] & torch.tensor(
            [name == "stairs_down" for name in self.tasks], device=self.device)[self.task_ids[ids]]
        self.stairs_up_buffers.task_mask[ids] = self.mask[ids] & torch.tensor(
            [name == "stairs_up" for name in self.tasks], device=self.device)[self.task_ids[ids]]
        yaw = torch.deg2rad(self.base_euler[ids, 2])
        forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        crossing_speed = forward[:, 0] * self._value("direction")[ids]
        if (self.traversal[ids] & (crossing_speed <= 1e-4)).any():
            raise ValueError("traversal reset yaw must face across the terrain, within 90 degrees of its travel direction")
        self.task_origin_xy[ids] = self.base_pos[ids, :2]
        self.task_forward_xy[ids] = forward
        # 先沿出生航向到达真实障碍末沿，再沿同一方向前进 landing_length。
        local_start_x = self.base_pos[ids, 0] - self.terrain_spawn_centers[ids, 0]
        distance_to_edge = (self._value("obstacle_end")[ids] - local_start_x) * self._value("direction")[ids]
        self.task_goal_distance[ids] = distance_to_edge / crossing_speed.clamp_min(1e-4) + self._value("landing_length")[ids]
        wheels = self.robot.get_links_pos(self.wheel_links_idx)
        self.farthest_x[ids] = self._task_progress_position(wheels)[ids]
        self.episode_start_x[ids] = self.farthest_x[ids]
        self.progress_anchor[ids] = self.farthest_x[ids]
        self.progress[ids] = 0
        self.stall_steps[ids] = 0
        self.task_steps[ids] = 0
        self.success[ids] = False
        self.outcome_success[ids] = False
        self.tilt_termination_steps = (self._value("tilt_termination_duration_s") / self.dt).ceil().long().clamp_min(1)
        self.base_contact_termination_steps = (self._value("base_contact_termination_duration_s") / self.dt).ceil().long().clamp_min(1)

        # 专用状态：与初始化保持相同的任务顺序。
        self.stairs_down_buffers.reset(env_idx)
        self.stairs_up_buffers.reset(env_idx)
        state = self.stairs_up_buffers
        tiles = self.terrain_tile_index[ids]
        state.approach_start_height[ids] = state.bottom_height(self)[ids]
        state.approach_peak_height[ids] = torch.minimum(
            state.approach_start_height[ids],
            state.tables["approach_top_height"][tiles] + state.tables["approach_clearance_m"][tiles])
        target_angle = torch.deg2rad(state.tables["backswing_angle_deg"][tiles])
        state.backswing_peak[ids] = (self.leg_angle[ids].clamp_min(0) / target_angle[:, None]).clamp_max(1)
        self.intermittent.reset(self, env_idx)
