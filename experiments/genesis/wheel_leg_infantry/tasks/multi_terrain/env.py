"""共享策略，按固定任务分组应用课程、命令、奖励和终止。"""

from copy import deepcopy
import math

import torch

from ..locomotion.env import LocomotionEnv
from .buffers import LocomotionBuffers, StairsDownBuffers, StairsUpBuffers
from .config import TASK_NAMES
from .curriculum import allocate_counts, initial_task_counts, resolve_task_command_ranges, resolve_tasks
from .rewards import MultiTerrainRewards
from .terrain import TaskTerrain
from .intermittent import IntermittentTraversal
from .parking import ParkingCurriculum


class MultiTerrainEnv(MultiTerrainRewards, LocomotionEnv):
    EVENT_REWARDS = {"task_progress", "task_success", "death", "intermittent_stop", "stairs_up_support", "stairs_up_retract",
                     "stairs_up_takeoff", "stairs_up_landing_distance", "stairs_up_approach_lift", "stairs_up_backswing",
                     "stairs_up_forward_swing", "stairs_up_extend", "stairs_up_wheel_support"}

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, *, evaluation=False, parking_evaluation=False, **kwargs):
        self.evaluation = evaluation
        self.parking_evaluation = parking_evaluation
        self.tasks = resolve_tasks(env_cfg["multi_terrain"], require_locomotion=not evaluation)
        self.task_terrain = TaskTerrain(self.tasks, initial_task_counts(self.tasks, num_envs, kwargs.get("curriculum_cfg")))
        reward_cfg = deepcopy(reward_cfg)
        reward_cfg["terrain_reward_scales"] = {}
        names = set().union(*(stage["rewards"] for stage in self.task_terrain.stages))
        for name in names:
            if not callable(getattr(self, "_reward_" + name, None)):
                raise ValueError(f"unknown task reward: {name}")
        reward_cfg["reward_scales"] = dict.fromkeys(sorted(names), 0.0)
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, **kwargs)

    def _register_task_curriculum(self):
        # 在 Scene 选定碰撞容量前替换地形；几何/任务快照保存在 multi_terrain。
        self.terrain = self.task_terrain
        self.curriculum.register_target("task_weights", self._apply_task_weights)
        self.curriculum.register_target("task_command_ranges", self._apply_task_command_ranges)

    def _apply_task_weights(self, weights):
        self.terrain.set_counts(allocate_counts(self.tasks, self.num_envs, weights))

    def _apply_task_command_ranges(self, values):
        """按任务覆盖各难度的采样范围；下一次命令重采样生效。"""
        commands = resolve_task_command_ranges(self.tasks, values)
        stage_commands = [(cfg, self._build_command_limits(cfg)) for cfg in commands]
        self._task_command_ranges = deepcopy(values)
        self.stage_commands = stage_commands

    def _add_terrain(self):
        self.terrain_entity = self.terrain.add_to_scene(self.scene)
        self.friction_terrain_entities = self.terrain_entity

    def _initialize_task_buffers(self):
        # 所有任务共用：分组、进度、计时、回合结果。
        threshold = self.obs_cfg.get("critic_task_one_hot_vx_threshold", 0.5)
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or threshold < 0):
            raise ValueError("critic_task_one_hot_vx_threshold must be finite and nonnegative, in m/s")
        self.task_one_hot_vx_threshold = threshold
        self.task_ids = self.terrain.fixed_types
        # task_ids 使用当前任务子集的局部编号；观测始终使用完整 TASK_NAMES 编码。
        self.task_one_hot_table = torch.eye(len(TASK_NAMES), device=self.device, dtype=self.base_pos.dtype)[
            [TASK_NAMES.index(name) for name in self.tasks]]
        self.traversal = torch.tensor([name != "locomotion" for name in self.tasks], device=self.device)[self.task_ids]
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
        self.locomotion_buffers = LocomotionBuffers.create(self)
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
        self.resample_step = 1
        self.intermittent = IntermittentTraversal(self)
        self.parking = ParkingCurriculum(self, evaluation=(getattr(self, "evaluation", False)
                                                           and not getattr(self, "parking_evaluation", False)))
        if self.parking.enabled:
            for i, (track, stage) in enumerate(zip(self.terrain.tracks, self.terrain.stages)):
                if track.kind == "locomotion":
                    continue
                approach = self.intermittent.tables["approach_speed_m_s"][i].item()
                if min(approach, stage["commands"]["lin_vel_range"][1]) <= self.parking.config["required_speed_m_s"]:
                    raise ValueError(f"{track.kind}: parking approach command must exceed required_speed_m_s")

    def step(self, actions):
        result = super().step(actions)
        # 固定 PPO 采样边界同步，不能在局部 reset 分支中调用 collective。
        if not self.evaluation and self.global_step % self.steps_per_iteration == 0:
            self.parking.synchronize()
        return result

    def _sample_base_reset_state(self, env_ids):
        state = super()._sample_base_reset_state(env_ids)
        self.parking.select(self, env_ids, state[-1])
        return state

    def _value(self, name):
        return self.tables[name][self.terrain_tile_index]

    def _get_task_privileged_observation_components(self):
        # 旧 run 的保存配置没有此字段，保留原 critic 输入维度以支持完整续训。
        components = {}
        if self.obs_cfg.get("critic_task_one_hot", False):
            # 使用当前原始命令（m/s），停车指令生效时立即清空地形掩码。
            active = self.commands[:, 0] > self.task_one_hot_vx_threshold
            components["privileged_task_one_hot"] = self.task_one_hot_table[self.task_ids] * active[:, None]
        return components

    def _task_progress_position(self, wheels):
        # 上台阶以双轮中心为进度参考，避免轮子不动时机身前倾也获得前进奖励。
        world_xy = torch.where(self.stairs_up_buffers.task_mask[:, None], wheels[..., :2].mean(dim=1), self.base_pos[:, :2])
        return ((world_xy - self.task_origin_xy) * self.task_forward_xy).sum(dim=-1)

    def _task_path_coordinates(self, world_xy):
        """把机身、轮心或接触点投影到本回合出生航向，返回前进距离和横向偏移。"""
        shape = (self.num_envs,) + (1,) * (world_xy.ndim - 2) + (2,)
        delta = world_xy - self.task_origin_xy.reshape(shape)
        forward = self.task_forward_xy.reshape(shape)
        along = (delta * forward).sum(dim=-1)
        lateral = delta[..., 1] * forward[..., 0] - delta[..., 0] * forward[..., 1]
        return along, lateral

    def _task_in_corridor(self, world_xy):
        """走廊随出生 yaw 倾斜，同时限制在实际地形的安全边界内。"""
        _, lateral = self._task_path_coordinates(world_xy)
        shape = (self.num_envs,) + (1,) * (world_xy.ndim - 2)
        local = world_xy - self.terrain_spawn_centers.reshape(shape + (2,))
        return ((lateral.abs() <= self._value("corridor_half_width").reshape(shape))
                & (local[..., 0] >= self._value("min_x").reshape(shape))
                & (local[..., 0] <= self._value("max_x").reshape(shape))
                & (local[..., 1].abs() <= self._value("half_y").reshape(shape)))

    def _reset_task_buffers(self, env_idx):
        # 通用回合状态。
        ids = slice(None) if env_idx is None else env_idx
        self.traversal[ids] = torch.tensor([name != "locomotion" for name in self.tasks], device=self.device)[self.task_ids[ids]]
        self.stairs_down_buffers.task_mask[ids] = torch.tensor(
            [name == "stairs_down" for name in self.tasks], device=self.device)[self.task_ids[ids]]
        self.stairs_up_buffers.task_mask[ids] = torch.tensor(
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
        self.locomotion_buffers.reset(env_idx)
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

    def _resample_commands(self, envs_idx):
        selected = torch.ones_like(self.traversal) if envs_idx is None else envs_idx
        selected = selected & (self.task_steps % self._value("resample_steps").long() == 0)
        # 间断回合持有 reset 时的行驶命令，不能被周期重采样覆盖。
        selected &= ~self.intermittent.enabled | (self.task_steps == 0)
        if not selected.any():
            self.intermittent.apply_commands(self)
            return
        for i, (cfg, limits) in enumerate(self.stage_commands):
            mask = selected & (self.terrain_tile_index == i)
            if not mask.any():
                continue
            # 采样支持静止概率、高速转向及高机身限速，写回仅限本任务、本回合。
            commands = self._sample_commands(cfg, limits)
            intermittent = mask & self.intermittent.enabled
            if intermittent.any() and cfg.get("standing_probability", 0):
                moving = self._sample_commands({**cfg, "standing_probability": 0.0}, limits)
                commands[intermittent] = moving[intermittent]
            self.commands[mask] = commands[mask]
            self.intermittent.move_commands[intermittent] = commands[intermittent]
        self.intermittent.apply_commands(self)

    def _update_task_state(self):
        self.task_steps += 1
        direction = self._value("direction")
        wheels = self.robot.get_links_pos(self.wheel_links_idx)
        x = torch.minimum(self._task_progress_position(wheels), self.task_goal_distance)
        self.progress.copy_((x - self.farthest_x).clamp_min(0) * self.traversal)
        self.farthest_x.copy_(torch.maximum(self.farthest_x, x))
        advanced = self.farthest_x - self.progress_anchor >= self._value("progress_epsilon")
        self.progress_anchor.copy_(torch.where(advanced, self.farthest_x, self.progress_anchor))
        self.stall_steps.copy_(torch.where(advanced, 0, self.stall_steps + 1))
        wheel_heights = self.terrain.height_at(wheels[:, :, :2].reshape(-1, 2)).reshape(self.num_envs, 2)
        wheel_clear = ((wheels[:, :, 0] - self.terrain_spawn_centers[:, :1]) * direction[:, None]
                       >= (self._value("obstacle_end") * direction + self._value("wheel_clearance_m"))[:, None]).all(dim=1)
        wheel_supported = ((wheels[:, :, 2] - wheel_heights - self.wheel_radius).abs()
                           <= self._value("wheel_surface_tolerance_m")[:, None]).all(dim=1)
        along, _ = self._task_path_coordinates(self.base_pos[:, :2])
        self.success.copy_(
            self.traversal & (along >= self.task_goal_distance) & wheel_clear & wheel_supported
            & self._task_in_corridor(self.base_pos[:, :2]) & self._task_in_corridor(wheels[..., :2]).all(dim=1)
            & (self.wheel_contact > 0.5).all(dim=1) & (self.base_contact < 0.5)
            & (self.base_euler[:, 0].abs() <= self._value("max_roll_deg"))
            & (self.base_euler[:, 1].abs() <= self._value("max_pitch_deg")))
        # 专用状态：与初始化/重置保持相同的任务顺序。
        self._update_locomotion_tracking()
        self._update_stairs_down_landing(wheels, wheel_heights)
        self._update_stairs_up_ascent(wheels)
        self.intermittent.update(self, wheels)
        self.stall_steps.masked_fill_(self.intermittent.reward_paused | self.intermittent.paused, 0)
        self.success &= self.intermittent.success_mask()
        self.success.copy_(torch.where(self.parking.drill, self.intermittent.stop_event & ~self.intermittent.failed,
                                       self.success))

    def _update_stairs_up_ascent(self, wheels):
        state = self.stairs_up_buffers
        if "stairs_up" not in self.tasks:
            return
        cfg = {key: value[self.terrain_tile_index] for key, value in state.tables.items()}
        lower_contact = self._stairs_up_lower_contact(wheels, cfg)
        # 先确认双轮实际站在台下，避免 reset 后接触信号尚未更新或空中出生误发奖。
        state.lower_ground_seen |= state.task_mask & lower_contact.all(dim=1)
        lifted = (wheels[..., 2] - self.wheel_radius >
                  (cfg["lower_height"] + cfg["wheel_bottom_tolerance_m"])[:, None]).all(dim=1)
        state.takeoff_event.copy_(state.task_mask & state.lower_ground_seen & lifted & ~state.takeoff_awarded)
        state.takeoff_awarded |= state.takeoff_event
        # 轮子仍可抵着立面；离地看实际高度，重新落回则必须有台下地面接触。
        state.fallen_off |= state.task_mask & state.takeoff_awarded & lower_contact.any(dim=1)
        self._update_stairs_up_approach(cfg, lower_contact.all(dim=1))
        contacts = self.robot.get_contacts(exclude_self_contact=True)
        self._update_stairs_up_landing_distance(contacts, cfg)
        self._update_stairs_up_wheel_wall_force(contacts, cfg)
        # Genesis 的接触 link 索引是全场景索引；base_link_idx 是机器人局部索引。
        base_link = self.robot.link_start + self.base_link_idx
        is_a = contacts["link_a"] == base_link
        is_b = contacts["link_b"] == base_link
        force_z = torch.where(is_a, contacts["force_a"][..., 2], contacts["force_b"][..., 2])
        position = contacts["position"]
        local_x = position[..., 0] - self.terrain_spawn_centers[:, :1]
        surface_contact = (
            contacts["valid_mask"] & (is_a | is_b)
            & (local_x >= self._value("obstacle_end")[:, None])
            & (local_x <= self._value("max_x")[:, None])
            & self._task_in_corridor(position[..., :2])
            & ((position[..., 2] - cfg["top_height"][:, None]).abs()
               <= cfg["support_surface_tolerance_m"][:, None])
            & (contacts["normal"][..., 2].abs() >= cfg["support_normal_z"][:, None])
            & (self.base_pos[:, 2:3] > position[..., 2])
        )
        # 记录首次轻触台面；后摆可持续到落台阶段，收腿仍须等待稳定支撑。
        touching_top = state.task_mask & (surface_contact & (force_z > 0)).any(dim=1)
        state.top_contact_seen |= touching_top
        top_contact = surface_contact & (
            local_x >= (self._value("obstacle_end") + cfg["support_margin_m"])[:, None])
        upward_force = torch.where(top_contact, force_z.clamp_min(0), 0).sum(dim=1)
        vertical_speed = (self.base_lin_vel * self.projected_gravity).sum(dim=1).abs()
        quiet = ((vertical_speed <= cfg["max_vertical_speed_m_s"])
                 & (self.base_ang_vel[:, :2].norm(dim=1) <= cfg["max_roll_pitch_speed_rad_s"]))
        supported = state.task_mask & quiet & (upward_force >= cfg["support_force_n"])
        state.support_steps.copy_(torch.where(supported, state.support_steps + 1, 0))
        stable_support = supported & (state.support_steps >= cfg["support_steps"])
        state.support_event.copy_(stable_support & ~state.support_awarded)
        # 以首次稳定支撑时的腿长为基准，之前的缩短量不补奖。
        state.retract_start_length.copy_(torch.where(state.support_event[:, None], self.leg_length, state.retract_start_length))
        state.support_awarded |= state.support_event

        wheel_x = wheels[..., 0] - self.terrain_spawn_centers[:, :1]
        wheel_bottom = wheels[..., 2] - self.wheel_radius
        # 首次双轮接触台面即可关闭收腿，不等整轮后缘越沿或最终站稳。
        on_top = (
            (wheel_x >= (self._value("obstacle_end") + cfg["wheel_edge_margin_m"])[:, None])
            & (wheel_x <= self._value("max_x")[:, None])
            & self._task_in_corridor(wheels[..., :2])
            & ((wheel_bottom - cfg["top_height"][:, None]).abs() <= cfg["wheel_bottom_tolerance_m"][:, None])
            & (self.wheel_contact > 0.5)
        ).all(dim=1)
        self._update_stairs_up_landing(wheels, contacts, cfg, touching_top, on_top)
        state.wheels_landed |= state.task_mask & on_top
        # 轮底已抬到台面高度后转入落台引导，避免收腿和伸腿奖励相互竞争。
        started = state.task_mask & state.support_awarded & ~state.wheels_landed & ~state.landing_started
        eligible = started & stable_support
        score = ((state.retract_start_length - self.leg_length) / cfg["retract_shortening_m"][:, None]).clamp(0, 1)
        state.retract_progress.copy_((score - state.retract_peak).clamp_min(0).mean(dim=1) * eligible)
        # 失去支撑时暂停奖励并重新计时；期间的缩短不能在重新支撑后补领。
        state.retract_peak.copy_(torch.where(started[:, None], torch.maximum(state.retract_peak, score), state.retract_peak))
        state.retract_awarded |= eligible & (state.retract_peak >= 1 - 1e-6).all(dim=1)
        # 落台后允许向上弹跳；任一轮明显跌到台面以下即失败，无需等落到低处地面。
        below_top = (wheel_bottom < (cfg["top_height"] - cfg["fall_height_tolerance_m"])[:, None]).any(dim=1)
        state.fallen_off |= state.task_mask & state.wheels_landed & below_top
        # 先更新前摆阶段和双轮落台，再计算后摆，保证阶段切换当拍互斥。
        self._update_stairs_up_backswing(wheels, cfg)

        # 最终成功独立于中间奖励：直接登上台阶的策略也可以完成任务。
        standing = state.task_mask & self.success & quiet & (self.base_height >= cfg["standing_height_m"])
        state.success_steps.copy_(torch.where(standing, state.success_steps + 1, 0))
        self.success &= ~state.task_mask | (state.success_steps >= cfg["success_steps"])

    def _update_stairs_up_landing(self, wheels, contacts, cfg, touching_top, on_top):
        """机身搭台且双轮抬够后，同时引导前摆与伸腿，不设前摆角度的先后门槛。"""
        state = self.stairs_up_buffers
        wheel_x = wheels[..., 0] - self.terrain_spawn_centers[:, :1]
        wheel_bottom = wheels[..., 2] - self.wheel_radius
        cleared = (wheel_bottom >= (cfg["top_height"] - cfg["wheel_bottom_tolerance_m"])[:, None]).all(dim=1)
        in_corridor = self._task_in_corridor(wheels[..., :2]).all(dim=1)
        first = (state.task_mask & cfg["landing_enabled"] & touching_top & cleared & in_corridor & ~on_top
                 & ~state.landing_started & ~state.wheels_landed & ~self._tilt_failure())
        angle = torch.deg2rad(cfg["forward_swing_angle_deg"])
        state.forward_start_angle.copy_(torch.where(first[:, None], self.leg_angle, state.forward_start_angle))
        # 从进入阶段时的角度开始奖励前摆，不必先越过竖直腿角才产生学习信号。
        span = state.forward_start_angle + angle[:, None]
        forward_score = torch.where(span > 1e-6,
                                    ((state.forward_start_angle - self.leg_angle) / span.clamp_min(1e-6)).clamp(0, 1), 0)
        state.extend_start_length.copy_(torch.where(first[:, None], self.leg_length, state.extend_start_length))
        state.landing_started |= first
        active = state.task_mask & state.landing_started & ~state.wheels_landed
        valid = active & in_corridor & ~self._tilt_failure() & ~state.fallen_off
        state.forward_progress.copy_((forward_score - state.forward_peak).clamp_min(0).mean(dim=1) * valid)
        state.forward_peak.copy_(torch.where(active[:, None], torch.maximum(state.forward_peak, forward_score), state.forward_peak))

        # 伸腿只需轮子处于台面上方，不要求先达到前摆目标角度；左右腿分别判断。
        extend_score = ((self.leg_length - state.extend_start_length) / cfg["landing_extension_m"][:, None]).clamp(0, 1)
        over_top = ((wheel_x >= (self._value("obstacle_end") + cfg["wheel_edge_margin_m"])[:, None])
                    & (wheel_x <= self._value("max_x")[:, None])
                    & (wheel_bottom >= (cfg["top_height"] - cfg["wheel_bottom_tolerance_m"])[:, None]))
        state.extend_progress.copy_(((extend_score - state.extend_peak).clamp_min(0) * over_top).mean(dim=1) * valid)
        state.extend_peak.copy_(torch.where(active[:, None], torch.maximum(state.extend_peak, extend_score), state.extend_peak))

        # 两个主轮都须有来自顶部台面的真实向上接触力；侧壁、空中或单轮接触不算支撑。
        links = torch.as_tensor(self.wheel_links_idx, device=self.device) + self.robot.link_start
        is_a = contacts["link_a"][..., None] == links
        is_b = contacts["link_b"][..., None] == links
        force_z = torch.where(is_a, contacts["force_a"][..., 2, None], contacts["force_b"][..., 2, None])
        surface = (contacts["valid_mask"]
                   & ((contacts["position"][..., 2] - cfg["top_height"][:, None]).abs()
                      <= cfg["support_surface_tolerance_m"][:, None])
                   & (contacts["normal"][..., 2].abs() >= cfg["support_normal_z"][:, None])
                   & (contacts["position"][..., 0] - self.terrain_spawn_centers[:, :1] >= self._value("obstacle_end")[:, None])
                   & (contacts["position"][..., 0] - self.terrain_spawn_centers[:, :1] <= self._value("max_x")[:, None])
                   & self._task_in_corridor(contacts["position"][..., :2]))
        supported = (surface[..., None] & (is_a | is_b) & (force_z > 0)).any(dim=1).all(dim=1)
        state.wheel_support_event.copy_(valid & on_top & supported & ~state.wheel_support_awarded)
        state.wheel_support_awarded |= state.wheel_support_event

    def _update_stairs_up_landing_distance(self, contacts, cfg):
        """第一次任一轮真实触台时，按接触点距台阶入口沿的水平距离发一次奖励。"""
        state = self.stairs_up_buffers
        state.landing_distance_event.zero_()
        if contacts["valid_mask"].shape[1] == 0:
            return
        wheel_links = torch.as_tensor(self.wheel_links_idx, device=self.device) + self.robot.link_start
        is_a = (contacts["link_a"][..., None] == wheel_links).any(dim=-1)
        is_b = (contacts["link_b"][..., None] == wheel_links).any(dim=-1)
        force_z = torch.where(is_a, contacts["force_a"][..., 2], contacts["force_b"][..., 2])
        position = contacts["position"]
        local_x = position[..., 0] - self.terrain_spawn_centers[:, :1]
        distance = local_x - self._value("obstacle_end")[:, None]
        on_top = (
            contacts["valid_mask"] & (is_a | is_b) & (force_z > 0)
            & (distance >= 0) & (local_x <= self._value("max_x")[:, None])
            & self._task_in_corridor(position[..., :2])
            & ((position[..., 2] - cfg["top_height"][:, None]).abs()
               <= cfg["support_surface_tolerance_m"][:, None])
            & (contacts["normal"][..., 2].abs() >= cfg["support_normal_z"][:, None])
        )
        first = state.task_mask & on_top.any(dim=1) & ~state.first_landing_seen
        # 同拍多个轮/接触点落台时取最近沿的落点，避免用较远轮掩盖贴沿落地。
        nearest = torch.where(on_top, distance, torch.inf).amin(dim=1)
        state.landing_distance_event.copy_(torch.where(first, nearest, 0))
        state.first_landing_distance.copy_(torch.where(first, nearest, state.first_landing_distance))
        # 即使第一次恰好落在边沿（距离为 0），也必须锁存，后续滚动不能补领。
        state.first_landing_seen |= first

    def _update_stairs_up_wheel_wall_force(self, contacts, cfg):
        """只累计轮子与台阶竖直立面的法向接触力，排除台面支撑及切向摩擦。"""
        state = self.stairs_up_buffers
        wheel_links = torch.as_tensor(self.wheel_links_idx, device=self.device) + self.robot.link_start
        is_a = (contacts["link_a"][..., None] == wheel_links).any(dim=-1)
        is_b = (contacts["link_b"][..., None] == wheel_links).any(dim=-1)
        force = torch.where(is_a[..., None], contacts["force_a"], contacts["force_b"])
        position = contacts["position"]
        local_x = position[..., 0] - self.terrain_spawn_centers[:, :1]
        tolerance = cfg["wall_surface_tolerance_m"][:, None]
        on_wall = (
            contacts["valid_mask"] & (is_a | is_b)
            & (contacts["normal"][..., 2].abs() <= cfg["wall_normal_z_max"][:, None])
            & (local_x >= cfg["approach_edge"][:, None] - tolerance)
            & (local_x <= self._value("obstacle_end")[:, None] + tolerance)
            & self._task_in_corridor(position[..., :2])
            & (position[..., 2] >= cfg["lower_height"][:, None] - tolerance)
            & (position[..., 2] <= cfg["top_height"][:, None] + tolerance)
        )
        # 逐接触点投影后求和，双轮/多个接触点的力不会因方向相反而抵消。
        normal_force = (force * contacts["normal"]).sum(dim=-1).abs()
        state.wheel_wall_force.copy_(torch.where(on_wall, normal_force, 0).sum(dim=1) * state.task_mask)

    def _stairs_up_lower_contact(self, wheels, cfg):
        """逐轮确认台下地面接触，排除台面、腾空和立面上的高处接触。"""
        wheel_x = wheels[..., 0] - self.terrain_spawn_centers[:, :1]
        return (
            (wheel_x < cfg["approach_edge"][:, None])
            & self._task_in_corridor(wheels[..., :2])
            & ((wheels[..., 2] - self.wheel_radius - cfg["lower_height"][:, None]).abs()
               <= cfg["wheel_bottom_tolerance_m"][:, None])
            & (self.wheel_contact > 0.5)
        )

    def _update_stairs_up_approach(self, cfg, on_lower_ground):
        """近台阶伸腿抬升的有界增量；全回合最高值防止反复升降或离开区域刷分。"""
        state = self.stairs_up_buffers
        local = self.base_pos[:, :2] - self.terrain_spawn_centers
        distance = cfg["approach_edge"] - local[:, 0]
        eligible = (
            state.task_mask & ~state.support_awarded & (self.base_contact < 0.5)
            & (distance > 0) & (distance <= cfg["approach_distance_m"])
            & self._task_in_corridor(self.base_pos[:, :2]) & on_lower_ground
            & (self.base_euler[:, 0].abs() <= self._value("max_roll_deg"))
            & (self.base_euler[:, 1].abs() <= self._value("max_pitch_deg"))
        )
        # 底盘下底面略高于眼前台阶即可，不要求固定机体原点高度，也不奖励继续抬高。
        target = cfg["approach_top_height"] + cfg["approach_clearance_m"]
        height = torch.minimum(state.bottom_height(self), target)
        remaining_height = (target - state.approach_start_height).clamp_min(1e-6)
        state.approach_lift_progress.copy_(
            (height - state.approach_peak_height).clamp_min(0) / remaining_height * eligible)
        # 区域外或腾空抬升也更新最高值，随后进入区域/落地不能补领那段奖励。
        state.approach_peak_height.copy_(torch.maximum(state.approach_peak_height, height))

    def _update_stairs_up_backswing(self, wheels, cfg):
        """进入前摆阶段前的后摆增量；触台后不再要求接近距离或轮子尚未越沿。"""
        state = self.stairs_up_buffers
        local = self.base_pos[:, :2] - self.terrain_spawn_centers
        distance = cfg["approach_edge"] - local[:, 0]
        wheel_x = wheels[..., 0] - self.terrain_spawn_centers[:, :1]
        approaching = ((distance.abs() <= cfg["approach_distance_m"])
                       & (wheel_x < cfg["approach_edge"][:, None]).all(dim=1)
                       & (self.base_contact < 0.5))
        eligible = (
            state.task_mask & ~state.landing_started & ~state.wheels_landed & ~state.fallen_off
            & (state.top_contact_seen | approaching) & ~self._tilt_failure()
            & self._task_in_corridor(self.base_pos[:, :2])
            & self._task_in_corridor(wheels[..., :2]).all(dim=1)
        )
        # 正腿角代表轮子相对机体向 -x 后摆；与机体世界俯仰角无关。
        target = torch.deg2rad(cfg["backswing_angle_deg"])
        score = (self.leg_angle.clamp_min(0) / target[:, None]).clamp_max(1)
        state.backswing_progress.copy_((score - state.backswing_peak).clamp_min(0).mean(dim=1) * eligible)
        # 仅在有效区域/阶段累计峰值；离开再进入不会清空已奖励进度。
        state.backswing_peak.copy_(torch.where(
            eligible[:, None], torch.maximum(state.backswing_peak, score), state.backswing_peak))

    def _update_locomotion_tracking(self):
        self.locomotion_buffers.tracking_errors += torch.stack((
            self._reward_tracking_lin_vel(), self._reward_tracking_ang_vel(), self._reward_base_height()), dim=1)

    def _update_stairs_down_landing(self, wheels, wheel_heights):
        state = self.stairs_down_buffers
        # 不要求曾经双轮腾空，兼容单轮先落地或保持接触的平顺下台阶。
        wheel_x = wheels[:, :, 0] - self.terrain_spawn_centers[:, :1]
        beyond_edge = wheel_x > self._value("obstacle_end")[:, None]
        on_surface = ((wheels[:, :, 2] - wheel_heights - self.wheel_radius).abs()
                      <= self._value("wheel_surface_tolerance_m")[:, None])
        lower_contact = (beyond_edge & on_surface & (self.wheel_contact > 0.5)).any(dim=1)
        landed = state.task_mask & lower_contact & ~state.landing_started & (self.task_steps > 1)
        state.landing_steps_left.sub_(1).clamp_min_(0)
        state.landing_steps_left.copy_(torch.where(
            landed, state.landing_window_steps[self.terrain_tile_index], state.landing_steps_left))
        state.landing_started |= landed
        # 窗口内再次弹起或机身触地也继续计入振荡；每回合只启动一次。
        state.landing_gate.copy_((state.landing_steps_left > 0).to(self.base_pos.dtype))

    def _tilt_failure(self):
        return ((self.base_euler[:, 0].abs() > self._value("termination_if_roll_greater_than"))
                | (self.base_euler[:, 1].abs() > self._value("termination_if_pitch_greater_than")))

    def _episode_timeout(self):
        # 独立计时不受 runner.init_at_random_ep_len 修改的 episode_length_buf 干扰。
        duration = self._value("episode_length_s") + self.intermittent.timeout_extension(self)
        return self.task_steps >= (duration / self.dt).ceil().long()

    def _task_termination(self):
        outside = ~self._task_in_corridor(self.base_pos[:, :2])
        stalled = self.stall_steps * self.dt >= self._value("stall_timeout_s")
        return self.traversal & (outside | (stalled & ~self.success) | (self.stairs_up_buffers.fallen_off & ~self.parking.drill)
                                 | self.intermittent.failed)

    def _task_success(self):
        return self.success

    def _get_reward_scale(self, name):
        scale = self.reward_tables[name][self.terrain_tile_index]
        if name in {"task_progress", "task_success", "stairs_up_support", "stairs_up_retract",
                    "stairs_up_takeoff", "stairs_up_landing_distance", "stairs_up_approach_lift",
                    "stairs_up_backswing", "stairs_up_forward_swing", "stairs_up_extend",
                    "stairs_up_wheel_support", "stairs_up_wheel_forward_action"}:
            scale = scale * ~self.intermittent.reward_paused * ~self.parking.drill
        return scale

    def _record_task_outcomes(self):
        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if not len(ids):
            return
        passed = self.success & ~self.terminated_buf
        for i, stage in enumerate(self.terrain.stages):
            if stage["terrain"]["preset"] != "plane":
                continue
            cfg = stage["tracking_success"]
            limits = self.base_pos.new_tensor([cfg["lin_vel_rmse"], cfg["ang_vel_rmse"], cfg["height_rmse"]]).square()
            mse = self.locomotion_buffers.tracking_errors / self.task_steps.clamp_min(1)[:, None]
            passed |= ((self.terrain_tile_index == i) & ~self.terminated_buf & self._episode_timeout()
                       & (mse <= limits).all(dim=1))
        drills = ids[self.parking.drill[ids]]
        self.parking.observe(self.terrain_tile_index[drills], passed[drills])
        # 停车达标不是跨越成功；不能让简单的停车回合推动地形难度升级。
        crossing_ids = ids[~self.parking.drill[ids]]
        rows = torch.stack((self.terrain_tile_index[crossing_ids], self.terrain.assignment_epoch[crossing_ids],
                            passed[crossing_ids].long()), dim=1).cpu().tolist()
        groups = {}
        for tile, epoch, success in rows:
            task, level = self.terrain.variants[tile]
            wins, count = groups.get((task, level, epoch), (0, 0))
            groups[task, level, epoch] = wins + success, count + 1
        for (task, level, epoch), (wins, count) in groups.items():
            self.terrain.controller.observe(task, level, epoch, wins, count)
        self.outcome_success.copy_(passed)

    def _reset_idx(self, env_idx=None):
        metrics = {}
        if env_idx is not None and env_idx.any():
            for i, name in enumerate(self.tasks):
                mask = env_idx & (self.task_ids == i)
                metrics[f"task/{name}/env_fraction"] = (self.task_ids == i).float().mean().item()
                metrics[f"task/{name}/difficulty"] = self.terrain.controller.levels[i]
                metrics[f"task/{name}/curriculum_success_rate"] = self.terrain.controller.rates[i]
                metrics[f"task/{name}/window_episodes"] = self.terrain.controller.counts[i]
                if mask.any():
                    crossing = mask & ~self.parking.drill
                    if crossing.any():
                        metrics[f"task/{name}/success"] = self.outcome_success[crossing].float()
                    if self.parking.enabled and name != "locomotion":
                        tile = self.terrain.lookup[i, self.terrain.controller.levels[i]]
                        prefix = f"task/{name}/parking_"
                        metrics[prefix + "unlocked"] = self.parking.unlocked[tile].float().item()
                        metrics[prefix + "pre_window_rate"] = self.parking.rates[tile].item()
                        metrics[prefix + "pre_window_count"] = self.parking.counts[tile].item()
                        selected = mask & self.parking.drill
                        if selected.any():
                            metrics[prefix + "pre_success"] = self.outcome_success[selected].float()
                    metrics[f"task/{name}/failure"] = self.terminated_buf[mask].float()
                    metrics[f"task/{name}/duration_s"] = self.task_steps[mask] * self.dt
                    if name != "locomotion":
                        metrics[f"task/{name}/progress_m"] = self.farthest_x[mask] - self.episode_start_x[mask]
                        state = self.intermittent
                        metrics[f"task/{name}/intermittent_fraction"] = state.enabled[mask].float()
                        intermittent = crossing & state.enabled
                        ordinary = crossing & ~state.enabled
                        if ordinary.any():
                            metrics[f"task/{name}/ordinary_success"] = self.outcome_success[ordinary].float()
                        if intermittent.any():
                            prefix = f"task/{name}/intermittent_"
                            metrics[prefix + "success"] = self.outcome_success[intermittent].float()
                            metrics[prefix + "pre_stop_completed"] = (state.completed_stops[intermittent] >= 1).float()
                            metrics[prefix + "stop_failure"] = state.failed[intermittent].float()
                            metrics[prefix + "max_stop_drift_m"] = state.max_stop_drift[intermittent]
                    if name == "stairs_up":
                        state = self.stairs_up_buffers
                        metrics[f"task/{name}/support_reached"] = state.support_awarded[mask].float()
                        metrics[f"task/{name}/retract_reached"] = state.retract_awarded[mask].float()
                        metrics[f"task/{name}/takeoff_reached"] = state.takeoff_awarded[mask].float()
                        metrics[f"task/{name}/first_landing_reached"] = state.first_landing_seen[mask].float()
                        metrics[f"task/{name}/first_landing_distance_m"] = state.first_landing_distance[mask]
                        metrics[f"task/{name}/landing_phase_reached"] = state.landing_started[mask].float()
                        metrics[f"task/{name}/forward_swing_progress"] = state.forward_peak[mask].mean(dim=1)
                        metrics[f"task/{name}/extension_progress"] = state.extend_peak[mask].mean(dim=1)
                        metrics[f"task/{name}/wheel_support_reached"] = state.wheel_support_awarded[mask].float()
                    if name == "locomotion":
                        mse = self.locomotion_buffers.tracking_errors[mask] / self.task_steps[mask].clamp_min(1)[:, None]
                        for j, key in enumerate(("lin_vel", "ang_vel", "height")):
                            metrics[f"task/{name}/{key}_mse"] = mse[:, j]
        super()._reset_idx(env_idx)
        if metrics:
            self.extras["episode"].update(metrics)

    def terrain_state_dict(self):
        return self.terrain.controller.state_dict()

    def parking_state_dict(self):
        return self.parking.state_dict()

    def reset(self):
        saved = getattr(self, "_pending_terrain_state", None)
        if saved is not None:
            self.terrain.controller.load_state_dict(saved)
            self._pending_terrain_state = None
        parking = getattr(self, "_pending_parking_state", None)
        if parking is not None:
            self.parking.load_state_dict(parking)
            self._pending_parking_state = None
        return super().reset()
