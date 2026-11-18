"""越障组件：按地形管理状态、结果统计和附加奖励。"""

from copy import deepcopy
import math

import torch

from .buffers import TerrainCrossBuffers
from .rewards import TerrainCrossRewards
from .reset_sampling import TerrainResetSampler
from ..metrics import TaskMetrics
from ...utils.tensor_utils import as_range_tensors, sample_uniform
from ...utils.tracking_gate import DEFAULT_TRACKING_GATE_CFG, merge_tracking_gate_config


class TerrainCross:
    def __init__(self, env, config):
        self.env, self.config = env, deepcopy(config)
        self.names = tuple(dict.fromkeys(name for name in env.terrain.tile_types if name != "plane"))
        goals = []
        errors = []
        margin = env.terrain.config["boundary_margin"] or 0.0
        limit = env.terrain.tile_size[0] / 2 - margin
        for name, geometry in zip(env.terrain.tile_types, env.terrain.geometries):
            goal = 0.0 if name == "plane" else geometry.traversal_end + self.config["finish_distance_m"]
            goals.append(goal)
            if name != "plane" and not 0 < goal < limit:
                errors.append(f"{name}: finish x={goal:g} must lie inside (0, {limit:g})")
        if (not math.isfinite(self.config["success_hold_s"]) or self.config["success_hold_s"] < 0
                or not math.isfinite(self.config["finish_distance_m"]) or self.config["finish_distance_m"] < 0
                or not 0 < self.config["max_tilt_deg"] < 90):
            errors.append("success_hold_s/finish_distance_m must be finite and non-negative; max_tilt_deg must be in (0, 90)")
        if errors:
            raise ValueError("Invalid terrain_cross config:\n- " + "\n- ".join(errors))
        self._tile_goals = env.commands.new_tensor(goals)
        self.buffers = {
            name: TerrainCrossBuffers(
                env, self.config,
                landing_config=self.config["terrains"][name].get("landing_stability") if name == "stairs" else None,
            )
            for name in self.names
        }
        self.rewards = {
            name: TerrainCrossRewards(
                env, self.buffers[name], self.config["terrains"][name]["reward_scales"],
                wheel_wall_config=self.config["terrains"][name].get("wheel_wall_force"),
            )
            for name in self.names
        }
        self.metrics = TaskMetrics(env.device, self.names)
        level_keys = tuple(dict.fromkeys((name, level) for name, level in zip(env.terrain.tile_types, env.terrain.tile_difficulties) if name != "plane"))
        self.level_metrics = TaskMetrics(env.device, tuple(f"{name}/level_{level}" if level is not None else f"{name}/default"
                                                        for name, level in level_keys))
        self.value = torch.zeros_like(env.reward_buf)
        self.terrain_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._tile_groups = self.terrain_ids.new_tensor([
            self.names.index(name) if name != "plane" else -1 for name in env.terrain.tile_types
        ])
        self._tile_level_groups = self.terrain_ids.new_tensor([
            level_keys.index((name, level)) if name != "plane" else -1
            for name, level in zip(env.terrain.tile_types, env.terrain.tile_difficulties)
        ])
        self.reset_sampler = TerrainResetSampler(env, self.config["terrains"], self.names, self._tile_groups)
        self.commands = torch.zeros_like(env.commands)
        self.parking = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.command_limits = {name: self._command_limits(name) for name in self.names}
        self._configure_tracking_gate()

    @property
    def assigned(self):
        return self.terrain_ids >= 0

    def special_tiles(self, tiles):
        return self._tile_groups[tiles] >= 0

    def prepare_reset(self, env_ids, tiles):
        """先确定停车回合，出生位置和后续命令使用同一次抽样。"""
        self.parking[env_ids] = False
        if getattr(self.env, "manual_control", False):
            return
        groups = self._tile_groups[tiles]
        for index, name in enumerate(self.names):
            ids = env_ids[groups == index]
            probability = self.config["terrains"][name]["commands"]["standing_probability"]
            if probability is None:
                probability = self.env.command_cfg["standing_probability"]
            if ids.numel() and probability > 0.0:
                self.parking[ids] = torch.rand(len(ids), device=self.env.device) < probability

    def adjust_reset(self, positions, rpy, lin_vel, ang_vel, tiles, env_ids):
        self.prepare_reset(env_ids, tiles)
        self.reset_sampler.apply(positions, rpy, lin_vel, ang_vel, tiles, self.parking[env_ids])

    def _configure_tracking_gate(self):
        configs = [merge_tracking_gate_config(None, self.config["terrains"][name]["tracking_gate"]) for name in self.names]
        self._tracking_gate_table = self.env.commands.new_tensor([
            [config[key] for key in DEFAULT_TRACKING_GATE_CFG] for config in configs
        ])

    def tracking_gate_parameters(self, defaults):
        """平地跟随全局课程，特殊地形使用独立门控，持续至物理回合结束。"""
        if not self.names:
            return defaults
        parameters = self._tracking_gate_table[self.terrain_ids.clamp_min(0)]
        assigned = self.assigned
        return {key: torch.where(assigned, parameters[:, index], defaults[key])
                for index, key in enumerate(DEFAULT_TRACKING_GATE_CFG)}

    def _command_limits(self, name):
        config = self.config["terrains"][name]["commands"]
        probability = config["standing_probability"]
        if probability is not None and not 0.0 <= float(probability) <= 1.0:
            raise ValueError(f"terrain_cross.{name}.commands.standing_probability must be in [0, 1] or None")
        return as_range_tensors(
            [config[key] for key in ("lin_vel_range", "ang_vel_range", "base_height_range")],
            3, f"terrain_cross.{name}.commands", device=self.env.device, dtype=self.env.commands.dtype,
        )

    def apply_commands(self, commands):
        if getattr(self.env, "manual_control", False):
            return
        commands[self.assigned] = self.commands[self.assigned]

    def reset(self, env, mask):
        self.terrain_ids[mask] = self._tile_groups[env.terrain_tile_index[mask]]
        self.commands[mask] = 0.0
        goals = self._tile_goals[env.terrain_tile_index]
        for index, name in enumerate(self.names):
            selected = mask & (self.terrain_ids == index)
            count = int(selected.sum())
            if count:
                # 特殊地形命令仅在 reset 采样，普通越障和停车均持有整回合。
                self.commands[selected] = sample_uniform(*self.command_limits[name], (count,))
                self.commands[selected & self.parking, :2] = 0.0
            # 停车回合不发越障事件、不记录越障失败，也不推进地形难度。
            self.buffers[name].reset(env, mask, (self.terrain_ids == index) & ~self.parking, goals)
            self.rewards[name].reset(mask)

    def update(self, env):
        if not self.names:
            return
        wheels = env.robot.get_links_pos(env.wheel_links_idx)
        rear_x = (wheels[..., 0] - env.terrain_spawn_centers[:, :1]).amin(dim=1)
        half_y = env.terrain.tile_size[1] / 2 - (env.terrain.config["boundary_margin"] or 0.0)
        in_lane = ((wheels[..., 1] - env.terrain_spawn_centers[:, 1:2]).abs() < half_y).all(dim=1)
        in_lane &= ~env.terrain.out_of_bounds(env.base_pos[:, :2], env.terrain_spawn_centers)
        level_ids = self._tile_level_groups[env.terrain_tile_index]
        for buffers in self.buffers.values():
            buffers.update(env, rear_x, in_lane)
            if buffers.landing is not None:
                edge_x = self._tile_goals[env.terrain_tile_index] - self.config["finish_distance_m"]
                buffers.landing.update(env, buffers, wheels, edge_x)
            self.metrics.start(buffers.started_event, self.terrain_ids)
            finished = (buffers.success_event > 0) | (buffers.failure_event > 0)
            self.metrics.finish(finished, buffers.success, invalid=env.terminated_buf, modes=self.terrain_ids)
            self.level_metrics.start(buffers.started_event, level_ids)
            self.level_metrics.finish(finished, buffers.success, invalid=env.terminated_buf, modes=level_ids)
            if not env.evaluation:
                env.terrain.record_results(env.terrain_tile_index, finished, buffers.success)

    def reward_scale_overrides(self, scales):
        """整回合按所属地形覆盖同名 locomotion 权重；未配置的项继承全局。"""
        overrides = {}
        for index, name in enumerate(self.names):
            terrain_scales = self.rewards[name].scales
            for key in terrain_scales.keys() & scales.keys():
                overrides[key] = torch.where(
                    self.terrain_ids == index, terrain_scales[key], overrides.get(key, scales[key]),
                )
        return overrides

    def compute_rewards(self, *, exclude=()):
        self.value.zero_()
        for name, rewards in self.rewards.items():
            buffers = self.buffers[name]
            self.value.add_(rewards.compute(buffers.was_active | (buffers.failure_event > 0), exclude=exclude))
        return self.value

    def episode_metrics(self, mask):
        result = {}
        for name, buffers in self.buffers.items():
            prefix = f"terrain_cross/{name}"
            assigned = mask & (self.terrain_ids == self.names.index(name))
            if assigned.any():
                result[f"{prefix}/parking"] = self.parking[assigned].float()
            selected = mask & buffers.started
            if not selected.any():
                continue
            result.update(self.rewards[name].metrics(selected, prefix))
            result[f"{prefix}/progress"] = buffers.max_progress[selected].clone()
            result[f"{prefix}/duration_s"] = buffers.elapsed_steps[selected].clone() * self.env.dt
            result[f"{prefix}/success"] = buffers.success[selected].float()
        return result

    def apply_curriculum(self, values):
        for name, options in values["terrains"].items():
            for section, overrides in options.items():
                self.config["terrains"][name][section].update(deepcopy(overrides))
            if name not in self.buffers:
                continue
            if "reward_scales" in options:
                self.rewards[name].set_scales(options["reward_scales"])
            if "reset" in options:
                self.reset_sampler.configure(name, self.config["terrains"][name]["reset"])
            if "commands" in options:
                self.command_limits[name] = self._command_limits(name)
            if "landing_stability" in options and self.buffers[name].landing is not None:
                self.buffers[name].landing.configure(self.config["terrains"][name]["landing_stability"])
            if "wheel_wall_force" in options:
                self.rewards[name].configure_wheel_wall_force(self.config["terrains"][name]["wheel_wall_force"])
        if any("tracking_gate" in options for options in values["terrains"].values()):
            self._configure_tracking_gate()

    def state_dict(self):
        return {name: rewards.state_dict() for name, rewards in self.rewards.items()}

    def load_state_dict(self, state):
        for name, rewards in self.rewards.items():
            if name in state:
                rewards.load_state_dict(state[name])
