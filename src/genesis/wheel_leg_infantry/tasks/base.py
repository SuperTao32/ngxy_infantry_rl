"""任务组件协议：共享物理状态，独立持有配置、奖励和回合状态。

所有任务使用统一 batch 的索引。写入公共 buffer 时必须限制到 ``mask``；
专用 buffer 可以覆盖整个 batch，任务管理器仅收集本任务对应的行。
"""

from copy import deepcopy
import math

import torch

from .task_config import TaskConfiguration
from ..curriculum import CurriculumManager
from ..rewards import CommonRewards
from ..terrains.manager import TerrainManager


class Task(CommonRewards, TaskConfiguration):
    EVENT_REWARDS = {"death", "stand_up_success"}

    def __init__(self, env, name, config, env_ids):
        self.env, self.name, self.env_ids = env, name, env_ids
        self.mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.mask[env_ids] = True
        self.env_cfg = deepcopy({key: value for key, value in env.env_cfg.items() if key != "tasks"})
        self.env_cfg.update(deepcopy(config["env_cfg"]))
        self.command_cfg = deepcopy(env.command_cfg)
        self.command_cfg.update(deepcopy(config["command_cfg"]))
        self.reward_cfg = deepcopy(env.reward_cfg)
        self.reward_cfg.update(deepcopy(config["reward_cfg"]))
        self.obs_cfg = env.obs_cfg
        self.obs_scales = env.obs_scales
        self.curriculum_cfg = deepcopy(config["curriculum_cfg"])
        self.evaluation = env.evaluation
        self.stand_up = None
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)
        self.terrain_tile_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.commands_limit = self._build_command_limits()
        self.reward_functions, self.reward_scales, self.raw_reward_scales, self.episode_sums = {}, {}, {}, {}
        self.tracking_gate_cfg = {}
        self._apply_tracking_gate(self.reward_cfg.get("tracking_gate", {}))
        self.tilt_termination_duration_s = self.env_cfg["tilt_termination_duration_s"]
        self.base_contact_termination_duration_s = self.env_cfg.get("base_contact_termination_duration_s")
        self._apply_termination_limits({})
        self._apply_reset_ranges({key: self.env_cfg[key] for key in (
            "base_init_pos_range", "base_init_rpy_offset_range_deg",
            "base_init_lin_vel_range", "base_init_ang_vel_range",
        )})
        self._apply_reward_scales(self.reward_cfg["reward_scales"])
        self.terrain = self.create_terrain()
        self.curriculum = CurriculumManager(self.curriculum_cfg)
        for target in ("command_ranges", "reward_scales", "tracking_gate", "standing_reward",
                       "termination_limits", "reset_ranges", "terrain"):
            self.curriculum.register_target(target, getattr(self, "_apply_" + target))
        self._register_task_curriculum()
        self.curriculum.update(0, force=True)

    def __getattr__(self, name):
        # Only state is shared; configuration helpers below bind to this task.
        return getattr(self.env, name)

    def create_terrain(self):
        return TerrainManager(self.env_cfg.get("terrain"))

    def bind_buffers(self):
        reset_positions = {**self.env_cfg["default_joint_pos"], **self.env_cfg.get("reset_joint_pos", {})}
        unknown = reset_positions.keys() - {joint.name for joint in self.robot.joints if joint.n_qs == 1}
        if unknown:
            raise ValueError(f"{self.name}: reset positions require named scalar joints: {sorted(unknown)}")
        self.init_dof_pos = self.env.init_dof_pos.clone()
        offset = 0
        for joint in self.robot.joints[1:]:
            if not joint.n_qs:
                continue
            if joint.n_qs != 1:
                raise ValueError("Task reset poses require scalar robot joints")
            self.init_dof_pos[offset] = reset_positions.get(joint.name, 0.0)
            offset += 1
        self._initialize_task_buffers()

    def _apply_reward_scales(self, values):
        for name, value in values.items():
            function = getattr(self, "_reward_" + name)
            scale = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if not torch.isfinite(scale).all():
                raise ValueError(f"{self.name}: nonfinite reward scale {name}")
            self.raw_reward_scales[name] = deepcopy(value)
            self.reward_cfg["reward_scales"][name] = deepcopy(value)
            self.reward_scales[name] = (float(scale) if scale.ndim == 0 else scale) * (1.0 if name in self.EVENT_REWARDS else self.dt)
            self.reward_functions[name] = function
            if name not in self.episode_sums:
                self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
                self.env.episode_sums[f"{self.name}/{name}"] = self.episode_sums[name]

    def _get_reward_scale(self, name):
        return self.reward_scales[name]

    def compute_rewards(self):
        for name, function in self.reward_functions.items():
            values = function() * self._get_reward_scale(name)
            # where also isolates invalid intermediate values from inactive rows.
            values = torch.where(self.mask, values, 0.0)
            self.env.reward_buf.add_(values)
            self.episode_sums[name].add_(values)

    def _apply_terrain(self, values):
        self._apply_terrain_curriculum(values)

    def _apply_terrain_curriculum(self, values):
        if values:
            raise ValueError(f"{self.name}: geometry changes require a task terrain handler")

    def _register_task_curriculum(self):
        pass

    def _initialize_task_buffers(self):
        pass

    def _reset_task_buffers(self, mask):
        pass

    def _resample_commands(self, mask):
        interval = max(1, math.ceil(self.env_cfg["resampling_time_s"] / self.dt))
        selected = mask & (self.episode_length_buf % interval == 0)
        if selected.any():
            commands = self._sample_commands(self.command_cfg, self.commands_limit)
            self.commands[selected] = commands[selected]

    def _update_task_state(self):
        pass

    def _apply_task_control(self):
        pass

    def _episode_timeout(self):
        return self.episode_length_buf >= math.ceil(self.env_cfg["episode_length_s"] / self.dt)

    def _task_termination(self):
        return torch.zeros_like(self.mask)

    def _task_success(self):
        return torch.zeros_like(self.mask)

    def _record_task_outcomes(self):
        pass

    def episode_metrics(self, mask):
        return {}

    def after_step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
