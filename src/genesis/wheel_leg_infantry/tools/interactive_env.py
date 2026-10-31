"""连续运行的部署式仿真：用户命令驱动行为，不运行训练任务回合。"""

import math
from copy import deepcopy

import genesis as gs
import torch

from ..env import WheelLegInfantryEnv
from ..tasks.manager import TaskManager
from ..tasks.recovery.buffers import RecoveryBuffers
from ..tasks.jump.config import MODE_NAMES
from ..tasks.jump.height_reference import sample_height_reference, validate_height_reference
from ..tasks.jump.phase import phase_encoding, validate_phase_durations
from ..utils.reset import ensure_reset_clearance


class InteractiveEnv(WheelLegInfantryEnv):
    def __init__(self, num_envs, *, control_configs, recovery_reset_configs=None, **kwargs):
        if num_envs != 1 or not kwargs.get("evaluation"):
            raise ValueError("InteractiveEnv requires one evaluation robot")
        self.control_configs = control_configs
        # Inject already-parsed configurations; pose files/providers belong to
        # the configuration layer, not the simulation or viewer callbacks.
        if recovery_reset_configs is None:
            recovery_reset_configs = [control_configs["recovery"]["env_cfg"]]
        if not recovery_reset_configs:
            raise ValueError("recovery_reset_configs must contain at least one pose")
        self.recovery_reset_configs = deepcopy(recovery_reset_configs)
        self.control_mode = "drive"
        self.drive_command = (0.0, 0.0, 0.22)
        self.jump_mode = 0
        self.jump_steps = 0
        self.jump_velocity = 0.0
        super().__init__(num_envs, **kwargs)
        self.command_ranges = {key: tuple(value) for key, value in
                               self.task_manager.tasks["locomotion"].command_cfg.items()
                               if key in ("lin_vel_range", "ang_vel_range", "base_height_range")}
        # Manual control must be able to ramp from/return to zero, including
        # checkpoints trained with a positive-only speed sampling interval.
        for key in ("lin_vel_range", "ang_vel_range"):
            low, high = self.command_ranges[key]
            self.command_ranges[key] = (min(0.0, low), max(0.0, high))

    def _register_task_curriculum(self):
        # Reuse common reset/configuration plumbing with one fixed component.
        # Neither the scheduler nor the terrain changes when commands change.
        self.task_manager = TaskManager(self, self.env_cfg["tasks"])
        self.terrain = self.task_manager.tasks["locomotion"].terrain

    def _add_terrain(self):
        super()._add_terrain()
        # Plane returns one entity; segmented terrain returns multiple boxes.
        if not isinstance(self.friction_terrain_entities, (tuple, list)):
            self.friction_terrain_entities = (self.friction_terrain_entities,)

    def _initialize_task_buffers(self):
        super()._initialize_task_buffers()
        recovery_cfg = self.control_configs["recovery"]["env_cfg"]["stand_up"]
        self.recovery = RecoveryBuffers(recovery_cfg, self.num_envs, self.dt, self.device)
        self.recovery.active.zero_()
        jump_cfg = self.control_configs["jump"]["env_cfg"]
        durations = validate_phase_durations(jump_cfg["jump_phase_durations_s"])
        self.jump_cycle_s = sum(durations)
        self.jump_horizon = math.ceil(self.jump_cycle_s / self.dt)
        self.height_references = []
        for points in jump_cfg["jump_modes"]["height_references"]:
            times, heights = validate_height_reference(points, self.jump_cycle_s)
            self.height_references.append((self.commands.new_tensor(times), self.commands.new_tensor(heights)))
        self.jump_mode_one_hot = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.jump_phase = torch.zeros((self.num_envs, 6), device=self.device, dtype=gs.tc_float)

    def set_drive_command(self, velocity, yaw_rate, height):
        self.drive_command = (float(velocity), float(yaw_rate), float(height))
        self._write_commands()

    def request_jump(self, mode=0):
        if not 0 <= mode < len(MODE_NAMES):
            raise ValueError(f"jump mode must be in [0, {len(MODE_NAMES) - 1}]")
        if self.control_mode == "jump":
            return
        self.return_to_drive()
        self.control_mode = "jump"
        self.jump_mode = mode
        self.jump_steps = 0
        self.jump_velocity = self.drive_command[0]
        self._write_commands()
        print(f"[eval] jump: {MODE_NAMES[mode]}")

    def request_recovery(self, reset_cfg=None):
        """随机（或显式指定）复位姿态后起身；调用方须同步重置策略历史。"""
        label = "explicit"
        if reset_cfg is None:
            index = int(torch.randint(len(self.recovery_reset_configs), (1,), device=self.device).item())
            reset_cfg = self.recovery_reset_configs[index]
            label = f"{index + 1}/{len(self.recovery_reset_configs)}"
        self.reset(pose_cfg=reset_cfg)
        self.control_mode = "recovery"
        self.recovery.reset()
        self.drive_command = (0.0, 0.0, self.recovery.config["height"])
        self._write_commands()
        print(f"[eval] recovery: reset pose {label}")

    def return_to_drive(self):
        self.control_mode = "drive"
        self.recovery.active.zero_()
        self.jump_mode_one_hot.zero_()
        self.jump_phase.zero_()

    def reset(self, *, pose_cfg=None):
        """共用物理/传感器复位路径；临时姿态不覆盖正常出生配置。"""
        self.return_to_drive()
        self.drive_command = (0.0, 0.0, 0.22)
        if pose_cfg is None:
            return super().reset()
        task = self.task_manager.tasks["locomotion"]
        original_cfg, original_dof_pos = task.env_cfg, task.init_dof_pos
        range_keys = ("base_init_pos_range", "base_init_rpy_offset_range_deg",
                      "base_init_lin_vel_range", "base_init_ang_vel_range")
        task.env_cfg = deepcopy(original_cfg)
        try:
            task._apply_reset_ranges({key: pose_cfg[key] for key in range_keys if key in pose_cfg})
            task.env_cfg["reset_joint_pos"] = deepcopy(pose_cfg.get("reset_joint_pos", {}))
            task.bind_buffers()
            return super().reset()
        finally:
            task.env_cfg = original_cfg
            task._apply_reset_ranges({key: original_cfg[key] for key in range_keys})
            task.init_dof_pos = original_dof_pos

    def _sample_base_reset_state(self, env_ids):
        state = super()._sample_base_reset_state(env_ids)
        # R 键的旧 JSON 也按当前模型校正高度，避免模型更新后首帧穿地。
        ensure_reset_clearance(self.robot, env_ids, state)
        return state

    def _write_commands(self):
        self.commands[:] = self.commands.new_tensor(self.drive_command)
        self.jump_mode_one_hot.zero_()
        self.jump_phase.zero_()
        if self.control_mode == "recovery":
            self.recovery.enforce_commands(self.commands)
        elif self.control_mode == "jump":
            elapsed = self.commands.new_full((self.num_envs,), self.jump_steps * self.dt)
            times, heights = self.height_references[self.jump_mode]
            self.commands[:, 0] = self.jump_velocity
            self.commands[:, 1] = 0.0
            self.commands[:, 2] = sample_height_reference(elapsed, times, heights)
            self.jump_mode_one_hot[:, self.jump_mode] = 1.0
            self.jump_phase.copy_(phase_encoding(elapsed, self.jump_cycle_s))

    def _resample_commands(self, envs_mask):
        self._write_commands()

    def _get_jump_observations(self):
        return self.jump_mode_one_hot, self.jump_phase

    def _constrain_joint_targets(self, targets):
        return targets if self.control_mode == "jump" else super()._constrain_joint_targets(targets)

    def _update_task_state(self):
        if self.control_mode == "recovery":
            self.recovery.update(self)
            if not bool(self.recovery.active[0]):
                self.return_to_drive()
                print("[eval] standing stable; drive commands enabled")
        elif self.control_mode == "jump":
            self.jump_steps += 1
            if self.jump_steps >= self.jump_horizon:
                self.return_to_drive()
                print("[eval] jump command complete; drive commands enabled")

    def _update_termination(self):
        # Tilt, contact, success, elapsed time, and terrain boundaries are
        # diagnostic states in deployment, not reasons to teleport the robot.
        if bool(self.scene.rigid_solver.get_error_envs_mask().any()):
            raise RuntimeError("Physics solver error; continuous evaluation stopped")
        self.terminated_buf.zero_()
        self.reset_buf.zero_()
        self.extras["task_successes"] = torch.zeros_like(self.reset_buf)
        return torch.zeros_like(self.reset_buf)
