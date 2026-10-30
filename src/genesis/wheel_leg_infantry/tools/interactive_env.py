"""连续运行的部署式仿真：用户命令驱动行为，不运行训练任务回合。"""

import math

import genesis as gs
import torch

from ..env import WheelLegInfantryEnv
from ..tasks.manager import TaskManager
from ..tasks.recovery.buffers import RecoveryBuffers
from ..tasks.jump.config import MODE_NAMES
from ..tasks.jump.height_reference import sample_height_reference, validate_height_reference
from ..tasks.jump.phase import phase_encoding, validate_phase_durations


class InteractiveEnv(WheelLegInfantryEnv):
    def __init__(self, num_envs, *, control_configs, **kwargs):
        if num_envs != 1 or not kwargs.get("evaluation"):
            raise ValueError("InteractiveEnv requires one evaluation robot")
        self.control_configs = control_configs
        self.control_mode = "drive"
        self.drive_command = (0.0, 0.0, 0.22)
        self.jump_mode = 0
        self.jump_steps = 0
        self.jump_velocity = 0.0
        self._normal_motor_params = None
        super().__init__(num_envs, **kwargs)
        self._normal_motor_params = {key: value.clone() for key, value in self.domain_rand.motor_nominals.items()}
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
        cfg = self.control_configs["jump"]["env_cfg"]
        params = dict(cfg.get("jump_motor_params", {}))
        if mode == 2 and cfg.get("jump_step_40cm_joint_kd") is not None:
            params["joint_kd"] = cfg["jump_step_40cm_joint_kd"]
        self._apply_motor_params({key: self.commands.new_tensor(value) for key, value in params.items()
                                  if value is not None})
        self._write_commands()
        print(f"[eval] jump: {MODE_NAMES[mode]}")

    def request_recovery(self):
        self.return_to_drive()
        self.control_mode = "recovery"
        self.recovery.reset()
        self.drive_command = (0.0, 0.0, self.recovery.config["height"])
        self._write_commands()
        print("[eval] recovery requested at current pose")

    def return_to_drive(self):
        if self.control_mode == "jump" and self._normal_motor_params is not None:
            self._apply_motor_params(self._normal_motor_params)
        self.control_mode = "drive"
        self.recovery.active.zero_()
        self.jump_mode_one_hot.zero_()
        self.jump_phase.zero_()

    def reset(self):
        # Only startup and the explicit reset key use this path. Behavior
        # requests never reset qpos, velocities, sensors, action, or GRU state.
        self.return_to_drive()
        self.drive_command = (0.0, 0.0, 0.22)
        return super().reset()

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
