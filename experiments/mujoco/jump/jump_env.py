"""共享一个 MuJoCo 物理状态，在 locomotion 与 jump actor 之间交接。"""

from copy import deepcopy
import math

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from experiments.genesis.wheel_leg_infantry.tasks.jump.height_reference import (
    sample_height_reference, validate_height_reference,
)
from experiments.genesis.wheel_leg_infantry.tasks.jump.phase import phase_encoding, validate_phase_durations
from experiments.genesis.wheel_leg_infantry.tasks.jump.warm_start import validate_locomotion_source
from experiments.mujoco.locomotion.locomotion_env import MujocoLocomotionEnv, tensor


class MujocoJumpEnv:
    def __init__(self, configs, source_configs, vx=0.0, handoff="landing"):
        validate_locomotion_source(source_configs)
        self.configs = deepcopy(configs)
        cfg = self.configs["env_cfg"]
        # Teacher sees the same prefix and action interface used during jump training.
        for key in ("robot_mjcf", "joint_names", "wheel_names", "default_joint_pos",
                    "joint_pos_scale", "wheel_vel_scale", "simulate_action_latency"):
            if cfg[key] != source_configs["env_cfg"][key]:
                raise ValueError(f"Jump/source locomotion contract differs: {key}")
        for key in ("imu", "velocity_estimator"):
            if self.configs["obs_cfg"].get(key) != source_configs["obs_cfg"].get(key):
                raise ValueError(f"Jump/source locomotion observation contract differs: {key}")
        for key, value in source_configs["obs_cfg"]["obs_scales"].items():
            if self.configs["obs_cfg"]["obs_scales"].get(key) != value:
                raise ValueError(f"Jump/source locomotion observation scale differs: {key}")
        if self.configs["obs_cfg"].get("num_policy_obs") != 44:
            raise ValueError("Expected the 44-dimensional jump observation contract")
        durations = validate_phase_durations(cfg["jump_phase_durations_s"])
        self.cycle_s = sum(durations)
        if not math.isclose(self.cycle_s, cfg["episode_length_s"], abs_tol=1e-6):
            raise ValueError("Jump phase durations must sum to episode_length_s")
        times, heights = validate_height_reference(cfg["height_reference"], self.cycle_s)
        self.reference_times, self.reference_heights = tensor(times), tensor(heights)
        self.standing_height = float(np.mean(cfg["locomotion_warmup"]["command_ranges"]["base_height_range"]))
        self.command_vx = float(vx)
        if handoff not in ("landing", "horizon"):
            raise ValueError("handoff must be landing or horizon")
        self.handoff = handoff
        self.sim = MujocoLocomotionEnv(self.configs, (vx, 0, self.standing_height))
        self.dt = self.sim.dt
        self.horizon = round(self.cycle_s / self.dt)
        self.motor_defaults = {}
        self.motor_jump = {}
        counts = {"joint_kp": 4, "joint_kd": 4, "wheel_kd": 2, "joint_force_limit": 4, "wheel_force_limit": 2}
        for key, value in cfg.get("jump_motor_params", {}).items():
            if key not in counts:
                raise ValueError(f"Unknown jump motor parameter: {key}")
            if value is None:
                continue
            array = np.asarray(value)
            if (array.shape not in ((), (counts[key],)) or not np.isfinite(array).all()
                    or np.any(array < 0) or (key.endswith("force_limit") and np.any(array <= 0))):
                raise ValueError(f"Invalid jump motor parameter: {key}={value}")
            self.motor_defaults[key] = deepcopy(cfg[key])
            self.motor_jump[key] = deepcopy(value)
        self.wheel_ids = [self.sim.model.body(n).id for n in cfg["wheel_link_names"]]
        self.jump_count = 0
        self.reset()

    def reset(self):
        self.sim.constrain_leg_targets_enabled = True
        self.sim.cfg.update(deepcopy(self.motor_defaults))
        self.mode = "locomotion"
        self.jump_step = 0
        self.last_actions = torch.zeros(6)
        self.has_taken_off = self.has_landed = False
        self.takeoff_contact_vz = 0.0
        self.peak_clearance = 0.0
        self.last_result = None
        self.sim.commands[:] = tensor([self.command_vx, 0, self.standing_height])
        self.sim.reset()
        self._read_contacts()
        return self.observations()

    def _read_contacts(self):
        sim = self.sim
        forces = np.zeros((sim.model.nbody, 3))
        for i, contact in enumerate(sim.data.contact):
            if contact.efc_address < 0:
                continue
            local_force = np.zeros(6)
            mujoco.mj_contactForce(sim.model, sim.data, i, local_force)
            force = contact.frame.reshape(3, 3).T @ local_force[:3]
            forces[sim.model.geom_bodyid[contact.geom1]] -= force
            forces[sim.model.geom_bodyid[contact.geom2]] += force
        self.wheel_contact = np.linalg.norm(forces[self.wheel_ids], axis=1) > sim.cfg.get("wheel_contact_force_threshold", 1.0)
        self.base_contact = bool(np.linalg.norm(forces[sim.base_id]) > sim.cfg.get("base_contact_force_threshold", 5.0))
        radius = sim.estimator.get("wheel_radius", .06)
        bottoms = sim.data.xpos[self.wheel_ids, 2] - radius
        self.clearance = float(np.min(bottoms))
        self.base_to_wheel_bottom = float(sim.data.xpos[sim.base_id, 2] - np.mean(bottoms))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_XBODY, sim.base_id, velocity, 0)
        self.vz = float(velocity[5])

    def _update_commands(self):
        if self.mode == "jump":
            height = sample_height_reference(torch.tensor(self.jump_step * self.dt), self.reference_times, self.reference_heights)
            self.sim.commands[:] = tensor([self.locked_vx, 0, height.item()])
        else:
            self.sim.commands[:] = tensor([self.command_vx, 0, self.standing_height])

    def jump_observations(self):
        prefix = self.sim.observations()["policy"]
        phase = phase_encoding(torch.tensor([self.jump_step * self.dt]), self.cycle_s)
        return TensorDict({"policy": torch.cat((prefix, self.last_actions[None], phase), dim=-1)}, batch_size=[1])

    def observations(self):
        self._update_commands()
        return self.jump_observations() if self.mode == "jump" else self.sim.observations()

    def trigger_jump(self):
        if self.mode != "locomotion":
            return False  # Space during jump is ignored, not queued for landing.
        self.mode = "jump"
        self.sim.constrain_leg_targets_enabled = False
        self.jump_count += 1
        self.jump_step = 0
        self.locked_vx = self.command_vx
        self.has_taken_off = self.has_landed = False
        self.takeoff_contact_vz = 0.0
        self.peak_clearance = max(0.0, self.clearance)
        self.last_result = None
        # At begin_jump_rollout Genesis has already copied actions into last_actions.
        self.last_actions = self.sim.actions.clone()
        self.sim.cfg.update(deepcopy(self.motor_jump))
        self._update_commands()
        return True

    def _finish_jump(self, reason):
        self.sim.constrain_leg_targets_enabled = True
        self.last_result = {"jump": self.jump_count, "reason": reason, "duration": self.jump_step * self.dt,
                            "taken_off": self.has_taken_off, "landed": self.has_landed,
                            "peak_clearance": self.peak_clearance}
        self.mode = "locomotion"
        self.sim.cfg.update(deepcopy(self.motor_defaults))
        self._update_commands()

    def step(self, action):
        self._update_commands()
        previous_actions = self.sim.actions.clone()
        self.sim.step(action)
        # Returned Genesis jump obs contains a_t in actions and a_(t-1) in last_actions.
        self.last_actions = previous_actions
        self._read_contacts()
        if self.mode == "jump":
            self.jump_step += 1
            self.peak_clearance = max(self.peak_clearance, self.clearance)
            if np.any(self.wheel_contact) and not self.base_contact and not self.has_taken_off:
                self.takeoff_contact_vz = self.vz
            # MuJoCo can briefly report zero support force while wheels are still at the
            # surface during push-off. Require 5 mm clearance before arming landing handoff.
            if (not np.any(self.wheel_contact) and not self.base_contact
                    and self.takeoff_contact_vz > 0 and self.clearance >= .005):
                self.has_taken_off = True
            if self.has_taken_off and (np.any(self.wheel_contact) or self.base_contact):
                self.has_landed = True
            if self.handoff == "landing" and self.has_taken_off and np.all(self.wheel_contact):
                self._finish_jump("both_wheels_landed")
            elif self.jump_step >= self.horizon:
                self._finish_jump("horizon")
        return self.observations()

    def diagnostics(self):
        return {**self.sim.diagnostics(), "mode": self.mode, "jump": self.jump_count,
                "jump_time": self.jump_step * self.dt, "clearance": self.clearance,
                "peak_clearance": self.peak_clearance, "base_to_wheel_bottom": self.base_to_wheel_bottom,
                "vz": self.vz, "left_contact": bool(self.wheel_contact[0]),
                "right_contact": bool(self.wheel_contact[1]), "base_contact": self.base_contact,
                "taken_off": self.has_taken_off, "landed": self.has_landed}
