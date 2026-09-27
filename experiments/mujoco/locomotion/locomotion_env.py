"""Genesis locomotion 的 MuJoCo 平地推理适配器（不依赖 Genesis）。"""

from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from experiments.genesis.wheel_leg_infantry.core.kinematics import (
    compute_leg_angle, compute_leg_length, constrain_leg_targets,
)
from experiments.genesis.wheel_leg_infantry.tasks.locomotion.velocity_estimator import (
    complementary_forward_velocity_update, gravity_compensated_forward_acceleration,
    wheel_forward_velocity,
)

ROOT = Path(__file__).resolve().parents[3]


def tensor(value):
    return torch.tensor(np.asarray(value).copy(), dtype=torch.float32)


class MujocoLocomotionEnv:
    dt = 0.02
    substeps = 20

    def __init__(self, configs, commands=(0.0, 0.0, 0.22)):
        self.cfg = configs["env_cfg"]
        self.obs_cfg = configs["obs_cfg"]
        self.scales = self.obs_cfg["obs_scales"]
        self.estimator = self.obs_cfg.get("velocity_estimator", {})
        if self.cfg["num_actions"] != 6 or configs["command_cfg"]["num_commands"] != 3:
            raise ValueError("Only the 6-action, 3-command locomotion contract is supported")
        imu = self.obs_cfg.get("imu", {})
        if any(np.any(np.asarray(imu.get(k, 0)) != 0) for k in (
            "acc_noise", "acc_bias", "acc_random_walk", "gyro_noise", "gyro_bias",
            "gyro_random_walk", "delay", "jitter",
        )):
            raise ValueError("This baseline requires noise-free, zero-delay IMU configuration")
        robot = Path(self.cfg["robot_mjcf"])
        if not robot.is_absolute():
            robot = ROOT / robot
        # Load the exact training robot and add a plane without modifying its asset file.
        spec = mujoco.MjSpec.from_file(str(robot))
        spec.option.timestep = self.dt / self.substeps
        spec.option.gravity = [0, 0, -9.81]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        spec.option.iterations = 100
        spec.option.tolerance = 1e-8
        spec.worldbody.add_geom(name="sim2sim_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, 0.05], friction=[1, 0.01, 0.01], rgba=[0.25, 0.3, 0.35, 1])
        spec.worldbody.add_light(pos=[0, -2, 4], dir=[0, 0, -1])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body(self.cfg.get("base_link_name", "base_link")).id
        site = self.model.site("base_link_site")
        if self.model.body(imu.get("link_name", "base_link")).id != site.bodyid[0]:
            raise ValueError("IMU link must match base_link_site")
        self.model.site_pos[site.id] = imu.get("pos_offset", [0, 0, 0])
        # Disable robot self collisions, matching Genesis enable_self_collision=False.
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] != 0 and (
                self.model.geom_contype[geom_id] or self.model.geom_conaffinity[geom_id]
            ):
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 1
        self.joint_names = self.cfg["joint_names"]
        self.jq, self.jd = self._addresses(self.joint_names)
        self.wq, self.wd = self._addresses(self.cfg["wheel_names"])
        self.sq, self.sd = self._addresses(self.cfg.get("spring_names", ["left_spring2_joint", "right_spring2_joint"]))
        self.front = [self.joint_names.index(n) for n in self.cfg.get("leg_front_joint_names", ["left_front1_joint", "right_front1_joint"])]
        self.rear = [self.joint_names.index(n) for n in self.cfg.get("leg_rear_joint_names", ["left_rear1_joint", "right_rear1_joint"])]
        self.default = tensor([self.cfg["default_joint_pos"][n] for n in self.joint_names])
        # Apply generalized forces directly: PD limits come from the saved training config.
        self.model.actuator_gainprm[:, :] = 0
        self.model.actuator_biasprm[:, :] = 0
        self.commands = tensor(commands)
        self.reset()

    def _addresses(self, names):
        joints = [self.model.joint(n) for n in names]
        return ([int(j.qposadr[0]) for j in joints], [int(j.dofadr[0]) for j in joints])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Expected exactly one floating base")
        adr = self.model.jnt_qposadr[free[0]]
        self.data.qpos[adr:adr + 3] = np.mean(self.cfg["base_init_pos_range"], axis=1)
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                self.data.qpos[self.model.jnt_qposadr[j]] = self.cfg["default_joint_pos"].get(self.model.joint(j).name, 0)
        self.actions = torch.zeros(6)
        self.estimated_velocity = torch.zeros(1)
        self.acc = torch.zeros(3)
        self.gyro = torch.zeros(3)
        self.target_joint = self.default.numpy().copy()
        self.target_wheel = np.zeros(2)
        self.spring_force = np.zeros(2)
        mujoco.mj_forward(self.model, self.data)
        return self.observations()

    def observations(self):
        q = tensor(self.data.qpos[self.jq])
        self.gravity = tensor(self.data.xmat[self.base_id].reshape(3, 3).T @ [0, 0, -1])
        s = self.scales
        self.components = {
            "imu_ang_vel": self.gyro * s["ang_vel"],
            "imu_lin_acc": self.acc * s.get("lin_acc", 1 / 9.81),
            "projected_gravity": self.gravity,
            "commands": self.commands * tensor([s["lin_vel"], s["ang_vel"], s["base_height"]]),
            "joint_pos_offset": (q - self.default) * s["joint_pos"],
            "joint_vel": tensor(self.data.qvel[self.jd]) * s["joint_vel"],
            "wheel_vel": tensor(self.data.qvel[self.wd]) * s["wheel_vel"],
            "leg_length": compute_leg_length(q, self.front, self.rear, self.cfg.get("leg_upper_link_length", .21),
                                             self.cfg.get("leg_lower_link_length", .25)) * s["leg_length"],
            "leg_angle": compute_leg_angle(q, self.front, self.rear) * s["leg_angle"],
            "actions": self.actions,
        }
        obs = torch.cat(tuple(self.components.values())).unsqueeze(0)
        if not torch.isfinite(obs).all():
            raise FloatingPointError("Non-finite policy observation")
        return TensorDict({"policy": obs}, batch_size=[1])

    def set_action(self, action):
        action = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
        if action.shape != (6,) or not torch.isfinite(action).all():
            raise ValueError("Policy action must contain six finite values")
        clipped = torch.cat((action[:4].clamp(-self.cfg["clip_joint_action"], self.cfg["clip_joint_action"]),
                             action[4:].clamp(-self.cfg["clip_wheel_action"], self.cfg["clip_wheel_action"])))
        executed = self.actions if self.cfg["simulate_action_latency"] else clipped
        limits = self.cfg.get("leg_angle_limit_range", [-np.pi / 4, np.pi / 4])
        self.target_joint = constrain_leg_targets(
            self.default + executed[:4] * self.cfg["joint_pos_scale"], self.front, self.rear,
            *limits, np.pi - self.cfg.get("min_upper_link_angle", np.pi / 2),
        ).numpy()
        self.target_wheel = (executed[4:] * self.cfg["wheel_vel_scale"]).numpy()
        self.actions = clipped.clone()
        # Genesis calculates spring force once per control tick and holds it over substeps.
        compression_max = self.cfg.get("gas_spring_max_compression", .06)
        compression = np.clip(compression_max - self.data.qpos[self.sq], 0, compression_max)
        self.spring_force = (self.cfg.get("gas_spring_preload_force", 420)
                             + self.cfg.get("gas_spring_stiffness", 1400) * compression
                             - self.cfg.get("gas_spring_damping", 50) * self.data.qvel[self.sd])

    def _forces(self):
        c, d = self.cfg, self.data
        d.qfrc_applied[:] = 0
        d.qfrc_applied[self.jd] = np.clip(
            np.asarray(c["joint_kp"]) * (self.target_joint - d.qpos[self.jq])
            - np.asarray(c["joint_kd"]) * d.qvel[self.jd],
            -np.asarray(c.get("joint_force_limit", 54)), c.get("joint_force_limit", 54))
        d.qfrc_applied[self.wd] = np.clip(np.asarray(c["wheel_kd"]) * (self.target_wheel - d.qvel[self.wd]),
                                                    -np.asarray(c.get("wheel_force_limit", 5)), c.get("wheel_force_limit", 5))
        d.qfrc_applied[self.sd] = self.spring_force

    def step(self, action):
        self.set_action(action)
        for _ in range(self.substeps):
            self._forces()
            mujoco.mj_step(self.model, self.data)
        self._forces()
        mujoco.mj_forward(self.model, self.data)
        if any(w.number for w in self.data.warning):
            raise FloatingPointError("MuJoCo reported a simulation warning; stop and inspect dynamics")
        self.acc = tensor(self.data.sensor("accelerometer").data)
        self.gyro = tensor(self.data.sensor("gyro").data)
        self.gravity = tensor(self.data.xmat[self.base_id].reshape(3, 3).T @ [0, 0, -1])
        v = self.estimator
        acceleration = gravity_compensated_forward_acceleration(self.acc, self.gravity, v.get("gravity_magnitude", 9.81))
        wheel_v = wheel_forward_velocity(tensor(self.data.qvel[self.wd]), v.get("wheel_radius", .06), v.get("wheel_velocity_sign", 1.0))
        self.estimated_velocity = complementary_forward_velocity_update(
            self.estimated_velocity, acceleration.reshape(1), wheel_v.reshape(1), dt=self.dt,
            wheel_correction_time_constant_s=v.get("wheel_correction_time_constant_s", .5),
            max_abs_velocity=v.get("max_abs_velocity", 3.0))
        return self.observations()

    def diagnostics(self):
        velocity = np.zeros(6)
        # BODY uses the principal-inertia frame (ximat); XBODY uses the robot frame (xmat).
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.base_id, velocity, 1)
        return {"time": float(self.data.time), "height": float(self.data.xpos[self.base_id, 2]),
                "x": float(self.data.xpos[self.base_id, 0]), "y": float(self.data.xpos[self.base_id, 1]),
                "vx": float(velocity[3]), "wz": float(velocity[2]),
                "estimated_vx": float(self.estimated_velocity[0]),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(-self.gravity[2].item(), -1, 1))))}
