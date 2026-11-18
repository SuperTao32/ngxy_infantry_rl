"""轮腿 locomotion 环境：场景、控制、传感器及恢复/跳跃组件接口。"""

import math
from copy import deepcopy

import numpy as np
import genesis as gs
from genesis.utils.geom import (
    inv_quat,
    pos_lookat_up_to_T,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)
from genesis.vis.keybindings import Key, KeyAction, Keybind

import torch
from tensordict import TensorDict

from .curriculum import CurriculumManager
from .sensors.base_contact import BaseContactRegions
from .robust.dynamics_randomization import DynamicsRandomizationManager, MOTOR_NAMES, SPRING_NAMES
from .robust.randomization_config import normalize_randomization_config
from .robust.sensor_randomization import SensorRandomizationManager
from .robust.wheel_slip import WheelSlipRandomizer
from .sensors.tof import GenesisToF, ToFHistory, ToFSamplingClock, resolve_tof_cfg
from .utils.kinematics import compute_leg_angle, compute_leg_length, constrain_leg_targets
from .utils.tensor_utils import as_gain_tensor, as_range_tensors
from .terrains.manager import TerrainManager
from .configuration import EnvironmentConfiguration
from .rewards import LocomotionRewards
from .reward_buffer import RewardBuffer
from .tasks.recovery import Recovery
from .tasks.jump import Jump
from .tasks.terrain_cross import TerrainCross
from .utils.tracking_gate import DEFAULT_TRACKING_GATE_CFG


class WheelLegInfantryEnv(LocomotionRewards, EnvironmentConfiguration):
    """一个 Genesis Scene 和 locomotion 控制循环，组合 reset 恢复与跳跃命令输入。"""

    # ============ 初始化与场景 ============
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        curriculum_cfg=None,
        steps_per_iteration=24,
        show_viewer=False,
        *,
        evaluation=False,
        initial_training_iteration=0,
    ):
        self.evaluation = evaluation
        ############################# 配置阶段 ###############################
        # 确定设备
        self.device = gs.device
        # 保存cfg
        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.curriculum_cfg = {"enabled": False, "stages": []} if curriculum_cfg is None else curriculum_cfg
        self.steps_per_iteration = int(steps_per_iteration)  # PPO runner 每采满 steps_per_iteration 个控制步完成一轮
        self.tracking_gate_cfg = {}
        self._apply_tracking_gate(reward_cfg.get("tracking_gate", DEFAULT_TRACKING_GATE_CFG))
        self.terrain = TerrainManager(env_cfg["terrain"])
        # 记录 TerrainManager 补齐后的配置，保证课程、日志与运行状态一致。
        self.env_cfg["terrain"] = self.terrain.config

        randomization = normalize_randomization_config(env_cfg, obs_cfg)
        self.domain_rand = DynamicsRandomizationManager(randomization["dynamics"])
        randomization["dynamics"] = self.domain_rand.config
        self.sensor_noise = SensorRandomizationManager(randomization["sensors"])
        self.wheel_slip = WheelSlipRandomizer(randomization["wheel_slip"])
        # 共享域随机化需要逐环境电机参数，统一使用 batched DOF 信息。
        self.batch_dofs_info = True

        # 观测、奖励和指令配置
        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.commands_scale = self._build_commands_scale()
        self.commands_limit = self._build_command_limits()

        # 定义训练配置和参数
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_joints = env_cfg["num_joints"]
        self.num_wheels = env_cfg["num_wheels"]
        self.num_commands = command_cfg["num_commands"]

        # 定义训练参数
        self.dt = float(env_cfg.get("control_dt", 0.01))
        self.resample_step = max(1, math.ceil(self.env_cfg["resampling_time_s"] / self.dt))
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # IMU配置
        self.imu_cfg = self.sensor_noise.imu_options(obs_cfg.get("imu", {}))
        self.imu_acc_scale = float(obs_cfg["obs_scales"].get("lin_acc", 1.0 / 9.81))
        self.tof_cfg = resolve_tof_cfg(obs_cfg.get("tof"))
        self.tof_observation_enabled = False
        self.tof_clock = ToFSamplingClock(self.tof_cfg, self.dt, num_envs, self.device) if self.tof_cfg["enabled"] else None

        # 地形接触几何和跳跃奖励共用的物理参数。
        self.wheel_radius = float(env_cfg["wheel_radius"])
        self.gravity_magnitude = float(env_cfg["gravity_magnitude"])

        # 气弹簧配置
        self.spring_names = tuple(env_cfg.get("spring_names", ("left_spring2_joint", "right_spring2_joint")))
        self.gas_spring_preload_force = float(env_cfg.get("gas_spring_preload_force", 420.0))
        self.gas_spring_stiffness = float(env_cfg.get("gas_spring_stiffness", 1400.0))
        self.gas_spring_damping = float(env_cfg.get("gas_spring_damping", 50.0))
        self.gas_spring_max_compression = float(env_cfg.get("gas_spring_max_compression", 0.06))

        # 接触判定和持续终止阈值
        self.wheel_contact_force_threshold = float(env_cfg.get("wheel_contact_force_threshold", 1.0))
        self.base_contact_force_threshold = float(env_cfg.get("base_contact_force_threshold", 5.0))
        base_contact_duration = env_cfg.get("base_contact_termination_duration_s")
        self.base_contact_termination_duration_s = None if base_contact_duration is None else float(base_contact_duration)
        self.base_contact_termination_steps = (
            None if self.base_contact_termination_duration_s is None else max(1, math.ceil(self.base_contact_termination_duration_s / self.dt))
        )
        self.tilt_termination_duration_s = float(env_cfg.get("tilt_termination_duration_s", self.dt))
        self.tilt_termination_steps = max(1, math.ceil(self.tilt_termination_duration_s / self.dt))

        # 腿部相关参数
        joint_names = tuple(self.env_cfg["joint_names"])
        wheel_names = tuple(self.env_cfg["wheel_names"])
        wheel_link_names = tuple(self.env_cfg.get("wheel_link_names", ("left_wheel_link", "right_wheel_link")))
        base_link_name = self.env_cfg.get("base_link_name", "base_link")
        front_joint_names = tuple(self.env_cfg.get("leg_front_joint_names", ("left_front1_joint", "right_front1_joint")))
        rear_joint_names = tuple(self.env_cfg.get("leg_rear_joint_names", ("left_rear1_joint", "right_rear1_joint")))
        self.leg_front_joint_indices = [joint_names.index(name) for name in front_joint_names]
        self.leg_rear_joint_indices = [joint_names.index(name) for name in rear_joint_names]
        # motor 控制参数
        self.joint_kp = as_gain_tensor(self.env_cfg["joint_kp"], self.num_joints, device=self.device)
        self.joint_kd = as_gain_tensor(self.env_cfg["joint_kd"], self.num_joints, device=self.device)
        self.wheel_kd = as_gain_tensor(self.env_cfg["wheel_kd"], self.num_wheels, device=self.device)
        self.joint_force_limit = as_gain_tensor(
            self.env_cfg.get("joint_force_limit", 54.0),
            self.num_joints,
            device=self.device,
        )
        self.wheel_force_limit = as_gain_tensor(
            self.env_cfg.get("wheel_force_limit", 5.0),
            self.num_wheels,
            device=self.device,
        )
        # 腿部连杆几何参数
        self.upper_links_min_angle = float(self.env_cfg.get("upper_links_min_angle", 0.5 * math.pi))
        self.leg_upper_link_length = float(env_cfg.get("leg_upper_link_length", 0.21))
        self.leg_lower_link_length = float(env_cfg.get("leg_lower_link_length", 0.25))
        self.max_motor_separation = math.pi - self.upper_links_min_angle
        half_separation = 0.5 * self.max_motor_separation
        self.max_leg_length = self.leg_upper_link_length * math.sin(half_separation) + math.sqrt(
            max(self.leg_lower_link_length**2 - (self.leg_upper_link_length * math.cos(half_separation)) ** 2, 0.0)
        )
        # 动作零点与复位姿态分开：起身任务还需要还原被动关节和轮子的角度。
        self.reset_joint_pos = dict(self.env_cfg["default_joint_pos"])
        self.reset_joint_pos.update(self.env_cfg.get("reset_joint_pos", {}))
        self.default_joint_pos = torch.tensor(
            [self.env_cfg["default_joint_pos"][name] for name in joint_names],
            dtype=gs.tc_float,
            device=self.device,
        )
        self.init_joint_pos = torch.tensor(
            [self.reset_joint_pos[name] for name in joint_names],
            dtype=gs.tc_float,
            device=self.device,
        )
        self.init_leg_length = compute_leg_length(
            self.init_joint_pos,
            self.leg_front_joint_indices,
            self.leg_rear_joint_indices,
            self.leg_upper_link_length,
            self.leg_lower_link_length,
        )
        self.init_leg_angle = compute_leg_angle(
            self.init_joint_pos,
            self.leg_front_joint_indices,
            self.leg_rear_joint_indices,
        )

        # 初始状态随机化范围
        # 场景构建使用原点和单位姿态；reset 时再从当前课程范围独立采样。
        self.global_gravity_dir = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=self.device)
        self.init_base_pos = torch.zeros(3, dtype=gs.tc_float, device=self.device)
        self.init_base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=self.device)
        self.init_base_quat_inv = gs.inv_quat(self.init_base_quat)
        legacy_base_init_pos = env_cfg.get("base_init_pos", [0.0, 0.0, 0.2])
        legacy_pos_xy = env_cfg.get("base_init_pos_xy", legacy_base_init_pos[:2])
        legacy_pos_z = env_cfg.get("base_init_pos_z_range", [legacy_base_init_pos[2], legacy_base_init_pos[2]])
        default_pos_range = [
            [legacy_pos_xy[0], legacy_pos_xy[0]],
            [legacy_pos_xy[1], legacy_pos_xy[1]],
            list(legacy_pos_z),
        ]
        self.base_init_pos_lower, self.base_init_pos_upper = as_range_tensors(
            env_cfg.get("base_init_pos_range", default_pos_range),
            3,
            "base_init_pos_range",
            device=self.device,
        )
        self.base_init_rpy_lower, self.base_init_rpy_upper = as_range_tensors(
            env_cfg.get("base_init_rpy_offset_range_deg", [[0.0, 0.0]] * 3),
            3,
            "base_init_rpy_offset_range_deg",
            device=self.device,
        )
        self.base_init_lin_vel_lower, self.base_init_lin_vel_upper = as_range_tensors(
            env_cfg.get("base_init_lin_vel_range", [[0.0, 0.0]] * 3),
            3,
            "base_init_lin_vel_range",
            device=self.device,
        )
        self.base_init_ang_vel_lower, self.base_init_ang_vel_upper = as_range_tensors(
            env_cfg.get("base_init_ang_vel_range", [[0.0, 0.0]] * 3),
            3,
            "base_init_ang_vel_range",
            device=self.device,
        )

        self.rewards = RewardBuffer(
            self.num_envs, self.device, self.dt, lambda name: getattr(self, "_reward_" + name),
            self.reward_cfg["reward_scales"], event_rewards={"death"},
        )
        # 课程注册及第 0 阶段配置
        self.training_iteration = int(initial_training_iteration)
        self.global_step = self.training_iteration * self.steps_per_iteration
        self.curriculum = CurriculumManager(self.curriculum_cfg)
        self.curriculum.register_target("command_ranges", self._apply_command_ranges)
        self.curriculum.register_target("tracking_gate", self._apply_tracking_gate)
        self.curriculum.register_target("reward_scales", self._apply_reward_scales)
        self.curriculum.register_target("standing_reward", self._apply_standing_reward)
        self.curriculum.register_target("termination_limits", self._apply_termination_limits)
        self.curriculum.register_target("reset_ranges", self._apply_reset_ranges)
        self.curriculum.register_target("recovery", self._apply_recovery_curriculum)
        self.curriculum.register_target("jump", self._apply_jump_curriculum)
        self.curriculum.register_target("domain_rand", self.domain_rand.apply_curriculum)
        self.curriculum.register_target("sensor_noise", self._apply_sensor_noise)
        self.curriculum.register_target("wheel_slip", self.wheel_slip.apply_curriculum)
        self.curriculum.register_target("tof", self._apply_tof_curriculum)
        self.curriculum.register_target("terrain_cross", self._apply_terrain_cross_curriculum)
        self.curriculum.register_target("terrain_weights", self._apply_terrain_weights)
        if self.curriculum.update(self.training_iteration, force=True):
            print(f"[curriculum] stage={self.curriculum.current_stage_name} iteration={self.training_iteration}")

        ############################# 初始化阶段 ###############################
        # 创建 scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=20,
            ),
            rigid_options=gs.options.RigidOptions(
                batch_dofs_info=self.batch_dofs_info,
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=self.terrain.max_collision_pairs,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.8, -2.8, 1.5),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
                refresh_rate=60,
                realtime_factor=env_cfg.get("viewer_realtime_factor", 1.0 if show_viewer else None),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            profiling_options=gs.options.ProfilingOptions(show_FPS=env_cfg.get("show_FPS", True)),
            show_viewer=show_viewer,
        )

        # 添加实体
        self._add_terrain()
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=self.env_cfg.get(
                    "robot_mjcf",
                    "assets/robot/wheelbipeV14_2/mjcf/wheelbipeV14_2.xml",
                ),
                pos=self.init_base_pos.tolist(),
                quat=self.init_base_quat.tolist(),
            ),
        )

        # actor 使用 IMU、轮速和 ToF 观测，仿真真值只供 critic 和诊断使用。
        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.robot.idx,
                link_idx_local=self.robot.get_link(self.imu_cfg.get("link_name", "base_link")).idx_local,
                pos_offset=tuple(self.imu_cfg.get("pos_offset", (0.0, 0.0, 0.0))),
                acc_noise=self.imu_cfg.get("acc_noise", 0.0),
                acc_bias=self.imu_cfg.get("acc_bias", 0.0),
                acc_random_walk=self.imu_cfg.get("acc_random_walk", 0.0),
                gyro_noise=self.imu_cfg.get("gyro_noise", 0.0),
                gyro_bias=self.imu_cfg.get("gyro_bias", 0.0),
                gyro_random_walk=self.imu_cfg.get("gyro_random_walk", 0.0),
                delay=self.imu_cfg.get("delay", 0.0),
                jitter=self.imu_cfg.get("jitter", 0.0),
            )
        )

        # tof
        self.tof = (
            GenesisToF(self.scene, self.robot, self.tof_cfg, self.sensor_noise.config["tof"], terrain=self.terrain)
            if self.tof_cfg["enabled"] else None
        )
        self.tof_distances = torch.empty(
            (num_envs, len(self.tof_cfg.get("sensors", ()))),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.tof_history = ToFHistory(self.tof_distances, self.tof_cfg) if self.tof_cfg["include_in_observation"] else None
        self.true_tof_distances = torch.empty_like(self.tof_distances)
        self.true_tof_history = ToFHistory(self.true_tof_distances, self.tof_cfg) if self.tof_history is not None else None

        # build环境
        self.scene.build(n_envs=num_envs)
        self.sensor_noise.bind_imu(self.imu)
        # Genesis 在 build 后才生成 terrain_hf，此时才能绑定高度场供 reset/step 查询。
        self.terrain.bind_entity(self.terrain_entity, device=self.device, dtype=gs.tc_float)

        ## 创建索引
        self.joints_dof_idx = self._joint_dof_indices(joint_names)
        self.springs_dof_idx = self._joint_dof_indices(self.spring_names)
        self.wheels_dof_idx = self._joint_dof_indices(wheel_names)
        controlled_names = set((*joint_names, *wheel_names, *self.spring_names))
        self.passive_joint_names = tuple(
            joint.name for joint in self.robot.joints if joint.type == gs.JOINT_TYPE.REVOLUTE and joint.name not in controlled_names
        )
        self.passive_dof_idx = self._joint_dof_indices(self.passive_joint_names)
        self.wheel_links_idx = torch.tensor(
            [self.robot.get_link(name).idx_local for name in wheel_link_names],
            dtype=gs.tc_int,
            device=self.device,
        )
        self.base_link_idx = self.robot.get_link(base_link_name).idx_local
        self.body_contact_links_idx = self._body_contact_link_indices()

        # PD 参数
        self.robot.set_dofs_kp(self.joint_kp.tolist(), self.joints_dof_idx)
        self.robot.set_dofs_kv(self.joint_kd.tolist(), self.joints_dof_idx)
        self.robot.set_dofs_kv(self.wheel_kd.tolist(), self.wheels_dof_idx)
        self.robot.set_dofs_force_range((-self.joint_force_limit).tolist(), self.joint_force_limit.tolist(), self.joints_dof_idx)
        self.robot.set_dofs_force_range((-self.wheel_force_limit).tolist(), self.wheel_force_limit.tolist(), self.wheels_dof_idx)

        self.domain_rand.bind(
            robot=self.robot,
            base_link=self.robot.get_link(base_link_name),
            friction_entities=(self.robot, *self.friction_terrain_entities),
            motor_params={name: getattr(self, name) for name in MOTOR_NAMES},
            joint_indices=self.joints_dof_idx,
            wheel_indices=self.wheels_dof_idx,
            num_envs=self.num_envs,
            batched_dofs=self.batch_dofs_info,
            spring_params={name: getattr(self, f"gas_spring_{name}") for name in SPRING_NAMES},
            num_springs=len(self.spring_names),
            passive_joint_names=self.passive_joint_names,
            passive_indices=self.passive_dof_idx,
            dt=self.dt,
            push_links={
                "base": self.robot.get_link(base_link_name),
                "left_wheel": self.robot.get_link(wheel_link_names[0]),
                "right_wheel": self.robot.get_link(wheel_link_names[1]),
            },
        )
        self.wheel_slip.bind(
            robot=self.robot,
            wheel_links=[self.robot.get_link(name) for name in wheel_link_names],
            terrain_entities=self.friction_terrain_entities,
            baseline_ratio=self.domain_rand.friction_ratio,
            dt=self.dt,
        )
        # 诊断参数与 manager 的实际参数共用 buffer，始终带并行环境维度。
        for name, value in self.domain_rand.motor_values.items():
            setattr(self, name, value)
        for name, value in self.domain_rand.gas_spring_values.items():
            setattr(self, f"gas_spring_{name}", value)

        # 初始化joint和dof的初始位置
        unknown_reset_joints = self.reset_joint_pos.keys() - {joint.name for joint in self.robot.joints if joint.n_qs == 1}
        if unknown_reset_joints:
            raise ValueError(f"Reset positions require named scalar joints: {sorted(unknown_reset_joints)}")
        init_dof_pos_list = []
        for joint in self.robot.joints[1:]:
            if joint.n_qs == 0:
                continue
            init_pos = self.reset_joint_pos.get(joint.name, 0.0)
            init_dof_pos_list.append(init_pos)

        self.init_dof_pos = torch.tensor(init_dof_pos_list, dtype=gs.tc_float, device=self.device)
        # robot的qpos总共是base的pos+quat共7个加上joint+wheel的qpos
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))

        ############################# buffers 创建 ###############################
        # RL 交互：动作、指令、奖励和 episode 状态
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.target_joint_pos = torch.empty((self.num_envs, self.num_joints), dtype=gs.tc_float, device=gs.device)
        self.target_wheel_vel = torch.empty((self.num_envs, self.num_wheels), dtype=gs.tc_float, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.reward_buf = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.bool, device=gs.device)
        self.terminated_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=self.device)
        self.extras = {}

        # 执行器与腿部状态
        self.joint_pos = torch.empty((self.num_envs, self.num_joints), dtype=gs.tc_float, device=gs.device)
        self.joint_vel = torch.empty_like(self.joint_pos)
        self.wheel_vel = torch.empty((self.num_envs, self.num_wheels), dtype=gs.tc_float, device=gs.device)
        self.sensor_noise.bind(self._encoder_truth())
        for name, value in self.sensor_noise.measurements.items():
            setattr(self, "measured_" + name, value)
        self.measured_leg_length = torch.empty((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.leg_length = torch.empty((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.leg_angle = torch.empty((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.gas_spring_force = torch.empty((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)

        # 机身与地形真值：供 reward、critic 和诊断使用
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.terrain_height = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.base_height = torch.empty_like(self.terrain_height)
        self.terrain_spawn_centers = torch.empty((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.terrain_tile_index = torch.empty((self.num_envs,), dtype=torch.long, device=gs.device)

        # actor 的 IMU 测量及 critic 使用的无噪声真值。
        self.imu_lin_acc = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.imu_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.true_imu_lin_acc = torch.empty_like(self.imu_lin_acc)
        self.true_imu_ang_vel = torch.empty_like(self.imu_ang_vel)

        # 接触状态
        self.wheel_contact = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.wheel_normal_force = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=gs.device)
        self.wheel_tangential_force = torch.zeros_like(self.wheel_normal_force)
        # 保留 base_contact 名称供奖励、门控、终止和旧配置使用，判定涵盖基座与腿部。
        self.base_contact = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.base_contact_regions = BaseContactRegions(
            self.robot.get_link(base_link_name), self.num_envs,
            device=self.device, dtype=gs.tc_float,
            force_threshold=self.base_contact_force_threshold,
        )

        # 奖励门控和终止计数
        self.height_gate = torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.attitude_gate = torch.ones_like(self.height_gate)
        self.tracking_gate_raw = torch.ones_like(self.height_gate)
        self.tracking_gate = torch.ones_like(self.height_gate)
        self.alive_gate = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.tilt_out_steps = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.base_contact_steps = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        self.episode_metric_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("height_gate", "attitude_gate", "tracking_gate", "tracking_gate_fully_open", "base_contact")
        }
        self.locomotion_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.termination_counts = {
            name: torch.zeros((), device=self.device)
            for name in ("tilt", "body_contact", "solver_error", "recovery_timeout", "time_limit", "out_of_bounds")
        }
        # 组件只持有自己的状态，不复制 env 的物理 buffer 或创建场景。
        self.recovery = Recovery(self, self.env_cfg["recovery"])
        self.jump = Jump(self, self.env_cfg["jump"])
        self.terrain_cross = TerrainCross(self, self.env_cfg["terrain_cross"])

        # 初始reset环境
        self.reset()
        if show_viewer:
            # 只在启动或按 ENTER 时对准机器人；逐帧 follow_entity 会覆盖鼠标调整。
            self.scene.viewer.register_keybinds(Keybind("camera_focus_robot", Key.ENTER, key_action=KeyAction.PRESS, callback=self.focus_viewer))
            self.focus_viewer()


    def _joint_dof_indices(self, names):
        """将场景自由度索引转换为实体控制接口需要的局部索引。"""
        return torch.tensor(
            [self.robot.get_joint(name).dof_start - self.robot.dof_start for name in names],
            dtype=gs.tc_int,
            device=self.device,
        )

    def _add_terrain(self):
        self.terrain_entity = self.terrain.add_to_scene(self.scene)
        self.friction_terrain_entities = (
            tuple(self.terrain_entity) if isinstance(self.terrain_entity, (tuple, list)) else (self.terrain_entity,)
        )


    # ============ 训练与评估接口 ============
    def reset(self, *, recovery=None):
        """重置并返回观测；recovery 可仅对本次复位强制使用/跳过姿态池。"""
        sampler = self.recovery.reset_sampler
        if recovery is True and sampler is None:
            raise ValueError("Recovery reset requires a configured recovery pose pool")
        probability = sampler.probability if sampler is not None else None
        try:
            if sampler is not None and recovery is not None:
                sampler.probability = float(recovery)
            self.recovery.metrics.clear()
            self.jump.metrics.clear()
            self.terrain_cross.metrics.clear()
            self.terrain_cross.level_metrics.clear()
            for count in self.termination_counts.values():
                count.zero_()
            self._reset_idx()
            self._update_observations()
            return self.get_observations()
        finally:
            if sampler is not None:
                sampler.probability = probability

    def step(self, actions):
        """执行一个控制步，返回观测、奖励、重置标志和附加统计。"""
        self.global_step += 1

        ########### 执行动作 ###########
        # 接收下一拍动作时再推进历史；重复读取/组装观测不能改变两拍动作。
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        target_joint_pos = exec_actions[:, : self.num_joints] * self.env_cfg["joint_pos_scale"]
        target_wheel_vel = exec_actions[:, self.num_joints : self.num_joints + self.num_wheels] * self.env_cfg["wheel_vel_scale"]

        target_joint_pos = target_joint_pos + self.default_joint_pos
        target_joint_pos = self.domain_rand.offset_joint_targets(target_joint_pos)
        target_joint_pos = self._constrain_joint_targets(target_joint_pos)
        self.target_joint_pos.copy_(target_joint_pos)
        self.target_wheel_vel.copy_(target_wheel_vel)

        self.robot.control_dofs_position(target_joint_pos, self.joints_dof_idx)
        self.robot.control_dofs_velocity(target_wheel_vel, self.wheels_dof_idx)
        self._apply_gas_spring_compensation()
        self.domain_rand.before_step()
        self.wheel_slip.before_step()
        self.scene.step()

        ########### 更新buffer ###########
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.terrain_height.copy_(self.terrain.height_at(self.base_pos[:, :2]))
        self.base_height.copy_(self.base_pos[:, 2] - self.terrain_height)
        self.base_quat = self.robot.get_quat()
        # init_base_quat 为单位四元数：这里是世界水平面基准的绝对姿态，不扣除地形坡角。
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.base_quat, self.init_base_quat_inv),
            rpy=True,
            degrees=True,
        )
        # 机器人的速度转化到自身坐标系下
        base_quat_inv = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), base_quat_inv)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), base_quat_inv)
        # 世界重力在机身坐标系中的方向，保留绝对 roll/pitch 信息，不包含 yaw。
        self.projected_gravity = transform_by_quat(self.global_gravity_dir, base_quat_inv)

        self.joint_pos = self.robot.get_dofs_position(self.joints_dof_idx)
        self.joint_vel = self.robot.get_dofs_velocity(self.joints_dof_idx)
        self.wheel_vel = self.robot.get_dofs_velocity(self.wheels_dof_idx)
        self._update_encoder_measurements()
        if self.tof is not None:
            due = self.tof_clock.advance()
            if due.any():
                # Raycaster 的内部 eager cache 仍随物理步刷新；仅发布到期的新测距帧。
                self.tof_distances[due] = self.tof.read(env_ids=due, strength=self.sensor_noise.strength)
                if self.tof_history is not None:
                    self.tof_history.append(self.tof_distances, due)
                    self.true_tof_distances[due] = self.tof.read_ground_truth(env_ids=due)
                    self.true_tof_history.append(self.true_tof_distances, due)
        self._update_imu_measurements()
        self._update_wheel_contact()

        self.leg_length = compute_leg_length(
            self.joint_pos,
            self.leg_front_joint_indices,
            self.leg_rear_joint_indices,
            self.leg_upper_link_length,
            self.leg_lower_link_length,
        )
        self.leg_angle = compute_leg_angle(
            self.joint_pos,
            self.leg_front_joint_indices,
            self.leg_rear_joint_indices,
        )
        self._update_tracking_gate()
        self.recovery.update(self)
        self.jump.update(self)

        ########### 判断终止 ###########
        time_out = self._update_termination()
        self.terrain_cross.update(self)

        ########### 计算奖励 ###########
        self.reward_buf.zero_()
        self._compute_rewards()
        self._update_episode_metrics()

        ########### 重采样指令 ###########
        self.jump.apply_commands(self.commands)
        self.terrain_cross.apply_commands(self.commands)
        resample_commands = (self.episode_length_buf % self.resample_step == 0) | self.recovery.buffers.just_completed
        # 刚结束 jump 的行已应用缓存命令，本拍不再次采样。
        self._resample_commands(resample_commands & ~self.reset_buf & ~self.jump.buffers.just_completed)

        ########### 计算timeout ###########
        # 失败与截断同拍发生时按失败处理，不对终止状态 bootstrap。
        self.extras["time_outs"] = (time_out & ~self.terminated_buf).to(dtype=gs.tc_float)

        ########### 重置环境（如果需要）###########
        self._reset_training_step()

        ########### 更新观测值 ###########
        self._update_observations()

        return self.get_observations(), self.reward_buf, self.reset_buf, self.extras

    def _update_termination(self):
        self.tilt_out_steps.copy_(torch.where(self._tilt_failure(), self.tilt_out_steps + 1, 0))
        self.base_contact_steps.copy_(torch.where(self._base_contact_failure(), self.base_contact_steps + 1, 0))
        persistent_tilt = self.tilt_out_steps >= self.tilt_termination_steps
        persistent_contact = torch.zeros_like(self.reset_buf) if self.base_contact_termination_steps is None else (
            self.base_contact_steps >= self.base_contact_termination_steps
        )
        solver_error = self.scene.rigid_solver.get_error_envs_mask().bool()
        recovery_timeout = self.recovery.termination()
        self.terminated_buf.copy_(persistent_tilt | persistent_contact | solver_error | recovery_timeout)
        time_limit = self._episode_timeout()
        out_of_bounds = self.terrain.out_of_bounds(self.base_pos[:, :2], self.terrain_spawn_centers)
        time_out = time_limit | out_of_bounds
        time_out &= ~self.terminated_buf
        # 失败原因可同时命中；时间/边界截断只统计未发生物理终止的行。
        causes = dict(tilt=persistent_tilt, body_contact=persistent_contact, solver_error=solver_error,
                      recovery_timeout=recovery_timeout, time_limit=time_limit & time_out, out_of_bounds=out_of_bounds & time_out)
        for name, cause in causes.items():
            self.termination_counts[name].add_(cause.sum())
        self.reset_buf.copy_(self.terminated_buf | time_out)
        return time_out

    def get_observations(self):
        """以 TensorDict 返回策略观测和特权 critic 观测。"""
        return TensorDict(
            {
                "policy": self.obs_buf,
                "critic": self.critic_obs_buf,
            },
            batch_size=[self.num_envs],
        )

    def _update_observations(self):
        """所有 actor/critic 观测在此显式列出；字典顺序就是网络输入顺序。"""
        jump_mode, jump_phase = self._get_jump_observations()
        commands = torch.concatenate((self.commands * self.commands_scale, jump_mode), dim=-1)

        # actor：可部署的传感器测量、动作、命令及跳跃条件。
        self.obs_components = {
            "imu_ang_vel": self.imu_ang_vel * self.obs_scales["ang_vel"],  # 3
            "imu_lin_acc": self.imu_lin_acc * self.imu_acc_scale,  # 3，IMU 比力（含重力）
            "projected_gravity": self.projected_gravity,  # 3
            "joint_pos_offset": (self.measured_joint_pos - self.default_joint_pos) * self.obs_scales["joint_pos"],  # 4
            "joint_vel": self.measured_joint_vel * self.obs_scales["joint_vel"],  # 4
            "wheel_vel": self.measured_wheel_vel * self.obs_scales["wheel_vel"],  # 2
            "leg_length": self.measured_leg_length * self.obs_scales["leg_length"],  # 2
            "actions": self.actions,  # 6
            "tof_distance": self._tof_observation(self.tof_history),
            "commands": commands,  # 5：速度、偏航速度、高度目标、两维 jump 模式 one-hot
            "jump_phase": jump_phase,  # 4：时间相位编码
        }
        self.obs_buf = torch.concatenate(tuple(self.obs_components.values()), dim=-1)

        # critic：与 actor 对齐的通道显式使用真值，再追加特权状态。
        # 未启用 jump 时保留零通道；ToF 课程同时控制 actor/critic 的固定宽度输入。
        self.critic_obs_components = {
            "imu_ang_vel": self.true_imu_ang_vel * self.obs_scales["ang_vel"],  # 3
            "imu_lin_acc": self.true_imu_lin_acc * self.imu_acc_scale,  # 3
            "projected_gravity": self.projected_gravity,  # 3
            "joint_pos_offset": (self.joint_pos - self.default_joint_pos) * self.obs_scales["joint_pos"],  # 4
            "joint_vel": self.joint_vel * self.obs_scales["joint_vel"],  # 4
            "wheel_vel": self.wheel_vel * self.obs_scales["wheel_vel"],  # 2
            "leg_length": self.leg_length * self.obs_scales["leg_length"],  # 2
            "actions": self.actions,  # 6
            "tof_distance": self._tof_observation(self.true_tof_history),
            "commands": commands,  # 5
            "jump_phase": jump_phase,  # 4
            # critic特权观测
            "privileged_base_lin_vel": self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
            "privileged_wheel_normal_force": self.wheel_normal_force.flatten(1) * self.obs_scales["normal_force"],  # 6
            "privileged_wheel_tangential_force": self.wheel_tangential_force.flatten(1) * self.obs_scales["tangential_force"],  # 6
            "privileged_wheel_slip": self.wheel_slip.active.to(dtype=self.obs_buf.dtype),  # 2
            "privileged_base_region_contact": self.base_contact_regions.contact,  # 3
            "privileged_base_region_normal_force": (
                (self.base_contact_regions.normal_force * self.obs_scales["base_normal_force"]).clamp(0.0, self.obs_cfg["base_contact_force_clip"])
            ),  # 3
            "privileged_base_height": self.base_pos[:, 2:3] * self.obs_scales["base_height"],  # 1：baselink 原点的世界坐标 z
        }
        self.critic_obs_buf = torch.concatenate(tuple(self.critic_obs_components.values()), dim=-1)

    def _get_jump_observations(self):
        return self.jump.observations()

    def set_training_iteration(self, iteration):
        self.training_iteration = int(iteration)
        self.global_step = self.training_iteration * self.steps_per_iteration
        self.curriculum.update(self.training_iteration, force=True)
        self.reset()

    def training_state_dict(self):
        return {
            "iteration": self.training_iteration,
            "locomotion": self.rewards.state_dict(),
            "recovery": self.recovery.rewards.state_dict(),
            "jump": self.jump.rewards.state_dict(),
            "terrain_cross": self.terrain_cross.state_dict(),
            "terrain_difficulty": self.terrain.state_dict(),
        }

    def load_training_state_dict(self, state, *, restore_reward_scales=True):
        if "terrain_difficulty" in state:
            self.terrain.load_state_dict(state["terrain_difficulty"])
        if restore_reward_scales:
            self.rewards.load_state_dict(state["locomotion"])
            self.recovery.rewards.load_state_dict(state["recovery"])
            self.jump.rewards.load_state_dict(state["jump"])
            if "terrain_cross" in state:
                self.terrain_cross.load_state_dict(state["terrain_cross"])
        self.set_training_iteration(state["iteration"])


    def _apply_reward_scales(self, values):
        self.rewards.set_scales(values)
        self.reward_cfg["reward_scales"].update(deepcopy(values))

    def _apply_recovery_curriculum(self, values):
        reset_pos = self.env_cfg["recovery"].get("reset_pos")
        if reset_pos is not None:
            reset_pos["probability"] = values["probability"]
        recovery = self.__dict__.get("recovery")
        if recovery is not None:
            recovery.apply_curriculum(values)

    def _apply_jump_curriculum(self, values):
        self.env_cfg["jump"]["probability"] = values["probability"]
        jump = self.__dict__.get("jump")
        if jump is not None:
            jump.apply_curriculum(values)

    def _apply_tof_curriculum(self, values):
        enabled = bool(values["enabled"])
        if enabled and not (self.tof_cfg["enabled"] and self.tof_cfg["include_in_observation"]):
            raise ValueError("ToF curriculum requires tof.enabled=True and include_in_observation=True at scene creation")
        self.tof_observation_enabled = enabled

    def _apply_terrain_cross_curriculum(self, values):
        for name, options in values["terrains"].items():
            for section, overrides in options.items():
                self.env_cfg["terrain_cross"]["terrains"][name][section].update(deepcopy(overrides))
        terrain_cross = self.__dict__.get("terrain_cross")
        if terrain_cross is not None:
            terrain_cross.apply_curriculum(values)

    def _apply_terrain_weights(self, values):
        if self.terrain.is_mixture:
            self.terrain.set_weights(values)

    def _tof_observation(self, history):
        if history is None:
            return self.base_pos.new_empty((self.num_envs, 0))
        observation = history.observation()
        return observation if self.tof_observation_enabled else torch.zeros_like(observation)

    def _update_episode_metrics(self):
        mask = ~self.recovery.buffers.was_active & ~self.jump.buffers.was_active
        self.locomotion_steps.add_(mask)
        for name, total in self.episode_metric_sums.items():
            values = (self.tracking_gate_raw >= 0.95).float() if name == "tracking_gate_fully_open" else getattr(self, name)
            total.add_(torch.where(mask, values, 0.0))

    # ============ 回合重置与命令 ============
    def _reset_training_step(self):
        self._reset_idx(self.reset_buf)
        if not self.evaluation and self.global_step % self.steps_per_iteration == 0:
            metrics = self.recovery.metrics.collect("recovery")
            metrics.update(self.jump.metrics.collect("jump"))
            metrics.update(self.terrain_cross.metrics.collect("terrain_cross"))
            metrics.update(self.terrain_cross.level_metrics.collect("terrain_cross"))
            metrics.update(self.terrain.update_curriculum())
            for name, count in self.termination_counts.items():
                metrics[f"termination/{name}"] = count.clone()
                count.zero_()
            self.training_iteration = self.global_step // self.steps_per_iteration
            if self.curriculum.update(self.training_iteration):
                print(f"[curriculum] stage={self.curriculum.current_stage_name} iteration={self.training_iteration}")
            metrics.update(self.terrain.sampling_metrics())
            self.extras.setdefault("episode", {}).update(metrics)

    def _reset_idx(self, env_idx=None, *, record_episode=None):
        """统一按布尔 mask 重置；None 为手动全量重置，不记录结束回合。"""
        record_episode = env_idx is not None if record_episode is None else record_episode
        reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device) if env_idx is None else env_idx
        if not reset_mask.any():
            self.extras.pop("episode", None)
            return
        reset_env_ids = reset_mask.nonzero(as_tuple=False).flatten()

        # 在状态清零前保存结束回合的统计；手动 reset 不产生 episode 日志。
        episode = self._collect_episode_metrics(reset_mask) if record_episode else None

        self.base_contact_regions.reset(reset_mask)
        reset_state = self._sample_base_reset_state(reset_env_ids)
        self.recovery.ensure_reset_clearance(self, reset_env_ids, reset_state)
        (
            reset_qpos,
            reset_base_pos,
            reset_base_quat,
            reset_base_lin_vel,
            reset_base_ang_vel,
            reset_projected_gravity,
            reset_terrain_height,
            reset_terrain_centers,
            reset_terrain_tile_index,
        ) = reset_state
        self.terrain.reset(reset_env_ids)
        self.domain_rand.reset(reset_env_ids)
        self.wheel_slip.reset(reset_env_ids)

        reset_quat_inv = inv_quat(reset_base_quat)
        reset_base_lin_vel_body = transform_by_quat(reset_base_lin_vel, reset_quat_inv)
        reset_base_ang_vel_body = transform_by_quat(reset_base_ang_vel, reset_quat_inv)
        self.robot.set_qpos(reset_qpos, envs_idx=reset_env_ids, zero_velocity=True, skip_forward=self.tof is None)
        # 浮动基座 qvel 的前 3 维是世界系线速度，后 3 维是机体系角速度。
        self.robot.set_dofs_velocity(
            torch.concatenate((reset_base_lin_vel, reset_base_ang_vel_body), dim=-1),
            dofs_idx_local=slice(0, 6), envs_idx=reset_env_ids, skip_forward=True,
        )
        reset_base_euler = quat_to_xyz(
            transform_quat_by_quat(reset_base_quat, self.init_base_quat_inv), rpy=True, degrees=True,
        )
        reset_joint_pos = self.robot.get_dofs_position(self.joints_dof_idx)[reset_env_ids]
        reset_leg_length = compute_leg_length(
            reset_joint_pos, self.leg_front_joint_indices, self.leg_rear_joint_indices,
            self.leg_upper_link_length, self.leg_lower_link_length,
        )
        reset_leg_angle = compute_leg_angle(reset_joint_pos, self.leg_front_joint_indices, self.leg_rear_joint_indices)

        # 全量与局部 reset 使用同一套状态写回。
        self.base_pos[reset_env_ids] = reset_base_pos
        self.terrain_height[reset_env_ids] = reset_terrain_height
        self.base_height[reset_env_ids] = reset_base_pos[:, 2] - reset_terrain_height
        self.terrain_spawn_centers[reset_env_ids] = reset_terrain_centers
        self.terrain_tile_index[reset_env_ids] = reset_terrain_tile_index
        self.base_quat[reset_env_ids] = reset_base_quat
        self.base_euler[reset_env_ids] = reset_base_euler
        self.base_lin_vel[reset_env_ids] = reset_base_lin_vel_body
        self.base_ang_vel[reset_env_ids] = reset_base_ang_vel_body
        self.projected_gravity[reset_env_ids] = reset_projected_gravity
        self.imu_ang_vel[reset_env_ids] = reset_base_ang_vel_body
        self.true_imu_ang_vel[reset_env_ids] = reset_base_ang_vel_body
        self.joint_pos[reset_env_ids] = reset_joint_pos
        self.leg_length[reset_env_ids] = reset_leg_length
        self.leg_angle[reset_env_ids] = reset_leg_angle
        self.target_joint_pos[reset_env_ids] = self.default_joint_pos
        self.reset_buf[reset_env_ids] = True
        for name in ("height_gate", "attitude_gate", "tracking_gate_raw", "tracking_gate"):
            getattr(self, name)[reset_env_ids] = 1.0
        for name in (
            "imu_lin_acc", "true_imu_lin_acc", "joint_vel", "wheel_vel", "gas_spring_force",
            "wheel_contact", "wheel_normal_force", "wheel_tangential_force", "base_contact", "alive_gate",
            "tilt_out_steps", "base_contact_steps", "actions", "last_actions", "target_wheel_vel", "episode_length_buf",
        ):
            getattr(self, name)[reset_env_ids] = 0

        self._update_encoder_measurements(reset_env_ids)
        self.jump.reset(reset_mask)
        self.recovery.reset(self, reset_mask)
        self.terrain_cross.reset(self, reset_mask)
        if self.tof is not None:
            self.tof.reset(reset_env_ids, strength=self.sensor_noise.strength)
            self.tof_distances[reset_env_ids] = self.tof.read(
                after_reset=True, env_ids=reset_env_ids, strength=self.sensor_noise.strength,
            )
            self.tof_clock.reset(reset_env_ids)
            if self.tof_history is not None:
                self.tof_history.reset(self.tof_distances, reset_env_ids)
                # Measured and nominal rays share geometry: read() just rebuilt
                # the global BVH, and no pose changed between these two queries.
                self.true_tof_distances[reset_env_ids] = self.tof.read_ground_truth(
                    after_reset=True, env_ids=reset_env_ids, refresh_context=False,
                )
                self.true_tof_history.reset(self.true_tof_distances, reset_env_ids)

        if record_episode:
            self.extras["episode"] = episode
        else:
            self.extras.pop("episode", None)
        self.rewards.reset(reset_mask)
        for value in self.episode_metric_sums.values():
            value[reset_mask] = 0.0
        self.locomotion_steps[reset_mask] = 0
        self._resample_commands(reset_mask, allow_jump=False)

    def _collect_episode_metrics(self, reset_mask):
        episode = self.rewards.metrics(reset_mask, "locomotion")
        episode.update(self.recovery.rewards.metrics(reset_mask, "recovery"))
        episode.update(self.jump.rewards.metrics(reset_mask, "jump"))
        episode.update(self.terrain_cross.episode_metrics(reset_mask))
        lengths = self.locomotion_steps[reset_mask].clamp_min(1)
        for name, total in self.episode_metric_sums.items():
            episode["locomotion/metric_" + name] = total[reset_mask] / lengths
        episode["episode_duration_s"] = self.episode_length_buf[reset_mask].clone() * self.dt
        episode["recovery/duration_s"] = self.recovery.buffers.elapsed_steps[reset_mask].clone() * self.dt
        episode["shared/curriculum/stage"] = torch.full_like(self.reward_buf[reset_mask], float(self.curriculum.current_stage_index))
        return episode


    def _resample_commands(self, mask, *, allow_jump=True):
        """仅平地周期采样普通命令/跳跃；特殊地形持有 reset 时的命令。"""
        selected = mask & ~self.terrain_cross.assigned & ~self.recovery.buffers.active & ~self.jump.buffers.active
        if selected.any():
            previous_vx = self.commands[:, 0].clone() if allow_jump else None
            commands = self._sample_commands(self.command_cfg, self.commands_limit)
            self.commands[selected] = commands[selected]
            if allow_jump:
                self.jump.on_resample(self, selected, previous_vx)
        self.terrain_cross.apply_commands(self.commands)
        self.recovery.apply_commands(self.commands)

    # ============ 奖励和终止 ============


    def _compute_rewards(self):
        locomotion = ~self.recovery.buffers.was_active & ~self.jump.buffers.was_active
        overrides = self.terrain_cross.reward_scale_overrides(self.rewards.scales)
        self.reward_buf.copy_(self.rewards.compute(locomotion, scale_overrides=overrides))
        self.reward_buf.add_(self.recovery.compute_rewards())
        self.reward_buf.add_(self.jump.compute_rewards())
        self.reward_buf.add_(self.terrain_cross.compute_rewards(exclude=self.rewards.scales))

    def _base_contact_failure(self):
        return (self.base_contact > 0.5) & ~self.recovery.buffers.was_active

    def _tilt_failure(self):
        return super()._tilt_failure() & ~self.recovery.buffers.was_active

    def _episode_timeout(self):
        return self.episode_length_buf >= self.max_episode_length


    # ============ 执行器与传感器 ============
    def _constrain_joint_targets(self, target_joint_pos):
        constrained = constrain_leg_targets(
            target_joint_pos, self.leg_front_joint_indices, self.leg_rear_joint_indices, self.max_motor_separation,
        )
        active = self.jump.buffers.active
        constrained[active] = target_joint_pos[active]
        return constrained

    def _apply_gas_spring_compensation(self):
        """根据弹簧压缩量和速度施加预载、刚度及阻尼补偿力。"""
        spring_pos = self.robot.get_dofs_position(self.springs_dof_idx)
        spring_vel = self.robot.get_dofs_velocity(self.springs_dof_idx)
        compression = torch.clamp(
            self.gas_spring_max_compression - spring_pos,
            0.0,
            self.gas_spring_max_compression,
        )
        force = self.gas_spring_preload_force + self.gas_spring_stiffness * compression - self.gas_spring_damping * spring_vel
        self.gas_spring_force.copy_(force)
        self.robot.control_dofs_force(force, self.springs_dof_idx)

    def _encoder_truth(self):
        return {"joint_pos": self.joint_pos, "joint_vel": self.joint_vel, "wheel_vel": self.wheel_vel}

    def _update_encoder_measurements(self, reset_env_ids=None):
        # 每个控制步只采样一次；reset 仅刷新被重置的环境。
        if reset_env_ids is None:
            self.sensor_noise.update(self._encoder_truth())
        else:
            self.sensor_noise.reset(reset_env_ids, self._encoder_truth())
        selected = slice(None) if reset_env_ids is None else reset_env_ids
        measured_pos = self.measured_joint_pos[selected]
        self.measured_leg_length[selected] = compute_leg_length(
            measured_pos,
            self.leg_front_joint_indices,
            self.leg_rear_joint_indices,
            self.leg_upper_link_length,
            self.leg_lower_link_length,
        )

    def _update_imu_measurements(self):
        """读取 actor 的 IMU 测量和 critic 使用的无噪声真值。"""
        imu_data = self.imu.read()
        self.imu_lin_acc.copy_(imu_data.lin_acc)
        self.imu_ang_vel.copy_(imu_data.ang_vel)
        imu_truth = self.imu.read_ground_truth()
        self.true_imu_lin_acc.copy_(imu_truth.lin_acc)
        self.true_imu_ang_vel.copy_(imu_truth.ang_vel)

    def _update_wheel_contact_forces(self, contacts=None):
        """逐接触点分解法向/切向力，再按轮求和并转到机身坐标系，单位 N。"""
        if contacts is None:
            contacts = self.robot.get_contacts(exclude_self_contact=True)
        normal = contacts["normal"]
        normal_squared = normal.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        base_quat_inv = inv_quat(self.base_quat)
        for side, link in enumerate(self.wheel_links_idx):
            # Genesis 接触索引为全场景编号，轮子索引为机器人局部编号。
            is_a = contacts["link_a"] == self.robot.link_start + link
            is_b = contacts["link_b"] == self.robot.link_start + link
            valid = contacts["valid_mask"] & (is_a | is_b)
            force = torch.where(is_a[..., None], contacts["force_a"], contacts["force_b"])
            normal_force = (force * normal).sum(dim=-1, keepdim=True) * normal / normal_squared
            tangential_force = force - normal_force
            self.wheel_normal_force[:, side] = transform_by_quat(
                torch.where(valid[..., None], normal_force, 0.0).sum(dim=1), base_quat_inv)
            self.wheel_tangential_force[:, side] = transform_by_quat(
                torch.where(valid[..., None], tangential_force, 0.0).sum(dim=1), base_quat_inv)

    def _body_contact_link_indices(self):
        """基座与双腿共用碰撞惩罚/终止；轮子和云台不属于腿部。"""
        leg_prefixes = tuple(f"{side}_{part}" for side in ("left", "right") for part in ("front", "rear", "spring"))
        return torch.tensor(
            [link.idx_local for link in self.robot.links
             if link.idx_local == self.base_link_idx or link.name.startswith(leg_prefixes)],
            dtype=gs.tc_int,
            device=self.device,
        )

    def _update_wheel_contact(self):
        """更新轮子、基座/腿部接触和存活门控。"""
        all_contact_forces = self.robot.get_links_net_contact_force()
        contact_forces = all_contact_forces[:, self.wheel_links_idx, :]
        self.wheel_contact.copy_((torch.linalg.vector_norm(contact_forces, dim=-1) > self.wheel_contact_force_threshold).to(dtype=gs.tc_float))
        contacts = self.robot.get_contacts(exclude_self_contact=True)
        self._update_wheel_contact_forces(contacts)
        self.base_contact_regions.update(contacts, self.base_pos, self.base_quat)
        body_contact_forces = torch.linalg.vector_norm(all_contact_forces[:, self.body_contact_links_idx, :], dim=-1)
        self.base_contact.copy_((body_contact_forces > self.base_contact_force_threshold).any(dim=1).to(dtype=gs.tc_float))
        # 任一基座/腿部 link 碰撞都会关闭存活奖励；轮子离地由 wheel_airborne 单独约束。
        self.alive_gate.copy_((self.base_contact < 0.5).to(dtype=gs.tc_float))

    def _apply_sensor_noise(self, values):
        self.sensor_noise.apply_curriculum(values)
        # 第 0 阶段在创建 IMU 前执行，必须同步它的初始配置。
        self.imu_cfg = self.sensor_noise.imu_options(self.obs_cfg.get("imu", {}))

    # ============ 诊断与可视化 ============
    def get_observation_components(self, env_idx=0):
        """返回指定环境中组成策略输入的、已经缩放并命名的观测分量。"""
        return {name: value[env_idx].detach() for name, value in self.obs_components.items()}

    def get_pd_diagnostics(self, env_idx=0):
        """返回配置的 PD 增益、目标状态，以及由速度计算的阻尼估算项。"""
        joint_vel = self.joint_vel[env_idx].detach()
        wheel_vel = self.wheel_vel[env_idx].detach()
        return {
            "joint_names": tuple(self.env_cfg["joint_names"]),
            "joint_kp": self.joint_kp[env_idx],
            "joint_kd": self.joint_kd[env_idx],
            "joint_force_limit": self.joint_force_limit[env_idx],
            "joint_pos": self.joint_pos[env_idx].detach(),
            "joint_target_pos": self.target_joint_pos[env_idx].detach(),
            "joint_vel": joint_vel,
            # 这里只是 Kd 对应的阻尼分量，不是仿真器测得的完整执行器力矩。
            "joint_kd_damping": -self.joint_kd[env_idx] * joint_vel,
            "wheel_names": tuple(self.env_cfg["wheel_names"]),
            "wheel_kd": self.wheel_kd[env_idx],
            "wheel_force_limit": self.wheel_force_limit[env_idx],
            "wheel_target_vel": self.target_wheel_vel[env_idx].detach(),
            "wheel_vel": wheel_vel,
            "wheel_kd_damping": -self.wheel_kd[env_idx] * wheel_vel,
        }

    def focus_viewer(self):
        """对准 0 号机器人并同步旋转中心，之后保留鼠标自由视角。"""
        viewer = self.scene.viewer
        if viewer is None:
            return
        lookat = self.robot.get_pos(relative=False)[0].detach().cpu().numpy().copy()
        pos = lookat + (1.8, -2.8, 1.2)
        pose = pos_lookat_up_to_T(pos, lookat, up=np.array([0.0, 0.0, 1.0]))
        with viewer.lock:
            viewer.set_camera_pose(pose=pose)
            # Genesis 1.2 的 set_camera_pose 只改位姿，不更新 trackball 的旋转中心。
            # 出生在其他地形 patch 时也必须围绕机器人旋转，不能继续绕世界原点。
            trackball = viewer._pyrender_viewer._trackball
            trackball._target = lookat.copy()
            trackball._n_target = lookat.copy()
