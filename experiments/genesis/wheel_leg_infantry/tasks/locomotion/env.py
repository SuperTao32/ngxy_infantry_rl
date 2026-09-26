import math

import numpy as np
import genesis as gs
from genesis.utils.geom import (
    inv_quat,
    pos_lookat_up_to_T,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
    xyz_to_quat,
)
from genesis.vis.keybindings import Key, KeyAction, Keybind

import torch
from tensordict import TensorDict

from ...core.curriculum import CurriculumManager
from ...core.domain_randomization import DomainRandomizationManager, MOTOR_NAMES, SPRING_NAMES
from ...core.kinematics import compute_leg_angle, compute_leg_length, constrain_leg_targets
from ...core.tensor_utils import as_gain_tensor, as_range_tensors, sample_uniform
from ...core.terrain import TerrainManager
from .rewards import LocomotionRewards
from .tracking_gate import DEFAULT_TRACKING_GATE_CFG, merge_tracking_gate_config, smooth_gate
from .velocity_estimator import (
    complementary_forward_velocity_update,
    gravity_compensated_forward_acceleration,
    wheel_forward_velocity,
)


class LocomotionEnv(LocomotionRewards):
    """轮腿机器人 locomotion 环境，集中管理仿真、控制、观测、奖励和课程状态。"""

    # ============ 初始化与构建辅助：只准备配置、索引和场景实体 ============
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
    ):
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
        self.steps_per_iteration = int(steps_per_iteration) # PPO runner 每采满 steps_per_iteration 个控制步完成一轮
        self.tracking_gate_cfg = {}
        self._apply_tracking_gate(reward_cfg.get("tracking_gate", DEFAULT_TRACKING_GATE_CFG))
        self.terrain = TerrainManager(env_cfg.get("terrain"))
        # 记录 TerrainManager 补齐后的配置，保证课程、日志与运行状态一致。
        self.env_cfg["terrain"] = self.terrain.config

        self.domain_rand = DomainRandomizationManager(env_cfg.get("domain_rand"))
        self.env_cfg["domain_rand"] = self.domain_rand.config
        self.batch_dofs_info = (
            self.domain_rand.requires_batched_dofs
            or bool(env_cfg.get("handoff_on_landing", False))
            or self._requires_batched_motor_params()
        )

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
        self.dt = 0.02
        self.resample_step = self.env_cfg["resampling_time_s"] / self.dt
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # IMU配置
        self.imu_cfg = dict(obs_cfg.get("imu", {}))
        self.imu_acc_scale = float(obs_cfg["obs_scales"].get("lin_acc", 1.0 / 9.81))

        # 速度估计器配置
        self.velocity_estimator_cfg = dict(obs_cfg.get("velocity_estimator", {}))
        self.wheel_radius = float(self.velocity_estimator_cfg.get("wheel_radius", 0.06))
        self.wheel_velocity_sign = float(self.velocity_estimator_cfg.get("wheel_velocity_sign", 1.0))
        self.gravity_magnitude = float(self.velocity_estimator_cfg.get("gravity_magnitude", 9.81))
        self.wheel_correction_time_constant_s = float(self.velocity_estimator_cfg.get("wheel_correction_time_constant_s", 0.5))
        self.max_estimated_velocity = float(self.velocity_estimator_cfg.get("max_abs_velocity", 3.0))

        # 气弹簧配置
        self.spring_names = tuple(env_cfg.get("spring_names", ("left_spring2_joint", "right_spring2_joint")))
        self.gas_spring_preload_force = float(env_cfg.get("gas_spring_preload_force", 420.0))
        self.gas_spring_stiffness = float(env_cfg.get("gas_spring_stiffness", 1400.0))
        self.gas_spring_damping = float(env_cfg.get("gas_spring_damping", 50.0))
        self.gas_spring_max_compression = float(env_cfg.get("gas_spring_max_compression", 0.06))

        # 着陆惩罚配置
        self.wheel_contact_force_threshold = float(env_cfg.get("wheel_contact_force_threshold", 1.0))
        self.base_contact_force_threshold = float(env_cfg.get("base_contact_force_threshold", 5.0))
        self.landing_penalty_enabled = self._landing_penalty_window_enabled()
        self.landing_penalty_duration_s = float(env_cfg.get("landing_penalty_duration_s", 0.30)) if self.landing_penalty_enabled else 0.0
        self.landing_penalty_steps = max(1, math.ceil(self.landing_penalty_duration_s / self.dt)) if self.landing_penalty_enabled else 1
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
        leg_angle_limits = tuple(self.env_cfg.get("leg_angle_limit_range", (-0.25 * math.pi, 0.25 * math.pi)))
        self.leg_front_joint_indices = [joint_names.index(name) for name in front_joint_names]
        self.leg_rear_joint_indices = [joint_names.index(name) for name in rear_joint_names]
        self.leg_angle_lower, self.leg_angle_upper = map(float, leg_angle_limits)
        # motor 控制参数
        self.joint_kp = as_gain_tensor(
            self.env_cfg["joint_kp"], self.num_joints, "joint_kp", device=self.device
        )
        self.joint_kd = as_gain_tensor(
            self.env_cfg["joint_kd"], self.num_joints, "joint_kd", device=self.device
        )
        self.wheel_kd = as_gain_tensor(
            self.env_cfg["wheel_kd"], self.num_wheels, "wheel_kd", device=self.device
        )
        self.joint_force_limit = as_gain_tensor(
            self.env_cfg.get("joint_force_limit", 54.0),
            self.num_joints,
            "joint_force_limit",
            device=self.device,
        )
        self.wheel_force_limit = as_gain_tensor(
            self.env_cfg.get("wheel_force_limit", 5.0),
            self.num_wheels,
            "wheel_force_limit",
            device=self.device,
        )
        # 腿部连杆几何参数
        self.min_upper_link_angle = float(self.env_cfg.get("min_upper_link_angle", 0.5 * math.pi))
        self.leg_upper_link_length = float(env_cfg.get("leg_upper_link_length", 0.21))
        self.leg_lower_link_length = float(env_cfg.get("leg_lower_link_length", 0.25))
        self.max_motor_separation = math.pi - self.min_upper_link_angle
        # 初始关节位置、腿长和腿角
        self.init_joint_pos = torch.tensor(
            [self.env_cfg["default_joint_pos"][name] for name in joint_names],
            dtype=gs.tc_float,
            device=self.device,
        )
        self.default_joint_pos = self.init_joint_pos.clone()
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

        # 验证以上配置合法
        self._validate_configuration()

        # 奖励函数与 episode 统计
        self.reward_functions, self.episode_sums = {}, {}
        self.raw_reward_scales: dict[str, float | list[float]] = dict(reward_cfg["reward_scales"])
        self.reward_scales: dict[str, float | torch.Tensor] = {}
        self._apply_reward_scales(self.raw_reward_scales)
        self.episode_metric_sums = {
            "height_gate": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
            "attitude_gate": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
            "tracking_gate": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
            "tracking_gate_fully_open": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
            "velocity_estimator_abs_error": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
            "alive_gate": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
            "landing_event": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
            **(
                {"landing_penalty_gate": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device)} if self.landing_penalty_enabled else {}
            ),
            "base_contact": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
        }

        # 课程注册及第 0 阶段配置
        self.global_step = 0
        self.training_iteration = 0
        self.curriculum = CurriculumManager(self.curriculum_cfg)
        self.curriculum.register_target("command_ranges", self._apply_command_ranges)
        self.curriculum.register_target("reward_scales", self._apply_reward_scales)
        self.curriculum.register_target("tracking_gate", self._apply_tracking_gate)
        self.curriculum.register_target("termination_limits", self._apply_termination_limits)
        self.curriculum.register_target("action_limits", self._apply_action_limits)
        self.curriculum.register_target("reset_ranges", self._apply_reset_ranges)
        self.curriculum.register_target("terrain", self._apply_terrain_curriculum)
        self.curriculum.register_target("domain_rand", self.domain_rand.apply_curriculum)
        if self.curriculum.update(0, force=True):
            print(f"[curriculum] stage={self.curriculum.current_stage_name} step=0")

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
                max_collision_pairs=20,
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

        # actor 固定使用 IMU 和轮速融合估计，仿真真值只供 critic 和诊断使用。
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

        # build环境
        self.scene.build(n_envs=num_envs)
        # Genesis 在 build 后才生成 terrain_hf，此时才能绑定高度场供 reset/step 查询。
        self.terrain.bind_entity(
            self.terrain_entity,
            device=self.device,
            dtype=gs.tc_float,
        )

        ## 创建索引
        self.joints_dof_idx = self._joint_dof_indices(joint_names)
        self.springs_dof_idx = self._joint_dof_indices(self.spring_names)
        self.wheels_dof_idx = self._joint_dof_indices(wheel_names)
        controlled_names = set((*joint_names, *wheel_names, *self.spring_names))
        self.passive_joint_names = tuple(
            joint.name for joint in self.robot.joints
            if joint.type == gs.JOINT_TYPE.REVOLUTE and joint.name not in controlled_names
        )
        self.passive_dof_idx = self._joint_dof_indices(self.passive_joint_names)
        self.wheel_links_idx = torch.tensor(
            [self.robot.get_link(name).idx_local for name in wheel_link_names],
            dtype=gs.tc_int,
            device=self.device,
        )
        self.base_link_idx = self.robot.get_link(base_link_name).idx_local

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
        # 诊断参数与 manager 的实际参数共用 buffer，始终带并行环境维度。
        for name, value in self.domain_rand.motor_values.items():
            setattr(self, name, value)
        for name, value in self.domain_rand.gas_spring_values.items():
            setattr(self, f"gas_spring_{name}", value)

        # 初始化joint和dof的初始位置
        init_dof_pos_list = []
        for joint in self.robot.joints[1:]:
            if joint.n_qs == 0:
                continue
            init_pos = self.env_cfg["default_joint_pos"].get(joint.name, 0.0)
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
        self.reward_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.bool, device=gs.device)
        self.terminated_buf = torch.empty((self.num_envs,), dtype=torch.bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=self.device)
        self.extras = {}

        # 执行器与腿部状态
        self.joint_pos = torch.empty((self.num_envs, self.num_joints), dtype=gs.tc_float, device=gs.device)
        self.joint_vel = torch.empty_like(self.joint_pos)
        self.wheel_vel = torch.empty((self.num_envs, self.num_wheels), dtype=gs.tc_float, device=gs.device)
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

        # actor 可部署的 IMU/轮速估计量
        self.imu_lin_acc = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.imu_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.forward_kinematic_acc = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.wheel_forward_vel = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.estimated_base_lin_vel = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        # 接触状态
        self.wheel_contact = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.base_contact = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.previous_both_wheels_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=gs.device)

        # 奖励门控、着陆事件和终止计数
        self.height_gate = torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.attitude_gate = torch.ones_like(self.height_gate)
        self.tracking_gate_raw = torch.ones_like(self.height_gate)
        self.tracking_gate = torch.ones_like(self.height_gate)
        self.alive_gate = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.landing_event = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.landing_penalty_steps_left = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.landing_penalty_gate = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.tilt_out_steps = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.base_contact_steps = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        # 通用 buffer 就绪后，由子任务追加自己的状态。
        self._initialize_task_buffers()

        # 初始reset环境
        self.reset()
        if show_viewer:
            # 只在启动或按 ENTER 时对准机器人；逐帧 follow_entity 会覆盖鼠标调整。
            self.scene.viewer.register_keybinds(Keybind("camera_focus_robot", Key.ENTER, key_action=KeyAction.PRESS, callback=self.focus_viewer))
            self.focus_viewer()

    def _validate_configuration(self) -> None:
        """集中校验构造环境所需的配置类型、数值和维度约束。"""
        joint_names = tuple(self.env_cfg["joint_names"])
        front_joint_names = tuple(self.env_cfg.get("leg_front_joint_names", ("left_front1_joint", "right_front1_joint")))
        rear_joint_names = tuple(self.env_cfg.get("leg_rear_joint_names", ("left_rear1_joint", "right_rear1_joint")))
        leg_angle_limits = tuple(self.env_cfg.get("leg_angle_limit_range", (-0.25 * math.pi, 0.25 * math.pi)))

        if self.steps_per_iteration <= 0:
            raise ValueError("steps_per_iteration must be positive")
        if self.num_actions != self.num_joints + self.num_wheels:
            raise ValueError("num_actions must equal num_joints + num_wheels")
        if len(joint_names) != self.num_joints:
            raise ValueError("joint_names length must equal num_joints")
        if len(self.env_cfg["wheel_names"]) != self.num_wheels:
            raise ValueError("wheel_names length must equal num_wheels")
        if len(front_joint_names) != 2 or len(rear_joint_names) != 2:
            raise ValueError("leg_front_joint_names and leg_rear_joint_names must each contain left and right joints")
        if any(name not in joint_names for name in (*front_joint_names, *rear_joint_names)):
            raise ValueError("leg-angle joints must also be present in joint_names")
        if len(leg_angle_limits) != 2 or not all(math.isfinite(value) for value in leg_angle_limits):
            raise ValueError("leg_angle_limit_range must contain two finite values")
        if leg_angle_limits[0] > leg_angle_limits[1]:
            raise ValueError("leg_angle_limit_range lower bound must not exceed upper bound")
        if not 0.0 < self.min_upper_link_angle <= math.pi:
            raise ValueError("min_upper_link_angle must be in (0, pi]")
        if self.wheel_radius <= 0.0:
            raise ValueError("velocity_estimator.wheel_radius must be positive")
        if self.wheel_velocity_sign not in {-1.0, 1.0}:
            raise ValueError("velocity_estimator.wheel_velocity_sign must be -1.0 or 1.0")
        if self.gravity_magnitude <= 0.0:
            raise ValueError("velocity_estimator.gravity_magnitude must be positive")
        if self.wheel_correction_time_constant_s < 0.0:
            raise ValueError("velocity_estimator.wheel_correction_time_constant_s cannot be negative")
        if self.max_estimated_velocity <= 0.0:
            raise ValueError("velocity_estimator.max_abs_velocity must be positive")
        if len(self.spring_names) != 2:
            raise ValueError("spring_names must contain the left and right gas-spring joints")
        if len(self.env_cfg.get("wheel_link_names", ("left_wheel_link", "right_wheel_link"))) != 2:
            raise ValueError("wheel_link_names must contain the left and right wheel links")
        if self.gas_spring_max_compression <= 0.0:
            raise ValueError("gas_spring_max_compression must be positive")
        if self.wheel_contact_force_threshold < 0.0:
            raise ValueError("wheel_contact_force_threshold cannot be negative")
        if self.base_contact_force_threshold < 0.0:
            raise ValueError("base_contact_force_threshold cannot be negative")
        if self.leg_upper_link_length <= 0.0 or self.leg_lower_link_length <= 0.0:
            raise ValueError("leg link lengths must be positive")
        if self.landing_penalty_enabled and self.landing_penalty_duration_s <= 0.0:
            raise ValueError("landing_penalty_duration_s must be positive")
        if self.tilt_termination_duration_s <= 0.0:
            raise ValueError("tilt_termination_duration_s must be positive")
        if self.base_contact_termination_duration_s is not None and self.base_contact_termination_duration_s <= 0.0:
            raise ValueError("base_contact_termination_duration_s must be positive")
        if torch.any(self.joint_force_limit <= 0.0) or torch.any(self.wheel_force_limit <= 0.0):
            raise ValueError("joint_force_limit and wheel_force_limit must be positive")

    def _joint_dof_indices(self, names):
        """按配置名称顺序构造关节自由度索引。"""
        return torch.tensor(
            [self.robot.get_joint(name).dof_start for name in names],
            dtype=gs.tc_int,
            device=self.device,
        )

    def _add_terrain(self):
        """在 scene.build 前创建配置指定的地形实体。"""
        self.terrain_entity = self.terrain.add_to_scene(self.scene)
        self.friction_terrain_entities = (
            self.terrain_entity if isinstance(self.terrain_entity, tuple) else (self.terrain_entity,)
        )

    def _requires_batched_motor_params(self):
        """任务可在场景构建前声明需要逐环境设置电机参数。"""
        return False

    def _apply_motor_params(self, params, envs_idx=None):
        self.domain_rand.apply_motor_params(params, envs_idx)

    # ============ 公共环境接口：训练器调用的 reset / step / 观测与恢复入口 ============
    def reset(self):
        """重置全部并行环境并返回重置后的 actor/critic 观测。"""
        self._reset_idx()
        self._update_observations()
        return self.get_observations()

    def step(self, actions):
        """执行一个控制步，返回观测、奖励、重置标志和附加统计。"""
        # PPO runner 每采满 steps_per_iteration 个控制步完成一轮；课程只在轮次边界切换。
        # jump 的 locomotion warmup 会覆盖该 hook，保证被丢弃的 teacher 步不推进训练时钟。
        if self._should_advance_training_clock():
            if self.global_step % self.steps_per_iteration == 0:
                self.training_iteration = self.global_step // self.steps_per_iteration
                if self.curriculum.update(self.training_iteration):
                    print(f"[curriculum] stage={self.curriculum.current_stage_name} " f"iteration={self.training_iteration} step={self.global_step}")
            self.global_step += 1

        ########### 执行动作 ###########
        joint_actions = torch.clip(
            actions[:, : self.num_joints],
            -self.env_cfg["clip_joint_action"],
            self.env_cfg["clip_joint_action"],
        )
        wheel_actions = torch.clip(
            actions[:, self.num_joints : self.num_joints + self.num_wheels],
            -self.env_cfg["clip_wheel_action"],
            self.env_cfg["clip_wheel_action"],
        )
        self.actions = torch.concatenate((joint_actions, wheel_actions), dim=-1)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        target_joint_pos = exec_actions[:, : self.num_joints] * self.env_cfg["joint_pos_scale"]
        target_wheel_vel = exec_actions[:, self.num_joints : self.num_joints + self.num_wheels] * self.env_cfg["wheel_vel_scale"]

        target_joint_pos = target_joint_pos + self.default_joint_pos
        target_joint_pos = self.domain_rand.offset_joint_targets(target_joint_pos)
        target_joint_pos = constrain_leg_targets(
            target_joint_pos,
            self.leg_front_joint_indices,
            self.leg_rear_joint_indices,
            self.leg_angle_lower,
            self.leg_angle_upper,
            self.max_motor_separation,
        )
        self.target_joint_pos.copy_(target_joint_pos)
        self.target_wheel_vel.copy_(target_wheel_vel)

        self.robot.control_dofs_position(target_joint_pos, self.joints_dof_idx)
        self.robot.control_dofs_velocity(target_wheel_vel, self.wheels_dof_idx)
        self._apply_task_control()
        self._apply_gas_spring_compensation()
        self.domain_rand.before_step()
        self.scene.step()

        ########### 更新buffer ###########
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.terrain_height.copy_(self.terrain.height_at(self.base_pos[:, :2]))
        self.base_height.copy_(self.base_pos[:, 2] - self.terrain_height)
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.base_quat, self.init_base_quat_inv),
            rpy=True,
            degrees=True,
        )
        # 机器人的速度转化到自身坐标系下
        base_quat_inv = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), base_quat_inv)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), base_quat_inv)
        self.projected_gravity = transform_by_quat(self.global_gravity_dir, base_quat_inv)

        self.joint_pos = self.robot.get_dofs_position(self.joints_dof_idx)
        self.joint_vel = self.robot.get_dofs_velocity(self.joints_dof_idx)
        self.wheel_vel = self.robot.get_dofs_velocity(self.wheels_dof_idx)
        self._update_velocity_estimator()
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
        self._update_task_state()

        ########### 判断终止 ###########
        roll_out = torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        pitch_out = torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        tilt_out = roll_out | pitch_out
        self.tilt_out_steps.copy_(torch.where(tilt_out, self.tilt_out_steps + 1, 0))
        self.base_contact_steps.copy_(torch.where(self.base_contact > 0.5, self.base_contact_steps + 1, 0))
        persistent_tilt = self.tilt_out_steps >= self.tilt_termination_steps
        if self.base_contact_termination_steps is None:
            persistent_base_contact = torch.zeros_like(self.base_contact, dtype=torch.bool)
        else:
            persistent_base_contact = self.base_contact_steps >= self.base_contact_termination_steps
        time_out = self.episode_length_buf > self.max_episode_length
        # patch 接缝不属于任何一种训练地形，接近边界时按 timeout 重置并允许 bootstrap。
        terrain_out = self.terrain.out_of_bounds(self.base_pos[:, :2], self.terrain_spawn_centers)
        time_out = time_out | terrain_out
        solver_error = self.scene.rigid_solver.get_error_envs_mask().bool()

        self.terminated_buf = persistent_tilt | persistent_base_contact | solver_error | self._task_termination()
        self.reset_buf = self.terminated_buf | time_out

        ########### 计算奖励 ###########
        self.reward_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            r = reward_func() * self._get_reward_scale(name)
            self.reward_buf += r
            self.episode_sums[name] += r
        self.episode_metric_sums["height_gate"] += self.height_gate
        self.episode_metric_sums["attitude_gate"] += self.attitude_gate
        self.episode_metric_sums["tracking_gate"] += self.tracking_gate
        self.episode_metric_sums["alive_gate"] += self.alive_gate
        self.episode_metric_sums["landing_event"] += self.landing_event
        if self.landing_penalty_enabled:
            self.episode_metric_sums["landing_penalty_gate"] += self.landing_penalty_gate
        self.episode_metric_sums["base_contact"] += self.base_contact
        self.episode_metric_sums["tracking_gate_fully_open"] += (self.tracking_gate_raw >= 0.95).to(dtype=gs.tc_float)
        self.episode_metric_sums["velocity_estimator_abs_error"] += torch.abs(self.estimated_base_lin_vel - self.base_lin_vel[:, 0])
        self._accumulate_task_metrics()

        ########### 重采样指令 ###########
        self._resample_commands(self.episode_length_buf % self.resample_step == 0)

        ########### 计算timeout ###########
        self.extras["time_outs"] = time_out.to(dtype=gs.tc_float)

        ########### 重置环境（如果需要）###########
        self._reset_idx(self.reset_buf)

        ########### 更新观测值 ###########
        self._update_observations()

        ########### 更新旧值 ###########
        self.last_actions.copy_(self.actions)

        return self.get_observations(), self.reward_buf, self.reset_buf, self.extras

    def get_observations(self):
        """以 TensorDict 返回策略观测和特权 critic 观测。"""
        return TensorDict(
            {
                "policy": self.obs_buf,
                "critic": self.critic_obs_buf,
            },
            batch_size=[self.num_envs],
        )

    def set_training_iteration(self, iteration: int) -> None:
        """将环境和课程推进到指定 PPO 轮次，并按该阶段重新初始化。"""
        iteration = int(iteration)
        if iteration < 0:
            raise ValueError("training iteration cannot be negative")

        self.training_iteration = iteration
        self.global_step = iteration * self.steps_per_iteration
        self.curriculum.update(iteration, force=True)
        print(f"[curriculum] resumed stage={self.curriculum.current_stage_name} " f"iteration={iteration} step={self.global_step}")
        # 构造环境时已经按第 0 阶段 reset 过；这里再次 reset，确保初始状态、
        # 指令范围和 episode 统计都来自恢复点对应的课程阶段。
        self.reset()

    # ============ 重置与采样：先采样完整状态，再同步 robot 和环境 buffer ============
    def _reset_idx(self, env_idx=None):
        """重置布尔掩码选中的环境；None 表示重置全部环境。"""
        if env_idx is not None and not env_idx.any():
            self.extras.pop("episode", None)
            return

        finished_episode_lengths = None
        if env_idx is not None and env_idx.any():
            finished_episode_lengths = self.episode_length_buf[env_idx].clone().clamp_min(1).to(gs.tc_float)

        # 重置机器人状态
        reset_env_ids = (
            torch.arange(self.num_envs, dtype=torch.long, device=gs.device) if env_idx is None else torch.nonzero(env_idx, as_tuple=False).flatten()
        )

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
        ) = self._sample_base_reset_state(reset_env_ids)
        self.domain_rand.reset(reset_env_ids)

        reset_quat_inv = inv_quat(reset_base_quat)
        reset_base_lin_vel_body = transform_by_quat(reset_base_lin_vel, reset_quat_inv)
        reset_base_ang_vel_body = transform_by_quat(reset_base_ang_vel, reset_quat_inv)

        self.robot.set_qpos(reset_qpos, envs_idx=env_idx, zero_velocity=True, skip_forward=True)
        # Genesis 浮动基座 qvel 的前 3 维是世界系线速度，后 3 维是机体系角速度。
        self.robot.set_dofs_velocity(
            torch.concatenate((reset_base_lin_vel, reset_base_ang_vel_body), dim=-1),
            dofs_idx_local=slice(0, 6),
            envs_idx=env_idx,
            skip_forward=True,
        )

        reset_base_euler = quat_to_xyz(
            transform_quat_by_quat(reset_base_quat, self.init_base_quat_inv),
            rpy=True,
            degrees=True,
        )

        # 重置buffers
        if env_idx is None:
            # 状态相关
            self.base_pos.copy_(reset_base_pos)
            self.terrain_height.copy_(reset_terrain_height)
            self.base_height.copy_(reset_base_pos[:, 2] - reset_terrain_height)
            self.terrain_spawn_centers.copy_(reset_terrain_centers)
            self.terrain_tile_index.copy_(reset_terrain_tile_index)
            self.base_quat.copy_(reset_base_quat)
            self.base_euler.copy_(reset_base_euler)
            self.base_lin_vel.copy_(reset_base_lin_vel_body)
            self.base_ang_vel.copy_(reset_base_ang_vel_body)
            self.projected_gravity.copy_(reset_projected_gravity)
            self.imu_lin_acc.zero_()
            self.imu_ang_vel.copy_(reset_base_ang_vel_body)
            self.forward_kinematic_acc.zero_()
            self.wheel_forward_vel.zero_()
            # 不把仿真真值泄漏给 IMU/轮速估计器；真机 reset 时也不知道绝对线速度。
            self.estimated_base_lin_vel.zero_()
            self.height_gate.fill_(1.0)
            self.attitude_gate.fill_(1.0)
            self.tracking_gate_raw.fill_(1.0)
            self.tracking_gate.fill_(1.0)
            self.joint_pos.copy_(self.init_joint_pos)
            self.joint_vel.zero_()
            self.wheel_vel.zero_()
            self.gas_spring_force.zero_()
            self.wheel_contact.zero_()
            self.base_contact.zero_()
            self.alive_gate.zero_()
            self.previous_both_wheels_contact.zero_()
            self.landing_event.zero_()
            self.landing_penalty_steps_left.zero_()
            self.landing_penalty_gate.zero_()
            self.tilt_out_steps.zero_()
            self.base_contact_steps.zero_()
            self.leg_length.copy_(self.init_leg_length)
            self.leg_angle.copy_(self.init_leg_angle)
            # 其他
            self.reset_buf.fill_(True)
            self.actions.zero_()
            self.last_actions.zero_()
            self.target_joint_pos.copy_(self.default_joint_pos)
            self.target_wheel_vel.zero_()
            self.episode_length_buf.zero_()
        else:
            self.base_pos[env_idx] = reset_base_pos
            self.terrain_height[env_idx] = reset_terrain_height
            self.base_height[env_idx] = reset_base_pos[:, 2] - reset_terrain_height
            self.terrain_spawn_centers[env_idx] = reset_terrain_centers
            self.terrain_tile_index[env_idx] = reset_terrain_tile_index
            self.base_quat[env_idx] = reset_base_quat
            self.base_euler[env_idx] = reset_base_euler
            self.base_lin_vel[env_idx] = reset_base_lin_vel_body
            self.base_ang_vel[env_idx] = reset_base_ang_vel_body
            self.projected_gravity[env_idx] = reset_projected_gravity
            self.height_gate.masked_fill_(env_idx, 1.0)
            self.attitude_gate.masked_fill_(env_idx, 1.0)
            self.tracking_gate_raw.masked_fill_(env_idx, 1.0)
            self.tracking_gate.masked_fill_(env_idx, 1.0)
            torch.where(env_idx[:, None], self.init_joint_pos, self.joint_pos, out=self.joint_pos)
            self.imu_lin_acc.masked_fill_(env_idx[:, None], 0.0)
            self.imu_ang_vel[env_idx] = reset_base_ang_vel_body
            self.forward_kinematic_acc.masked_fill_(env_idx, 0.0)
            self.wheel_forward_vel.masked_fill_(env_idx, 0.0)
            self.estimated_base_lin_vel.masked_fill_(env_idx, 0.0)
            self.joint_vel.masked_fill_(env_idx[:, None], 0.0)
            self.wheel_vel.masked_fill_(env_idx[:, None], 0.0)
            self.gas_spring_force.masked_fill_(env_idx[:, None], 0.0)
            self.wheel_contact.masked_fill_(env_idx[:, None], 0.0)
            self.base_contact.masked_fill_(env_idx, 0.0)
            self.alive_gate.masked_fill_(env_idx, 0.0)
            self.previous_both_wheels_contact.masked_fill_(env_idx, False)
            self.landing_event.masked_fill_(env_idx, 0.0)
            self.landing_penalty_steps_left.masked_fill_(env_idx, 0)
            self.landing_penalty_gate.masked_fill_(env_idx, 0.0)
            self.tilt_out_steps.masked_fill_(env_idx, 0)
            self.base_contact_steps.masked_fill_(env_idx, 0)
            torch.where(env_idx[:, None], self.init_leg_length, self.leg_length, out=self.leg_length)
            torch.where(env_idx[:, None], self.init_leg_angle, self.leg_angle, out=self.leg_angle)
            self.reset_buf.masked_fill_(env_idx, True)
            self.actions.masked_fill_(env_idx[:, None], 0.0)
            self.last_actions.masked_fill_(env_idx[:, None], 0.0)
            torch.where(env_idx[:, None], self.default_joint_pos, self.target_joint_pos, out=self.target_joint_pos)
            self.target_wheel_vel.masked_fill_(env_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(env_idx, 0)

        self._reset_task_buffers(env_idx)

        # 更新extras和episoded的reward
        if env_idx is not None and env_idx.any():
            self.extras["episode"] = {}
            for key, value in self.episode_sums.items():
                # 保存真实 episode 累计贡献，避免短 episode 被固定时长除法掩盖。
                self.extras["episode"]["reward_" + key] = value[env_idx]
                # 只清空已经结束的环境
                value.masked_fill_(env_idx, 0.0)
            for key, value in self.episode_metric_sums.items():
                self.extras["episode"]["metric_" + key] = value[env_idx] / finished_episode_lengths
                value.masked_fill_(env_idx, 0.0)
            if self.curriculum.enabled:
                self.extras["episode"]["curriculum_stage"] = torch.full_like(self.reward_buf[env_idx], float(self.curriculum.current_stage_index))
            self.extras["episode"]["episode_duration_s"] = finished_episode_lengths * self.dt
        else:
            # 没有 episode 结束，就不要让 logger 收到虚假的零
            self.extras.pop("episode", None)

            # env_idx=None 表示初始化或手动重置全部环境
            if env_idx is None:
                for value in self.episode_sums.values():
                    value.zero_()
                for value in self.episode_metric_sums.values():
                    value.zero_()

        # 重选指令
        self._resample_commands(env_idx)

    def _sample_base_reset_state(self, env_ids):
        """为本次 reset 的每个子环境独立采样 base pose 和速度。"""
        num_resets = env_ids.numel()
        reset_base_pos = sample_uniform(
            self.base_init_pos_lower,
            self.base_init_pos_upper,
            (num_resets,),
        )
        terrain_centers, terrain_tile_index = self.terrain.sample_spawn_tiles(env_ids)
        # 配置中的 x/y 是相对 patch 中心的扰动，z 是相对当地地面的初始高度。
        reset_base_pos[:, :2] += terrain_centers
        terrain_height = self.terrain.height_at(reset_base_pos[:, :2])
        reset_base_pos[:, 2] += terrain_height

        rpy_offset_deg = sample_uniform(
            self.base_init_rpy_lower,
            self.base_init_rpy_upper,
            (num_resets,),
        )
        quat_offset = xyz_to_quat(rpy_offset_deg, rpy=True, degrees=True)
        nominal_quat = self.init_base_quat.expand(num_resets, -1)
        reset_base_quat = transform_quat_by_quat(quat_offset, nominal_quat)

        reset_base_lin_vel = sample_uniform(
            self.base_init_lin_vel_lower,
            self.base_init_lin_vel_upper,
            (num_resets,),
        )
        reset_base_ang_vel = sample_uniform(
            self.base_init_ang_vel_lower,
            self.base_init_ang_vel_upper,
            (num_resets,),
        )
        reset_projected_gravity = transform_by_quat(
            self.global_gravity_dir.expand(num_resets, -1),
            inv_quat(reset_base_quat),
        )
        reset_qpos = torch.concatenate(
            (
                reset_base_pos,
                reset_base_quat,
                self.init_dof_pos.expand(num_resets, -1),
            ),
            dim=-1,
        )
        return (
            reset_qpos,
            reset_base_pos,
            reset_base_quat,
            reset_base_lin_vel,
            reset_base_ang_vel,
            reset_projected_gravity,
            terrain_height,
            terrain_centers,
            terrain_tile_index,
        )

    def _resample_commands(self, envs_idx):
        """为全部环境采样指令，仅将选中行写回；None 表示全部写回。"""
        commands = sample_uniform(*self.commands_limit, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)
        return

    # ============ 控制与状态更新：控制发生在 scene.step 前，状态更新发生在其后 ============
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

    def _update_velocity_estimator(self):
        """更新 actor 使用的前向速度；仿真真值只留给 reward、critic 与诊断。"""
        self.wheel_forward_vel.copy_(
            wheel_forward_velocity(
                self.wheel_vel,
                self.wheel_radius,
                self.wheel_velocity_sign,
            )
        )

        imu_data = self.imu.read()
        self.imu_lin_acc.copy_(imu_data.lin_acc)
        self.imu_ang_vel.copy_(imu_data.ang_vel)
        self.forward_kinematic_acc.copy_(
            gravity_compensated_forward_acceleration(
                self.imu_lin_acc,
                self.projected_gravity,
                self.gravity_magnitude,
            )
        )
        self.estimated_base_lin_vel.copy_(
            complementary_forward_velocity_update(
                self.estimated_base_lin_vel,
                self.forward_kinematic_acc,
                self.wheel_forward_vel,
                dt=self.dt,
                wheel_correction_time_constant_s=self.wheel_correction_time_constant_s,
                max_abs_velocity=self.max_estimated_velocity,
            )
        )

    def _update_wheel_contact(self):
        """更新轮子/基座接触、存活门控和落地事件。"""
        all_contact_forces = self.robot.get_links_net_contact_force()
        contact_forces = all_contact_forces[:, self.wheel_links_idx, :]
        self.wheel_contact.copy_((torch.linalg.vector_norm(contact_forces, dim=-1) > self.wheel_contact_force_threshold).to(dtype=gs.tc_float))
        base_contact_force = torch.linalg.vector_norm(all_contact_forces[:, self.base_link_idx, :], dim=-1)
        self.base_contact.copy_((base_contact_force > self.base_contact_force_threshold).to(dtype=gs.tc_float))
        both_wheels_contact = torch.all(self.wheel_contact > 0.5, dim=1)
        self.alive_gate.copy_((both_wheels_contact & (self.base_contact < 0.5)).to(dtype=gs.tc_float))

        # 双轮由“未同时接触”切换为“同时接触”视为落地。首个仿真步只用于
        # 初始化接触状态，避免机器人 reset 后本来就在地面却被误判为落地。
        landed = both_wheels_contact & ~self.previous_both_wheels_contact & (self.base_contact < 0.5) & (self.episode_length_buf > 1)
        self.landing_event.copy_(landed.to(dtype=gs.tc_float))
        if self.landing_penalty_enabled:
            self.landing_penalty_steps_left.sub_(1).clamp_min_(0)
            self.landing_penalty_steps_left.copy_(
                torch.where(
                    landed,
                    torch.full_like(self.landing_penalty_steps_left, self.landing_penalty_steps),
                    self.landing_penalty_steps_left,
                )
            )
            landing_gate = (self.landing_penalty_steps_left > 0) & (self.base_contact < 0.5)
            self.landing_penalty_gate.copy_(landing_gate.to(dtype=gs.tc_float))
        else:
            self.landing_penalty_steps_left.zero_()
            self.landing_penalty_gate.zero_()
        self.previous_both_wheels_contact.copy_(both_wheels_contact)

    def _update_tracking_gate(self):
        """根据高度目标和机身倾角更新平滑 AND 门控。"""
        height_error = torch.abs(self.base_height - self.commands[:, 2])
        self.height_gate = smooth_gate(
            height_error,
            self.tracking_gate_cfg["height_full_error"],
            self.tracking_gate_cfg["height_zero_error"],
        )

        # projected_gravity 的水平分量模长等于 sin(tilt)，可连续表示 roll/pitch 合成倾角。
        sin_tilt = torch.linalg.vector_norm(self.projected_gravity[:, :2], dim=1).clamp(0.0, 1.0)
        tilt = torch.asin(sin_tilt)
        self.attitude_gate = smooth_gate(
            tilt,
            math.radians(self.tracking_gate_cfg["attitude_full_angle_deg"]),
            math.radians(self.tracking_gate_cfg["attitude_zero_angle_deg"]),
        )

        self.tracking_gate_raw = self.height_gate * self.attitude_gate
        gate_floor = self.tracking_gate_cfg["floor"]
        self.tracking_gate = gate_floor + (1.0 - gate_floor) * self.tracking_gate_raw

    def _update_observations(self):
        """按固定顺序组装 actor 观测及包含仿真真值的 critic 观测。"""
        # 字典插入顺序就是策略输入的拼接顺序；调整顺序或维度后旧模型将不再兼容。
        self.obs_components = {
            "estimated_base_lin_vel": self.estimated_base_lin_vel.unsqueeze(-1) * self.obs_scales["lin_vel"],  # 1
            "imu_ang_vel": self.imu_ang_vel * self.obs_scales["ang_vel"],  # 3
            "imu_lin_acc": self.imu_lin_acc * self.imu_acc_scale,  # 3，IMU 比力（含重力）
            "projected_gravity": self.projected_gravity,  # 3
            **self._get_command_observation_components(),
            "joint_pos_offset": (self.joint_pos - self.default_joint_pos) * self.obs_scales["joint_pos"],
            "joint_vel": self.joint_vel * self.obs_scales["joint_vel"],
            "wheel_vel": self.wheel_vel * self.obs_scales["wheel_vel"],
            "leg_length": self.leg_length * self.obs_scales["leg_length"],  # 2
            "leg_angle": self.leg_angle * self.obs_scales["leg_angle"],  # 2
            "actions": self.actions,
            **self._get_task_observation_components(),
        }
        self.obs_buf = torch.concatenate(tuple(self.obs_components.values()), dim=-1)
        self.critic_obs_components = {
            # critic 继承 actor 的完整观测，其中已经包含三轴 imu_lin_acc；
            # 下方只追加训练时可用、部署时不可直接获得的仿真特权信息。
            **self.obs_components,
            "privileged_base_lin_vel": self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
            "privileged_base_ang_vel": self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            # 仿真接触真值只提供给 critic，actor/实机接口不依赖它。
            "privileged_wheel_contact": self.wheel_contact,  # 2: left, right
            "privileged_base_contact": self.base_contact.unsqueeze(-1),  # 1
            **self._get_landing_privileged_observation_components(),
            **self._get_task_privileged_observation_components(),
        }
        self.critic_obs_buf = torch.concatenate(tuple(self.critic_obs_components.values()), dim=-1)
        return

    # ============ 课程配置应用：更新运行参数，不执行物理步进或 reset ============
    def _build_commands_scale(self):
        """构造 locomotion 的 [vx, wz, base_height] 命令缩放。"""
        return torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["base_height"]],
            dtype=gs.tc_float,
            device=self.device,
        )

    def _build_command_limits(self):
        """构造 [vx, wz, base_height] 的采样上下限。"""
        return tuple(
            torch.tensor(items, dtype=gs.tc_float, device=gs.device)
            for items in zip(
                self.command_cfg["lin_vel_range"],
                self.command_cfg["ang_vel_range"],
                self.command_cfg["base_height_range"],
            )
        )

    def _apply_command_ranges(self, values):
        """应用课程指令范围，并重建采样使用的上下限 Tensor。"""
        allowed = {"lin_vel_range", "ang_vel_range", "base_height_range"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported command range curriculum keys: {sorted(unknown)}")
        for name, limits in values.items():
            if len(limits) != 2 or limits[0] > limits[1]:
                raise ValueError(f"{name} must be [lower, upper], got {limits}")
            self.command_cfg[name] = list(limits)
        self.commands_limit = self._build_command_limits()

    def _get_reward_scale(self, name):
        return self.reward_scales[name]

    def _apply_reward_scales(self, values):
        """应用原始奖励权重；除死亡奖励外，运行时权重统一乘以 dt。"""
        for name, raw_scale in values.items():
            reward_function = getattr(self, "_reward_" + name)
            raw_scale = float(raw_scale)
            self.raw_reward_scales[name] = raw_scale
            self.reward_cfg["reward_scales"][name] = raw_scale
            self.reward_scales[name] = raw_scale if name == "death" else raw_scale * self.dt
            self.reward_functions[name] = reward_function
            if name not in self.episode_sums:
                self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    def _apply_tracking_gate(self, values):
        """应用速度跟踪门控参数。"""
        self.tracking_gate_cfg = merge_tracking_gate_config(self.tracking_gate_cfg, values)
        self.reward_cfg["tracking_gate"] = dict(self.tracking_gate_cfg)

    def _apply_termination_limits(self, values):
        """应用姿态/接触终止条件，并同步刷新由持续时间换算出的步数。"""
        angle_keys = {"termination_if_roll_greater_than", "termination_if_pitch_greater_than"}
        duration_keys = {"tilt_termination_duration_s", "base_contact_termination_duration_s"}
        allowed = angle_keys | duration_keys
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported termination curriculum keys: {sorted(unknown)}")

        normalized = {}
        for name in angle_keys.intersection(values):
            angle = float(values[name])
            if not math.isfinite(angle) or angle <= 0.0:
                raise ValueError(f"{name} must be a positive finite angle")
            normalized[name] = angle

        tilt_duration = self.tilt_termination_duration_s
        if "tilt_termination_duration_s" in values:
            tilt_duration = float(values["tilt_termination_duration_s"])
            if not math.isfinite(tilt_duration) or tilt_duration <= 0.0:
                raise ValueError("tilt_termination_duration_s must be positive and finite")
            normalized["tilt_termination_duration_s"] = tilt_duration

        base_contact_duration = self.base_contact_termination_duration_s
        if "base_contact_termination_duration_s" in values:
            raw_duration = values["base_contact_termination_duration_s"]
            base_contact_duration = None if raw_duration is None else float(raw_duration)
            if base_contact_duration is not None and (not math.isfinite(base_contact_duration) or base_contact_duration <= 0.0):
                raise ValueError("base_contact_termination_duration_s must be positive and finite, or None")
            normalized["base_contact_termination_duration_s"] = base_contact_duration

        self.env_cfg.update(normalized)
        self.tilt_termination_duration_s = tilt_duration
        self.base_contact_termination_duration_s = base_contact_duration
        self.tilt_termination_steps = max(1, math.ceil(tilt_duration / self.dt))
        self.base_contact_termination_steps = None if base_contact_duration is None else max(1, math.ceil(base_contact_duration / self.dt))

    def _apply_action_limits(self, values):
        """应用每一步都会从 env_cfg 读取的动作裁剪范围。"""
        allowed = {"clip_joint_action", "clip_wheel_action"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported action-limit curriculum keys: {sorted(unknown)}")
        self.env_cfg.update(values)

    def _apply_reset_ranges(self, values):
        """应用课程阶段的 base 初始位置、姿态和速度随机化范围。"""
        allowed = {
            "base_init_pos_range",
            "base_init_rpy_offset_range_deg",
            "base_init_lin_vel_range",
            "base_init_ang_vel_range",
        }
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported reset range curriculum keys: {sorted(unknown)}")

        tensor_attributes = {
            "base_init_pos_range": ("base_init_pos_lower", "base_init_pos_upper"),
            "base_init_rpy_offset_range_deg": ("base_init_rpy_lower", "base_init_rpy_upper"),
            "base_init_lin_vel_range": ("base_init_lin_vel_lower", "base_init_lin_vel_upper"),
            "base_init_ang_vel_range": ("base_init_ang_vel_lower", "base_init_ang_vel_upper"),
        }
        for name, limits in values.items():
            lower, upper = as_range_tensors(limits, 3, name, device=self.device)
            lower_attr, upper_attr = tensor_attributes[name]
            setattr(self, lower_attr, lower)
            setattr(self, upper_attr, upper)
            # 同步运行时配置，便于日志和调试输出反映当前课程阶段。
            self.env_cfg[name] = [list(axis_limits) for axis_limits in limits]

    def _apply_terrain_curriculum(self, values):
        """更新允许出生的地形难度，不改变 actor/critic 接口。"""
        self.terrain.apply_curriculum(values)
        self.env_cfg["terrain"]["max_difficulty"] = self.terrain.max_difficulty

    # ============ 诊断与可视化：查询状态或调整相机，不推进训练时钟 ============
    def get_observation_components(self, env_idx=0):
        """返回指定环境中组成策略输入的、已经缩放并命名的观测分量。"""
        return {name: value[env_idx].detach() for name, value in self.obs_components.items()}

    def get_velocity_estimator_diagnostics(self, env_idx=0):
        """返回同一环境的真值、轮速估计和融合估计，便于量化部署接口误差。"""
        return {
            "source": "imu_wheel_estimator",
            "true_forward_velocity": self.base_lin_vel[env_idx, 0].detach(),
            "estimated_forward_velocity": self.estimated_base_lin_vel[env_idx].detach(),
            "wheel_forward_velocity": self.wheel_forward_vel[env_idx].detach(),
            "imu_specific_force": self.imu_lin_acc[env_idx].detach(),
            "gravity_compensated_forward_acceleration": self.forward_kinematic_acc[env_idx].detach(),
        }

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

    # ============ 子任务扩展点：由 jump 等子类覆盖；调用位置由 step / reset 固定 ============
    def _initialize_task_buffers(self):
        """任务扩展点：在首次 reset 前分配额外 buffer。"""

    def _should_advance_training_clock(self):
        """任务可在非 PPO 控制段暂停课程与 global_step。"""
        return True

    def _apply_task_control(self):
        """任务扩展点：在基础关节/轮控制之后、物理步进之前施加额外控制。"""

    def _update_task_state(self):
        """任务扩展点：物理状态与接触更新后、奖励计算前更新任务状态。"""

    def _task_termination(self):
        """任务扩展点：额外终止条件 """
        return torch.zeros_like(self.episode_length_buf, dtype=torch.bool)

    def _reset_task_buffers(self, env_idx):
        """任务扩展点：与 locomotion buffer 同步重置任务状态。"""

    def _accumulate_task_metrics(self):
        """任务扩展点：累计额外诊断量。"""

    def _landing_penalty_window_enabled(self):
        """子任务可关闭 locomotion 的落地后惩罚窗口。"""
        return True

    def _get_command_observation_components(self):
        """返回基础命令观测；子任务可保留该前缀并追加独立命令。"""
        return {"commands": self.commands * self.commands_scale}

    def _get_landing_privileged_observation_components(self):
        return {
            "privileged_landing_phase": (self.landing_penalty_steps_left.unsqueeze(-1).to(dtype=gs.tc_float) / self.landing_penalty_steps),
        }

    def _get_task_observation_components(self):
        """任务扩展点：追加 actor 可部署观测，默认不改变 locomotion 契约。"""
        return {}

    def _get_task_privileged_observation_components(self):
        """任务扩展点：追加仅供 critic 使用的仿真真值。"""
        return {}
