"""步兵轮腿车 locomotion 的训练配置。

配置按物理、观测、奖励、命令和课程拆开。locomotion 训练先学习原地站立，
再逐步扩大速度、转向、高度命令和 reset 难度。
"""

import math
from copy import deepcopy

from ...core.terrain import default_terrain_cfg
from ...core.randomization import default_randomization_cfg


def get_cfgs():
    """返回 locomotion 环境需要的五组配置。"""
    env_cfg = _env_cfg()
    obs_cfg = _obs_cfg()
    reward_cfg = _reward_cfg()
    command_cfg = _command_cfg()
    curriculum_cfg = _curriculum_cfg()
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg


def _env_cfg() -> dict:
    return {
        "num_actions": 6,
        "num_joints": 4,
        "num_wheels": 2,
        "robot_mjcf": "assets/robot/wheelbipeV14_2/mjcf/wheelbipeV14_2.xml",
        # 地形由所选配置决定；多地形训练可在此使用 "mixed"。
        "terrain": default_terrain_cfg("plane"),
        "randomization": default_randomization_cfg(dynamics_enabled=True, sensors_enabled=True),
        "default_joint_pos": {
            "left_front1_joint": 0.0,
            "left_rear1_joint": 0.0,
            "right_front1_joint": 0.0,
            "right_rear1_joint": 0.0,
        },
        # 动作顺序也是 actor 输出顺序，修改后旧 checkpoint 不再兼容。
        "joint_names": [
            "left_front1_joint",
            "right_front1_joint",
            "left_rear1_joint",
            "right_rear1_joint",
        ],
        "leg_front_joint_names": ["left_front1_joint", "right_front1_joint"],
        "leg_rear_joint_names": ["left_rear1_joint", "right_rear1_joint"],
        "wheel_names": ["left_wheel_joint", "right_wheel_joint"],
        "wheel_link_names": ["left_wheel_link", "right_wheel_link"],
        "spring_names": ["left_spring2_joint", "right_spring2_joint"],
        "base_link_name": "base_link",
        # 虚拟腿仅作为观测和诊断；站立目标使用平地上的实际 base z。
        "leg_upper_link_length": 0.21,
        "leg_lower_link_length": 0.25,
        # 同时用于目标摆角限幅与实际摆角越界惩罚，单位 rad。
        "leg_angle_limit_range": [-0.25 * math.pi, 0.25 * math.pi],
        "min_upper_link_angle": 0.5 * math.pi,
        # 气弹簧：F = F0 + k * compression - c * velocity。
        "gas_spring_preload_force": 420.0,
        "gas_spring_stiffness": 1400.0,
        "gas_spring_damping": 50.0,
        "gas_spring_max_compression": 0.06,
        "wheel_contact_force_threshold": 1.0,
        "base_contact_force_threshold": 5.0,
        "landing_penalty_duration_s": 0.30,
        "joint_kp": 60.0,
        "joint_kd": 3.0,
        "wheel_kd": 0.25,
        "joint_force_limit": 40.0,
        "wheel_force_limit": 5.0,
        "joint_pos_scale": 2.0,
        "wheel_vel_scale": 70.0,
        "clip_joint_action": 1.0,
        "clip_wheel_action": 1.0,
        # 固定一拍动作延迟；不属于 domain_rand，本轮仍保持原有动作路径。
        "simulate_action_latency": True,
        # 只有连续超限一段时间才终止，给策略留下可学习的恢复窗口。
        "termination_if_roll_greater_than": 20.0,
        "termination_if_pitch_greater_than": 20.0,
        "tilt_termination_duration_s": 0.30,
        "base_contact_termination_duration_s": 0.25,
        # 从 0.22 m 起步只保留很小的落地行程，避免策略还没输出就先自由落体。
        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "episode_length_s": 100.0,
        "resampling_time_s": 10.0,
    }


def _obs_cfg() -> dict:
    return {
        # actor 使用 IMU、轮速等真机观测；融合速度估计仅用于诊断，不占用输入维度。
        "imu": {
            "link_name": "base_link",
            "pos_offset": [0.0, 0.0, 0.0],
        },
        "velocity_estimator": {
            "wheel_radius": 0.06,
            "wheel_velocity_sign": 1.0,
            "gravity_magnitude": 9.81,
            "wheel_correction_time_constant_s": 0.5,
            "max_abs_velocity": 4.2,
        },
        "obs_scales": {
            "lin_vel": 1.0 / 4.2,
            "lin_acc": 1.0 / 9.81,
            "ang_vel": 0.5,
            "joint_pos": 1.0,
            "joint_vel": 0.1,
            "wheel_vel": 1.0 / 70.0,
            "base_height": 1.0 / 0.35,
            "leg_length": 1.0 / 0.35,
            "leg_angle": 1.0,
        },
    }


def _reward_cfg() -> dict:
    return {
        "tracking_sigma": 0.25,
        # 初学站立保留运动跟踪容差；课程通过 standing_reward 逐步收紧。
        "standing_tracking_sigma": 0.25,
        "standing_lin_vel_threshold": 0.02,  # m/s，命令判定阈值
        "standing_ang_vel_threshold": 0.02,  # rad/s，命令判定阈值
        "standing_drift_deadband": 0.02,  # m/s，每个水平轴允许的平衡微动
        "landing_base_ang_vel_weight": 0.25,
        "tracking_gate": {
            "height_full_error": 0.03,
            "height_zero_error": 0.10,
            "attitude_full_angle_deg": 5.0,
            "attitude_zero_angle_deg": 15.0,
            "floor": 0.30,
        },
        "reward_scales": {
            "tracking_lin_vel": -0.5,
            "tracking_ang_vel": -0.5,
            "standing_drift": 0.0,  # 先学平衡，后续课程再逐步启用漂移惩罚。
            "gated_tracking_lin_vel": 5.0,
            "gated_tracking_ang_vel": 5.0,
            "base_balance": -5.0,
            "leg_symmetry": -5.0,
            "leg_angle_limits": -5.0,
            "leg_symmetry_bonus": 2.0,
            "base_height": -10.0,
            "height_gate": 5.0,
            "joint_vel": -0.005,
            "landing_base_oscillation": 0.0,
            "landing_joint_vel": 0.0,
            # 每个离地轮子持续扣分；接触由 wheel_contact_force_threshold 判定。
            "wheel_airborne": -5.0,
            "base_contact": -10.0,
            "alive": 5.0,
            "death": -100.0,
        },
    }


def _command_cfg() -> dict:
    return {
        "num_commands": 3,
        "lin_vel_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        "base_height_range": [0.22, 0.22],
        # |vx| <= 1 m/s 使用课程角速度范围；更快时收紧到 ±1 rad/s。
        # 设为 None 可恢复线速度与角速度独立采样。
        "high_speed_ang_vel": {"lin_vel_threshold": 1.0, "max_abs_ang_vel": 1.0},
        # 默认 25% 概率令 vx=wz=0；高度仍按当前课程采样，概率可由课程覆盖。
        "standing_probability": 0.25,
    }


def _curriculum_cfg() -> dict:
    # 阶段采用累计覆盖：这里只写相对上一阶段发生变化的字段。
    return {
        "enabled": True,
        "stages": [
            {
                "name": "low",
                "start_iteration": 0,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.04},
                    "domain_rand": {"strength": 0.6},
                    "sensor_noise": {"strength": 0.6},
                    "terrain": {"max_difficulty": 1},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.24]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-2.5, 2.5],
                        "ang_vel_range": [-4.0, 4.0],
                        "base_height_range": [0.20, 0.34],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.015,
                        "height_zero_error": 0.05,
                        "attitude_full_angle_deg": 2.0,
                        "attitude_zero_angle_deg": 8.0,
                        "floor": 0.05,
                    },
                    "termination_limits": {
                        "termination_if_roll_greater_than": 25.0,
                        "termination_if_pitch_greater_than": 25.0,
                        "tilt_termination_duration_s": 0.50,
                        "base_contact_termination_duration_s": 0.5,
                    },
                    "reward_scales": {
                        "standing_drift": -1.0,
                        "gated_tracking_lin_vel": 20.0,
                        "gated_tracking_ang_vel": 20.0,
                        "base_balance": -20.0,
                        "base_contact": -50.0,
                        "base_height": -15.0,
                        "height_gate": 15.0,
                        "landing_base_oscillation": -0.3,
                        "landing_joint_vel": -0.01,
                        "joint_vel": -0.01,
                    },
                },
            },
            {
                "name": "middle",
                "start_iteration": 4000,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.02},
                    "domain_rand": {"strength": 0.8},
                    "sensor_noise": {"strength": 0.8},
                    "terrain": {"max_difficulty": 1},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.24]],
                        "base_init_rpy_offset_range_deg": [[-3.0, 3.0], [-3.0, 3.0], [-5.0, 5.0]],
                        "base_init_lin_vel_range": [[-1.5, 1.5], [-0.05, 0.05], [-0.1, 0.1]],
                        "base_init_ang_vel_range": [[-0.1, 0.1], [-0.1, 0.1], [-3.0, 3.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-3.2, 3.2],
                        "ang_vel_range": [-4.0, 4.0],
                        "base_height_range": [0.20, 0.38],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.015,
                        "height_zero_error": 0.025,
                        "attitude_full_angle_deg": 1.5,
                        "attitude_zero_angle_deg": 5.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -2.0,
                        "gated_tracking_lin_vel": 30.0,
                        "gated_tracking_ang_vel": 30.0,
                        "base_contact": -40.0,
                        "leg_symmetry": -15.0,
                        "base_height": -15.0,
                        "height_gate": 15.0,
                        "landing_base_oscillation": -0.5,
                        "landing_joint_vel": -0.03,
                        "joint_vel": -0.02,
                    },
                },
            },
            {
                "name": "full_range",
                "start_iteration": 8000,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.01},
                    "domain_rand": {"strength": 1.0},
                    "sensor_noise": {"strength": 1.0},
                    "terrain": {"max_difficulty": 1},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.24]],
                        "base_init_rpy_offset_range_deg": [[-3.0, 3.0], [-3.0, 3.0], [-5.0, 5.0]],
                        "base_init_lin_vel_range": [[-2.5, 2.5], [-0.05, 0.05], [-0.1, 0.1]],
                        "base_init_ang_vel_range": [[-0.1, 0.1], [-0.1, 0.1], [-5.1, 5.1]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-3.8, 3.8],
                        "ang_vel_range": [-4.0, 4.0],
                        "base_height_range": [0.20, 0.38],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.01,
                        "height_zero_error": 0.025,
                        "attitude_full_angle_deg": 1.5,
                        "attitude_zero_angle_deg": 3.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -3.0,
                        "gated_tracking_lin_vel": 30.0,
                        "gated_tracking_ang_vel": 30.0,
                        "base_contact": -40.0,
                        "leg_symmetry": -15.0,
                        "base_height": -15.0,
                        "height_gate": 15.0,
                        "landing_base_oscillation": -0.7,
                        "landing_joint_vel": -0.05,
                        "joint_vel": -0.03,
                    },
                },
            },
        ],
    }


def get_final_command_cfg(command_cfg: dict, curriculum_cfg: dict) -> dict:
    """返回 eval 使用的命令配置（含静止采样率），累计应用到课程最终阶段。"""
    resolved = deepcopy(command_cfg)
    if not curriculum_cfg.get("enabled", False):
        return resolved

    for stage in curriculum_cfg.get("stages", []):
        command_ranges = stage.get("targets", {}).get("command_ranges", {})
        for name, value in command_ranges.items():
            resolved[name] = deepcopy(value)
    return resolved
