"""步兵轮腿车 locomotion 的训练配置。

配置按物理、观测、奖励、命令和课程拆开。locomotion 训练先学习原地站立，
再逐步扩大速度、转向、高度命令和 reset 难度。
"""

import math
from copy import deepcopy

from ...core.terrain import default_terrain_cfg
from ...core.domain_randomization import default_domain_rand_cfg


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
        "domain_rand": default_domain_rand_cfg(enabled=True),
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
        "leg_angle_limit_range": [-math.pi / 6 , math.pi / 6],
        "min_upper_link_angle": 0.5 * math.pi,
        # 气弹簧：F = F0 + k * compression - c * velocity。
        "gas_spring_preload_force": 420.0,
        "gas_spring_stiffness": 0.0,
        "gas_spring_damping": 0.0,
        "gas_spring_max_compression": 0.06,
        "wheel_contact_force_threshold": 1.0,
        "base_contact_force_threshold": 5.0,
        "landing_penalty_duration_s": 0.30,
        "joint_kp": 60.0,
        "joint_kd": 3.0,
        "wheel_kd": 0.3,
        "joint_force_limit": 40.0,
        "wheel_force_limit": 5.0,
        "joint_pos_scale": 1.2,
        "wheel_vel_scale": 70.0,
        "clip_joint_action": 1.0,
        "clip_wheel_action": 1.0,
        # 固定一拍动作延迟；不属于 domain_rand，本轮仍保持原有动作路径。
        "simulate_action_latency": True,
        # 只有连续超限一段时间才终止，给策略留下可学习的恢复窗口。
        "termination_if_roll_greater_than": 10.0,
        "termination_if_pitch_greater_than": 10.0,
        "tilt_termination_duration_s": 0.20,
        "base_contact_termination_duration_s": 0.1,
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
        # actor 使用 IMU、轮速等真机观测
        "imu": {
            "link_name": "base_link",
            "pos_offset": [0.0, 0.0, 0.0],
            "acc_noise": 0.0,
            "acc_bias": 0.0,
            "acc_random_walk": 0.0,
            "gyro_noise": 0.0,
            "gyro_bias": 0.0,
            "gyro_random_walk": 0.0,
            "delay": 0.0,
            "jitter": 0.0,
        },
        "obs_scales": {
            "lin_vel": 1.0 / 3.5,
            "lin_acc": 1.0 / 9.81,
            "ang_vel": 1.0 / 4.0,
            "joint_pos": 1.2,
            "joint_vel": 0.05,
            "wheel_vel": 1.0 / 50.0,
            "base_height": 1.0 / 0.35,
            "leg_length": 1.0 / 0.35,
            "leg_angle": 1.0,
        },
    }


def _reward_cfg() -> dict:
    return {
        "tracking_sigma": 0.25,
        "landing_base_ang_vel_weight": 0.25,
        "tracking_gate": {
            "height_full_error": 0.03,
            "height_zero_error": 0.05,
            "attitude_full_angle_deg": 5.0,
            "attitude_zero_angle_deg": 10.0,
            "floor": 0.30,
        },
        "reward_scales": {
            "tracking_lin_vel": -0.5,
            "tracking_ang_vel": -0.5,
            "gated_tracking_lin_vel": 2.0,
            "gated_tracking_ang_vel": 2.0,
            "base_balance": -5.0,
            "leg_symmetry": -5.0,
            "leg_angle_limits": -5.0,
            "leg_symmetry_bonus": 1.0,
            "base_height": -10.0,
            "height_gate": 5.0,
            # 已训练的同模型任务使用 5e-3 量级；原来的 1.0 让冲击速度
            # 惩罚淹没了全部站立收益。
            "joint_vel": -0.005,
            "wheel_action_rate": -1.0,
            "leg_action_rate": -0.1,
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
        "high_speed_ang_vel": {"lin_vel_threshold": 1.0, "max_abs_ang_vel": 1.0},
        # 静止站立采样
        "standing_probability": 0.10,
    }


def _curriculum_cfg() -> dict:
    # 阶段采用累计覆盖：这里只写相对上一阶段发生变化的字段。
    return {
        "enabled": True,
        "stages": [
            {
                "name": "stand",
                "start_iteration": 0,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "terrain": {"max_difficulty": 0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "lin_vel_range": [0.0, 0.0],
                        "ang_vel_range": [0.0, 0.0],
                        "base_height_range": [0.22, 0.22],
                    },
                    "termination_limits": {
                        "termination_if_roll_greater_than": 20.0,
                        "termination_if_pitch_greater_than": 20.0,
                        "tilt_termination_duration_s": 0.30,
                        "base_contact_termination_duration_s": 0.25,
                    },
                },
            },
            {
                "name": "locomotion",
                "start_iteration": 500,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "terrain": {"max_difficulty": 0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "lin_vel_range": [-1.0, 1.0],
                        "ang_vel_range": [-2.0, 2.0],
                        "base_height_range": [0.20, 0.30],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.03,
                        "height_zero_error": 0.09,
                        "attitude_full_angle_deg": 1.0,
                        "attitude_zero_angle_deg": 2.0,
                        "floor": 0.10,
                    },
                    "reward_scales": {
                        "tracking_lin_vel": -2.0,
                        "tracking_ang_vel": -3.0,
                        "leg_symmetry": -10.0,
                        "base_balance": -10.0,
                    },
                },
            },
            {
                "name": "locomotion2",
                "start_iteration": 2000,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "terrain": {"max_difficulty": 0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "lin_vel_range": [-2.0, 2.0],
                        "ang_vel_range": [-3.0, 3.0],
                        "base_height_range": [0.20, 0.34],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.02,
                        "height_zero_error": 0.05,
                        "attitude_full_angle_deg": 0.8,
                        "attitude_zero_angle_deg": 4.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "gated_tracking_lin_vel": 5.0,
                        "gated_tracking_ang_vel": 7.0,
                        "base_balance": -15.0,
                        "base_contact": -15.0,
                        "landing_base_oscillation": -0.3,
                        "landing_joint_vel": -0.01,
                    },
                },
            },
            {
                "name": "locomotion_rand",
                "start_iteration": 4000,
                "targets": {
                    "domain_rand": {"strength": 0.4},
                    "terrain": {"max_difficulty": 0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.24]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [-1.0, 1.0]],
                    },
                    "command_ranges": {
                        "lin_vel_range": [-3.0, 3.0],
                        "ang_vel_range": [-5.0, 5.0],
                        "base_height_range": [0.20, 0.36],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.01,
                        "height_zero_error": 0.025,
                        "attitude_full_angle_deg": 0.5,
                        "attitude_zero_angle_deg": 2.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "joint_vel": -0.01,
                        "wheel_action_rate": -1.5,
                        "leg_action_rate": -0.15,
                    },
                },
            },
            {
                "name": "full_range",
                "start_iteration": 7000,
                "targets": {
                    "domain_rand": {"strength": 0.6},
                    "terrain": {"max_difficulty": 0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.24]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [-1.0, 1.0]],
                    },
                    "command_ranges": {
                        "lin_vel_range": [-4.0, 4.0],
                        "ang_vel_range": [-6.0, 6.0],
                        "base_height_range": [0.20, 0.36],
                    },
                    "reward_scales": {
                    },
                },
            },
        ],
    }


def get_final_command_cfg(command_cfg: dict, curriculum_cfg: dict) -> dict:
    """返回 eval 使用的命令范围，并累计应用到课程最终阶段。"""
    resolved = deepcopy(command_cfg)
    if not curriculum_cfg.get("enabled", False):
        return resolved

    for stage in curriculum_cfg.get("stages", []):
        command_ranges = stage.get("targets", {}).get("command_ranges", {})
        for name, limits in command_ranges.items():
            resolved[name] = list(limits)
    return resolved
