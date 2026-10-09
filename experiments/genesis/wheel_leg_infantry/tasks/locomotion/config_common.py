"""三个训练阶段共用的机器人、动作和观测配置，以及命令合并工具。

每次调用返回新的字典；奖励、命令、课程和任务覆盖项由各阶段独立维护。
"""

import math
from copy import deepcopy

from ...terrains import default_terrain_cfg
from ...core.randomization_config import default_randomization_cfg
from ...core.tof import default_tof_cfg


def get_env_cfg() -> dict:
    return {
        "num_actions": 6,
        "num_joints": 4,
        "num_wheels": 2,
        "robot_mjcf": "assets/robot/wheelbipeV14_2/mjcf/wheelbipeV14_2.xml",
        # 地形由所选配置决定，也可通过 --terrain 覆盖。
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


def get_obs_cfg() -> dict:
    return {
        # actor 使用 IMU、轮速等真机观测
        "imu": {
            "link_name": "base_link",
            "pos_offset": [0.0, 0.0, 0.0],
        },
        # True: (32 + 4 * history_frames)D actor；False: 32D actor，供 jump warmup。
        "tof": default_tof_cfg(
            include_in_observation=True,
            history_frames=3,
            update_hz=50.0,  # ToF 更新频率；历史仅在新测距帧到来时推进。
            max_range_m=1.2,  # 测距量程，单位 m；独立于观测缩放参考距离。
            # 观测 = clamp(距离 / 参考距离, 0, 1)，单位 m。
            forward_reference_distance_m=1.0,
            downward_reference_distance_m=0.25,
        ),
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
