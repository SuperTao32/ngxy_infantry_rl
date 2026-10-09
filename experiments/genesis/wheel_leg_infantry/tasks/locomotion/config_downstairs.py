"""接收混合地形权重的 downstairs 训练配置。

共享机器人和观测定义，独立维护奖励、命令与课程。
课程依次训练梯形波、楼梯和不同等级的平台凸台。
"""

from .config_common import get_env_cfg, get_obs_cfg, get_final_command_cfg


def get_cfgs():
    """返回 locomotion 环境需要的五组配置。"""
    env_cfg = _env_cfg()
    obs_cfg = _obs_cfg()
    reward_cfg = _reward_cfg()
    command_cfg = _command_cfg()
    curriculum_cfg = _curriculum_cfg()
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg


def _env_cfg() -> dict:
    env = get_env_cfg()
    env.update(
        base_contact_force_threshold=20.0,
        termination_if_roll_greater_than=90.0,
        termination_if_pitch_greater_than=90.0,
        tilt_termination_duration_s=20.0,
        base_contact_termination_duration_s=20.0,
    )
    return env


def _obs_cfg() -> dict:
    obs = get_obs_cfg()
    # 仅用于诊断，不占用策略输入维度。
    obs["velocity_estimator"] = {
        "wheel_radius": 0.06,
        "wheel_velocity_sign": 1.0,
        "gravity_magnitude": 9.81,
        "wheel_correction_time_constant_s": 0.5,
        "max_abs_velocity": 4.2,
    }
    return obs


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
            "height_full_error": 0.05,
            "height_zero_error": 0.10,
            "attitude_full_angle_deg": 0.6,
            "attitude_zero_angle_deg": 2.0,
            "floor": 0.2,
        },
        "reward_scales": {
            "tracking_lin_vel": -100.0,
            "tracking_ang_vel": -100.0,
            "standing_drift": -10.0,
            "gated_tracking_lin_vel": 5.0,
            "gated_tracking_ang_vel": 5.0,
            "base_balance": -30.0,
            "leg_symmetry": -10.0,
            "leg_angle_limits": -5.0,
            "leg_symmetry_bonus": 2.0,
            "base_height": -10.0,
            "height_gate": 2.0,
            "joint_vel": -0.01,
            "wheel_action_rate": -1.5,
            "leg_action_rate": -0.15,
            "landing_base_oscillation": -0.5,
            "landing_joint_vel": -0.02,
            # 每个离地轮子持续扣分；接触由 wheel_contact_force_threshold 判定。
            "wheel_airborne": -50.0,
            "base_contact": -50.0,
            "alive": 5.0,
            "death": -10.0,
        },
    }


def _command_cfg() -> dict:
    # 基础命令字段直接放在顶层；只有课程 targets 中使用 command_ranges。
    return {
        "num_commands": 3,
        "lin_vel_range": [0.5, 2.0],
        "ang_vel_range": [-0.5, 0.5],
        "base_height_range": [0.25, 0.36],
        "high_speed_ang_vel": {
            "lin_vel_threshold": 1.5,
            "max_abs_ang_vel": 2.0,
        },
        "standing_probability": 0.0,
    }


def _curriculum_cfg() -> dict:
    # 几何参数集中在 terrains/；课程只选择类型和等级。
    return {
        "enabled": True,
        "stages": [
            {
                "name": "trapezoidal_wave",
                "start_iteration": 0,
                "targets": {
                    "terrain": {"preset": "trapezoidal_wave", "difficulty": 4},
                    "command_ranges": {
                        "standing_probability": 0.1,
                        "lin_vel_range": [-3.0, 3.0],
                        "ang_vel_range": [-4.0, 4.0],
                        "base_height_range": [0.22, 0.36],
                    },
                },
            },
            {
                "name": "stairs",
                "start_iteration": 2000,
                "targets": {
                    "terrain": {"preset": "stairs", "difficulty": 4},
                    "command_ranges": {
                        "standing_probability": 0.0,
                        "lin_vel_range": [1.0, 3.0],
                        "ang_vel_range": [-0.2, 0.2],
                        "base_height_range": [0.25, 0.36],
                    },
                },
            },
            {
                "name": "low_platform_ridge",
                "start_iteration": 3000,
                "targets": {
                    "terrain": {"preset": "platform_ridge", "difficulty": 0},
                    "command_ranges": {"lin_vel_range": [0.5, 2.0], "ang_vel_range": [-0.1, 0.1], "base_height_range": [0.25, 0.36]},
                },
            },
            {"name": "low_middle_platform_ridge", "start_iteration": 5000, "targets": {"terrain": {"preset": "platform_ridge", "difficulty": 1}}},
            {"name": "middle_platform_ridge", "start_iteration": 6000, "targets": {"terrain": {"preset": "platform_ridge", "difficulty": 2}}},
            {"name": "middle_high_platform_ridge", "start_iteration": 7000, "targets": {"terrain": {"preset": "platform_ridge", "difficulty": 3}}},
            {"name": "high_platform_ridge", "start_iteration": 8000, "targets": {"terrain": {"preset": "platform_ridge", "difficulty": 4}}},
        ],
    }
