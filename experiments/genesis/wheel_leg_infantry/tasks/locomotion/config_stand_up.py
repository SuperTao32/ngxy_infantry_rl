"""统一起身 → locomotion 课程；每回合均先站稳，再执行当前课程命令。"""

from pathlib import Path

from .config_locomotion import get_cfgs as get_locomotion_cfgs
from .config_locomotion import get_final_command_cfg
from .reset_pose import apply_reset_pose

# 可改为 "default"（0.22 m 默认站姿）或其他 Viewer JSON 快照。
RESET_POSE = Path(__file__).with_name("reset_pos.json")
LOCOMOTION_START_ITERATION = 2000


def get_cfgs():
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = get_locomotion_cfgs()
    env_cfg.update(
        {
            "clip_joint_action": 2.0,
            "episode_length_s": 20.0,
            "resampling_time_s": 5.0,
            "termination_if_roll_greater_than": 60.0,
            "termination_if_pitch_greater_than": 60.0,
            "tilt_termination_duration_s": 1.0,
            # 所有课程阶段都需要容许从触地姿态起身。
            "base_contact_termination_duration_s": None,
            "stand_up": {
                "height": 0.22,
                "height_tolerance": 0.025,
                "max_tilt_deg": 10.0,
                "max_leg_angle_deg": 10.0,
                "max_lin_vel": 1.0,
                "max_ang_vel": 1.0,
                "hold_time_s": 0.5,
            },
        }
    )
    command_cfg.update(
        {
            "lin_vel_range": [0.0, 0.0],
            "ang_vel_range": [0.0, 0.0],
            "base_height_range": [0.22, 0.22],
            "standing_probability": 1.0,
        }
    )
    reward_cfg.update(
        {
            "stand_up_height_sigma": 0.02,
            "standing_tracking_sigma": 0.10,
            # 缩到目标长度后不再惩罚，避免要求无限收腿；单位 m。
            "stand_up_leg_length_target": 0.14,
            "stand_up_leg_length_sigma": 0.10,
            "stand_up_leg_angle_sigma": 0.5,  # rad
        }
    )
    reward_cfg["reward_scales"].update(
        {
            "stand_up_posture": 0.0,
            "stand_up_success": 500.0,
            "stand_up_leg_length": -10.0,
            "stand_up_leg_vertical": -10.0,
            "wheel_airborne": -100.0,
            "base_contact": -1.0,
            "base_balance": -10.0,
            "gated_tracking_lin_vel": 1.0,
            "gated_tracking_ang_vel": 1.0,
            "leg_angle_limits": 0.0,
            "alive": 5.0,
            "death": -100.0,
            "leg_symmetry": -5.0,
            "leg_symmetry_bonus": 1.0,
            "height_gate": 5.0,
            "joint_vel": -0.002,
        }
    )

    curriculum_cfg = _curriculum_cfg()
    apply_reset_pose(env_cfg, curriculum_cfg, RESET_POSE)
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg


def _curriculum_cfg() -> dict:
    """阶段参数累计覆盖；复位姿态和起身终止条件由 env_cfg 统一设置。"""
    # 在配置顶部调整起身训练时长，后续阶段相对首次行走迭代数设置。
    locomotion_start_iteration = LOCOMOTION_START_ITERATION
    if not isinstance(locomotion_start_iteration, int) or isinstance(locomotion_start_iteration, bool) or locomotion_start_iteration <= 0:
        raise ValueError("LOCOMOTION_START_ITERATION must be a positive integer")
    return {
        "enabled": True,
        "stages": [
            {
                "name": "stand_up",
                "start_iteration": 0,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "command_ranges": {
                        "standing_probability": 1.0,
                        "lin_vel_range": [0.0, 0.0],
                        "ang_vel_range": [0.0, 0.0],
                        "base_height_range": [0.22, 0.22],
                    },
                },
            },
            {
                "name": "locomotion",
                "start_iteration": locomotion_start_iteration,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.1},
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "command_ranges": {
                        "standing_probability": 0.1,
                        "lin_vel_range": [-1.0, 1.0],
                        "ang_vel_range": [-2.0, 2.0],
                        "base_height_range": [0.2, 0.3],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.03,
                        "height_zero_error": 0.09,
                        "attitude_full_angle_deg": 1.0,
                        "attitude_zero_angle_deg": 2.0,
                        "floor": 0.1,
                    },
                    # 显式设置行走权重；起身专属奖励继续由基础配置提供。
                    "reward_scales": {
                        "tracking_lin_vel": -2.0,
                        "tracking_ang_vel": -3.0,
                        "standing_drift": -1.0,
                        "gated_tracking_lin_vel": 2.0,
                        "gated_tracking_ang_vel": 2.0,
                        "base_balance": -10.0,
                        "leg_symmetry": -10.0,
                        "leg_angle_limits": -5.0,
                        "leg_symmetry_bonus": 1.0,
                        "base_height": -10.0,
                        "height_gate": 5.0,
                        "joint_vel": -0.005,
                        "wheel_action_rate": -1.0,
                        "leg_action_rate": -0.1,
                        "landing_base_oscillation": 0.0,
                        "landing_joint_vel": 0.0,
                        "wheel_airborne": -5.0,
                        "base_contact": -10.0,
                        "alive": 5.0,
                        "death": -100.0,
                    },
                },
            },
            {
                "name": "locomotion2",
                "start_iteration": locomotion_start_iteration + 1500,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.05},
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "command_ranges": {
                        "standing_probability": 0.1,
                        "lin_vel_range": [-2.0, 2.0],
                        "ang_vel_range": [-3.0, 3.0],
                        "base_height_range": [0.2, 0.34],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.02,
                        "height_zero_error": 0.05,
                        "attitude_full_angle_deg": 0.8,
                        "attitude_zero_angle_deg": 4.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -2.0,
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
                "start_iteration": locomotion_start_iteration + 3500,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.02},
                    "domain_rand": {"strength": 0.4},
                    "sensor_noise": {"strength": 0.4},
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-3.0, 3.0],
                        "ang_vel_range": [-5.0, 5.0],
                        "base_height_range": [0.2, 0.36],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.02,
                        "height_zero_error": 0.04,
                        "attitude_full_angle_deg": 0.6,
                        "attitude_zero_angle_deg": 2.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -3.0,
                        "joint_vel": -0.01,
                        "wheel_action_rate": -1.5,
                        "leg_action_rate": -0.15,
                    },
                },
            },
            {
                "name": "full_range",
                "start_iteration": locomotion_start_iteration + 6500,
                "targets": {
                    "domain_rand": {"strength": 1.0},
                    "sensor_noise": {"strength": 1.0},
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-3.7, 3.7],
                        "ang_vel_range": [-6.0, 6.0],
                        "base_height_range": [0.2, 0.36],
                    },
                    "reward_scales": {},
                },
            },
        ],
    }
