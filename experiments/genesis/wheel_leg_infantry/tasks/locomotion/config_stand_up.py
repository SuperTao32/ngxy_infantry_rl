"""独立起身训练；起身成功后继续保持站立。"""

from pathlib import Path

from .config_locomotion import get_cfgs as get_locomotion_cfgs
from .reset_pose import apply_reset_pose

# 可改为 "default"（0.22 m 默认站姿）或其他 Viewer JSON 快照。
RESET_POSE = Path(__file__).with_name("reset_pos.json")


def get_cfgs():
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = get_locomotion_cfgs()
    env_cfg.update(
        {
            # 独立起身的动作范围，不随 locomotion 的默认值变化。
            "wheel_vel_scale": 35.0,
            "clip_joint_action": 2.0,
            "clip_wheel_action": 2.0,
            "episode_length_s": 5.0,
            "resampling_time_s": 5.0,
            "termination_if_roll_greater_than": 60.0,
            "termination_if_pitch_greater_than": 60.0,
            "tilt_termination_duration_s": 1.0,
            # 起身和站稳期间均容许机身触地，不因此终止。
            "base_contact_termination_duration_s": None,
            "stand_up": {
                "height": 0.22,
                "height_tolerance": 0.025,
                "max_tilt_deg": 5.0,
                "max_leg_angle_deg": 5.0,
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
            "tracking_gate": {
                "height_full_error": 0.02,
                "height_zero_error": 0.08,
                "attitude_full_angle_deg": 1.0,
                "attitude_zero_angle_deg": 10.0,
                "floor": 0.30,
            },
            "reward_scales": {
                "stand_up_success": 500.0,
                "stand_up_leg_length": -10.0,
                "stand_up_leg_vertical": -10.0,
                "wheel_airborne": -100.0,
                "base_contact": -1.0,
                "base_balance": -20.0,
                "gated_tracking_lin_vel": 1.0,
                "gated_tracking_ang_vel": 1.0,
                "leg_angle_limits": 0.0,
                "leg_symmetry": -5.0,
                "leg_symmetry_bonus": 1.0,
                "height_gate": 5.0,
                "joint_vel": -0.003,
                "alive": 5.0,
                "death": -100.0,
            },
        }
    )

    curriculum_cfg = _curriculum_cfg()
    apply_reset_pose(env_cfg, curriculum_cfg, RESET_POSE)
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg


def _curriculum_cfg() -> dict:
    """阶段参数累计覆盖；复位姿态和起身终止条件由 env_cfg 统一设置。"""
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
        ],
    }
