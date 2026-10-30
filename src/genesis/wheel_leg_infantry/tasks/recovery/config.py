"""恢复任务独立配置；连续站稳后结束回合。"""

from pathlib import Path

from .reset_pose import SIDE_LYING_ANGLE_DEG, get_reset_pose_cfg, load_reset_pose_pool

# 可改为 "default"（0.22 m 默认站姿）或其他 Viewer JSON 快照。
RESET_POSE = Path(__file__).with_name("reset_pos.json")

# 交互评估的姿态来源；可追加其他 Viewer JSON 快照或 "default"。
# 配置层负责解析，仿真和按键层只接收数值复位配置。
RECOVERY_RESET_POSES = (RESET_POSE,)

# 新训练自动纳入采集脚本的输出；CLI --recovery-poses 可指定其他目录/文件。
RESET_POSE_DIR = Path(__file__).with_name("reset_pos")
DYNAMIC_RESET_PROBABILITY = 0.2


def get_recovery_training_reset_cfg(sources=None):
    """将训练快照解析为数值配置；续训不再读取原始采集文件。"""
    if sources is None:
        sources = (RESET_POSE, *sorted(RESET_POSE_DIR.rglob("pose_*.json")))
    return {
        "pose_pool": load_reset_pose_pool(sources),
        "side_lying_angle_deg": SIDE_LYING_ANGLE_DEG,
        "dynamic_probability": DYNAMIC_RESET_PROBABILITY,
        "dynamic": {
            "lift_height_range": [0.05, 0.15],
            "pitch_offset_range_deg": [-10.0, 10.0],
            # 基座线速度与角速度均为世界系；关节初始速度仍为零。
            "base_lin_vel_range": [[-0.15, 0.15], [-0.15, 0.15], [-0.2, 0.2]],
            "base_ang_vel_range": [[-0.1, 0.1], [-0.5, 0.5], [-0.1, 0.1]],
        },
    }


def get_recovery_reset_cfgs():
    """返回 R 键可随机采样的复位配置，不改变训练的默认复位姿态。"""
    if not RECOVERY_RESET_POSES:
        raise ValueError("RECOVERY_RESET_POSES must contain at least one pose")
    return [get_reset_pose_cfg(source) for source in RECOVERY_RESET_POSES]


def get_cfgs():
    env_cfg = {
        "episode_length_s": 5.0,
        "resampling_time_s": 5.0,
        "termination_if_roll_greater_than": 90.0,
        "termination_if_pitch_greater_than": 90.0,
        "tilt_termination_duration_s": 3.0,
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
    env_cfg.update(get_reset_pose_cfg(RESET_POSE))
    env_cfg["recovery_reset"] = get_recovery_training_reset_cfg()
    reward_cfg = {
        "standing_tracking_sigma": 0.1,
        "stand_up_leg_length_target": 0.14,
        "stand_up_leg_length_sigma": 0.1,
        "stand_up_leg_angle_sigma": 0.5,
        "tracking_gate": {
            "height_full_error": 0.02,
            "height_zero_error": 0.08,
            "attitude_full_angle_deg": 1.0,
            "attitude_zero_angle_deg": 10.0,
            "floor": 0.3,
        },
        "reward_scales": {
            "stand_up_success": 50.0,
            "wheel_airborne": -10.0,
            "base_contact": -10.0,
            "base_balance": -50.0,
            "gated_tracking_lin_vel": 1.0,
            "gated_tracking_ang_vel": 1.0,
            "base_height": -10.0,
            "height_gate": 5.0,
            "joint_vel": -0.005,
            "death": -50.0,
        },
    }
    command_cfg = {
        "lin_vel_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        "base_height_range": [0.22, 0.22],
        "high_speed_ang_vel": {
            "lin_vel_threshold": 1.5,
            "max_abs_ang_vel": 2.0,
        },
        "high_base_height_velocity_limits": {
            "base_height_threshold": 0.32,
            "max_abs_lin_vel": 2.0,
            "max_abs_ang_vel": 3.0,
        },
        "standing_probability": 1.0,
    }
    return dict(env_cfg=env_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, curriculum_cfg=_curriculum_cfg())


def _curriculum_cfg() -> dict:
    """阶段参数累计覆盖；复位姿态和起身终止条件由 env_cfg 统一设置。"""
    return {
        "enabled": True,
        "stages": [
            {
                "name": "stand_up",
                "start_iteration": 0,
                "targets": {
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
