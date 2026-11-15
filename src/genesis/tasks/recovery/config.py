"""reset 后从初始姿态恢复的配置；不分配独立训练环境。"""

from .reset_pose import load_reset_pose_pool

# 相对 tasks/recovery/reset_pos/ 的显式清单，可改成单个文件或绝对路径。
RESET_POSE_FILES = (
    "pose_leg_bb.json",
    "pose_leg_bf.json",
    "pose_leg_fb.json",
    "pose_leg_ff.json",
    # "pose_pitch90.json",
    # "pose_pitch-90.json",
)


def get_reset_pos_cfg(sources=None, *, probability=0.0):
    return {
        "pose_pool": load_reset_pose_pool(RESET_POSE_FILES if sources is None else sources),
        "probability": probability,  # 每个待 reset 环境独立决定；未命中时使用 env 默认复位姿态。
        # 将现有 reset_ranges 转成对快照位置的扰动，默认 z=0.22 对应零扰动。
        "reference_base_pos": [0.0, 0.0, 0.22],
        "clearance_m": 1e-4,  # 只向上修正穿地；None 关闭几何校正。
    }


def get_cfg():
    return {
        "reset_pos": get_reset_pos_cfg(),
        "max_duration_s": 5.0,
        "stand_up": {
            "height": 0.22,
            "height_tolerance": 0.025,
            "max_tilt_deg": 5.0,
            "max_leg_angle_deg": 10.0,
            "max_lin_vel": 0.2,
            "max_ang_vel": 0.2,
            "hold_time_s": 0.01,
        },
        "reward_scales": {
            "height": -10.0,
            "balance": -50.0,
            "leg_angle": 1.0,
            "speed": 50.0,
            # "stand_up_wheel_push": 0.0,
            # "wheel_ground_distance": 0.0,
            "tracking_lin_vel": -2.0,
            "tracking_ang_vel": -3.0,
            "leg_action_rate": -1.0,
            "wheel_action_rate": -10.0,
            "leg_symmetry": -5.0,
            "wheel_airborne": -5.0,
            "base_contact": -10.0,
            "joint_vel": -0.05,
            "success": 50.0,
            "failure": -50.0,
        },
        "stand_up_height_reference_m": 0.22,
        "stand_up_wheel_push_force_reference_n": 250.0,
        "leg_angle_tolerance_deg": 10.0,
        "leg_angle_sigma": 0.5,
    }
