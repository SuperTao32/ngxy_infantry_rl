"""恢复任务独立配置；连续站稳后结束回合。"""

from pathlib import Path

from .reset_pose import SIDE_LYING_ANGLE_DEG, get_reset_pose_cfg, load_reset_pose_pool, resolve_reset_selection

RESET_POSE = Path(__file__).with_name("reset_pos.json")

# 配置中列出的 pose 文件位于此目录；CLI --recovery-poses 可显式指定其他来源。
RESET_POSE_DIR = Path(__file__).with_name("reset_pos")
DYNAMIC_RESET_PROBABILITY = 0.2

# 默认训练池的完整清单：文件名 -> 难度，非负整数，越大越难。
# reset_pos.json 使用 RESET_POSE，其余文件相对 RESET_POSE_DIR；不扫描目录。
# 以下仅为初始分级，可根据各 pose 的实际起身成功率调整；不修改快照 JSON。
RESET_POSE_DIFFICULTIES = {
    "reset_pos.json": 0,
    "pose_leg_bb.json": 0,
    "pose_leg_bf.json": 0,
    "pose_leg_fb.json": 0,
    "pose_leg_ff.json": 0,
    "pose_pitch90.json": 1,
    "pose_pitch-90.json": 1,
}

# 交互 eval 的独立清单：文件名 -> 难度；路径规则与训练清单一致。
# 可以改为 RESET_POSE_DIFFICULTIES.copy()，复用训练清单中的文件和难度。
# 也支持 "default"、绝对路径，或旧式路径 tuple/list（不按难度筛选）。
RECOVERY_RESET_POSES = {
    "pose_leg_bb.json": 0,
    "pose_leg_bf.json": 0,
    "pose_leg_fb.json": 0,
    "pose_leg_ff.json": 0,
}
RECOVERY_RESET_DIFFICULTY_RANGE = None  # None 全选；[0, 0] 只选 0 级，含上下界。


def get_recovery_training_reset_cfg(sources=None):
    """只加载配置清单或显式来源；续训不再读取原始快照文件。"""
    if sources is None:
        sources = [RESET_POSE if name == "reset_pos.json" else RESET_POSE_DIR / name for name in RESET_POSE_DIFFICULTIES]
    return {
        "pose_pool": load_reset_pose_pool(sources, difficulties=RESET_POSE_DIFFICULTIES),
        "difficulty_range": None,
        "side_lying_angle_deg": SIDE_LYING_ANGLE_DEG,
        "dynamic_probability": DYNAMIC_RESET_PROBABILITY,
        "dynamic": {
            "lift_height_range": [0.05, 0.15],
            "pitch_offset_range_deg": [-10.0, 10.0],
            # 基座线速度与角速度均为世界系；关节初始速度仍为零。
            "base_lin_vel_range": [[-0.15, 0.15], [-0.15, 0.15], [-0.2, 0.0]],
            "base_ang_vel_range": [[-0.1, 0.1], [-0.5, 0.5], [-0.1, 0.1]],
        },
    }


def get_recovery_reset_cfgs():
    """按 eval 清单与难度生成 R 键候选，不扫描目录或修改训练姿态池。"""
    if not RECOVERY_RESET_POSES:
        raise ValueError("RECOVERY_RESET_POSES must contain at least one pose")
    if not isinstance(RECOVERY_RESET_POSES, dict):
        # 兼容已有显式路径 tuple/list；沿用原路径解析行为。
        return [get_reset_pose_cfg(source) for source in RECOVERY_RESET_POSES]
    if any(not isinstance(name, str) or not name.strip() for name in RECOVERY_RESET_POSES):
        raise ValueError("RECOVERY_RESET_POSES keys must be non-empty filenames, paths, or 'default'")
    entries = [dict(source=name, difficulty=difficulty) for name, difficulty in RECOVERY_RESET_POSES.items()]
    try:
        indices, _ = resolve_reset_selection(
            dict(
                pose_pool=entries,
                difficulty_range=RECOVERY_RESET_DIFFICULTY_RANGE,
                dynamic_probability=0.0,
            )
        )
    except ValueError as exc:
        raise ValueError(f"Invalid eval RECOVERY_RESET_POSES / RECOVERY_RESET_DIFFICULTY_RANGE: {exc}") from exc
    configs = []
    for index in indices:
        name = entries[index]["source"]
        if name == "reset_pos.json":
            source = RESET_POSE
        elif name == "default":
            source = "default"
        else:
            source = Path(name).expanduser()
            if not source.is_absolute() and source.parent == Path("."):
                source = RESET_POSE_DIR / source
        configs.append(get_reset_pose_cfg(source))
    return configs


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
            "max_leg_angle_deg": 10.0,
            "max_lin_vel": 0.2,
            "max_ang_vel": 0.2,
            "hold_time_s": 0.01,
        },
    }
    env_cfg.update(get_reset_pose_cfg(RESET_POSE))
    env_cfg["recovery_reset"] = get_recovery_training_reset_cfg()
    reward_cfg = {
        "stand_up_height_reference_m": 0.22,  # 倾倒时高度惩罚的 H_REF。
        "stand_up_leg_angle_tolerance_deg": 10.0,  # 双腿角度均在 ±10° 内时满奖励。
        "stand_up_leg_angle_sigma": 0.5,
        "reward_scales": {
            "stand_up_success": 50.0,
            "stand_up_speed": 50.0,
            "stand_up_leg_angle": 1.0,
            "tracking_lin_vel": -2.0,
            "tracking_ang_vel": -3.0,
            "leg_symmetry": -5.0,
            "wheel_airborne": -5.0,
            "base_contact": -10.0,
            "base_balance": -50.0,
            "base_height": -10.0,
            "joint_vel": -0.01,
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
    """阶段参数累计覆盖；recovery_reset 可筛选难度及调整动态比例。"""
    return {
        "enabled": True,
        "stages": [
            {
                "name": "difficulty_0",
                "start_iteration": 0,
                "targets": {
                    # None 使用全部难度；[0, 1] 选 0/1 级，[2, 2] 只选 2 级。
                    "recovery_reset": {"difficulty_range": [0, 0]},
                    "command_ranges": {
                        "standing_probability": 1.0,
                        "lin_vel_range": [0.0, 0.0],
                        "ang_vel_range": [0.0, 0.0],
                        "base_height_range": [0.22, 0.22],
                    },
                },
            },
            {
                "name": "difficulty_1",
                "start_iteration": 500,
                "targets": {
                    # None 使用全部难度；[0, 1] 选 0/1 级，[2, 2] 只选 2 级。
                    "recovery_reset": {"difficulty_range": [0, 1]},
                    "command_ranges": {
                        "standing_probability": 1.0,
                        "lin_vel_range": [0.0, 0.0],
                        "ang_vel_range": [0.0, 0.0],
                        "base_height_range": [0.22, 0.22],
                    },
                },
            },
            {
                "name": "difficulty_2",
                "start_iteration": 1000,
                "targets": {
                    # None 使用全部难度；[0, 1] 选 0/1 级，[2, 2] 只选 2 级。
                    "recovery_reset": {"difficulty_range": [0, 2]},
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
