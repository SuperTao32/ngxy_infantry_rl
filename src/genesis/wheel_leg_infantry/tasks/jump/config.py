"""条件跳跃的阶段、目标、奖励和课程；公共观测与动作由根配置统一。"""

import math
from typing import Mapping

from .height_reference import validate_height_reference

PHASE_NAMES = ("takeoff", "flight", "landing")
MODE_NAMES = ("flat", "step_20cm", "step_40cm")
MODE_HEIGHTS = (0.0, 0.20, 0.40)


def get_cfgs():
    durations = {"takeoff": 0.20, "flight": 0.40, "landing": 0.50}
    duration = sum(durations.values())
    course = _get_curriculum_cfg()
    modes = _get_jump_modes_cfg(duration)
    modes["mode_probabilities"] = list(course["stages"][0]["targets"]["terrain"]["mode_probabilities"])
    env = dict(
        episode_length_s=duration, resampling_time_s=duration,
        jump_phase_durations_s=durations, jump_modes=modes,
        takeoff_min_airborne_time_s=0.06,
        **_get_termination_cfg(),
    )
    commands = dict(lin_vel_range=[1.5, 2.0], ang_vel_range=[0.0, 0.0],
                    base_height_range=[0.22, 0.22], standing_probability=0.0,
                    high_speed_ang_vel=dict(lin_vel_threshold=1.5, max_abs_ang_vel=2.0),
                    high_base_height_velocity_limits=dict(base_height_threshold=0.32, max_abs_lin_vel=2.0, max_abs_ang_vel=3.0))
    validate_configs(env, course)
    return dict(env_cfg=env, reward_cfg=_get_reward_cfg(), command_cfg=commands, curriculum_cfg=course)


def _get_termination_cfg():
    """终止条件唯一调参入口；二次腾空和机身触地只惩罚，回合超时由阶段总时长决定。"""
    return {
        # jump 任务失败阈值：任意一拍机身倾角超过此值即锁存无效。
        "jump_max_tilt_deg": 10.0,
        "jump_termination": {
            # 过倾、撞台阶立面、错误落台中的任一项，立即提前终止。
            "on_invalid": True,
        },
        # 默认 jump 的 10° 过倾会先通过 on_invalid 立即终止。
        "termination_if_roll_greater_than": 90.0, 
        "termination_if_pitch_greater_than": 90.0,  
        "tilt_termination_duration_s": 3.0, 
        "base_contact_termination_duration_s": 5.0,  
    }


def _get_jump_modes_cfg(episode_length_s: float):
    """三种模式的 base 到轮底距离轨迹、起跳距离、场地与成功条件。"""
    # [触发 jump 后的时间(s), 目标 base 到两侧轮底的平均竖直距离(m)]。
    # 实测距离 = base_z - mean(wheel_center_z - wheel_radius)，沿世界 z 方向测量。
    # 节点间五次平滑插值；描述伸腿/收腿姿态，与地形高度及整体腾空高度无关。
    base_to_wheel_bottom_trajectories = {
        "flat": [
            [0.00, 0.22],
            [0.20, 0.60],
            [0.30, 0.20],
            [episode_length_s, 0.22],
        ],
        "step_20cm": [
            [0.00, 0.22],
            [0.20, 0.40],
            [0.30, 0.20],
            [episode_length_s, 0.22],
        ],
        "step_40cm": [
            [0.00, 0.22],
            [0.20, 0.70],
            [0.30, 0.20],
            [episode_length_s, 0.22],
        ],
    }
    # 输出中的逐模式列表统一按 MODE_NAMES 排序：平地、20 cm、40 cm。
    return {
        "mode_names": list(MODE_NAMES),
        "step_heights_m": list(MODE_HEIGHTS),
        "platform_enabled": [True, True, True],  # 课程可移走台阶，任务编码与跳高目标不变。
        "assignment": "random",  # cyclic 用于逐模式验证。
        # base 原点相对起跳面的峰值目标，与 base 到轮底的距离参考独立。
        "clearance_targets_m": [0.55, 0.45, 0.70],
        "min_forward_speeds_m_s": [0.0, 0.8, 1.5],
        "flat_stationary_probability": 0.1,
        # 每个模式的 [前向速度 m/s, 起跳距离 m]，节点间线性插值。
        # 这些是待训练/标定的初值，不是已验证的最优起跳位置。
        "distance_tables": [
            [[0.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.35], [0.8, 0.35], [1.0, 0.40], [2.0, 0.60], [3.0, 0.75]],
            [[0.0, 0.65], [1.5, 0.65], [2.0, 0.8], [3.0, 0.95]],
        ],
        "distance_jitter_m": 0.01,
        "height_references": [base_to_wheel_bottom_trajectories[name] for name in MODE_NAMES],
        # 场地几何
        "platform_length_m": 2.0,
        "platform_width_m": 1.0,
        "lane_spacing_m": 4.0,
        # 无台阶时的成功高度，按 flat / step_20cm / step_40cm 排序，单位 m。
        # 轮底取双轮较低侧的回合峰值；base_link 取首次腾空峰值，均须严格大于阈值。
        # 独立于 clearance_targets_m（奖励目标）；有台阶时仍使用落台成功条件。
        "ground_success_wheel_clearance_m": [0.32, 0.25, 0.40],
        "ground_success_base_height_m": [0.55, 0.45, 0.68],
        # 落台成功判定
        "landing_margin_m": 0.02,
        "landing_height_tolerance_m": 0.025,
        "landing_stable_time_s": 0.16,
        "landing_max_vertical_speed_m_s": 0.35,
        "landing_max_tilt_deg": 10.0,
    }


def _get_reward_cfg():
    """奖励形状参数、各阶段权重及禁用的 locomotion 奖励。"""
    return {
        # 落地 base 到轮底距离跟踪：容差单位 m，指数分母 sigma 单位 m²。
        "height_reference_tolerance_m": 0.02,
        "height_reference_sigma": 0.0025,
        "short_leg_length_target": 0.15,
        "leg_length_sigma": 0.01,
        "flight_height_sigma": 0.04,
        "flight_balance_sigma": 0.04,
        "landing_velocity_sigma": 0.05,
        "landing_tilt_sigma": 0.04,
        "landing_ang_vel_sigma": 0.02,
        "reference_forward_velocity_sigma": 0.25,
        "zero_yaw_rate_sigma": 0.02,
        "reward_scales": {
            # 复用 locomotion 的无门控奖励，在起跳、腾空和落地全程生效。
            "base_balance": -5.0,
            "leg_symmetry": -50.0,
            "leg_symmetry_bonus": 10.0,
            # 全程追踪切换时的前向速度，同时保持 yaw 角速度为零。
            "tracking_lin_vel": 5.0,
            "tracking_ang_vel": 10.0,
            "action_rate": -0.001,
            # 实测腿长达到 upper_links_min_angle 对应限位后，惩罚继续伸腿的目标角差平方。
            "leg_extension_at_limit": -100.0,
            "base_contact": -1.0,  # 碰撞期间逐步惩罚，仍乘 dt，不直接判失败。
            "death": -100.0,
            "jump_invalid": -200.0,
            # 起跳与峰值高度
            "flight_leg_vertical": 5.0,
            "takeoff_event": 200.0,
            "flight_height_shortfall": -2000.0,
            "flight_peak_height": 5000.0,
            "takeoff_upward_velocity": 2000.0,
            "takeoff_vertical_velocity": 3000.0,
            # 腾空
            "flight_airtime": 10.0,
            "flight_balance": 5.0,
            "flight_height_progress": 80.0,
            "flight_height_tracking": 0.0,
            "flight_wheel_clearance": 1500.0,
            # 落地
            "landing_vertical_velocity": -1.0,
            "soft_landing": 1.0,
            "landing_stability": 1.0,
            "landing_airborne": -1.0,  # 首次触地后双轮离地期间逐步惩罚。
            "target_landing": 30.0,
            "base_to_wheel_bottom_distance_tracking": 20.0,
            # 任务结果
            "task_success": 5000.0,  # 事件奖励沿用框架的 dt 缩放。
            "task_failure": -500.0,
        },
    }


def _get_curriculum_cfg():
    """按阶段配置任务采样比例、速度范围与奖励；比例顺序为平地、20 cm、40 cm。"""
    return {
        "enabled": True,
        "stages": [
            {
                "name": "flat_jump_high",
                "start_iteration": 0,
                "targets": {
                    "terrain": {
                        "mode_probabilities": [0.0, 0, 1.0],
                        "platform_enabled": [True, True, False],
                    },
                    "command_ranges": {
                        "lin_vel_range": [1.5, 2.0],
                    },
                },
            },
            {
                "name": "step_40cm",
                "start_iteration": 300,
                "targets": {
                    "terrain": {"platform_enabled": [True, True, True]},
                    "reward_scales": {
                        "base_contact": -200.0, 
                        "landing_airborne": -200.0,
                        "base_balance": -30.0,
                        "leg_symmetry": -20.0,
                        "leg_symmetry_bonus": 10.0,
                        "tracking_lin_vel": 10.0,
                        "tracking_ang_vel": 30.0,
                        # 落地
                        "soft_landing": 1000.0,
                        "landing_stability": 1000.0,
                        "leg_extension_at_limit": -200.0,
                        "flight_wheel_clearance": 500.0,
                        "landing_vertical_velocity": -100.0,
                    },
                },
            },
            {
                "name": "mixed_jump",
                "start_iteration": 600,
                "targets": {
                    "terrain": {
                        "mode_probabilities": [1 / 5, 1 / 5, 3 / 5],
                        "platform_enabled": [True, True, True],
                    },
                    "command_ranges": {"lin_vel_range": [1.0, 3.0]},
                    "reward_scales": {
                        "base_balance": -30.0,
                        "leg_symmetry": -20.0,
                        "leg_symmetry_bonus": 10.0,
                        "tracking_lin_vel": 10.0,
                        "tracking_ang_vel": 20.0,
                        # 落地
                        "soft_landing": 2000.0,
                        "landing_stability": 2000.0,
                        "action_rate": -0.07,
                        # 按 [平地, 20 cm, 40 cm] 指定权重，仅提高 40 cm 任务。
                        "flight_airtime": [20.0, 5.0, 20.0],
                        "flight_peak_height": [100.0, 0.0, 2000.0],
                        "takeoff_vertical_velocity": [200.0, 200.0, 2000.0],
                        "takeoff_upward_velocity": [0.0, 0.0, 1000.0],
                        "flight_height_progress": [0.0, 0.0, 100.0],
                        "flight_height_tracking": [1000.0, 500.0, 0.0],
                        "flight_height_shortfall": -100.0,
                        "leg_extension_at_limit": -300.0,
                        "flight_wheel_clearance": 100.0,
                        "flight_balance": 100.0,
                        "task_failure": -5000.0,
                    },
                },
            },
        ],
    }


def validate_platform_enabled(values):
    if len(values) != len(MODE_NAMES) or any(type(v) is not bool for v in values):
        raise ValueError("platform_enabled must contain three booleans")


def validate_mode_probabilities(values):
    """模式权重允许为零，但必须是三个有限非负数，且总权重大于零。"""
    if len(values) != len(MODE_NAMES) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("mode_probabilities must contain three finite non-negative values")
    if not math.isfinite(sum(values)) or sum(values) <= 0:
        raise ValueError("mode_probabilities must have a finite positive total weight")


def validate_configs(env_cfg, curriculum_cfg=None):
    termination = env_cfg.get("jump_termination", {})
    if not isinstance(termination, Mapping):
        raise ValueError("jump_termination must be a mapping")
    # 兼容旧存档的 on_rebound 字段：仍校验类型，但不再用于终止。
    if set(termination) - {"on_rebound", "on_invalid"}:
        raise ValueError("jump_termination only supports on_rebound and on_invalid")
    for key, value in termination.items():
        if type(value) is not bool:
            raise ValueError(f"jump_termination.{key} must be a boolean")
    cfg = env_cfg["jump_modes"]
    if tuple(cfg["mode_names"]) != MODE_NAMES or tuple(cfg["step_heights_m"]) != MODE_HEIGHTS:
        raise ValueError("jump_modes mode order must be flat, step_20cm, step_40cm")
    validate_mode_probabilities(cfg["mode_probabilities"])
    validate_platform_enabled(cfg.get("platform_enabled", [True, True, True]))
    for stage in (curriculum_cfg or {}).get("stages", []):
        terrain = stage.get("targets", {}).get("terrain", {})
        if "platform_enabled" in terrain:
            validate_platform_enabled(terrain["platform_enabled"])
        if "mode_probabilities" in terrain:
            validate_mode_probabilities(terrain["mode_probabilities"])
    for key in (
        "clearance_targets_m",
        "min_forward_speeds_m_s",
        "ground_success_wheel_clearance_m",
        "ground_success_base_height_m",
    ):
        values = cfg[key]
        if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"jump_modes.{key} must contain three finite non-negative values")
    if any(c <= h for c, h in zip(cfg["clearance_targets_m"], MODE_HEIGHTS)):
        raise ValueError("clearance targets must exceed the landing surface height, including flat")
    if any(v <= 0 for v in cfg["min_forward_speeds_m_s"][1:]):
        raise ValueError("step modes require a positive approach speed")
    if cfg["assignment"] not in ("random", "cyclic"):
        raise ValueError("jump_modes.assignment must be random or cyclic")
    if not 0 <= cfg["flat_stationary_probability"] <= 1:
        raise ValueError("flat_stationary_probability must be in [0, 1]")
    for key in (
        "platform_length_m",
        "platform_width_m",
        "lane_spacing_m",
        "landing_margin_m",
        "landing_height_tolerance_m",
        "landing_stable_time_s",
        "landing_max_vertical_speed_m_s",
        "landing_max_tilt_deg",
    ):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"jump_modes.{key} must be finite and positive")
    if cfg["lane_spacing_m"] <= cfg["platform_width_m"]:
        raise ValueError("platform lanes must not overlap")
    if 2 * cfg["landing_margin_m"] >= min(cfg["platform_width_m"], cfg["platform_length_m"]):
        raise ValueError("landing margin leaves no usable platform")
    jitter = cfg["distance_jitter_m"]
    if not math.isfinite(jitter) or jitter < 0:
        raise ValueError("distance_jitter_m must be finite and non-negative")
    if len(cfg["distance_tables"]) != 3 or len(cfg["height_references"]) != 3:
        raise ValueError("provide one distance table and height reference per mode")
    for mode, table in enumerate(cfg["distance_tables"]):
        if len(table) < 2 or any(len(point) != 2 for point in table):
            raise ValueError("distance tables need at least two [speed, distance] points")
        speeds, distances = zip(*table)
        if any(not math.isfinite(v) or v < 0 for v in (*speeds, *distances)):
            raise ValueError("distance table values must be finite and non-negative")
        if any(b <= a for a, b in zip(speeds, speeds[1:])):
            raise ValueError("distance table speeds must strictly increase")
        if mode and min(distances) <= jitter:
            raise ValueError("step trigger distances must remain positive after jitter")
        validate_height_reference(cfg["height_references"][mode], env_cfg["episode_length_s"])
