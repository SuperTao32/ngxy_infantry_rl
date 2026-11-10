"""条件跳跃的阶段、目标、奖励和课程；公共观测与动作由根配置统一。"""

import math
from typing import Mapping

from .height_reference import resolve_height_reference, resolve_retract_duration

PHASE_NAMES = ("takeoff", "flight", "landing")
MODE_NAMES = ("JUMP_20", "JUMP_40")
MODE_HEIGHTS = (0.20, 0.40)
REWARD_TERRAIN_NAMES = ("without_platform", "with_platform")


def get_cfgs():
    # 阶段时长与输入窗口独立配置；回合最长持续到两者都结束。
    durations = {"takeoff": 0.20, "flight": 0.40, "landing": 0.50}
    duration = sum(durations.values())
    course = _get_curriculum_cfg()
    modes = _get_jump_modes_cfg()
    modes["mode_probabilities"] = list(course["stages"][0]["targets"]["terrain"]["mode_probabilities"])
    env = dict(
        episode_length_s=duration,
        resampling_time_s=duration,
        jump_phase_durations_s=durations,
        jump_input_duration_s=[0.60, 0.80],  # 按 JUMP_20 / JUMP_40 配置 one-hot 和相位输入时长，单位 s。
        jump_modes=modes,
        # 两种模式共用：takeoff 内伸腿，随后按独立的 retract_duration_s 收腿并保持最低。
        jump_height_reference=dict(min_distance_m=0.22, max_distance_m=0.40, retract_duration_s=0.10),
        jump_approach=get_approach_cfg(),
        takeoff_min_airborne_time_s=0.06,
        takeoff_min_wheel_clearance_m=0.01,  # 双轮均须高于各自下方地面，避免卸载时接触力过小被误判腾空。
        **_get_termination_cfg(),
    )
    commands = dict(
        lin_vel_range=[1.0, 3.0],
        ang_vel_range=[0.0, 0.0],
        base_height_range=[0.22, 0.22],
    )
    validate_configs(env, course)
    return dict(env_cfg=env, reward_cfg=_get_reward_cfg(), command_cfg=commands, curriculum_cfg=course)


def get_approach_cfg():
    """空旷跑道加速，连续满足速度与站稳条件后才触发 jump。"""
    return {
        "spawn_x_m": -6.0,
        "spawn_y_m": -4.0,
        "speed_tolerance_m_s": 0.15,
        "lateral_speed_tolerance_m_s": 0.15,
        "vertical_speed_tolerance_m_s": 0.20,
        "height_tolerance_m": 0.04,
        "max_tilt_deg": 10.0,
        "max_yaw_deg": 5.0,
        "stable_time_s": 0.15,
        "timeout_s": 5.0,
    }


def _get_termination_cfg():
    """终止条件唯一调参入口；回合超时取阶段总时长和输入时长的较大值。"""
    return {
        # jump 任务失败阈值：任意一拍机身倾角超过此值即锁存无效。
        "jump_max_tilt_deg": 10.0,
        "jump_termination": {
            # 过倾、撞台阶立面、错误落台中的任一项，立即提前终止。
            "on_invalid": True,
        },
        # 过倾会先通过 on_invalid 立即终止。
        "termination_if_roll_greater_than": 10.0,
        "termination_if_pitch_greater_than": 10.0,
        "tilt_termination_duration_s": 1.0,
        "base_contact_termination_duration_s": 0.2,
    }


def _get_jump_modes_cfg():
    """JUMP_20 / JUMP_40 的起跳距离、场地与成功条件。"""
    # 输出中的逐模式列表统一按 MODE_NAMES 排序：20 cm、40 cm。
    return {
        "mode_names": list(MODE_NAMES),
        "step_heights_m": list(MODE_HEIGHTS),  # 模式最终高度；实际台阶由高度课程选择。
        "platform_height_levels_m": [[0.10, 0.15, 0.20], [0.25, 0.30, 0.35, 0.40]],
        "platform_enabled": [True, True],  # 课程可移走台阶，任务编码与跳高目标不变。
        "platform_probabilities": [0.5, 0.5],  # 各高度模式采样台阶的概率；其余回合采样平地。
        "assignment": "random",  # cyclic 用于逐模式验证。
        # base 原点相对起跳面的峰值目标，与 base 到轮底的距离参考独立。
        "clearance_targets_m": [0.45, 0.63],
        "min_forward_speeds_m_s": [0.8, 1.5],  # 仅台阶助跑使用，平地不设最低速度。
        # 台阶跳每个模式的 [前向速度 m/s, 起跳距离 m]，节点间线性插值；平地不查表。
        # 这些是待训练/标定的初值，不是已验证的最优起跳位置。
        "distance_tables": [
            [[0.0, 0.35], [0.8, 0.35], [1.0, 0.40], [2.0, 0.60], [3.0, 0.75]],
            [[0.0, 0.65], [1.5, 0.65], [2.0, 0.8], [3.0, 0.95]],
        ],
        "distance_jitter_m": 0.01,
        # 场地几何
        "platform_length_m": 2.0,
        "platform_width_m": 1.0,
        "lane_spacing_m": 4.0,
        # 平地回合的成功高度，按 JUMP_20 / JUMP_40 排序，单位 m。
        # 轮底取双轮较低侧的回合峰值；base_link 取首次腾空峰值，均须严格大于阈值。
        # 平地额外成功门槛；两类地形还须达到 clearance_targets_m，并满足落稳条件。
        "ground_success_wheel_clearance_m": [0.22, 0.42],
        "ground_success_base_height_m": [0.43, 0.63],
        # 平地和台阶共用 base 峰值上限：成功高度加缓冲，按新增超高量惩罚，不改变成功条件。
        "ground_base_height_excess_buffer_m": 0.03,
        # 落台成功判定
        "landing_margin_m": 0.02,
        "landing_height_tolerance_m": 0.025,
        "landing_stable_time_s": 0.16,
        "landing_max_vertical_speed_m_s": 0.35,
        "landing_max_tilt_deg": 10.0,
    }


def _get_reward_cfg():
    """奖励形状参数、共享奖励及各阶段权重。"""
    return {
        # 落地 base_link 高度与 base 到轮底距离跟踪：容差单位 m，指数分母 sigma 单位 m²。
        "height_reference_tolerance_m": 0.02,
        "height_reference_sigma": 0.0025,
        "landing_base_height_tracking_delay_s": 0.20,  # 从首次触地起留出缓冲时间，再恢复高度追踪奖励。
        "short_leg_length_target": 0.15,
        "leg_length_sigma": 0.01,
        "flight_height_sigma": 0.02,
        "flight_balance_sigma": 0.04,
        "flight_leg_angle_limit_deg": 3.0,  # 相对机身的虚拟腿摆角，±10° 内不惩罚。
        "landing_velocity_sigma": 0.05,
        "landing_tilt_sigma": 0.04,
        "landing_ang_vel_sigma": 0.02,
        "reference_forward_velocity_sigma": 0.2,
        "zero_yaw_rate_sigma": 0.02,
        "takeoff_wheel_push_force_reference_n": 400.0,  # 两轮向上法向支撑力之和的归一化参考，单位 N。
        "reward_scales": {
            # 每项均支持标量、[JUMP_20, JUMP_40]，或 {without_platform: ..., with_platform: ...}。
            # 助跑不结算奖励；共享奖励在 jump 信号更新后的起跳、腾空和落地生效。
            "base_balance": -5.0,
            "leg_symmetry": -5.0,
            "leg_symmetry_bonus": 2.0,
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 5.0,
            "joint_vel": -0.005,
            "airborne_wheel_vel": -0.01,  # 离地轮实际角速度平方和，(rad/s)^2；按 dt 累计。
            "wheel_action_rate": -1.0,
            "leg_action_rate": -0.1,
            # 实测腿长达到 upper_links_min_angle 对应限位后，惩罚继续伸腿的目标角差平方。
            "leg_extension_at_limit": -5.0,
            "base_contact": -10.0,  # 碰撞期间逐步惩罚，仍乘 dt，不直接判失败。
            "death": -100.0,  # 与 locomotion 一致，不乘 dt。
            "jump_invalid": -10.0,
            # 一次性奖励仍乘 dt：默认 0.01 s 下合格起跳 +1；高度进度按新增峰值累计。
            "takeoff_event": 100.0,
            "flight_height_shortfall": -300.0,
            "flight_height_excess": -50000.0,  # 平地和台阶新增超高 / 目标高度；超过缓冲后抵消峰值增益。
            # 持续项按每秒计，避免起跳速度和超高收益压过落稳成功。
            "takeoff_upward_velocity": 150.0,
            "takeoff_wheel_push": 100.0,  # 起跳接触期按 Fz / 参考力线性给分
            "takeoff_vertical_velocity": 2000.0,
            # 腾空
            "flight_airtime": 1.0,
            "flight_balance": 2.0,
            "flight_height_progress": 1200.0,
            "flight_height_tracking": 100.0,
            # 按本回合是否实际有台阶选权重；各组也可填 [JUMP_20, JUMP_40]。
            "flight_wheel_clearance": 100.0,
            "flight_leg_angle": -50.0,
            # 落地
            "landing_vertical_velocity": -20.0,
            "landing_base_acceleration": -0.01,  # 三轴加速度平方和；100 m/s² 持续一拍扣 1。
            "soft_landing": 10.0,
            "landing_stability": 50.0,
            "landing_base_height_tracking": 10.0,  # 首次触地后等待配置的缓冲时长，再逐拍跟踪高度参考。
            "landing_airborne": -10.0,  # 首次触地后，任一轮仍未着地即逐步惩罚。
            "base_to_wheel_bottom_distance_tracking": 0.0,
            # 任务结果
            "task_success": 5000.0,
            "task_failure": -500.0,
        },
    }


def _get_curriculum_cfg():
    """阶段控制台阶开关、采样和奖励；台阶高度按两个模式各自的成功率升降级。"""
    return {
        "enabled": True,
        # 仅统计开启台阶后的真实台阶回合；两个模式分别从最低高度开始。
        "height_curriculum": {
            "enabled": True,
            "window_episodes": 100,
            "promote_threshold": 0.80,
            "demote_threshold": 0.40,
        },
        "stages": [
            {
                "name": "without_platform",
                "start_iteration": 0,
                "targets": {
                    "terrain": {
                        "mode_probabilities": [1 / 3, 2 / 3],
                        "platform_enabled": [False, False],
                    },
                    "command_ranges": {
                        "lin_vel_range": [0.0, 2.0],
                    },
                },
            },
            {
                "name": "with_platform",
                "start_iteration": 300,
                "targets": {
                    "terrain": {
                        "mode_probabilities": [1 / 3, 2 / 3],
                        "platform_enabled": [True, True],
                    },
                    "command_ranges": {
                        "lin_vel_range": [0.0, 3.0],
                    },
                    "reward_scales": {
                        "base_balance": -10.0,
                        "airborne_wheel_vel": -0.03,
                        "leg_extension_at_limit": -10.0,
                        "base_contact": -20.0,
                        "flight_height_excess": -100000.0,
                        "takeoff_upward_velocity": 0.0,
                        "takeoff_wheel_push": 0.0,
                        "takeoff_vertical_velocity": 0.0,
                        # 腾空
                        "flight_airtime": {"with_platform": 0.0, "without_platform": 1.0},
                        "flight_balance": 5.0,
                        "flight_height_progress": {"with_platform": 100.0, "without_platform": 500.0},
                        "flight_height_tracking": {"with_platform": 0.0, "without_platform": 50.0},
                        "flight_wheel_clearance": {"without_platform": 0.0, "with_platform": 100.0},
                        # 落地
                        "landing_vertical_velocity": -30.0,
                        "landing_base_acceleration": -0.03,  # 三轴加速度平方和；100 m/s² 持续一拍扣 1。
                        "soft_landing": 50.0,
                        "landing_stability": 50.0,
                        # 任务结果
                        "task_success": {"with_platform": 3000.0, "without_platform": 2000.0},
                        "task_failure": {"with_platform": -100.0, "without_platform": -50.0},
                    },
                },
            },
        ],
    }


def validate_platform_enabled(values):
    if len(values) != len(MODE_NAMES) or any(type(v) is not bool for v in values):
        raise ValueError(f"platform_enabled must contain {len(MODE_NAMES)} booleans")


def validate_platform_probabilities(values):
    if len(values) != len(MODE_NAMES) or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"platform_probabilities must contain {len(MODE_NAMES)} probabilities in [0, 1]")


def validate_mode_probabilities(values):
    """每种模式一个有限非负权重，允许为零，但总权重大于零。"""
    if len(values) != len(MODE_NAMES) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"mode_probabilities must contain {len(MODE_NAMES)} finite non-negative values")
    if not math.isfinite(sum(values)) or sum(values) <= 0:
        raise ValueError("mode_probabilities must have a finite positive total weight")


def validate_configs(env_cfg, curriculum_cfg=None):
    from .phase import resolve_input_durations, validate_phase_durations

    validate_phase_durations(env_cfg["jump_phase_durations_s"])
    resolve_input_durations(env_cfg)
    resolve_height_reference(env_cfg)
    resolve_retract_duration(env_cfg)
    clearance = env_cfg.get("takeoff_min_wheel_clearance_m", 0.01)
    if isinstance(clearance, bool) or not isinstance(clearance, (int, float)) or not math.isfinite(clearance) or clearance < 0:
        raise ValueError("takeoff_min_wheel_clearance_m must be finite and non-negative")
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
        raise ValueError("jump_modes mode order must be JUMP_20, JUMP_40 with heights 0.20, 0.40 m")
    from .curriculum import JumpHeightCurriculum

    JumpHeightCurriculum(cfg, (curriculum_cfg or {}).get("height_curriculum"))
    validate_mode_probabilities(cfg["mode_probabilities"])
    validate_platform_enabled(cfg.get("platform_enabled", [True] * len(MODE_NAMES)))
    validate_platform_probabilities(cfg["platform_probabilities"])
    for stage in (curriculum_cfg or {}).get("stages", []):
        terrain = stage.get("targets", {}).get("terrain", {})
        if "platform_enabled" in terrain:
            validate_platform_enabled(terrain["platform_enabled"])
        if "mode_probabilities" in terrain:
            validate_mode_probabilities(terrain["mode_probabilities"])
        if "platform_probabilities" in terrain:
            validate_platform_probabilities(terrain["platform_probabilities"])
    for key in (
        "clearance_targets_m",
        "min_forward_speeds_m_s",
        "ground_success_wheel_clearance_m",
        "ground_success_base_height_m",
    ):
        values = cfg[key]
        if len(values) != len(MODE_NAMES) or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"jump_modes.{key} must contain {len(MODE_NAMES)} finite non-negative values")
    if any(c <= h for c, h in zip(cfg["clearance_targets_m"], MODE_HEIGHTS)):
        raise ValueError("clearance targets must exceed the landing surface height")
    minimum, _ = resolve_height_reference(env_cfg)
    if any(height <= minimum for height in cfg["ground_success_base_height_m"]):
        raise ValueError("ground_success_base_height_m must exceed jump_height_reference.min_distance_m")
    buffer = cfg.get("ground_base_height_excess_buffer_m", 0.03)
    if isinstance(buffer, bool) or not isinstance(buffer, (int, float)) or not math.isfinite(buffer) or buffer < 0:
        raise ValueError("jump_modes.ground_base_height_excess_buffer_m must be finite and non-negative")
    if any(v <= 0 for v in cfg["min_forward_speeds_m_s"]):
        raise ValueError("jump modes require a positive approach speed")
    if cfg["assignment"] not in ("random", "cyclic"):
        raise ValueError("jump_modes.assignment must be random or cyclic")
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
    if len(cfg["distance_tables"]) != len(MODE_NAMES):
        raise ValueError("provide one distance table per mode")
    for mode, table in enumerate(cfg["distance_tables"]):
        if len(table) < 2 or any(len(point) != 2 for point in table):
            raise ValueError("distance tables need at least two [speed, distance] points")
        speeds, distances = zip(*table)
        if any(not math.isfinite(v) or v < 0 for v in (*speeds, *distances)):
            raise ValueError("distance table values must be finite and non-negative")
        if any(b <= a for a, b in zip(speeds, speeds[1:])):
            raise ValueError("distance table speeds must strictly increase")
        if min(distances) <= jitter:
            raise ValueError("step trigger distances must remain positive after jitter")
