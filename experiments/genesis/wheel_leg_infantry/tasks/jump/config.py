"""Jump 唯一配置入口：平地、20 cm、40 cm 三维 one-hot 条件跳跃。"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping, Sequence

from ...core.domain_randomization import default_domain_rand_cfg
from ...core.randomization import normalize_randomization_config
from ...core.tof import resolve_tof_cfg
from ..locomotion.config_locomotion import get_cfgs as get_locomotion_cfgs
from .height_reference import validate_height_reference
from .observation import JUMP_POLICY_LAYOUT, LOCOMOTION_POLICY_LAYOUT, layout_dim

PHASE_NAMES = ("takeoff", "flight", "landing")
MODE_NAMES = ("flat", "step_20cm", "step_40cm")
MODE_HEIGHTS = (0.0, 0.20, 0.40)


def get_cfgs(locomotion_cfgs: Sequence[Mapping] | None = None):
    """返回 env、obs、reward、command、curriculum 五组配置。

    调参入口：本函数配置阶段、动力学与终止条件；下方按功能配置预热、
    三种跳跃模式、观测、奖励和课程。输出键名保持与训练及存档兼容。
    """
    if locomotion_cfgs is None:
        locomotion_cfgs = get_locomotion_cfgs()
    if len(locomotion_cfgs) < 4:
        raise ValueError("locomotion_cfgs must contain env, obs, reward and command configs")

    # 阶段时长：所有模式共用一个 rollout 周期。
    phase_durations_s = {
        "takeoff": 0.20,
        "flight": 0.40,
        "landing": 0.50,
    }
    episode_length_s = sum(phase_durations_s.values())
    warmup_cfg = _get_warmup_cfg()

    env_cfg = deepcopy(dict(locomotion_cfgs[0]))
    env_cfg.pop("landing_penalty_duration_s", None)
    env_cfg.update(
        {
            # 周期与交接：完成落地阶段后再交回 locomotion。
            "episode_length_s": episode_length_s,
            "resampling_time_s": episode_length_s,
            "jump_phase_durations_s": phase_durations_s,
            "handoff_on_landing": False,
            # jump 独立选择随机化范围；teacher 预热与跳跃共用本轮随机参数。
            # 仅 jump 阶段覆盖；预热使用 locomotion 参数，下一轮自动恢复。
            # None 表示继承。标量或列表：腿按 joint_names、轮按 wheel_names 排序。
            "jump_motor_params": {
                "joint_kp": 100.0,
                "joint_kd": 2.0,
                "wheel_kd": 0.3,
                "joint_force_limit": None,  # N·m，覆盖值必须 > 0
                "wheel_force_limit": None,  # N·m，覆盖值必须 > 0
            },
            # 001 模式全程覆盖关节 kd，包括无台阶高跳和 40 cm 台阶；None 关闭覆盖。
            # 标量或按 joint_names 排序的列表；其他模式使用上面的 joint_kd。
            "jump_step_40cm_joint_kd": 0.1,
            # 离地确认：高台面可能提前触地，短暂离地仍需连续确认。
            "takeoff_min_airborne_time_s": 0.06,
            # 终止参数统一在下方 _get_termination_cfg 中调整。
            **_get_termination_cfg(episode_length_s),
            # 初始状态
            "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
            "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "locomotion_warmup": warmup_cfg,
            "terrain": {"preset": "plane"},
            "jump_modes": _get_jump_modes_cfg(episode_length_s),
        }
    )

    # 预热指令范围为唯一入口；jump 开始后，第三项改为各模式的 base 到轮底距离参考。
    command_cfg = deepcopy(dict(locomotion_cfgs[3]))
    command_cfg.update({"num_commands": 3, **deepcopy(warmup_cfg["command_ranges"])})
    obs_cfg = _get_obs_cfg(locomotion_cfgs[1])
    normalize_randomization_config(env_cfg, obs_cfg)["dynamics"] = default_domain_rand_cfg(enabled=False)
    reward_cfg = _get_reward_cfg()
    curriculum_cfg = _get_curriculum_cfg()
    # 采样比例只在课程中配置；复制首阶段作为初始值，关闭课程时也使用这一比例。
    # jump_modes 保留运行时字段，兼容地形采样、旧存档和评估的单模式覆盖。
    env_cfg["jump_modes"]["mode_probabilities"] = list(curriculum_cfg["stages"][0]["targets"]["terrain"]["mode_probabilities"])
    validate_configs(env_cfg, obs_cfg, curriculum_cfg)
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg


def _get_termination_cfg(episode_length_s: float):
    """终止条件唯一调参入口；二次腾空和机身触地只惩罚，回合超时由阶段总时长决定。"""
    return {
        # jump 任务失败阈值：任意一拍机身倾角超过此值即锁存无效。
        # 这是相对竖直方向的合成倾角；不同于下方分别判断 roll / pitch。
        "jump_max_tilt_deg": 10.0,
        "jump_termination": {
            # 过倾、撞台阶立面、错误落台中的任一项，立即提前终止。
            "on_invalid": True,
        },
        # 上述开关仅控制提前终止；关闭后仍记录失败、给予惩罚，不能恢复成功。
        # 父环境的独立终止条件（也用于 teacher 预热），不受上述开关控制。
        # 默认 jump 的 10° 过倾会先通过 on_invalid 立即终止。
        "termination_if_roll_greater_than": 30.0,  # deg，绝对值严格超过
        "termination_if_pitch_greater_than": 30.0,  # deg，绝对值严格超过
        "tilt_termination_duration_s": 0.5,  # 连续超限持续时间，> 0
        "base_contact_termination_duration_s": None,  # 机身触地只惩罚，关闭父环境接触终止
        # solver error 始终终止并中止训练；episode 超时始终截断，不作为可关闭开关。
    }


def _get_warmup_cfg():
    """跳跃前 locomotion teacher 的指令与站稳条件。"""
    return {
        # 每轮重新采样，预热期间保持不变；固定工况写 [value, value]。
        "command_ranges": {
            "lin_vel_range": [0.0, 1.0],
            "ang_vel_range": [0.0, 0.0],
            "base_height_range": [0.22, 0.22],
        },
        "min_steps": 1,
        "max_steps": 200,
        "stable_steps": 10,
        "stable_fraction": 0.8,  # 避免整批环境被最慢的少数环境拖住。
        "forward_velocity_tolerance": 0.2,
        "yaw_rate_tolerance": 0.05,
        "base_height_tolerance": 0.03,
        "vertical_velocity_tolerance": 0.03,
        "tilt_tolerance_deg": 5.0,
        "require_both_wheels_contact": True,
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
        "warmup_distance_m": 50.0,
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


def _get_obs_cfg(locomotion_obs_cfg: Mapping):
    """继承 locomotion 观测前缀，追加 jump 观测与缩放。"""
    obs_cfg = deepcopy(dict(locomotion_obs_cfg))
    # jump 不使用 ToF，也不为无用的测距创建传感器；不修改来源 locomotion 配置。
    obs_cfg["tof"] = {"enabled": False, "include_in_observation": False}
    obs_cfg.setdefault("obs_scales", {})
    obs_cfg["obs_scales"].update(
        {
            "jump_base_height": 1.0 / 0.40,  # 缩放 base 原点相对起跳面的高度。
            "vertical_velocity": 1.0 / 2.0,
            "vertical_acceleration": 1.0 / 10.0,
        }
    )
    # 保留 locomotion 前缀和缩放；jump 的第三项为按时间变化的目标 base 到轮底距离。
    obs_cfg["locomotion_policy_obs_dim"] = layout_dim(LOCOMOTION_POLICY_LAYOUT)
    obs_cfg["num_policy_obs"] = layout_dim(JUMP_POLICY_LAYOUT)
    # critic 追加基础真值 9D、落地状态 1D、jump 真值 7D、距离/落稳进度 2D。
    obs_cfg["num_critic_obs"] = obs_cfg["num_policy_obs"] + 19
    return obs_cfg


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
            # 实测腿长达到 min_upper_link_angle 对应限位后，惩罚继续伸腿的目标角差平方。
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
        # 这些 locomotion 项会直接压制离地、伸腿或腾空，jump 中明确不注册。
        "disabled_locomotion_rewards": [
            "gated_tracking_lin_vel",
            "gated_tracking_ang_vel",
            "base_height",
            "height_gate",
            "joint_vel",
            "landing_base_oscillation",
            "landing_joint_vel",
            "alive",
        ],
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


def validate_configs(env_cfg, obs_cfg, curriculum_cfg=None):
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
        "warmup_distance_m",
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
    if env_cfg["handoff_on_landing"]:
        raise ValueError("jump_modes requires the full landing stabilization window")
    expected_policy = layout_dim(JUMP_POLICY_LAYOUT)
    if resolve_tof_cfg(obs_cfg.get("tof"))["include_in_observation"]:
        raise ValueError("jump does not support ToF observations; set tof.include_in_observation=False")
    if obs_cfg["num_policy_obs"] != expected_policy or obs_cfg["num_critic_obs"] != expected_policy + 19:
        raise ValueError("jump_modes observation dimensions must include 3D mode and 2D privileged state")
