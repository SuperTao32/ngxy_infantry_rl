"""轮腿车原地跳跃任务配置。

Jump 复用 locomotion 的机器人、执行器、动作顺序和基础观测，只覆盖单次跳跃
需要的时序、命令和奖励。这样训练入口可以从 locomotion actor 做有约束的热启动。

"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from ...core.domain_randomization import default_domain_rand_cfg
from ..locomotion.config import get_cfgs as get_locomotion_cfgs
from .observation import JUMP_ESTIMATOR_LAYOUT, LOCOMOTION_ESTIMATOR_LAYOUT, layout_dim

PHASE_NAMES = ("crouch", "takeoff", "flight", "landing")


def get_cfgs(locomotion_cfgs: Sequence[Mapping] | None = None):
    """返回 jump 的五组配置，可从某次 locomotion 的已保存配置派生。"""
    if locomotion_cfgs is None:
        locomotion_cfgs = get_locomotion_cfgs()
    if len(locomotion_cfgs) < 4:
        raise ValueError("locomotion_cfgs must contain env, obs, reward and command configs")

    env_cfg = deepcopy(dict(locomotion_cfgs[0]))
    # jump 独立选择随机化范围；以后启用时，teacher 预热和跳跃沿用同一组随机参数。
    env_cfg["domain_rand"] = default_domain_rand_cfg(enabled=False)
    env_cfg["handoff_on_landing"] = False  # jump 完成落地阶段后再交回 locomotion
    # 仅 jump 阶段覆盖；locomotion 站稳预热仍使用源配置，下一轮自动恢复。
    # None 表示继承。可填标量或列表：腿按 joint_names 顺序，轮按 wheel_names 顺序。
    env_cfg["jump_motor_params"] = {
        "joint_kp": 150.0,  # 腿部位置增益，例如 60.0 或 [60.0, 60.0, 60.0, 60.0]
        "joint_kd": 1.0,  # 腿部阻尼，例如 3.0
        "wheel_kd": 0.25,  # 轮子速度控制增益，例如 0.25 或 [0.25, 0.25]
        "joint_force_limit": None,  # 腿电机力矩上限，单位 N·m，必须 > 0
        "wheel_force_limit": None,  # 轮电机力矩上限，单位 N·m，必须 > 0
    }
    env_cfg.pop("landing_penalty_duration_s", None)
    obs_cfg = deepcopy(dict(locomotion_cfgs[1]))
    locomotion_reward_cfg = deepcopy(dict(locomotion_cfgs[2]))
    # 这是 jump 前 locomotion teacher 的用户配置入口。
    # 固定工况写 [value, value]；需要随机训练时再写 [lower, upper]。
    locomotion_warmup_command_ranges = {
        "lin_vel_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        "base_height_range": [0.22, 0.22],
    }

    phase_durations = {
        "crouch": 0.0,
        "takeoff": 0.15,
        "flight": 0.45,
        "landing": 0.80,
    }
    episode_length_s = sum(phase_durations.values())
    # [触发 jump 后的时间(s), 机身原点到双轮轮底的平均竖直距离(m)]。
    # 50cm 轮底峰值的时序初值：假设水平姿态，0.10s 离地时距离 0.40m，
    # 顶点收腿到 0.20m，则机身顶点约 0.70m，弹道上升量约 0.30m。
    # 简化弹道给出 vz≈2.43m/s、触发后 0.35s 达顶点、约 0.63s 触地。
    # 机身原点不等于质心，真实离地时间/速度需通过仿真验证；参考并非强制轨迹。
    # 节点间五次平滑插值；末节点必须等于 episode_length_s（1.20s）。
    height_reference = [
        [0.00, 0.22],  # 初始下蹲，与 warmup 指令一致
        [0.15, 0.40],  # 蹬地伸腿
        [0.30, 0.20],  # 上升段收腿
        [0.40, 0.20],  # 顶点附近保持收腿，避免过早伸腿降低轮底峰值
        [0.65, 0.30],  # 下降段提前伸腿，准备触地
        [0.70, 0.30],  # 覆盖名义触地时刻，随后缓慢恢复下蹲
        [episode_length_s, 0.22],  # 恢复下蹲
    ]
    command_cfg = deepcopy(dict(locomotion_cfgs[3]))
    command_cfg.update(
        {"num_commands": 3, **{name: list(limits) for name, limits in locomotion_warmup_command_ranges.items()}}
    )
    env_cfg.update(
        {
            "episode_length_s": episode_length_s,
            "jump_max_tilt_deg": 20.0,
            "resampling_time_s": episode_length_s,
            "jump_phase_durations_s": phase_durations,
            "wheel_clearance_target_m": 0.50,
            "height_reference": height_reference,
            # 离地前最后接触拍 vz>0 且连续双轮离地至少 0.25 s，
            # 才发放一次 takeoff_event 奖励。
            "takeoff_min_airborne_time_s": 0.25,
            # 固定四阶段作为采集上限；落地交接或死亡后不再采集该环境的 jump 数据。
            # 二次起跳立即死亡；solver error 仍会中止本轮训练。
            "termination_if_roll_greater_than": 20.0,
            "termination_if_pitch_greater_than": 20.0,
            "tilt_termination_duration_s": episode_length_s,
            "base_contact_termination_duration_s": episode_length_s,
            "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
            "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            # 每个 PPO iteration 都重新采样一次，warmup 期间保持不变。
            "locomotion_warmup": {
                "command_ranges": {name: list(limits) for name, limits in locomotion_warmup_command_ranges.items()},
                "min_steps": 1,
                "max_steps": 500,
                "stable_steps": 10,
                # 避免 8192 个环境被最慢的少数 outlier 拖住。
                "stable_fraction": 0.95,
                "forward_velocity_tolerance": 0.1,
                "yaw_rate_tolerance": 0.03,
                "base_height_tolerance": 0.012,
                "vertical_velocity_tolerance": 0.01,
                "tilt_tolerance_deg": 5.0,
                "require_both_wheels_contact": True,
            },
        }
    )

    obs_cfg.setdefault("obs_scales", {})
    obs_cfg["obs_scales"].update(
        {
            "wheel_clearance": 1.0 / 0.25,
            # 与 locomotion base_height 使用相同缩放，才能安全复用第一层对应权重。
            "base_to_wheel_bottom_distance": obs_cfg["obs_scales"]["base_height"],
            "vertical_velocity": 1.0 / 2.0,
            "vertical_acceleration": 1.0 / 10.0,
        }
    )
    # 保留 locomotion 前缀的形状和缩放；jump 的第三项改为机身到轮底距离参考。
    obs_cfg["locomotion_policy_obs_dim"] = layout_dim(LOCOMOTION_ESTIMATOR_LAYOUT)
    obs_cfg["num_policy_obs"] = layout_dim(JUMP_ESTIMATOR_LAYOUT)
    # critic 追加基础特权量 9 维、jump 落地状态 1 维，再追加 jump 真值 8 维。
    obs_cfg["num_critic_obs"] = obs_cfg["num_policy_obs"] + 18

    reward_cfg = {
        # 与 25cm 配置一致，宽松跟踪伸缩参考，2 cm 内不扣跟踪分。
        "height_reference_tolerance_m": 0.02,
        "height_reference_sigma": 0.04,
        "short_leg_length_target": 0.14,
        "leg_length_sigma": 0.01,
        "flight_height_sigma": 0.04,
        "flight_balance_sigma": 0.04,
        "landing_velocity_sigma": 0.25,
        "landing_tilt_sigma": 0.04,
        "landing_ang_vel_sigma": 1.0,
        "reference_forward_velocity_sigma": 0.25,
        "zero_yaw_rate_sigma": 0.25,
        "reward_scales": {
            # 复用 locomotion 的无门控奖励，在蓄力、起跳、腾空和落地全程生效。
            "base_balance": -2.0,
            "jump_invalid": -200.0,
            "landing_airborne": -100.0,
            "leg_symmetry": -2.0,
            "leg_symmetry_bonus": 2.0,
            "height_reference_tracking": 5.0,
            # 计划下蹲阶段双轮离地的持续惩罚。
            "crouch_airborne": -100.0,
            # 全程追踪切换时的前向速度，同时保持 yaw 角速度为零。
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 2.0,
            # 起跳
            "takeoff_event": 200.0,
            "flight_height_shortfall": -2000.0,
            "flight_peak_height": 3000.0,
            "takeoff_upward_velocity": 2000.0,
            "takeoff_vertical_velocity": 2000.0,
            # 腾空
            "flight_airtime": 2.0,
            "flight_balance": 5.0,
            "flight_height_progress": 80.0,
            "flight_height_tracking": 80.0,
            # 落地
            "soft_landing": 1.0,
            "landing_stability": 1.0,
            "action_rate": -0.001,
            "base_contact": -5.0,
            "death": -100.0,
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

    curriculum_cfg = {
        "enabled": True,
        "stages": [
            {
                "name": "static_jump",
                "start_iteration": 0,
                "targets": {
                    "command_ranges": {
                        **{name: list(limits) for name, limits in locomotion_warmup_command_ranges.items()},
                    },
                },
            },
            {
                "name": "slow_jump",
                "start_iteration": 500,
                "targets": {
                    "command_ranges": {
                        "lin_vel_range": [-0.5, 0.5],
                    },
                    "reward_scales": {
                        "base_balance": -10.0,
                        "leg_symmetry": -10.0,
                        "leg_symmetry_bonus": 10.0,
                        "tracking_lin_vel": 10.0,
                        "tracking_ang_vel": 10.0,
                        "flight_airtime": 5.0,
                        "flight_balance": 10.0,
                        # 落地
                        "soft_landing": 10.0,
                        "landing_stability": 10.0,
                        "action_rate": -0.001,
                        "base_contact": -20.0,
                    },
                },
            },
            {
                "name": "medium_jump",
                "start_iteration": 1000,
                "targets": {
                    "command_ranges": {
                        "lin_vel_range": [-1.0, 1.0],
                    },
                    "reward_scales": {
                        "base_balance": -20.0,
                        "leg_symmetry": -20.0,
                        "leg_symmetry_bonus": 20.0,
                        "tracking_lin_vel": 20.0,
                        "tracking_ang_vel": 20.0,
                        "flight_airtime": 10.0,
                        "flight_balance": 15.0,
                    },
                },
            },
            {
                "name": "fast_jump",
                "start_iteration": 1500,
                "targets": {
                    "command_ranges": {
                        "lin_vel_range": [-2.0, 2.0],
                    },
                    "reward_scales": {
                        "base_balance": -30.0,
                        "leg_symmetry": -30.0,
                        "leg_symmetry_bonus": 30.0,
                        "tracking_lin_vel": 30.0,
                        "tracking_ang_vel": 30.0,
                        # 腾空
                        "flight_balance": 30.0,
                        # 落地
                        "soft_landing": 50.0,
                        "landing_stability": 50.0,
                        "action_rate": -0.02,
                        "base_contact": -300.0,
                    },
                },
            },
        ],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg
