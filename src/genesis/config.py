"""环境配置入口：locomotion、机器人、PPO、课程与恢复/跳跃/越障组件。

env_cfg.recovery / jump / terrain_cross 分别配置恢复、跳跃和越障，不按任务划分环境。
"""

import math

from .robust.randomization_config import default_randomization_cfg
from .sensors.tof import default_tof_cfg
from .terrains import default_terrain_cfg
from .tasks.recovery.config import get_cfg as get_recovery_cfg
from .tasks.jump.config import get_cfg as get_jump_cfg
from .tasks.terrain_cross.config import get_cfg as get_terrain_cross_cfg

FRAMEWORK = "wheel_leg_infantry_v2"


def get_train_cfg(exp_name: str) -> dict:
    """单策略 actor/critic 网络、PPO 更新参数和运行记录设置。"""
    model = {
        "class_name": "RNNModel",
        "hidden_dims": (512, 256, 128),
        "activation": "elu",
        "obs_normalization": True,
        "rnn_type": "gru",
        "rnn_hidden_dim": 512,
        "rnn_num_layers": 1,
    }
    return {
        # actor 与 critic 使用相同网络结构，各自维护独立状态。
        "actor": {
            **model,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": dict(model),
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        # PPO 更新。
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        # 每轮每个环境采集的控制步数。
        "num_steps_per_env": 24,
        "max_iterations": 10000,
        "seed": 42,
        "clip_actions": None,
        # 日志与 checkpoint。
        "experiment_name": exp_name,
        "run_name": exp_name,
        "save_interval": 100,
        "logger": "tensorboard",
        "upload_model": False,
    }


def get_env_cfg() -> dict:
    """机器人、控制、重置和终止参数。"""
    return {
        # 机器人模型、自由度与关节命名。
        "num_actions": 6,
        "num_joints": 4,
        "num_wheels": 2,
        "robot_mjcf": "assets/robot/wheelbipeV14_2/mjcf/wheelbipeV14_2.xml",
        "default_joint_pos": {
            "left_front1_joint": 0.0,
            "left_rear1_joint": 0.0,
            "right_front1_joint": 0.0,
            "right_rear1_joint": 0.0,
        },
        # 动作顺序也是 actor 输出顺序，修改后旧 checkpoint 不再兼容。
        "joint_names": [
            "left_front1_joint",
            "right_front1_joint",
            "left_rear1_joint",
            "right_rear1_joint",
        ],
        "leg_front_joint_names": ["left_front1_joint", "right_front1_joint"],
        "leg_rear_joint_names": ["left_rear1_joint", "right_rear1_joint"],
        "wheel_names": ["left_wheel_joint", "right_wheel_joint"],
        "wheel_link_names": ["left_wheel_link", "right_wheel_link"],
        "spring_names": ["left_spring2_joint", "right_spring2_joint"],
        "base_link_name": "base_link",
        # 机器人几何与物理参数。
        "wheel_radius": 0.06,
        "gravity_magnitude": 9.81,
        # 虚拟腿仅作为观测和诊断；站立目标使用平地上的实际 base z。
        "leg_upper_link_length": 0.21,
        "leg_lower_link_length": 0.25,
        "upper_links_min_angle": 0.5 * math.pi,
        # 气弹簧：F = F0 + k * compression - c * velocity。
        "gas_spring_preload_force": 420.0,
        "gas_spring_stiffness": 0.0,
        "gas_spring_damping": 0.0,
        "gas_spring_max_compression": 0.06,
        # 电机参数与动作映射。
        "joint_kp": 50.0,
        "joint_kd": 2.0,
        "wheel_kd": 0.3,
        "joint_force_limit": 40.0,
        "wheel_force_limit": 5.0,
        "joint_pos_scale": 0.25,
        "wheel_vel_scale": 30.0,
        # 固定一拍动作延迟；不属于 domain_rand。
        "simulate_action_latency": True,
        # 预建各等级地块；按成功率升降级，仅影响后续 reset。eval 仍选择固定单地形。
        "terrain": {
            **default_terrain_cfg("mixture"),
            "curriculum": {"enabled": True, "min_completed": 100, "promote_threshold": 0.95, "demote_threshold": 0.1},
            "tiles": [
                {"preset": "plane", "difficulty": 0, "weight": 1.0},
                {"preset": "slope_up", "difficulty": 0, "weight": 0.0},
                {"preset": "slope_down", "difficulty": 0, "weight": 0.0},
                {"preset": "square_wave", "difficulty": 0, "weight": 0.0},
                {"preset": "stairs", "difficulty": 0, "weight": 0.0},
            ],
        },
        "randomization": default_randomization_cfg(dynamics_enabled=True, sensors_enabled=True),
        # 接触判定阈值；基座与腿部共用 base_contact 阈值、奖励和终止计时。
        "wheel_contact_force_threshold": 1.0,
        "base_contact_force_threshold": 1.0,
        # 回合、命令重采样与终止默认值；可由课程调整。
        "episode_length_s": 48.0,
        "resampling_time_s": 8.0,
        # 连续超限一段时间才终止，给策略留下恢复窗口。
        "termination_if_roll_greater_than": 60.0,
        "termination_if_pitch_greater_than": 60.0,
        "tilt_termination_duration_s": 3.20,
        "base_contact_termination_duration_s": 3.20,
        # 默认重置范围；可由课程调整。命中 recovery 姿态池时，位置减去其 reference_base_pos 后叠加到快照。
        # 从 0.22 m 起步只保留很小的落地行程，避免策略还没输出就先自由落体。
        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "control_dt": 0.01,
        "recovery": get_recovery_cfg(),
        "jump": get_jump_cfg(),
        "terrain_cross": get_terrain_cross_cfg(),
    }


def get_obs_cfg() -> dict:
    """传感器和观测缩放，actor 使用可在真机获取的观测。"""
    return {
        # IMU。
        "imu": {
            "link_name": "base_link",
            "pos_offset": [0.0, 0.0, 0.0],
        },
        # actor 包含 6D actions、5D 命令（运动目标与两维 jump 模式）及 4D 跳跃阶段编码。
        # 固定保留 ToF 通道，课程开启前输入全零；共 (36 + 4 * history_frames)D。
        "tof": default_tof_cfg(
            enabled=True,
            include_in_observation=True,
            history_frames=1,
            update_hz=50.0,  # ToF 更新频率；历史仅在新测距帧到来时推进。
            max_range_m=1.0,  # 测距量程，单位 m；独立于观测缩放参考距离。
            # 观测 = clamp(距离 / 参考距离, 0, 1)，单位 m。
            forward_reference_distance_m=1.0,
            downward_reference_distance_m=0.25,
        ),
        # 机身底部/前侧/后侧接触及对应法向力固定加入 critic，共 6D。
        "base_contact_force_clip": 5.0,  # 对缩放后的分区法向力限幅。
        # 共享观测缩放。
        "obs_scales": {
            "lin_vel": 1.0 / 3.5,
            "lin_acc": 1.0 / 9.81,
            "ang_vel": 1.0 / 5.0,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "wheel_vel": 1.0 / 50.0,
            "base_height": 1.0 / 0.35,
            "leg_length": 1.0 / 0.35,
            "normal_force": 1.0 / 100.0,  # 每轮法向合力
            "base_normal_force": 1.0 / 100.0,  # 每个机身分区的法向力大小之和。
            "tangential_force": 1.0 / 100.0,  # 每轮切向合力
        },
    }


def get_command_cfg() -> dict:
    return {
        "num_commands": 3,
        "lin_vel_range": [3.0, 3.0],
        "ang_vel_range": [-6.0, 6.0],
        "base_height_range": [0.2, 0.38],
        "standing_probability": 0.1,
        "high_speed_ang_vel": {
            "lin_vel_threshold": 1.5,
            "max_abs_ang_vel": 2.0,
        },
        "high_base_height_velocity_limits": {
            "base_height_threshold": 0.32,
            "max_abs_lin_vel": 2.0,
            "max_abs_ang_vel": 3.0,
        },
    }


def get_reward_cfg() -> dict:
    return {
        "tracking_sigma": 0.25,
        "standing_tracking_sigma": 0.25,
        "standing_lin_vel_threshold": 0.02,
        "standing_ang_vel_threshold": 0.02,
        "standing_drift_deadband": 0.01,
        "tracking_gate": {
            "height_full_error": 0.03,
            "height_zero_error": 0.05,
            "attitude_full_angle_deg": 5.0,
            "attitude_zero_angle_deg": 10.0,
            "floor": 0.3,
        },
        "reward_scales": {
            "tracking_lin_vel": -0.5,
            "tracking_ang_vel": -0.5,
            "standing_drift": 0.0,
            "gated_tracking_lin_vel": 2.0,
            "gated_tracking_ang_vel": 2.0,
            "base_balance": -5.0,
            "leg_symmetry": -5.0,
            "leg_symmetry_bonus": 1.0,
            "base_height": -10.0,
            "height_gate": 5.0,
            "joint_vel": -0.005,
            "wheel_action_rate": -1.0,
            "leg_action_rate": -0.1,
            "wheel_airborne": -10.0,
            "base_contact": -10.0,
            "alive": 5.0,
            "death": -100.0,
        },
    }


def get_curriculum_cfg() -> dict:
    return {
        "enabled": True,
        "stages": [
            {
                "name": "stand",
                "start_iteration": 0,
                "targets": {
                    # 概率为 0 时关闭；后续课程阶段可覆盖，不需要单独 enabled 开关。
                    "recovery": {"probability": 0.0},
                    "jump": {"probability": 0.0},
                    "tof": {"enabled": False},
                    "terrain_weights": {"plane": 1.0},  # 未列出的地形权重为 0，先学平地。
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 1.0,
                        "lin_vel_range": [0.0, 0.0],
                        "ang_vel_range": [0.0, 0.0],
                        "base_height_range": [0.22, 0.22],
                    },
                    "domain_rand": {
                        "strength": 0.0,
                    },
                    "sensor_noise": {
                        "strength": 0.0,
                    },
                    "wheel_slip": {
                        "strength": 0.0,
                    },
                },
            },
            {
                "name": "locomotion",
                "start_iteration": 500,
                "targets": {
                    "standing_reward": {
                        "tracking_sigma": 0.1,
                    },
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.1,
                        "lin_vel_range": [-1.2, 1.2],
                        "ang_vel_range": [-2.0, 2.0],
                        "base_height_range": [0.2, 0.32],
                    },
                    "tracking_gate": {
                        "attitude_full_angle_deg": 1.0,
                        "attitude_zero_angle_deg": 4.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -2.0,
                        "tracking_lin_vel": -2.0,
                        "tracking_ang_vel": -3.0,
                        "height_gate": 10.0,
                        "leg_symmetry": -10.0,
                        "base_balance": -20.0,
                    },
                },
            },
            {
                "name": "locomotion2",
                "start_iteration": 1000,
                "targets": {
                    "standing_reward": {
                        "tracking_sigma": 0.05,
                    },
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-3.2, 3.2],
                        "ang_vel_range": [-5.0, 5.0],
                        "base_height_range": [0.2, 0.38],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.02,
                        "height_zero_error": 0.05,
                        "attitude_full_angle_deg": 0.8,
                        "attitude_zero_angle_deg": 3.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -5.0,
                        "gated_tracking_lin_vel": 5.0,
                        "gated_tracking_ang_vel": 7.0,
                    },
                },
            },
            {
                "name": "locomotion_rand",
                "start_iteration": 2000,
                "targets": {
                    "recovery": {"probability": 0.20},
                    "standing_reward": {
                        "tracking_sigma": 0.02,
                    },
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.50]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [-1.0, 1.0]],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.01,
                        "height_zero_error": 0.04,
                        "attitude_full_angle_deg": 0.3,
                        "attitude_zero_angle_deg": 1.5,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -10.0,
                        "wheel_action_rate": -2.0,
                        "leg_action_rate": -0.3,
                        "gated_tracking_lin_vel": 5.0,
                        "gated_tracking_ang_vel": 7.0,
                        "base_balance": -15.0,
                        "base_contact": -15.0,
                    },
                },
            },
            {
                "name": "jump",
                "start_iteration": 4000,
                "targets": {
                    "jump": {"probability": 0.20},
                },
            },
            {
                "name": "terrain_cross",
                "start_iteration": 9000,
                "targets": {
                    "tof": {"enabled": True},
                    "terrain_weights": {"plane": 0.6, "stairs": 0.1, "slope_down": 0.1, "slope_up": 0.1, "square_wave": 0.1},
                },
            },
        ],
    }


def get_cfgs(exp_name):
    return {
        "framework": FRAMEWORK,
        "env_cfg": get_env_cfg(),
        "obs_cfg": get_obs_cfg(),
        "reward_cfg": get_reward_cfg(),
        "command_cfg": get_command_cfg(),
        "curriculum_cfg": get_curriculum_cfg(),
        "train_cfg": get_train_cfg(exp_name),
    }


def env_kwargs(configs):
    """提取环境构造参数；PPO 的采样步数等训练参数由入口单独传入。"""
    return {key: configs[key] for key in ("env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "curriculum_cfg")}
