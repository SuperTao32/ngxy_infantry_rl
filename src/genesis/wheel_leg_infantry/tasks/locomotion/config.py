"""步兵轮腿车 locomotion 的训练配置。

配置按物理、观测、奖励、命令和课程拆开。locomotion 训练先学习原地站立，
再逐步扩大速度、转向、高度命令和 reset 难度。
"""

def get_cfgs():
    """平地任务覆盖项；共享机器人、观测和训练参数由根配置提供。"""
    return dict(env_cfg={}, reward_cfg=_reward_cfg(), command_cfg=_command_cfg(),
                curriculum_cfg=_curriculum_cfg())


def _reward_cfg() -> dict:
    return {
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


def _command_cfg() -> dict:
    return {
        "lin_vel_range": [3.0, 3.0],
        "ang_vel_range": [-6.0, 6.0],
        "base_height_range": [0.20, 0.38],
        # |vx| > 1.5 m/s 时，偏航角速度收紧到 ±2 rad/s。
        "high_speed_ang_vel": {"lin_vel_threshold": 1.5, "max_abs_ang_vel": 2.0},
        # 目标机身高度超过阈值时，同时收紧线速度与偏航角速度。
        "high_base_height_velocity_limits": {"base_height_threshold": 0.32, "max_abs_lin_vel": 2.0, "max_abs_ang_vel": 3.0},
        # 静止站立采样；课程可通过 command_ranges.standing_probability 覆盖。
        "standing_probability": 0.10,
    }


def _curriculum_cfg() -> dict:
    # 阶段采用累计覆盖：这里只写相对上一阶段发生变化的字段。
    return {
        "enabled": True,
        "stages": [
            {
                "name": "stand",
                "start_iteration": 0,
                "targets": {
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
                },
            },
            {
                "name": "locomotion",
                "start_iteration": 1000,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.10},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.1,
                        "lin_vel_range": [-1.0, 1.0],
                        "ang_vel_range": [-2.0, 2.0],
                        "base_height_range": [0.20, 0.32],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.03,
                        "height_zero_error": 0.06,
                        "attitude_full_angle_deg": 1.0,
                        "attitude_zero_angle_deg": 4.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -1.0,
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
                "start_iteration": 2000,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.05},
                    "command_ranges": {
                        "standing_probability": 0.1,
                        "lin_vel_range": [-2.0, 2.0],
                        "ang_vel_range": [-3.0, 3.0],
                        "base_height_range": [0.20, 0.34],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.02,
                        "height_zero_error": 0.05,
                        "attitude_full_angle_deg": 0.8,
                        "attitude_zero_angle_deg": 3.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -2.0,
                        "gated_tracking_lin_vel": 5.0,
                        "gated_tracking_ang_vel": 7.0,
                        "base_balance": -15.0,
                        "base_contact": -15.0,
                    },
                },
            },
            {
                "name": "locomotion_rand",
                "start_iteration": 3000,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.02},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.24]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [-1.0, 1.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-3.0, 3.0],
                        "ang_vel_range": [-5.0, 5.0],
                        "base_height_range": [0.20, 0.38],
                    },
                    "tracking_gate": {
                        "height_full_error": 0.01,
                        "height_zero_error": 0.04,
                        "attitude_full_angle_deg": 0.5,
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
                "start_iteration": 4000,
                "targets": {
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-3.5, 3.5],
                        "ang_vel_range": [-6.0, 6.0],
                        "base_height_range": [0.20, 0.38],
                    },
                    "reward_scales": {},
                },
            },
        ],
    }
