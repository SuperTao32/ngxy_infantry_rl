"""接收 locomotion 权重的混合地形迁移训练配置。

共享机器人、动作与观测定义；奖励、命令和课程独立维护。
从第 0 次迭代开始混合地形训练，完成后可将权重交给 config_downstairs。
"""

from .config_common import get_env_cfg, get_obs_cfg, get_final_command_cfg


def get_cfgs():
    return _env_cfg(), get_obs_cfg(), _reward_cfg(), _command_cfg(), _curriculum_cfg()


def _env_cfg() -> dict:
    env = get_env_cfg()
    # 越障允许短暂倾斜，但持续失稳仍结束回合。
    env.update(
        termination_if_roll_greater_than=45.0,
        termination_if_pitch_greater_than=45.0,
        tilt_termination_duration_s=0.30,
        base_contact_termination_duration_s=0.5,
    )
    env["terrain"].update(_terrain_cfg())
    return env


def _terrain_cfg() -> dict:
    # 每次回合结束按 weight 重新采样；权重不必合计为 1。
    # difficulty 为初始等级；min/max 为自动升降范围。几何只在对应地形模块中编辑。
    return {
        "mixture": [
            {"preset": "plane", "weight": 10, "difficulty": 0},
            # {"preset": "stairs", "weight": 20, "difficulty": 0, "min_difficulty": 0, "max_difficulty": 4},
            # {"preset": "platform_ridge", "weight": 20, "difficulty": 0, "min_difficulty": 0, "max_difficulty": 4},
            {"preset": "trapezoidal_wave", "weight": 30, "difficulty": 0, "min_difficulty": 0, "max_difficulty": 4},
            {"preset": "square_wave", "weight": 30, "difficulty": 0, "min_difficulty": 0, "max_difficulty": 4},
            {"preset": "random_rough", "weight": 30, "difficulty": 0, "min_difficulty": 0, "max_difficulty": 4},
        ],
        "adaptive": {
            "enabled": True,
            "window_episodes": 100,
            "promote_threshold": 0.80,
            "demote_threshold": 0.30,
            "min_distance": 2.0,
            "min_duration_s": 2.0,
            "max_lin_vel_rmse": 0.5,
            "max_ang_vel_rmse": 1.0,
            "max_tilt_deg": 60.0,
        },
    }


def _reward_cfg() -> dict:
    # 迁移训练的初始参数：保留速度跟踪、平衡和动作平滑，允许越障姿态变化。
    # 这些数值需结合各地形成功率验证，不继承平地课程的逐阶段奖励覆盖。
    return {
        "tracking_sigma": 0.25,
        "standing_tracking_sigma": 0.05,
        "standing_lin_vel_threshold": 0.02,
        "standing_ang_vel_threshold": 0.02,
        "standing_drift_deadband": 0.02,
        "landing_base_ang_vel_weight": 0.25,
        "tracking_gate": {
            "height_full_error": 0.03,
            "height_zero_error": 0.09,
            "attitude_full_angle_deg": 5.0,
            "attitude_zero_angle_deg": 25.0,
            "floor": 0.20,
        },
        "reward_scales": {
            "tracking_lin_vel": -2.0,
            "tracking_ang_vel": -3.0,
            "standing_drift": 0.0,
            "gated_tracking_lin_vel": 5.0,
            "gated_tracking_ang_vel": 7.0,
            "base_balance": -10.0,
            "leg_symmetry": -5.0,
            "leg_angle_limits": -5.0,
            "leg_symmetry_bonus": 1.0,
            "base_height": -10.0,
            "height_gate": 5.0,
            "joint_vel": -0.005,
            "wheel_action_rate": -1.0,
            "leg_action_rate": -0.1,
            "landing_base_oscillation": -0.3,
            "landing_joint_vel": -0.01,
            "wheel_airborne": -5.0,
            "base_contact": -15.0,
            "alive": 5.0,
            "death": -100.0,
        },
    }


def _command_cfg() -> dict:
    return {
        "num_commands": 3,
        # 保持前进，使回合净位移能用于自动难度统计。
        "lin_vel_range": [2.5, 3.2],
        "ang_vel_range": [-0.2, 0.2],
        "base_height_range": [0.32, 0.32],
        "high_speed_ang_vel": {"lin_vel_threshold": 1.5, "max_abs_ang_vel": 2.0},
        "standing_probability": 0.0,
    }


def _curriculum_cfg() -> dict:
    # 迭代从载入权重后重新计数；地形难度由成功率控制，与这些阶段独立。
    # 后续阶段不重复覆盖 terrain，避免重置已学到的等级及统计窗口。
    return {
        "enabled": True,
        "stages": [
            {
                "name": "mixed_adaptation",
                "start_iteration": 0,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                },
            },
            {
                "name": "mixed_randomization",
                "start_iteration": 3000,
                "targets": {
                    "domain_rand": {"strength": 0.4},
                    "sensor_noise": {"strength": 0.4},
                },
            },
            {
                "name": "mixed_robustness",
                "start_iteration": 6000,
                "targets": {
                    "domain_rand": {"strength": 1.0},
                    "sensor_noise": {"strength": 1.0},
                },
            },
        ],
    }
