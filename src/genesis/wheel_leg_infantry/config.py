"""统一配置入口：机器人、观测、PPO 默认值和四任务训练配置组装。

任务专属的奖励、命令、重置和课程参数位于 tasks/<name>/config.py。
每次调用配置工厂均返回新的字典。
"""

import math

from .terrains import default_terrain_cfg
from .robust.randomization_config import default_randomization_cfg
from .sensors.tof import default_tof_cfg
from .tasks.manager import TASK_NAMES, task_configs


def get_env_cfg() -> dict:
    return {
        "num_actions": 6,
        "num_joints": 4,
        "num_wheels": 2,
        "robot_mjcf": "assets/robot/wheelbipeV14_2/mjcf/wheelbipeV14_2.xml",
        # 地形由所选配置决定，也可通过 --terrain 覆盖。
        "terrain": default_terrain_cfg("plane"),
        "randomization": default_randomization_cfg(dynamics_enabled=True, sensors_enabled=True),
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
        "wheel_radius": 0.06,
        "gravity_magnitude": 9.81,
        "spring_names": ["left_spring2_joint", "right_spring2_joint"],
        "base_link_name": "base_link",
        # 虚拟腿仅作为观测和诊断；站立目标使用平地上的实际 base z。
        "leg_upper_link_length": 0.21,
        "leg_lower_link_length": 0.25,
        "upper_links_min_angle": 0.5 * math.pi,
        # 气弹簧：F = F0 + k * compression - c * velocity。
        "gas_spring_preload_force": 420.0,
        "gas_spring_stiffness": 0.0,
        "gas_spring_damping": 0.0,
        "gas_spring_max_compression": 0.06,
        "joint_kp": 50.0,
        "joint_kd": 2.0,
        "wheel_kd": 0.3,
        "joint_force_limit": 40.0,
        "wheel_force_limit": 5.0,
        "joint_pos_scale": 0.25,
        "wheel_vel_scale": 30.0,
        # 固定一拍动作延迟；不属于 domain_rand，本轮仍保持原有动作路径。
        "simulate_action_latency": True,
        # 接触力判定相关
        "wheel_contact_force_threshold": 1.0,
        "base_contact_force_threshold": 5.0,
        # 只有连续超限一段时间才终止，给策略留下可学习的恢复窗口。
        "termination_if_roll_greater_than": 90.0,
        "termination_if_pitch_greater_than": 90.0,
        "tilt_termination_duration_s": 3.20,
        "base_contact_termination_duration_s": 5.0,
        # 从 0.22 m 起步只保留很小的落地行程，避免策略还没输出就先自由落体。
        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "episode_length_s": 100.0,
        "resampling_time_s": 10.0,
    }


def get_obs_cfg() -> dict:
    return {
        # 机身底部/前侧/后侧接触及对应法向力固定加入 critic，共 6D。
        "base_contact_force_clip": 5.0,  # 对缩放后的分区法向力限幅。
        # actor 使用 IMU、轮速等真机观测
        "imu": {
            "link_name": "base_link",
            "pos_offset": [0.0, 0.0, 0.0],
        },
        # actor 包含 6D actions、6D 命令（运动目标与 jump 模式）及 6D 跳跃阶段编码。
        # ToF 开启时为 (39 + 4 * history_frames)D；关闭时为 39D。
        "tof": default_tof_cfg(
            include_in_observation=True,
            history_frames=3,
            update_hz=50.0,  # ToF 更新频率；历史仅在新测距帧到来时推进。
            max_range_m=1.0,  # 测距量程，单位 m；独立于观测缩放参考距离。
            # 观测 = clamp(距离 / 参考距离, 0, 1)，单位 m。
            forward_reference_distance_m=1.0,
            downward_reference_distance_m=0.25,
        ),
        "obs_scales": {
            "lin_vel": 1.0 / 3.5,
            "lin_acc": 1.0 / 9.81,
            "ang_vel": 1.0 / 5.0,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "wheel_vel": 1.0 / 50.0,
            "base_height": 1.0 / 0.35,
            "leg_length": 1.0 / 0.35,
            "normal_force": 1.0 / 400.0,  # 每轮法向合力
            "base_normal_force": 1.0 / 400.0,  # 每个机身分区的法向力大小之和。
            "tangential_force": 1.0 / 200.0,  # 每轮切向合力
        },
    }


def validate_high_base_height_velocity_limits(command_cfg: dict) -> None:
    """校验按目标机身高度触发的速度上限；缺省或 None 表示关闭。"""
    cfg = command_cfg.get("high_base_height_velocity_limits")
    if cfg is None:
        return
    for key in ("base_height_threshold", "max_abs_lin_vel", "max_abs_ang_vel"):
        value = float(cfg[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"high_base_height_velocity_limits.{key} must be finite and nonnegative")


def get_command_cfg() -> dict:
    """共享三维命令接口；采样范围由每个任务独立配置。"""
    return {
        "num_commands": 3,
        "lin_vel_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        "base_height_range": [0.22, 0.22],
        "standing_probability": 1.0,
    }


def get_reward_cfg() -> dict:
    """公共机器人奖励的形状参数；各任务自行提供权重和必要覆盖。"""
    return {
        "tracking_sigma": 0.25,
        "standing_tracking_sigma": 0.25,
        "standing_lin_vel_threshold": 0.02,
        "standing_ang_vel_threshold": 0.02,
        "standing_drift_deadband": 0.02,
        "tracking_gate": {
            "height_full_error": 0.03,
            "height_zero_error": 0.05,
            "attitude_full_angle_deg": 5.0,
            "attitude_zero_angle_deg": 10.0,
            "floor": 0.3,
        },
        "reward_scales": {},
    }


def get_curriculum_cfg() -> dict:
    """共享随机化课程，独立于任何任务的采样和奖励课程。"""
    return {
        "enabled": True,
        "stages": [
            {
                "name": name,
                "start_iteration": iteration,
                "targets": {
                    "domain_rand": {"strength": strength},
                    "sensor_noise": {"strength": strength},
                    "wheel_slip": {"strength": strength},
                },
            }
            for name, iteration, strength in (
                ("warmup", 0, 0.0),
                ("randomization", 4500, 0.4),
                ("full_randomization", 7000, 1.0),
            )
        ],
    }


TASK_WEIGHTS = {"locomotion": 1.0, "recovery": 0.0, "jump": 0.0, "multi_terrain": 0.0}
# 任务可以覆盖回合和重置参数；机器人、动作和随机化参数只在根配置声明。
TASK_ENV_FIELDS = {
    "episode_length_s", "resampling_time_s", "termination_if_roll_greater_than",
    "termination_if_pitch_greater_than", "tilt_termination_duration_s",
    "base_contact_termination_duration_s", "base_init_pos_range",
    "base_init_rpy_offset_range_deg", "base_init_lin_vel_range", "base_init_ang_vel_range",
}


def get_train_cfg(exp_name: str) -> dict:
    model = dict(class_name="RNNModel", hidden_dims=(512, 256, 128), activation="elu",
                 obs_normalization=True, rnn_type="gru", rnn_hidden_dim=512, rnn_num_layers=1)
    return {
        "actor": {**model, "distribution_cfg": {
            "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar",
        }},
        "critic": dict(model),
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
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 24,
        "save_interval": 500,
        "max_iterations": 10000,
        "seed": 42,
        "clip_actions": None,
        "experiment_name": exp_name,
        "run_name": exp_name,
        "logger": "tensorboard",
        "upload_model": False,
    }


def get_cfgs(exp_name="wheel_leg_infantry"):
    env = get_env_cfg()
    tasks = {name: task_configs(name) for name in TASK_NAMES}
    shared_env_fields = set(env) - TASK_ENV_FIELDS
    for name, task in tasks.items():
        forbidden = set(task["env_cfg"]) & shared_env_fields
        if forbidden:
            raise ValueError(f"{name}: shared environment settings belong in config.py: {sorted(forbidden)}")
        for stage in task["curriculum_cfg"].get("stages", []):
            forbidden = set(stage.get("targets", {})) & {"domain_rand", "sensor_noise", "wheel_slip", "action_limits"}
            if forbidden:
                raise ValueError(f"{name}: shared curriculum targets belong in config.py: {sorted(forbidden)}")
        task["weight"] = TASK_WEIGHTS[name]
    env["tasks"] = tasks
    env["control_dt"] = 0.01
    return dict(env_cfg=env, obs_cfg=get_obs_cfg(), reward_cfg=get_reward_cfg(),
                command_cfg=get_command_cfg(), curriculum_cfg=get_curriculum_cfg(),
                train_cfg=get_train_cfg(exp_name), framework="wheel_leg_infantry_multitask_v1")


def select_tasks(configs, names):
    """用于聚焦采样/评估，观测维度和网络结构保持一致。"""
    unknown = set(names) - set(TASK_NAMES)
    if unknown or not names or len(names) != len(set(names)):
        raise ValueError(f"Select distinct tasks from {TASK_NAMES}; got {names}")
    tasks = configs["env_cfg"]["tasks"]
    if set(names) - tasks.keys():
        raise ValueError("Requested task was not part of the saved training configuration")
    configs["env_cfg"]["tasks"] = {name: tasks[name] for name in TASK_NAMES if name in names}


def env_kwargs(configs):
    return {key: configs[key] for key in ("env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "curriculum_cfg")}
