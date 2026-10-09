"""随机化配置入口：默认值、旧存档迁移及训练/评估命令行覆盖。

动力学实现位于 dynamics_randomization，传感器实现位于 sensor_randomization。
持久化配置键、CLI 和课程目标仍沿用 dynamics/sensors、domain_rand/sensor_noise。
"""

import argparse
from copy import deepcopy

from .dynamics_randomization import default_dynamics_randomization_cfg
from .sensor_randomization import (
    IMU_ERROR_KEYS, ENCODER_CHANNELS, default_sensor_randomization_cfg, resolve_sensor_randomization_cfg,
)


def default_randomization_cfg(*, dynamics_enabled=True, sensors_enabled=False):
    return {
        "dynamics": default_dynamics_randomization_cfg(enabled=dynamics_enabled),
        "sensors": default_sensor_randomization_cfg(enabled=sensors_enabled),
    }


def normalize_randomization_config(env_cfg, obs_cfg):
    """就地迁移；新入口优先。旧存档不自动开启此前没有的编码器噪声。"""
    config = deepcopy(env_cfg.get("randomization", {}))
    if not isinstance(config, dict):
        raise ValueError("randomization must be a dictionary")
    if config.keys() - {"dynamics", "sensors"}:
        raise ValueError("randomization supports dynamics and sensors")
    config.setdefault("dynamics", deepcopy(env_cfg.get("domain_rand", {"enabled": False})))
    imu = obs_cfg.get("imu", {})
    if "sensors" not in config and any(key in imu for key in IMU_ERROR_KEYS):
        legacy = default_sensor_randomization_cfg(enabled=True)
        legacy["tof"]["enabled"] = False
        legacy["imu"] = {key: deepcopy(imu.get(key, 0.0)) for key in IMU_ERROR_KEYS}
        for name in ENCODER_CHANNELS:
            legacy[name] = {"enabled": False, "std": 0.0, "bias_range": [0.0, 0.0]}
        config["sensors"] = legacy
    config["sensors"] = resolve_sensor_randomization_cfg(config.get("sensors"))
    env_cfg["randomization"] = config
    env_cfg.pop("domain_rand", None)
    if "imu" in obs_cfg:
        obs_cfg["imu"] = {key: deepcopy(value) for key, value in imu.items() if key not in IMU_ERROR_KEYS}
    return config


def add_randomization_arguments(parser, *, evaluation=False):
    parser.add_argument(
        "--domain-rand", action=argparse.BooleanOptionalAction,
        default=False if evaluation else None,
        help="override dynamics randomization (training: use config; evaluation: off)",
    )
    parser.add_argument(
        "--sensor-noise", action=argparse.BooleanOptionalAction, default=None,
        help="override sensor noise independently (default: use saved/selected config)",
    )


def apply_randomization_arguments(env_cfg, obs_cfg, args):
    config = normalize_randomization_config(env_cfg, obs_cfg)
    for argument, group in (("domain_rand", "dynamics"), ("sensor_noise", "sensors")):
        value = getattr(args, argument, None)
        if value is not None:
            config[group]["enabled"] = value
