"""物理单位下的编码器测量噪声；IMU 噪声仍由 Genesis 执行。"""

from copy import deepcopy
import math

import torch


IMU_ERROR_KEYS = (
    "acc_noise", "acc_bias", "acc_random_walk",
    "gyro_noise", "gyro_bias", "gyro_random_walk", "delay", "jitter",
)
ENCODER_CHANNELS = ("joint_pos", "joint_vel", "wheel_vel")


def default_sensor_noise_cfg(*, enabled=False):
    """幅度是待实机标定的起点；位置 rad，速度 rad/s，IMU 使用 SI 单位。"""
    return {
        "enabled": enabled,
        "strength": 1.0,
        "imu": {
            "acc_noise": 0.05, "acc_bias": 0.0, "acc_random_walk": 0.0001,
            "gyro_noise": 0.003, "gyro_bias": 0.0, "gyro_random_walk": 0.00001,
            "delay": 0.0, "jitter": 0.0,
        },
        "joint_pos": {"enabled": False, "std": 0.0, "bias_range": [0.0, 0.0]},
        "joint_vel": {"enabled": True, "std": 0.1, "bias_range": [0.0, 0.0]},
        "wheel_vel": {"enabled": True, "std": 0.2, "bias_range": [0.0, 0.0]},
    }


def _merge(config):
    result = default_sensor_noise_cfg()
    if config is not None and not isinstance(config, dict):
        raise ValueError("sensors must be a dictionary")
    for key, value in (config or {}).items():
        if key not in result:
            raise ValueError(f"Unknown sensors option: {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or value.keys() - result[key].keys():
                raise ValueError(f"Invalid sensors.{key} options")
            result[key].update(deepcopy(value))
        else:
            result[key] = value
    return result


def _finite(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative")


class SensorNoise:
    """每拍 update 一次；观测组装只读取缓存，局部 reset 不影响其他环境。"""

    def __init__(self, config=None):
        self.config = _merge(config)
        cfg = self.config
        if not isinstance(cfg["enabled"], bool):
            raise ValueError("sensors.enabled must be bool")
        _finite(cfg["strength"], "sensors.strength")
        if not 0 <= cfg["strength"] <= 1:
            raise ValueError("sensors.strength must be in [0, 1]")
        self.strength = float(cfg["strength"]) if cfg["enabled"] else 0.0
        for key, value in cfg["imu"].items():
            # Genesis 允许按三轴指定 IMU 参数。
            values = value if isinstance(value, (list, tuple)) else (value,)
            if isinstance(value, (list, tuple)) and (len(values) != 3 or key in ("delay", "jitter")):
                raise ValueError(f"Invalid sensors.imu.{key} shape")
            for item in values:
                _finite(item, f"sensors.imu.{key}", nonnegative=not key.endswith("bias"))
        if cfg["imu"]["jitter"] > cfg["imu"]["delay"]:
            raise ValueError("sensors.imu.jitter must not exceed delay")
        for name in ENCODER_CHANNELS:
            channel = cfg[name]
            if not isinstance(channel["enabled"], bool):
                raise ValueError(f"sensors.{name}.enabled must be bool")
            _finite(channel["std"], f"sensors.{name}.std", nonnegative=True)
            bounds = channel["bias_range"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"sensors.{name}.bias_range must contain two numbers")
            for value in bounds:
                _finite(value, f"sensors.{name}.bias_range")
            if bounds[0] > bounds[1]:
                raise ValueError(f"sensors.{name}.bias_range is reversed")
        self.measurements = {}
        self.biases = {}
        self.imu = None

    def apply_curriculum(self, values):
        """只覆盖幅度；总开关仍由配置/命令行控制，已有测量在下一拍更新。"""
        unknown = set(values).difference({"strength"})
        if unknown:
            raise KeyError(f"Unsupported sensor_noise curriculum keys: {sorted(unknown)}")
        strength = values.get("strength", self.config["strength"])
        _finite(strength, "sensor_noise.strength")
        if not 0 <= strength <= 1:
            raise ValueError("sensor_noise.strength must be in [0, 1]")
        previous = self.strength
        self.strength = float(strength) if self.config["enabled"] else 0.0
        if self.strength == previous:
            return
        if previous > 0:
            for bias in self.biases.values():
                bias.mul_(self.strength / previous)
        elif self.strength > 0:
            # 从无噪声阶段进入有噪声阶段时，为现有回合初始化偏置。
            self._sample_biases(slice(None))
        if self.imu is not None:
            self._update_imu_amplitudes()

    def bind_imu(self, imu):
        """在 Genesis scene.build 后绑定，使用公开 setter 更新幅度。"""
        self.imu = imu
        # 当前 Genesis 将单个 IMU 的参数 expand 到批次，公开 setter 写入时会
        # 命中 stride=0 的重叠存储。仅在绑定时实体化这三个共享参数 Tensor。
        metadata = imu._shared_metadata
        for name in ("noise", "bias", "random_walk"):
            setattr(metadata, name, getattr(metadata, name).clone())
        self._update_imu_amplitudes()

    def _update_imu_amplitudes(self):
        options = self.imu_options({})
        for kind in ("noise", "bias", "random_walk"):
            values = []
            for channel in ("acc", "gyro"):
                value = options[f"{channel}_{kind}"]
                values.extend(value if isinstance(value, (list, tuple)) else [value] * 3)
            # Genesis IMU 的公开 setter 使用 acc/gyro/mag 九轴顺序；本任务未配置磁力计误差。
            getattr(self.imu, f"set_{kind}")(values + [0.0] * 3)

    def imu_options(self, setup):
        """仅返回给 Genesis 的参数，不再对 IMU 读数二次加噪。"""
        result = {key: deepcopy(value) for key, value in setup.items() if key not in IMU_ERROR_KEYS}
        for key, value in self.config["imu"].items():
            # 时间参数不随幅度缩放，避免破坏传感器采样周期对齐。
            scale = float(self.config["enabled"]) if key in ("delay", "jitter") else self.strength
            result[key] = (
                [scale * item for item in value]
                if isinstance(value, (tuple, list)) else scale * value
            )
        return result

    def bind(self, true_values):
        self.measurements = {name: torch.empty_like(true_values[name]) for name in ENCODER_CHANNELS}
        self.biases = {name: torch.zeros_like(true_values[name]) for name in ENCODER_CHANNELS}

    def _sample_biases(self, env_ids):
        for name, bias in self.biases.items():
            channel = self.config[name]
            low, high = channel["bias_range"]
            if self.strength and channel["enabled"] and low != high:
                bias[env_ids] = torch.empty_like(bias[env_ids]).uniform_(low, high) * self.strength
            else:
                bias[env_ids] = low * self.strength if channel["enabled"] else 0.0

    def reset(self, env_ids, true_values):
        """逐环境逐通道采样偏置，并生成 reset 后的首帧测量。"""
        self._sample_biases(env_ids)
        self.update(true_values, env_ids)

    def update(self, true_values, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        for name, measured in self.measurements.items():
            truth = true_values[name][selected]
            channel = self.config[name]
            if self.strength and channel["enabled"]:
                value = truth + self.biases[name][selected]
                if channel["std"]:
                    value = value + torch.randn_like(truth) * (channel["std"] * self.strength)
                measured[selected] = value
            else:
                measured[selected] = truth
