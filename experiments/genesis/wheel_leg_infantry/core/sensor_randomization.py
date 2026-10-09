"""传感器随机化：配置解析、IMU/编码器测量，以及 ToF 测距/安装误差。

阅读顺序：默认配置 → 配置校验与解析 → IMU/编码器管理 → ToF 运行时。
配置解析不创建测量状态；两个运行时类分别按控制步和 ToF 测距帧更新。
本模块不依赖 Genesis、MuJoCo 或射线查询实现。
"""

from copy import deepcopy
import math

import torch


IMU_ERROR_KEYS = (
    "acc_noise", "acc_bias", "acc_random_walk",
    "gyro_noise", "gyro_bias", "gyro_random_walk", "delay", "jitter",
)
ENCODER_CHANNELS = ("joint_pos", "joint_vel", "wheel_vel")


# 默认配置：每次调用返回独立字典；幅度均使用物理单位。


def default_sensor_randomization_cfg(*, enabled=False):
    """幅度是待实机标定的起点；位置 rad，速度 rad/s，IMU 使用 SI 单位。"""
    return {
        "enabled": enabled,
        "strength": 1.0,
        "tof": default_tof_randomization_cfg(),
        "imu": {
            "acc_noise": 0.05, "acc_bias": 0.0, "acc_random_walk": 0.0001,
            "gyro_noise": 0.003, "gyro_bias": 0.0, "gyro_random_walk": 0.00001,
            "delay": 0.0, "jitter": 0.0,
        },
        "joint_pos": {"enabled": False, "std": 0.0, "bias_range": [0.0, 0.0]},
        "joint_vel": {"enabled": True, "std": 0.1, "bias_range": [0.0, 0.0]},
        "wheel_vel": {"enabled": True, "std": 0.2, "bias_range": [0.0, 0.0]},
    }


def default_tof_randomization_cfg():
    """测距和位置用米，安装角用度；范围按环境/传感器/轴独立采样。"""
    return {
        "enabled": True,
        "std_m": 0.01,
        "bias_range_m": [-0.01, 0.01],
        "position_range_m": [[-0.005, 0.005] for _ in range(3)],
        "rotation_range_deg": [[-2.0, 2.0] for _ in range(3)],
    }


# 配置解析：合并旧配置、检查形状/单位范围，不分配运行时 Tensor。


def _validate_finite(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_range(bounds, name):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"{name} must contain two numbers")
    for value in bounds:
        _validate_finite(value, name)
    if bounds[0] > bounds[1]:
        raise ValueError(f"{name} is reversed")


def _validate_strength(value, name):
    _validate_finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_imu_config(cfg):
    for key, value in cfg.items():
        # Genesis 允许三轴 IMU 参数；时间参数 delay/jitter 必须是标量。
        values = value if isinstance(value, (list, tuple)) else (value,)
        if isinstance(value, (list, tuple)) and (len(values) != 3 or key in ("delay", "jitter")):
            raise ValueError(f"Invalid sensors.imu.{key} shape")
        for item in values:
            _validate_finite(item, f"sensors.imu.{key}", nonnegative=not key.endswith("bias"))
    if cfg["jitter"] > cfg["delay"]:
        raise ValueError("sensors.imu.jitter must not exceed delay")


def _validate_encoder_config(cfg, name):
    if not isinstance(cfg["enabled"], bool):
        raise ValueError(f"sensors.{name}.enabled must be bool")
    _validate_finite(cfg["std"], f"sensors.{name}.std", nonnegative=True)
    _validate_range(cfg["bias_range"], f"sensors.{name}.bias_range")


def _merge_sensor_config(config):
    """逐组补齐默认值；拒绝未知键，复制嵌套值以免修改调用方配置。"""
    result = default_sensor_randomization_cfg()
    result["tof"]["enabled"] = False
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


def resolve_sensor_randomization_cfg(config=None):
    """返回独立、完整且已校验的配置；旧存档缺少 ToF 时保持关闭。"""
    cfg = _merge_sensor_config(config)
    cfg["tof"] = resolve_tof_randomization_cfg(cfg["tof"])
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("sensors.enabled must be bool")
    _validate_strength(cfg["strength"], "sensors.strength")
    _validate_imu_config(cfg["imu"])
    for name in ENCODER_CHANNELS:
        _validate_encoder_config(cfg[name], name)
    return cfg


def resolve_tof_randomization_cfg(value=None):
    """ToF 后端也可单独解析本组配置；未声明时不自动开启随机化。"""
    cfg = default_tof_randomization_cfg()
    cfg["enabled"] = False
    if value is not None:
        if not isinstance(value, dict) or value.keys() - cfg.keys():
            raise ValueError("Invalid sensors.tof options")
        cfg.update(deepcopy(value))
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("sensors.tof.enabled must be bool")
    _validate_finite(cfg["std_m"], "sensors.tof.std_m", nonnegative=True)
    _validate_range(cfg["bias_range_m"], "sensors.tof.bias_range_m")
    for name in ("position_range_m", "rotation_range_deg"):
        ranges = cfg[name]
        if not isinstance(ranges, (list, tuple)) or len(ranges) != 3:
            raise ValueError(f"sensors.tof.{name} must contain three [min, max] ranges")
        for bounds in ranges:
            _validate_range(bounds, f"sensors.tof.{name}")
    return cfg


# IMU/编码器运行时：课程控制幅度，编码器按控制步更新，IMU 参数交给后端。


class SensorRandomizationManager:
    """先 bind/bind_imu，再 reset/update；组装观测时只读取 measurements。

    编码器偏置在 reset 采样，课程变化会同步缩放已有偏置；白噪声每控制步采样。
    ToF 共用此处的 config/strength，但由 ToFRandomization 独立管理采样状态。
    """

    def __init__(self, config=None):
        self.config = resolve_sensor_randomization_cfg(config)
        self.strength = float(self.config["strength"]) if self.config["enabled"] else 0.0
        self.measurements = {}
        self.biases = {}
        self.imu = None

    def apply_curriculum(self, values):
        """只覆盖幅度；总开关仍由配置/命令行控制，已有测量在下一拍更新。"""
        unknown = set(values).difference({"strength"})
        if unknown:
            raise KeyError(f"Unsupported sensor_noise curriculum keys: {sorted(unknown)}")
        strength = values.get("strength", self.config["strength"])
        _validate_strength(strength, "sensor_noise.strength")
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

    # 编码器：分配缓存 → reset 采偏置 → 每控制步更新测量。

    def bind(self, true_values):
        self.measurements = {name: torch.empty_like(true_values[name]) for name in ENCODER_CHANNELS}
        self.biases = {name: torch.zeros_like(true_values[name]) for name in ENCODER_CHANNELS}

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

    def _sample_biases(self, env_ids):
        for name, bias in self.biases.items():
            channel = self.config[name]
            low, high = channel["bias_range"]
            if self.strength and channel["enabled"] and low != high:
                bias[env_ids] = torch.empty_like(bias[env_ids]).uniform_(low, high) * self.strength
            else:
                bias[env_ids] = low * self.strength if channel["enabled"] else 0.0

    # IMU：生成后端参数，并在绑定或课程变化时同步幅度。

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


# ToF 运行时：每回合固定安装/偏置，新测距帧独立添加白噪声。


def sanitize_tof_distances(distances, cfg):
    """最近交点不在量程内或不存在时统一返回 max_range_m。"""
    lo, hi = cfg["min_range_m"], cfg["max_range_m"]
    valid = torch.isfinite(distances) & (distances >= lo) & (distances < hi)
    return torch.where(valid, distances, hi)


class ToFRandomization:
    """reset 采样每环境/每传感器误差，measure 只处理本次发布的测距帧。

    bias 的形状为 (环境, 传感器)，position/rotation_deg 的末维是三轴。
    安装位置和四元数供后端改变射线；白噪声使用当前强度，回合误差保持 reset 时的值。
    """

    def __init__(self, cfg, num_envs, num_sensors, *, device=None, dtype=torch.float32):
        self.cfg = resolve_tof_randomization_cfg(cfg)
        self.bias = torch.zeros((num_envs, num_sensors), device=device, dtype=dtype)
        self.position = self.bias.new_zeros((num_envs, num_sensors, 3))
        self.rotation_deg = torch.zeros_like(self.position)
        self.rotation_quat = self.bias.new_zeros((num_envs, num_sensors, 4))
        self.rotation_quat[..., 0] = 1
        self.mount_active = False

    def reset(self, env_ids=None, *, strength=1.0):
        """安装位姿与测距偏置只在 reset 缩放和采样，课程变化不移动当前安装。"""
        selected = slice(None) if env_ids is None else env_ids
        scale = strength if self.cfg["enabled"] else 0.0
        self._sample_episode_errors(selected, scale)
        self._update_mount_state(selected)

    def measure(self, distances, tof_cfg, env_ids=None, *, strength=1.0):
        """输入是选中环境的原始米制距离；无回波保留最大量程标记。"""
        clean = sanitize_tof_distances(distances, tof_cfg)
        if not self.cfg["enabled"]:
            return clean
        selected = slice(None) if env_ids is None else env_ids
        measured = clean + self.bias[selected]
        if strength and self.cfg["std_m"]:
            measured = measured + torch.randn_like(clean) * (self.cfg["std_m"] * strength)
        measured = measured.clamp(tof_cfg["min_range_m"], tof_cfg["max_range_m"])
        return torch.where(clean < tof_cfg["max_range_m"], measured, clean)

    def _sample_episode_errors(self, selected, scale):
        for buffer, name in ((self.bias, "bias_range_m"), (self.position, "position_range_m"),
                             (self.rotation_deg, "rotation_range_deg")):
            bounds = buffer.new_tensor(self.cfg[name])
            low, high = bounds[..., 0], bounds[..., 1]
            if scale and torch.any(high != low):
                buffer[selected] = (low + torch.rand_like(buffer[selected]) * (high - low)) * scale
            else:
                buffer[selected] = low * scale

    def _update_mount_state(self, selected):
        self.mount_active = bool(torch.any(self.position != 0) | torch.any(self.rotation_deg != 0))
        # 安装误差绕 mounting link 的 X/Y/Z 轴，R = Rz(yaw) Ry(pitch) Rx(roll)。
        half = torch.deg2rad(self.rotation_deg[selected]) / 2
        cr, cp, cy = half.cos().unbind(-1)
        sr, sp, sy = half.sin().unbind(-1)
        self.rotation_quat[selected] = torch.stack((
            cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
        ), dim=-1)
