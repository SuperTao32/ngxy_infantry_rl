"""前向与下向单点 ToF 的安装、量程和观测契约。"""

from copy import copy, deepcopy
import math

import numpy as np
import torch

from .sensor_randomization import ToFRandomization, sanitize_tof_distances


def default_tof_cfg(
    *, include_in_observation=True, history_frames=1, update_hz=50.0, max_range_m=1.2,
    forward_reference_distance_m=1.0, downward_reference_distance_m=0.5,
):
    # base_link 坐标为 X 前、Y 左、Z 上；前向组位于前侧底缘，下向组位于底部左右。
    # 四路从 z=-0.15 向下外移 10 mm，避免位置随机化后贴入底盘碰撞箱。
    # 对 wheelbipeV14_2 实际 pos/quat/size，默认三轴 ±5 mm 下仍有至少 10 mm
    # 的底盘外侧余量；默认三轴 ±2° 下光轴始终朝该外侧平面之外。
    return {
        "enabled": True,
        "include_in_observation": include_in_observation,
        "history_frames": history_frames,
        "update_hz": update_hz,
        "link_name": "base_link",
        "min_range_m": 0.0,
        "max_range_m": max_range_m,
        "sensors": [
            {"name": "tof_left", "pos_offset": [0.23, 0.15, -0.16], "downward_angle_deg": 45.0,
             "reference_distance_m": forward_reference_distance_m},
            {"name": "tof_right", "pos_offset": [0.23, -0.15, -0.16], "downward_angle_deg": 45.0,
             "reference_distance_m": forward_reference_distance_m},
            {"name": "tof_down_left", "pos_offset": [0.10, 0.12, -0.16], "downward_angle_deg": 90.0,
             "reference_distance_m": downward_reference_distance_m},
            {"name": "tof_down_right", "pos_offset": [0.10, -0.12, -0.16], "downward_angle_deg": 90.0,
             "reference_distance_m": downward_reference_distance_m},
        ],
    }


def resolve_tof_cfg(value):
    # 旧 locomotion cfgs.pkl 未声明 ToF 时保留原有观测接口。
    cfg = deepcopy(value) if value is not None else {"enabled": False}
    if not isinstance(cfg, dict) or not isinstance(cfg.get("enabled"), bool):
        raise ValueError("tof.enabled must be a boolean")
    # 兼容之前仅用 enabled 控制观测的存档；新配置可独立关闭策略输入。
    cfg.setdefault("include_in_observation", cfg["enabled"])
    if not isinstance(cfg["include_in_observation"], bool):
        raise ValueError("tof.include_in_observation must be a boolean")
    cfg.setdefault("history_frames", 1)
    if type(cfg["history_frames"]) is not int or cfg["history_frames"] < 1:
        raise ValueError("tof.history_frames must be a positive integer (including the current frame)")
    rate = cfg.setdefault("update_hz", 50.0)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("tof.update_hz must be positive and finite")
    if not cfg["enabled"]:
        cfg["include_in_observation"] = False
        return cfg
    if not isinstance(cfg.get("link_name"), str) or not cfg["link_name"]:
        raise ValueError("tof.link_name must name the mounting link")
    lo, hi = cfg.get("min_range_m"), cfg.get("max_range_m")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi)) or not 0 <= lo < hi:
        raise ValueError("tof ranges must satisfy 0 <= min_range_m < max_range_m, in meters")
    sensors = cfg.get("sensors", [])
    if not isinstance(sensors, (list, tuple)) or not sensors or any(not isinstance(s, dict) for s in sensors):
        raise ValueError("tof.sensors must be a non-empty list of sensor configurations")
    names = [sensor.get("name") for sensor in sensors]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("tof sensor names must be non-empty and unique; list order defines observation order")
    for sensor in sensors:
        pos = np.asarray(sensor.get("pos_offset"), dtype=float)
        angle = sensor.get("downward_angle_deg")
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError(f"{sensor['name']}.pos_offset must contain three finite coordinates in meters")
        if not isinstance(angle, (int, float)) or not math.isfinite(angle) or not 0 < angle <= 90:
            raise ValueError(f"{sensor['name']}.downward_angle_deg must be in (0, 90]")
        # 旧存档没有独立参考距离，沿用按量程缩放的策略输入。
        reference = sensor.setdefault("reference_distance_m", hi)
        if (isinstance(reference, bool) or not isinstance(reference, (int, float))
                or not math.isfinite(reference) or reference <= 0):
            raise ValueError(f"{sensor['name']}.reference_distance_m must be positive and finite, in meters")
    return cfg


def tof_direction(sensor):
    angle = math.radians(sensor["downward_angle_deg"])
    return (math.cos(angle), 0.0, -math.sin(angle))


def tof_site_quat(sensor):
    # MuJoCo site 的 +Z 轴为光轴，绕 +Y 转 90° + 俯角。
    half_angle = math.radians(90.0 + sensor["downward_angle_deg"]) / 2
    return (math.cos(half_angle), 0.0, math.sin(half_angle), 0.0)


def tof_observation(distances, cfg):
    """按 sensors 配置顺序独立缩放距离，并截断到 [0, 1]。"""
    references = distances.new_tensor([
        sensor.get("reference_distance_m", cfg["max_range_m"])
        for sensor in cfg["sensors"]
    ])
    return (distances / references).clamp(0.0, 1.0)


class ToFSamplingClock:
    """各环境独立采样相位；非整数周期在下一控制步更新，保留余量防止漂移。"""

    def __init__(self, cfg, dt, num_envs=1, device=None):
        self.increment = cfg["update_hz"] * dt
        if self.increment > 1.0 + 1e-9:
            raise ValueError(f"tof.update_hz must not exceed the control frequency ({1 / dt:g} Hz)")
        self.phase = torch.zeros(num_envs, dtype=torch.float64, device=device)

    def reset(self, env_ids=None):
        self.phase[slice(None) if env_ids is None else env_ids] = 0

    def advance(self):
        self.phase += self.increment
        due = self.phase >= 1.0 - 1e-9
        self.phase -= due.to(self.phase.dtype)
        return due


class ToFHistory:
    """按传感器分组、组内从旧到新的归一化历史；仅在新测距帧和 reset 时写入。"""

    def __init__(self, distances, cfg):
        self.cfg = cfg
        self.buffer = distances.new_empty((*distances.shape, cfg["history_frames"]))

    def reset(self, distances, env_ids=None):
        # 首帧填满，避免零填充制造虚假的近距离障碍，也不保留上一回合数据。
        selected = slice(None) if env_ids is None else env_ids
        self.buffer[selected] = tof_observation(distances[selected], self.cfg).unsqueeze(-1)

    def append(self, distances, env_ids=None):
        if env_ids is not None:
            selected = self.buffer[env_ids]
            if selected.shape[-1] > 1:
                selected[..., :-1] = selected[..., 1:].clone()
            selected[..., -1] = tof_observation(distances[env_ids], self.cfg)
            self.buffer[env_ids] = selected
            return
        if self.buffer.shape[-1] > 1:
            self.buffer[..., :-1] = self.buffer[..., 1:].clone()
        self.buffer[..., -1] = tof_observation(distances, self.cfg)

    def observation(self):
        return self.buffer.flatten(-2)


class GenesisToF:
    """附着在机身上的单射线 Raycaster，以及无物理步进的复位刷新。"""

    def __init__(self, scene, robot, cfg, randomization_cfg=None):
        import genesis as gs

        self.robot = robot
        self.cfg = cfg
        self.randomization_cfg = randomization_cfg
        self.randomization = None
        self._mount_metadata = None
        self._link = robot.get_link(cfg["link_name"])
        self.sensors = [
            scene.add_sensor(gs.sensors.Raycaster(
                entity_idx=robot.idx,
                link_idx_local=robot.get_link(cfg["link_name"]).idx_local,
                pos_offset=tuple(sensor["pos_offset"]),
                pattern=gs.sensors.GridPattern(size=(0.0, 0.0), direction=tof_direction(sensor)),
                min_range=cfg["min_range_m"],
                max_range=cfg["max_range_m"],
                no_hit_value=cfg["max_range_m"],
                return_points=False,
            ))
            for sensor in cfg["sensors"]
        ]
        self._reset_raw = None

    def reset(self, env_ids=None, *, strength=1.0):
        if self.randomization is None:
            pos = self.robot.get_pos()
            self.randomization = ToFRandomization(
                self.randomization_cfg, pos.shape[0], len(self.sensors), device=pos.device, dtype=pos.dtype,
            )
        self.randomization.reset(env_ids, strength=strength)

    def _read_randomized_mounts(self, *, after_reset=False):
        # 当前 Genesis Raycaster 在 build 时把安装位姿烘焙到所有环境共享的射线。
        # set_pos/quat_offset 不会更新这些射线，因此用私有查询 metadata 为每个
        # 环境构造真实传感器坐标系；复用 Genesis BVH，不改共享 sensor cache。
        from genesis.utils.geom import transform_by_quat, transform_quat_by_quat

        first = self.sensors[0]
        if self._mount_metadata is None:
            metadata = copy(first._shared_metadata)
            metadata.solver_groups = []
            metadata.ray_starts = metadata.ray_starts.clone()
            batch = self.randomization.bias.shape[0]
            metadata.links_pos = self.randomization.bias.new_zeros((batch, metadata.n_sensors, 3))
            metadata.links_quat = self.randomization.bias.new_zeros((batch, metadata.n_sensors, 4))
            metadata.links_quat[..., 0] = 1
            self._mount_cols = [sensor._idx for sensor in self.sensors]
            for sensor in self.sensors:
                ray = metadata.sensors_ray_start_idx[sensor._idx]
                metadata.ray_starts[ray] = 0
            self._nominal_positions = self.randomization.position.new_tensor([
                sensor["pos_offset"] for sensor in self.cfg["sensors"]
            ])
            self._mount_metadata = metadata
            self._mount_raw = self.randomization.bias.new_empty((metadata.total_cache_size, batch))
        metadata = self._mount_metadata
        local_pos = self._nominal_positions + self.randomization.position
        link_pos = self._link.get_pos(relative=False)[:, None, :]
        link_quat = self._link.get_quat(relative=False)[:, None, :].expand_as(self.randomization.rotation_quat)
        metadata.links_pos[:, self._mount_cols] = link_pos + transform_by_quat(local_pos, link_quat)
        metadata.links_quat[:, self._mount_cols] = transform_quat_by_quat(self.randomization.rotation_quat, link_quat)
        if after_reset:
            first._shared_context.update()
        type(first)._update_raw_data(first._shared_context, metadata, self._mount_raw)
        return torch.cat([
            self._mount_raw[sensor._cache_offset:sensor._cache_offset + 1].T for sensor in self.sensors
        ], dim=-1)

    def read(self, *, after_reset=False, env_ids=None, strength=1.0):
        if self.randomization is not None and self.randomization.mount_active:
            distances = self._read_randomized_mounts(after_reset=after_reset)
        elif after_reset:
            # Genesis 的 read() 只读取 eager cache；set_qpos 不刷新传感器。
            # 当前 Genesis 没有公开的即时刷新接口。只重新调用 Raycaster 几何
            # 查询，使用独立缓存，不推进 IMU 历史、噪声或仿真时钟。
            first = self.sensors[0]
            metadata = first._shared_metadata
            if self._reset_raw is None:
                pos = self.robot.get_pos()
                self._reset_raw = torch.empty(
                    (metadata.total_cache_size, pos.shape[0]), device=pos.device, dtype=pos.dtype,
                )
            # 调用方用 skip_forward=False 设置复位姿态，确保 links/geoms 已更新。
            first._shared_context.update()
            type(first)._update_raw_data(first._shared_context, metadata, self._reset_raw)
            distances = torch.cat([
                self._reset_raw[sensor._cache_offset:sensor._cache_offset + 1].T
                for sensor in self.sensors
            ], dim=-1)
        else:
            distances = torch.cat([sensor.read().distances.flatten(1) for sensor in self.sensors], dim=-1)
        selected = slice(None) if env_ids is None else env_ids
        if self.randomization is not None:
            return self.randomization.measure(distances[selected], self.cfg, env_ids, strength=strength)
        return sanitize_tof_distances(distances[selected], self.cfg)
