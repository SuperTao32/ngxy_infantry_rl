"""两套 locomotion eval 共用的 ToF GUI 读数，单位为米。"""

from copy import deepcopy

from ..core.tof import default_tof_cfg, resolve_tof_cfg


def prepare_tof_for_viewer(obs_cfg):
    """旧权重没有测距时，只开启诊断测距，保持原有策略输入维度。"""
    saved = resolve_tof_cfg(obs_cfg.get("tof"))
    if saved["enabled"]:
        return
    cfg = default_tof_cfg(include_in_observation=False)
    cfg.update(deepcopy(obs_cfg.get("tof") or {}))
    cfg.update(enabled=True, include_in_observation=False)
    obs_cfg["tof"] = resolve_tof_cfg(cfg)


def tof_overlay_lines(env):
    """从当前测距 buffer 读取第一个环境，不触发额外测量或观测更新。"""
    cfg = env.tof_cfg
    if not cfg["enabled"]:
        return ("ToF: disabled",)
    sensors = cfg["sensors"]
    distances = env.tof_distances.detach().reshape(-1, len(sensors))[0].cpu().tolist()
    labels = {
        "tof_left": "Front L", "tof_right": "Front R",
        "tof_down_left": "Down L", "tof_down_right": "Down R",
    }
    values = []
    for sensor, distance in zip(sensors, distances):
        label = labels.get(sensor["name"], sensor["name"])
        status = " [no hit/range]" if distance >= cfg["max_range_m"] else ""
        values.append(f"{label}: {distance:.4f} m{status}")
    policy = "ON" if cfg["include_in_observation"] else "OFF"
    return (
        f"ToF  range {cfg['min_range_m']:.2f}..{cfg['max_range_m']:.2f} m | policy: {policy}",
        *("    ".join(values[i:i + 2]) for i in range(0, len(values), 2)),
    )
