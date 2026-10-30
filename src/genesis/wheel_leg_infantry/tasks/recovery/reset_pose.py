"""将可选复位快照解析为可保存的环境配置，不依赖 Genesis。"""

import json
import math
from pathlib import Path

BASE_NAMES = tuple(f"floating_base_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz"))
SIDE_LYING_ANGLE_DEG = 20.0
POSTURE_GROUPS = ("level", "front_down", "rear_down", "inverted", "tilted")


def read_reset_pose(reset_pose):
    """读取并规范化完整快照；也接受已经存档的数值映射。"""
    if str(reset_pose) == "default":
        pose = dict(zip(BASE_NAMES, (0.0, 0.0, 0.22, 1.0, 0.0, 0.0, 0.0)))
    else:
        pose = dict(reset_pose) if isinstance(reset_pose, dict) else json.loads(Path(reset_pose).expanduser().read_text(encoding="utf-8"))
    if not isinstance(pose, dict):
        raise ValueError("Reset pose must be a flat mapping of coordinate names to numbers")
    missing = set(BASE_NAMES) - pose.keys()
    if missing:
        raise ValueError(f"Missing reset coordinates: {sorted(missing)}")
    for name, value in pose.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Reset coordinate {name} must be a finite number")
    norm = math.hypot(*(pose[name] for name in BASE_NAMES[3:]))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Floating base quaternion must have a nonzero finite norm")
    pose = {name: float(value) for name, value in pose.items()}
    pose.update({name: pose[name] / norm for name in BASE_NAMES[3:]})
    return pose


def world_up_in_body(pose):
    """世界竖直方向在机身坐标系中的表示，不受全局 yaw 影响。"""
    w, x, y, z = (pose[name] for name in BASE_NAMES[3:])
    norm = math.hypot(w, x, y, z)
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))


def lateral_axis_vertical_angle_deg(pose):
    """机身左右轴与竖直线的锐角；左右侧平躺都接近 0°。"""
    return math.degrees(math.acos(min(1.0, abs(world_up_in_body(pose)[1]))))


def load_reset_pose_pool(sources, side_lying_angle_deg=SIDE_LYING_ANGLE_DEG):
    """展开 JSON/采集目录，过滤侧躺；返回可直接存档、不依赖文件的快照池。"""
    if isinstance(side_lying_angle_deg, bool) or not math.isfinite(side_lying_angle_deg) or not 0 < side_lying_angle_deg < 90:
        raise ValueError("side_lying_angle_deg must be between 0 and 90 (exclusive)")
    pool, seen = [], set()
    for source in sources:
        path = Path(source).expanduser()
        paths = sorted(path.rglob("pose_*.json")) if path.is_dir() else [source]
        if not paths:
            raise ValueError(f"{source}: no pose_*.json snapshots found")
        for candidate in paths:
            label = "default" if str(candidate) == "default" else str(Path(candidate).expanduser().resolve())
            if label in seen:
                continue
            seen.add(label)
            try:
                pose = read_reset_pose(candidate)
            except (OSError, ValueError) as exc:
                raise ValueError(f"{candidate}: {exc}") from exc
            if lateral_axis_vertical_angle_deg(pose) <= side_lying_angle_deg:
                continue
            up = world_up_in_body(pose)
            if up[2] < 0:
                group = "inverted"
            elif up[0] < -math.sin(math.radians(15)):
                group = "front_down"
            elif up[0] > math.sin(math.radians(15)):
                group = "rear_down"
            elif up[2] >= math.cos(math.radians(15)):
                group = "level"
            else:
                group = "tilted"
            pool.append(dict(source=label, pose=pose, group=group))
    if not pool:
        raise ValueError("Recovery reset pose pool is empty after filtering side-lying poses")
    return pool


def get_reset_pose_cfg(reset_pose):
    """读取任务自己的姿态覆盖项；输出纯数值，保存后不依赖原文件。"""
    pose = read_reset_pose(reset_pose)
    pos = [pose[name] for name in BASE_NAMES[:3]]
    rpy_deg = _quat_to_rpy_deg([pose[name] for name in BASE_NAMES[3:]])
    joint_positions = {name: value for name, value in pose.items() if name not in BASE_NAMES}

    return {
        "base_init_pos_range": [[value, value] for value in pos],
        "base_init_rpy_offset_range_deg": [[value, value] for value in rpy_deg],
        "base_init_lin_vel_range": [[0.0, 0.0] for _ in range(3)],
        "base_init_ang_vel_range": [[0.0, 0.0] for _ in range(3)],
        "reset_joint_pos": joint_positions,
    }


def _quat_to_rpy_deg(quat):
    """将单位 wxyz 四元数转成 RPY，不依赖 Genesis 初始化状态。"""
    w, x, y, z = quat
    sin_pitch = 2.0 * (w * y - x * z)
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    cos_pitch = math.hypot(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    pitch = math.atan2(sin_pitch, cos_pitch)
    if cos_pitch < 1e-10:
        roll = 0.0
        yaw = math.atan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (x * x + z * z))
    else:
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return [math.degrees(value) for value in (roll, pitch, yaw)]
