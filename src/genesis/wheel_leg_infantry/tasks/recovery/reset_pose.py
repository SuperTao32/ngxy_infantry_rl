"""将可选复位快照解析为可保存的环境配置，不依赖 Genesis。"""

import json
import math
from pathlib import Path


def get_reset_pose_cfg(reset_pose):
    """读取任务自己的姿态覆盖项；输出纯数值，保存后不依赖原文件。"""
    if str(reset_pose) == "default":
        pos, rpy_deg = [0.0, 0.0, 0.22], [0.0, 0.0, 0.0]
        joint_positions = {}
    else:
        path = Path(reset_pose).expanduser()
        pose = json.loads(path.read_text(encoding="utf-8"))
        base_names = [f"floating_base_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        if not isinstance(pose, dict):
            raise ValueError(f"{path}: expected a flat mapping of coordinate names to numbers")
        missing = set(base_names) - pose.keys()
        if missing:
            raise ValueError(f"{path}: missing reset coordinates: {sorted(missing)}")
        for name, value in pose.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: {name} must be a finite number")
        pos = [pose[name] for name in base_names[:3]]
        quat = [pose[name] for name in base_names[3:]]
        norm = math.hypot(*quat)
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"{path}: floating base quaternion must have a nonzero finite norm")
        rpy_deg = _quat_to_rpy_deg([value / norm for value in quat])
        joint_positions = {name: value for name, value in pose.items() if name not in base_names}

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
