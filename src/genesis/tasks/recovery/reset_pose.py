"""读取可存档的复位快照；路径只在创建配置时使用。"""

import json
import math
from pathlib import Path


BASE_NAMES = tuple(f"floating_base_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz"))
RESET_POSE_DIR = Path(__file__).with_name("reset_pos")


def read_reset_pose(source):
    """读取完整快照并归一化 wxyz 四元数。"""
    if isinstance(source, dict):
        pose = dict(source)
    else:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = RESET_POSE_DIR / path
        pose = json.loads(path.read_text(encoding="utf-8"))
    pose = {name: float(value) for name, value in pose.items()}
    if not all(math.isfinite(value) for value in pose.values()):
        raise ValueError("Recovery reset pose must contain finite coordinates")
    norm = math.hypot(*(pose[name] for name in BASE_NAMES[3:]))
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("Recovery reset quaternion must have a nonzero finite norm")
    pose.update({name: pose[name] / norm for name in BASE_NAMES[3:]})
    return pose


def load_reset_pose_pool(sources):
    """读取显式文件清单；保存配置后不再依赖 JSON 文件。"""
    if isinstance(sources, (str, Path)):
        sources = [sources]
    return [{"name": Path(source).stem, "pose": read_reset_pose(source)} for source in sources]
