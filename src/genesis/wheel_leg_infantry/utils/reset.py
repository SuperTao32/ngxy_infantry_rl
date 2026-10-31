"""完整姿态复位的碰撞几何离地校正。"""

import torch


def ensure_reset_clearance(robot, env_ids, state, clearance=1e-4):
    """按当前碰撞体向上修正旧快照，保持关节、朝向与速度；最终状态由环境写回。"""
    if not len(env_ids):
        return
    qpos, pos = state[:2]
    robot.set_qpos(qpos, envs_idx=env_ids, zero_velocity=True)
    bottom = robot.get_AABB(envs_idx=env_ids)[:, 0, 2]
    if not torch.isfinite(bottom).all():
        raise RuntimeError("Nonfinite collision geometry in pose reset")
    correction = (state[6] + clearance - bottom).clamp_min(0.0)
    qpos[:, 2] += correction
    pos[:, 2] += correction
