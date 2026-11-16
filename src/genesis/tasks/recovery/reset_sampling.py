"""恢复姿态模板与逐环境采样；reset_ranges 由 env 实时提供。"""

import torch

from .reset_pose import BASE_NAMES, read_reset_pose


class RecoveryResetSampler:
    def __init__(self, env, config):
        self.apply_curriculum({"probability": config.get("probability", 0.0)})
        pool = config["pose_pool"]
        if not pool:
            raise ValueError("recovery.reset_pos.pose_pool must contain at least one pose")
        self.reference_base_pos = env.init_base_pos.new_tensor(config["reference_base_pos"])
        if self.reference_base_pos.shape != (3,):
            raise ValueError("recovery.reset_pos.reference_base_pos must have three coordinates")
        self.clearance_m = config.get("clearance_m", 1e-4)
        if self.clearance_m is not None:
            self.clearance_m = float(self.clearance_m)

        robot = env.robot
        joints = [joint for joint in robot.joints if joint.n_qs == 1]
        if robot.n_qs != 7 + len(joints) or list(robot.joints[0].qs_idx_local) != list(range(7)):
            raise ValueError("Recovery snapshots require a floating base followed by scalar joints")
        joint_names = {joint.name for joint in joints}
        templates = []
        for entry in pool:
            pose = read_reset_pose(entry["pose"])
            supplied = set(pose) - set(BASE_NAMES)
            if supplied != joint_names:
                raise ValueError(
                    f"{entry['name']}: reset joints must match the robot; "
                    f"missing={sorted(joint_names - supplied)}, unknown={sorted(supplied - joint_names)}"
                )
            values = [0.0] * robot.n_qs
            values[:7] = [pose[name] for name in BASE_NAMES]
            for joint in joints:
                values[joint.qs_idx_local[0]] = pose[joint.name]
            templates.append(values)
        self.templates = env.init_qpos.new_tensor(templates)
        self.pose_names = tuple(entry["name"] for entry in pool)
        self.pose_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def apply_curriculum(self, values):
        probability = float(values["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("recovery.reset_pos.probability must be in [0, 1]")
        self.probability = probability

    def sample(self, env, env_ids, positions, *, eligible):
        """每行独立选择快照或默认复位，仅快照行叠加模板位置。"""
        count = len(env_ids)
        nominal_quat = env.init_base_quat.expand(count, -1).clone()
        dof_pos = env.init_dof_pos.expand(count, -1).clone()
        self.pose_ids[env_ids] = -1
        if self.probability == 0.0:
            return nominal_quat, dof_pos
        selected = torch.rand(count, device=self.templates.device) < self.probability
        selected &= eligible
        selected_rows = selected.nonzero(as_tuple=False).flatten()
        if not len(selected_rows):
            return nominal_quat, dof_pos
        indices = torch.randint(len(self.templates), (len(selected_rows),), device=self.templates.device)
        templates = self.templates[indices]
        self.pose_ids[env_ids[selected_rows]] = indices
        positions[selected_rows] += templates[:, :3] - self.reference_base_pos
        nominal_quat[selected_rows] = templates[:, 3:7]
        dof_pos[selected_rows] = templates[:, 7:]
        return nominal_quat, dof_pos

    def ensure_clearance(self, env, env_ids, state):
        """扰动可能使快照穿地，按当前模型碰撞包围盒向上修正；最终状态由 env 写回。"""
        if self.clearance_m is None or not len(env_ids):
            return
        selected = self.pose_ids[env_ids] >= 0
        if not selected.any():
            return
        selected_env_ids = env_ids[selected]
        qpos, positions = state[:2]
        env.robot.set_qpos(qpos[selected], envs_idx=selected_env_ids, zero_velocity=True)
        bottom = env.robot.get_AABB(envs_idx=selected_env_ids)[:, 0, 2]
        if not torch.isfinite(bottom).all():
            raise RuntimeError("Nonfinite collision geometry in recovery pose reset")
        correction = (state[6][selected] + self.clearance_m - bottom).clamp_min(0.0)
        qpos[selected, 2] += correction
        positions[selected, 2] += correction
