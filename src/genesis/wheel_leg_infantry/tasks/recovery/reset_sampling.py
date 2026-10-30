"""恢复任务的逐环境完整快照采样；静态与低难度动态初态共用关节构型。"""

import math

import torch
from genesis.utils.geom import inv_quat, transform_by_quat, transform_quat_by_quat, xyz_to_quat

from ...utils.tensor_utils import as_range_tensors, sample_uniform
from .reset_pose import BASE_NAMES, POSTURE_GROUPS, _quat_to_rpy_deg, read_reset_pose


class RecoveryResetSampler:
    def __init__(self, config, robot, num_envs, device, default_joint_pos):
        probability = config["dynamic_probability"]
        if isinstance(probability, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("recovery_reset.dynamic_probability must be in [0, 1]")
        self.dynamic_probability = probability
        self.device = device
        self.pose_pool = config["pose_pool"]
        if not self.pose_pool:
            raise ValueError("recovery_reset.pose_pool must contain at least one pose")
        joints = [joint for joint in robot.joints if joint.n_qs == 1]
        if robot.n_qs != 7 + len(joints) or robot.joints[0].qs_idx_local != list(range(7)):
            raise ValueError("Recovery snapshots require a floating base followed by scalar joints")
        joint_names = {joint.name for joint in joints}
        templates, rpy = [], []
        for entry in self.pose_pool:
            pose = read_reset_pose(entry["pose"])
            supplied = set(pose) - set(BASE_NAMES)
            if supplied - joint_names or (entry["source"] != "default" and supplied != joint_names):
                raise ValueError(
                    f"{entry['source']}: snapshot joints must match robot; missing={sorted(joint_names - supplied)}, "
                    f"unknown={sorted(supplied - joint_names)}"
                )
            values = [0.0] * robot.n_qs
            values[:7] = [pose[name] for name in BASE_NAMES]
            for joint in joints:
                values[joint.qs_idx_local[0]] = pose.get(joint.name, default_joint_pos.get(joint.name, 0.0))
            templates.append(values)
            rpy.append(_quat_to_rpy_deg(values[3:7]))
        self.templates = torch.tensor(templates, dtype=torch.float32, device=device)
        self.rpy = torch.tensor(rpy, dtype=torch.float32, device=device)
        self.group_names = POSTURE_GROUPS
        self.group_ids = torch.tensor([self.group_names.index(entry["group"]) for entry in self.pose_pool], device=device)
        self.pose_ids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.is_dynamic = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.landed = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.clearance = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.limits = {}
        for name, count in (("lift_height_range", 1), ("pitch_offset_range_deg", 1), ("base_lin_vel_range", 3), ("base_ang_vel_range", 3)):
            self.limits[name] = as_range_tensors(config["dynamic"][name], count, f"recovery_reset.dynamic.{name}", device=device)
        if self.limits["lift_height_range"][0].item() <= 0:
            raise ValueError("Dynamic lift heights must be positive")

    def sample(self, env_ids, terrain):
        """每次独立采样姿态和 Bernoulli 动静态标记，只更新本次 reset 的行。"""
        count = len(env_ids)
        pose_ids = torch.randint(len(self.pose_pool), (count,), device=self.device)
        dynamic = torch.rand(count, device=self.device) < self.dynamic_probability
        qpos = self.templates[pose_ids].clone()
        pos, rpy = qpos[:, :3].clone(), self.rpy[pose_ids].clone()
        centers, tiles = terrain.sample_spawn_tiles(env_ids)
        pos[:, :2] += centers
        terrain.adjust_spawn(pos, rpy, tiles)
        height = terrain.height_at(pos[:, :2])
        pos[:, 2] += height
        quat = xyz_to_quat(rpy, rpy=True, degrees=True)
        lift = sample_uniform(*self.limits["lift_height_range"], (count,)).squeeze(-1)
        lift = torch.where(dynamic, lift, 0.0)
        pos[:, 2] += lift
        pitch = sample_uniform(*self.limits["pitch_offset_range_deg"], (count,)).squeeze(-1)
        offset = torch.zeros((count, 3), device=self.device)
        offset[:, 1] = torch.where(dynamic, pitch, 0.0)
        # 右乘局部 pitch，前后倾扰动不依赖快照的全局 yaw，也不改变闭链关节。
        quat = transform_quat_by_quat(xyz_to_quat(offset, rpy=True, degrees=True), quat)
        lin_vel = sample_uniform(*self.limits["base_lin_vel_range"], (count,))
        ang_vel = sample_uniform(*self.limits["base_ang_vel_range"], (count,))
        lin_vel = torch.where(dynamic[:, None], lin_vel, 0.0)
        ang_vel = torch.where(dynamic[:, None], ang_vel, 0.0)
        qpos[:, :3], qpos[:, 3:7] = pos, quat
        gravity = transform_by_quat(pos.new_tensor([0.0, 0.0, -1.0]).expand(count, -1), inv_quat(quat))
        self.pose_ids[env_ids] = pose_ids
        self.is_dynamic[env_ids] = dynamic
        self.landed[env_ids] = ~dynamic
        self.clearance[env_ids] = lift
        return qpos, pos, quat, lin_vel, ang_vel, gravity, height, centers, tiles

    def ensure_dynamic_clearance(self, robot, env_ids, state):
        """只对动态行临时做 FK，抬高至碰撞几何真正离地；最终状态仍由环境统一写入。"""
        dynamic = self.is_dynamic[env_ids]
        ids = env_ids[dynamic]
        if not ids.numel():
            return
        qpos, pos = state[:2]
        robot.set_qpos(qpos[dynamic], envs_idx=ids, zero_velocity=True)
        bottom = robot.get_AABB(envs_idx=ids)[:, 0, 2]
        if not torch.isfinite(bottom).all():
            raise RuntimeError("Nonfinite collision geometry in recovery reset")
        correction = (state[6][dynamic] + self.clearance[ids] - bottom).clamp_min(0.0)
        qpos[dynamic, 2] += correction
        pos[dynamic, 2] += correction

    def update_landing(self, wheel_contact, base_contact):
        self.landed |= (wheel_contact > 0.5).any(dim=1) | (base_contact > 0.5)

    def episode_metrics(self, mask, success, duration):
        """只返回真实结束回合的样本，让 logger 按各分组自己的样本求均值。"""
        pose_ids = self.pose_ids[mask]
        dynamic = self.is_dynamic[mask]
        groups = self.group_ids[pose_ids]
        success, duration = success[mask], duration[mask]
        metrics = {"dynamic_fraction": dynamic.float()}

        def add(prefix, selected):
            if selected.any():
                metrics[f"{prefix}/success"] = success[selected]
                metrics[f"{prefix}/recovery_duration_s"] = duration[selected]

        for mode, value in (("static", False), ("dynamic", True)):
            selected = dynamic == value
            add(mode, selected)
            for index, group in enumerate(self.group_names):
                add(f"{mode}/{group}", selected & (groups == index))
        for index in pose_ids.unique().tolist():
            add(f"pose/{index:04d}", pose_ids == index)
        return metrics
