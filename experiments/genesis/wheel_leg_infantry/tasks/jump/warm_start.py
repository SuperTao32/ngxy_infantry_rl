"""从 locomotion checkpoint 迁移 actor，同时保持新增输入初始为零影响。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from ...core.tof import resolve_tof_cfg


@dataclass(frozen=True)
class WarmStartReport:
    copied_tensors: tuple[str, ...]
    expanded_input_tensor: str
    source_input_dim: int
    target_input_dim: int
    mapped_input_columns: int


def validate_locomotion_source(source_configs: Mapping) -> None:
    """拒绝无法证明与当前 jump 基础接口一致的旧 locomotion 配置。"""
    env_cfg = source_configs["env_cfg"]
    obs_cfg = source_configs["obs_cfg"]
    if resolve_tof_cfg(obs_cfg.get("tof"))["include_in_observation"]:
        raise ValueError(
            "jump requires a 32-dimensional locomotion teacher without ToF observations; "
            "train locomotion with tof.include_in_observation=False"
        )
    command_cfg = source_configs["command_cfg"]
    required_env_keys = {"num_actions", "joint_names", "wheel_names", "joint_pos_scale", "wheel_vel_scale"}
    missing = required_env_keys.difference(env_cfg)
    if missing:
        raise ValueError(f"locomotion config is missing action-contract keys: {sorted(missing)}")
    expected_joints = (
        "left_front1_joint",
        "right_front1_joint",
        "left_rear1_joint",
        "right_rear1_joint",
    )
    expected_wheels = ("left_wheel_joint", "right_wheel_joint")
    if (
        env_cfg["num_actions"] != 6
        or tuple(env_cfg["joint_names"]) != expected_joints
        or tuple(env_cfg["wheel_names"]) != expected_wheels
    ):
        raise ValueError("locomotion checkpoint must use the infantry 4 leg + 2 wheel action contract")
    if command_cfg.get("num_commands") != 3 or "base_height_range" not in command_cfg:
        raise ValueError(
            "locomotion checkpoint is too old: expected [vx, wz, base_height] commands; "
            "train the current locomotion task before jump"
        )
    required_obs_scales = {
        "lin_vel",
        "lin_acc",
        "ang_vel",
        "joint_pos",
        "joint_vel",
        "wheel_vel",
        "base_height",
        "leg_length",
        "leg_angle",
    }
    missing_obs_scales = required_obs_scales.difference(obs_cfg.get("obs_scales", {}))
    if missing_obs_scales:
        raise ValueError(f"locomotion config is missing observation scales: {sorted(missing_obs_scales)}")


def warm_start_actor(
    actor,
    checkpoint_path: str | Path,
    locomotion_obs_dim: int,
) -> WarmStartReport:
    """复制 actor；输入层保留 locomotion 前缀，新增 jump 输入初始为零。"""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    if "actor_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint has no actor_state_dict: {checkpoint_path}")

    source_state = checkpoint["actor_state_dict"]
    target_state = actor.state_dict()
    copied: list[str] = []
    expanded: list[str] = []
    incompatible: list[str] = []

    for name, target_value in target_state.items():
        source_value = source_state.get(name)
        if source_value is None:
            incompatible.append(f"{name}: missing from source")
            continue
        source_value = source_value.to(device=target_value.device, dtype=target_value.dtype)
        if source_value.shape == target_value.shape:
            target_state[name] = source_value
            copied.append(name)
            continue
        if (
            name.startswith("mlp.")
            and name.endswith(".weight")
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] == locomotion_obs_dim
            and target_value.shape[1] > locomotion_obs_dim
        ):
            expanded_value = torch.zeros_like(target_value)
            expanded_value[:, :locomotion_obs_dim].copy_(source_value)
            target_state[name] = expanded_value
            expanded.append(name)
            continue
        incompatible.append(
            f"{name}: source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
        )

    if len(expanded) != 1:
        source_shapes = {name: tuple(value.shape) for name, value in source_state.items() if value.ndim == 2}
        raise ValueError(
            "expected exactly one expandable actor input layer, "
            f"found {expanded}; locomotion_obs_dim={locomotion_obs_dim}, source matrices={source_shapes}"
        )
    if incompatible:
        raise ValueError(f"locomotion actor architecture is incompatible with jump actor: {incompatible}")
    actor.load_state_dict(target_state, strict=True)
    expanded_name = expanded[0]
    return WarmStartReport(
        copied_tensors=tuple(copied),
        expanded_input_tensor=expanded_name,
        source_input_dim=locomotion_obs_dim,
        target_input_dim=int(target_state[expanded_name].shape[1]),
        mapped_input_columns=locomotion_obs_dim,
    )
