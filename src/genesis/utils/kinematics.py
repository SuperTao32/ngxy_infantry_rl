"""步兵车左右虚拟腿的纯 Torch 几何函数。"""

import torch


def compute_leg_length(joint_pos, front_indices, rear_indices, upper_length, lower_length):
    """由 front1/rear1 电机角估算左右虚拟腿长度。"""
    front_angle = joint_pos[..., front_indices]
    rear_angle = joint_pos[..., rear_indices]
    separation = torch.atan2(torch.sin(front_angle - rear_angle), torch.cos(front_angle - rear_angle))
    half_separation = 0.5 * torch.abs(separation)

    upper_projection = upper_length * torch.sin(half_separation)
    half_joint_distance = upper_length * torch.cos(half_separation)
    lower_projection = torch.sqrt(torch.clamp_min(lower_length**2 - half_joint_distance**2, 0.0))
    return upper_projection + lower_projection


def compute_leg_angle(joint_pos, front_indices, rear_indices):
    """用 front1/rear1 的角平分线表示左右虚拟腿方向。"""
    angle = 0.5 * (joint_pos[..., front_indices] + joint_pos[..., rear_indices])
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def constrain_leg_targets(
    target_joint_pos,
    front_indices,
    rear_indices,
    max_motor_separation,
):
    """保留虚拟腿摆角，只限制两根上杆的电机角差。"""
    front_angle = target_joint_pos[..., front_indices]
    rear_angle = target_joint_pos[..., rear_indices]
    leg_angle = 0.5 * (front_angle + rear_angle)
    half_separation = 0.5 * (front_angle - rear_angle)

    half_separation = torch.clamp(
        half_separation,
        -0.5 * max_motor_separation,
        0.5 * max_motor_separation,
    )
    constrained = target_joint_pos.clone()
    constrained[..., front_indices] = leg_angle + half_separation
    constrained[..., rear_indices] = leg_angle - half_separation
    return constrained
