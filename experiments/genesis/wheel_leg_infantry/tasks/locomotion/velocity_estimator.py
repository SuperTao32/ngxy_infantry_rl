""" 面向真机观测接口的公共前向速度估计函数。"""

from __future__ import annotations

import math

import torch


def gravity_compensated_forward_acceleration(
    imu_specific_force: torch.Tensor,
    projected_gravity: torch.Tensor,
    gravity_magnitude: float,
) -> torch.Tensor:
    """把 IMU 比力换算为车体坐标系的前向运动加速度。

    Genesis 与真实加速度计给出的都是 ``R^T(a_world - g_world)``。因此车体
    倾斜时不能直接积分 x 轴读数，必须先加回车体坐标系中的重力分量。
    """
    if imu_specific_force.shape[-1] != 3 or projected_gravity.shape[-1] != 3:
        raise ValueError("imu_specific_force and projected_gravity must end with dimension 3")
    if imu_specific_force.shape != projected_gravity.shape:
        raise ValueError("imu_specific_force and projected_gravity must have the same shape")
    if gravity_magnitude <= 0.0:
        raise ValueError("gravity_magnitude must be positive")

    return imu_specific_force[..., 0] + gravity_magnitude * projected_gravity[..., 0]


def wheel_forward_velocity(
    wheel_angular_velocity: torch.Tensor,
    wheel_radius: float,
    wheel_velocity_sign: float = 1.0,
) -> torch.Tensor:
    """由左右轮角速度均值计算无打滑假设下的前向速度。"""
    if wheel_angular_velocity.shape[-1] != 2:
        raise ValueError("wheel_angular_velocity must contain left and right wheel velocities")
    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive")
    if wheel_velocity_sign not in {-1.0, 1.0}:
        raise ValueError("wheel_velocity_sign must be -1.0 or 1.0")

    return wheel_velocity_sign * wheel_angular_velocity.mean(dim=-1) * wheel_radius


def complementary_forward_velocity_update(
    previous_velocity: torch.Tensor,
    forward_acceleration: torch.Tensor,
    wheel_velocity: torch.Tensor,
    *,
    dt: float,
    wheel_correction_time_constant_s: float,
    max_abs_velocity: float,
) -> torch.Tensor:
    """预测一步 IMU 积分，并用轮速低频校正积分漂移。

    时间常数越小，估计越信任轮速；时间常数越大，越信任 IMU 的短时变化。
    这只是易于部署和验证的基线估计器，真机仍应根据打滑与传感器标定调整。
    """
    if previous_velocity.shape != forward_acceleration.shape or previous_velocity.shape != wheel_velocity.shape:
        raise ValueError("velocity estimator inputs must have the same shape")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if wheel_correction_time_constant_s < 0.0:
        raise ValueError("wheel_correction_time_constant_s cannot be negative")
    if max_abs_velocity <= 0.0:
        raise ValueError("max_abs_velocity must be positive")

    predicted_velocity = previous_velocity + forward_acceleration * dt
    if wheel_correction_time_constant_s == 0.0:
        estimated_velocity = wheel_velocity
    else:
        correction_gain = -math.expm1(-dt / wheel_correction_time_constant_s)
        estimated_velocity = predicted_velocity + correction_gain * (wheel_velocity - predicted_velocity)

    return torch.clamp(estimated_velocity, -max_abs_velocity, max_abs_velocity)
