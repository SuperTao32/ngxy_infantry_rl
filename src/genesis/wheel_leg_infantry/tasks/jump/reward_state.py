"""单次跳跃的奖励事件；与 Genesis 解耦，便于验证连续腾空和一次性结算。"""

import math

import torch


class JumpRewardState:
    def __init__(self, num_envs, *, dt, min_airborne_time_s, horizon, device, dtype):
        if not math.isfinite(min_airborne_time_s) or min_airborne_time_s <= 0.0:
            raise ValueError("takeoff_min_airborne_time_s must be finite and positive")
        self.min_airborne_steps = max(1, math.ceil(min_airborne_time_s / dt - 1e-9))
        if self.min_airborne_steps > horizon:
            raise ValueError("takeoff_min_airborne_time_s cannot exceed the jump horizon")
        self.horizon = horizon
        self.airborne_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_airborne_steps = torch.zeros_like(self.airborne_steps)
        self.takeoff_rewarded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.height_settled = torch.zeros_like(self.takeoff_rewarded)
        self.peak_clearance = torch.zeros(num_envs, dtype=dtype, device=device)
        self.revoked_peak_clearance = torch.zeros_like(self.peak_clearance)
        self.revoked_settled_clearance = torch.zeros_like(self.peak_clearance)
        self.previous_peak_clearance = torch.zeros_like(self.peak_clearance)
        self.takeoff_event = torch.zeros_like(self.peak_clearance)
        self.height_settlement_event = torch.zeros_like(self.peak_clearance)

    def reset(self, env_idx=None):
        for value in (
            self.airborne_steps, self.max_airborne_steps, self.takeoff_rewarded,
            self.height_settled, self.peak_clearance, self.revoked_peak_clearance,
            self.previous_peak_clearance,
            self.revoked_settled_clearance,
            self.takeoff_event, self.height_settlement_event,
        ):
            if env_idx is None:
                value.zero_()
            else:
                value.masked_fill_(env_idx, 0)

    def update(
        self, airborne, clearance, landing_event, episode_steps, *, takeoff_eligible, valid_jump=None
    ):
        # 高度进度已在腾空时发放；无论空中还是落地失稳都须一次性追回。
        # 缺高惩罚仅在此前已结算时补扣，否则结束时会按零高度完整结算。
        self.revoked_peak_clearance.zero_()
        self.revoked_settled_clearance.zero_()
        if valid_jump is not None:
            self.revoked_peak_clearance.copy_(torch.where(
                ~valid_jump, self.peak_clearance, 0.0
            ))
            self.revoked_settled_clearance.copy_(self.revoked_peak_clearance * self.height_settled)
            self.peak_clearance.mul_(valid_jump)
            airborne = airborne & valid_jump
            takeoff_eligible = takeoff_eligible & valid_jump
        # 结算后的二次腾空不能更新峰值或补发合格起跳奖励。
        airborne = airborne & ~self.height_settled
        # 一轮内不连续的离地片段不能拼成一次合格腾空。
        self.airborne_steps.copy_(torch.where(airborne, self.airborne_steps + 1, 0))
        self.max_airborne_steps.copy_(torch.maximum(self.max_airborne_steps, self.airborne_steps))
        # 仅无接触可能来自主动收腿、接触抖动或车体下落；必须确认最后一个
        # 轮地接触拍的 base-link 仍在向上，才算真正起跳。
        qualified = (
            (self.airborne_steps >= self.min_airborne_steps)
            & takeoff_eligible
            & ~self.takeoff_rewarded
        )
        self.takeoff_event.copy_(qualified)
        self.takeoff_rewarded.logical_or_(qualified)
        # 只计实际腾空时的base 原点相对起跳地面的高度，地面支撑时不累计峰值。
        self.previous_peak_clearance.copy_(self.peak_clearance)
        self.peak_clearance.copy_(
            torch.where(airborne, torch.maximum(self.peak_clearance, clearance), self.peak_clearance)
        )
        settle = ((landing_event > 0.5) | (episode_steps >= self.horizon)) & ~self.height_settled
        self.height_settlement_event.copy_(settle)
        self.height_settled.logical_or_(settle)
