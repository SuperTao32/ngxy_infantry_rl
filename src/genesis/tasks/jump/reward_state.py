"""单次跳跃的奖励事件；与 Genesis 解耦，便于验证连续腾空和一次性结算。"""

import math

import torch


class JumpRewardState:
    def __init__(self, num_envs, *, dt, min_airborne_time_s, device, dtype):
        self.min_airborne_steps = max(1, math.ceil(min_airborne_time_s / dt - 1e-9))
        self.airborne_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.takeoff_rewarded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.height_settled = torch.zeros_like(self.takeoff_rewarded)
        self.peak_clearance = torch.zeros(num_envs, dtype=dtype, device=device)
        self.pending_peak_clearance = torch.zeros_like(self.peak_clearance)
        self.revoked_peak_clearance = torch.zeros_like(self.peak_clearance)
        self.revoked_settled_clearance = torch.zeros_like(self.peak_clearance)
        self.previous_peak_clearance = torch.zeros_like(self.peak_clearance)
        self.takeoff_event = torch.zeros_like(self.peak_clearance)
        self.height_settlement_event = torch.zeros_like(self.peak_clearance)

    def reset(self, env_idx=None):
        for value in (
            self.airborne_steps, self.takeoff_rewarded,
            self.height_settled, self.peak_clearance, self.pending_peak_clearance, self.revoked_peak_clearance,
            self.previous_peak_clearance,
            self.revoked_settled_clearance,
            self.takeoff_event, self.height_settlement_event,
        ):
            if env_idx is None:
                value.zero_()
            else:
                value.masked_fill_(env_idx, 0)

    def clear_events(self):
        for value in (
            self.takeoff_event, self.height_settlement_event,
            self.revoked_peak_clearance, self.revoked_settled_clearance,
        ):
            value.zero_()

    def update(
        self, airborne, clearance, landing_event, *, takeoff_eligible, active, ended, valid_jump
    ):
        # 高度进度已在腾空时发放；无论空中还是落地失稳都须一次性追回。
        # 缺高惩罚仅在此前已结算时补扣，否则结束时会按零高度完整结算。
        self.clear_events()
        valid_jump = valid_jump | ~active
        self.revoked_peak_clearance.copy_(torch.where(
            ~valid_jump, self.peak_clearance, 0.0
        ))
        self.revoked_settled_clearance.copy_(self.revoked_peak_clearance * self.height_settled)
        self.peak_clearance.mul_(valid_jump)
        airborne = airborne & valid_jump
        takeoff_eligible = takeoff_eligible & valid_jump
        # 结算后的二次腾空不能更新峰值或补发合格起跳奖励。
        airborne = airborne & ~self.height_settled & active
        # 一轮内不连续的离地片段不能拼成一次合格腾空。
        self.airborne_steps.copy_(torch.where(airborne, self.airborne_steps + 1, 0))
        # 仅无接触可能来自主动收腿、接触抖动或车体下落；必须确认最后一个
        # 轮地接触拍的 base-link 仍在向上，才算真正起跳。
        qualified = (
            (self.airborne_steps >= self.min_airborne_steps)
            & takeoff_eligible
            & ~self.takeoff_rewarded
        )
        self.takeoff_event.copy_(qualified)
        self.takeoff_rewarded.logical_or_(qualified)
        # 短暂卸载不立即领取整段站立高度的峰值奖励；连续双轮腾空合格后，
        # 再一次性认可该连续片段的峰值，避免漏掉资格确认前已经经过的顶点。
        self.pending_peak_clearance.copy_(torch.where(
            airborne, torch.maximum(self.pending_peak_clearance, clearance), 0.0
        ))
        self.previous_peak_clearance.copy_(self.peak_clearance)
        self.peak_clearance.copy_(
            torch.where(
                airborne & self.takeoff_rewarded,
                torch.maximum(self.peak_clearance, self.pending_peak_clearance), self.peak_clearance,
            )
        )
        settle = active & ((landing_event > 0.5) | ended) & ~self.height_settled
        self.height_settlement_event.copy_(settle)
        self.height_settled.logical_or_(settle)
