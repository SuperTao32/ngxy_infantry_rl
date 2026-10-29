"""每种地形/难度的地形前停车门槛；PPO 轮次边界合并多卡统计。"""

from copy import deepcopy
import math

import torch
import torch.distributed as dist


DEFAULTS = dict(enabled=False, window_episodes=100, success_threshold=0.9,
                replay_probability=0.2, required_speed_m_s=0.2)


def resolve_parking(options):
    if not isinstance(options, dict) or set(options) - set(DEFAULTS):
        raise ValueError("invalid parking_curriculum options")
    cfg = {**DEFAULTS, **options}
    if type(cfg["enabled"]) is not bool or type(cfg["window_episodes"]) is not int or cfg["window_episodes"] < 1:
        raise ValueError("parking_curriculum requires boolean enabled and positive integer window_episodes")
    for key in ("success_threshold", "replay_probability", "required_speed_m_s"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid parking_curriculum.{key}")
        if key == "replay_probability":
            valid = 0 <= value < 1
        elif key == "success_threshold":
            valid = 0 < value <= 1
        else:
            valid = value > 0
        if not valid:
            raise ValueError(f"invalid parking_curriculum.{key}")
    return cfg


class ParkingCurriculum:
    def __init__(self, env, *, evaluation=False):
        self.config = resolve_parking(env.env_cfg["multi_terrain"].get("parking_curriculum", {}))
        self.enabled = self.config["enabled"] and not evaluation
        self.variants = list(env.terrain.variants)
        n = len(self.variants)
        self.unlocked = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.counts = torch.zeros(n, dtype=torch.long, device=env.device)
        self.wins = torch.zeros_like(self.counts)
        self.rates = torch.zeros(n, device=env.device)
        self.pending = torch.zeros(n, 2, dtype=torch.long, device=env.device)  # count, wins
        self.drill = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.motion_ready = torch.zeros_like(self.drill)

    def select(self, env, ids, tiles):
        """只在物理 reset 选择；已在途的练习不会因其他环境解锁而改变。"""
        self.drill[ids] = False
        self.motion_ready[ids] = False
        if not self.enabled:
            return
        traversal = torch.tensor([t.kind != "locomotion" for t in env.terrain.tracks], device=env.device)[tiles]
        self.drill[ids] = traversal & (~self.unlocked[tiles] | (torch.rand(len(ids), device=env.device) < self.config["replay_probability"]))

    def observe(self, tiles, success):
        if not self.enabled or tiles.numel() == 0:
            return
        index = tiles.long()
        self.pending[:, 0].scatter_add_(0, index, torch.ones_like(index))
        self.pending[:, 1].scatter_add_(0, index, success.long())

    def synchronize(self):
        """所有 worker 每轮都调用，包括没有结束回合的 worker。"""
        if not self.enabled:
            return
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.pending, op=dist.ReduceOp.SUM)
        self.counts += self.pending[..., 0]
        self.wins += self.pending[..., 1]
        self.pending.zero_()
        complete = self.counts >= self.config["window_episodes"]
        rates = self.wins.float() / self.counts.clamp_min(1)
        self.rates[complete] = rates[complete]
        self.unlocked |= complete & (rates >= self.config["success_threshold"])
        self.counts[complete] = 0
        self.wins[complete] = 0

    def state_dict(self):
        return dict(config=deepcopy(self.config), variants=list(self.variants),
                    **{key: getattr(self, key).cpu().tolist()
                       for key in ("unlocked", "counts", "wins", "rates", "pending")})

    def load_state_dict(self, state):
        if state.get("config") != self.config or state.get("variants") != self.variants:
            raise ValueError("saved parking curriculum differs; use --load-weights for a new course")
        restored = {}
        for key in ("unlocked", "counts", "wins", "rates", "pending"):
            target = getattr(self, key)
            value = torch.as_tensor(state[key], device=target.device)
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid parking checkpoint {key}")
            if key == "unlocked":
                valid = ((value == 0) | (value == 1)).all()
            elif key == "rates":
                valid = ((value >= 0) & (value <= 1)).all()
            else:
                valid = ((value >= 0) & (value == value.long())).all()
            if not valid:
                raise ValueError(f"invalid parking checkpoint {key}")
            restored[key] = value.to(target.dtype)
        if (restored["wins"] > restored["counts"]).any() or (restored["pending"][..., 1] > restored["pending"][..., 0]).any():
            raise ValueError("invalid parking checkpoint successes")
        for key, value in restored.items():
            getattr(self, key).copy_(value)
