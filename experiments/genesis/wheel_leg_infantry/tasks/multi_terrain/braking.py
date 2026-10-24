"""locomotion_v6/model_6100 的平地制动标定；运行时不依赖 logs 目录。

来源：version_0002/braking_eval/20261024_164833_149169，16 次成功试验。
每行：制动起点实际离地高度、实际速度、双轮最大前冲、含 0.5 s 确认的停车时间。
按目标高度分组，但查询使用每组实测高度范围；随机化关闭，每格仅一次。
"""

import torch


STOP_CONFIRMATION_S = 0.5
SAMPLES = (
    (
        (0.209614173, 0.489313932, 0.175640387, 2.870000000),
        (0.209529877, 1.010166817, 0.322170001, 2.570000000),
        (0.208968297, 1.506756623, 0.505879012, 2.500000000),
        (0.208360344, 2.020054454, 0.749282475, 2.570000000),
    ),
    (
        (0.255318433, 0.490003740, 0.169660255, 3.120000000),
        (0.254309148, 1.027870332, 0.318924018, 2.450000000),
        (0.253593594, 1.537979734, 0.504655801, 2.360000000),
        (0.252999455, 2.057480666, 0.749109336, 2.440000000),
    ),
    (
        (0.300257504, 0.495597023, 0.160616931, 2.200000000),
        (0.299645156, 1.026689539, 0.332486056, 1.670000000),
        (0.300211310, 1.541341177, 0.532704965, 1.760000000),
        (0.300012350, 2.054837115, 0.794840004, 1.880000000),
    ),
    (
        (0.379225105, 0.475250960, 0.170076796, 2.620000000),
        (0.379248619, 0.966162971, 0.340694986, 2.530000000),
        (0.379387200, 1.442933065, 0.543843155, 2.620000000),
        (0.379400134, 1.927560167, 0.825807365, 2.770000000),
    ),
)


class BrakingEnvelope:
    """速度线性插值，邻接高度取较大值；超出标定范围使用保守包络。"""

    def __init__(self, reference):
        self.samples = reference.new_tensor(SAMPLES)
        # 每组按速度变化的实际高度范围均被覆盖。
        self.height_low = self.samples[:, :, 0].amin(dim=1)
        self.height_high = self.samples[:, :, 0].amax(dim=1)

    def predict(self, speed, height):
        velocities = self.samples[:, :, 1].contiguous()
        query = speed.abs()[None, :].expand(self.samples.shape[0], -1).contiguous()
        hi = torch.searchsorted(velocities, query).clamp(1, velocities.shape[1] - 1)
        lo = hi - 1
        lower = velocities.gather(1, lo)
        upper = velocities.gather(1, hi)
        fraction = ((query - lower) / (upper - lower)).clamp(0, 1)
        distances = self.samples[:, :, 2]
        distance = distances.gather(1, lo).lerp(distances.gather(1, hi), fraction)
        times = self.samples[:, :, 3]
        # 时间并不随速度单调变化，使用邻接速度中较慢的停车时间。
        duration = torch.maximum(times.gather(1, lo), times.gather(1, hi))
        # 低速保留最低速标定量；超速不能静默截断制动距离。
        excess = (query / velocities[:, -1:]).clamp_min(1)
        distance = distance * excess.square()
        duration = duration * excess
        low = torch.searchsorted(self.height_low, height.contiguous(), right=True).sub(1).clamp(0, self.samples.shape[0] - 1)
        high = torch.searchsorted(self.height_high, height.contiguous()).clamp(0, self.samples.shape[0] - 1)
        index = torch.arange(speed.numel(), device=speed.device)
        d = torch.maximum(distance[low, index], distance[high, index])
        t = torch.maximum(duration[low, index], duration[high, index])
        outside = (height < self.height_low[0]) | (height > self.height_high[-1])
        return (torch.where(outside, distance.amax(dim=0), d),
                torch.where(outside, duration.amax(dim=0), t))
