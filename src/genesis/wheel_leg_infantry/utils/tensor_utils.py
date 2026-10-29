"""环境配置转换与批量随机采样使用的无状态张量工具。"""

import torch


def sample_uniform(lower, upper, batch_shape):
    """在逐元素上下限之间均匀采样指定批次的张量。"""
    if lower.shape != upper.shape:
        raise ValueError(
            f"lower and upper must have matching shapes, got {lower.shape} and {upper.shape}"
        )
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape),
        dtype=lower.dtype,
        device=lower.device,
    ) + lower


def as_gain_tensor(value, count, name, *, device, dtype=torch.float32):
    """将标量或定长序列转换为执行器增益张量。"""
    if isinstance(value, (int, float)):
        values = [float(value)] * count
    else:
        values = list(value)
        if len(values) != count:
            raise ValueError(f"{name} requires {count} values, got {len(values)}")
    return torch.tensor(values, dtype=dtype, device=device)


def as_range_tensors(values, num_axes, name, *, device, dtype=torch.float32):
    """校验各轴上下限并分别返回 lower、upper 张量。"""
    limits = torch.tensor(values, dtype=dtype, device=device)
    expected_shape = (2,) if num_axes == 1 else (num_axes, 2)
    if limits.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(limits.shape)}")
    if num_axes == 1:
        limits = limits.unsqueeze(0)
    if not torch.isfinite(limits).all():
        raise ValueError(f"{name} must contain only finite values")
    lower, upper = limits[:, 0], limits[:, 1]
    if torch.any(lower > upper):
        raise ValueError(f"every lower bound in {name} must be <= its upper bound")
    return lower, upper
