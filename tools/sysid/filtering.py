"""SysID 离线低通滤波，时间单位为秒。"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def lowpass(t, values, sample_rate_hz=500.0, cutoff_hz=50.0):
    """四阶 Butterworth 双向零相位滤波；cutoff_hz=None 关闭。

    沿第 0 维（时间）滤波，要求等间隔采样；允许 5% 时间戳抖动。
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("filter input contains NaN or infinity")
    if cutoff_hz is None:
        return values.copy()
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not np.isfinite(cutoff_hz) or not 0 < cutoff_hz < sample_rate_hz / 2:
        raise ValueError("filter_cutoff_hz must be between 0 and sample_rate_hz / 2")
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or values.ndim == 0 or len(t) != len(values):
        raise ValueError("filter timestamps must match signal length")
    if len(t) <= 15:
        raise ValueError("zero-phase filtering requires at least 16 samples")
    if not np.all(np.isfinite(t)) or not np.allclose(
        np.diff(t), 1.0 / sample_rate_hz, rtol=0.05, atol=1e-9
    ):
        raise ValueError("timestamps must be uniformly sampled at sample_rate_hz (5% tolerance)")
    sos = butter(4, cutoff_hz, fs=sample_rate_hz, output="sos")
    return sosfiltfilt(sos, values, axis=0)
