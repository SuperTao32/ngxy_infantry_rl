"""运行：.venv/bin/python -m unittest discover -s tools/sysid -p 'test_*.py'。"""
import unittest

import numpy as np

from filtering import lowpass


class FilteringTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(4000) / 1000.0

    def test_noise_reduction_and_phase(self):
        clean = np.sin(2 * np.pi * 5 * self.t)
        noise = 0.5 * np.sin(2 * np.pi * 200 * self.t)
        result = lowpass(self.t, clean + noise)
        # 避开端点，检查噪声抑制及低频信号幅值/相位保真。
        self.assertLess(np.max(np.abs(result[500:-500] - clean[500:-500])), 1e-4)

    def test_constant_and_disabled(self):
        constant = np.full((len(self.t), 2), 3.0)
        np.testing.assert_allclose(lowpass(self.t, constant), constant, atol=1e-12)
        raw = np.sin(self.t * 1000)
        np.testing.assert_array_equal(lowpass(self.t, raw, cutoff_hz=None), raw)

    def test_smoothed_angle_derivative(self):
        angle = np.sin(2 * np.pi * 5 * self.t) + 0.01 * np.sin(2 * np.pi * 200 * self.t)
        velocity = np.gradient(lowpass(self.t, angle), self.t)
        expected = 2 * np.pi * 5 * np.cos(2 * np.pi * 5 * self.t)
        self.assertLess(np.max(np.abs(velocity[500:-500] - expected[500:-500])), 0.01)

    def test_invalid_inputs(self):
        for cutoff in (0, -1, 500, float('nan')):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                lowpass(self.t, self.t, cutoff_hz=cutoff)
        with self.assertRaises(ValueError):
            lowpass(self.t * 2, self.t)
        with self.assertRaises(ValueError):
            lowpass(self.t[:10], self.t[:10])
        with self.assertRaises(ValueError):
            lowpass(self.t, np.full_like(self.t, np.nan))


if __name__ == '__main__':
    unittest.main()
