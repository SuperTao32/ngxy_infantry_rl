"""外力脉冲的时长、目标选择、局部重置和关闭兼容性。"""

import unittest
from unittest.mock import Mock

import torch

from experiments.genesis.wheel_leg_infantry.core.push import PushRandomizer, default_push_cfg


class PushTests(unittest.TestCase):
    def make_push(self, *, enabled=True, num_envs=4, **overrides):
        cfg = default_push_cfg()
        cfg.update(interval_s_range=[.1, .1], duration_s=.04,
                   force_x_range=[10., 10.], force_y_range=[-5., -5.], force_z_range=[3., 3.])
        cfg.update(overrides)
        push = PushRandomizer(cfg, enabled=enabled)
        push.bind(solver=Mock(), links={name: Mock(idx=idx) for name, idx in
                  (("base", 11), ("left_wheel", 27), ("right_wheel", 43))},
                  dt=.02, num_envs=num_envs, reference=torch.zeros(1))
        push.reset(push.env_ids)
        return push

    def test_pulse_duration_and_repeat_interval(self):
        push = self.make_push(targets=["left_wheel"])
        for _ in range(4):
            push.before_step()
        push.solver.apply_links_external_force.assert_not_called()
        push.before_step()
        first = push.last_applied_force.clone()
        torch.testing.assert_close(first, torch.tensor([[10., -5., 3.]]).expand(4, -1))
        call = push.solver.apply_links_external_force.call_args
        self.assertEqual(call.kwargs["links_idx"], [27])
        self.assertEqual(call.kwargs["ref"], "link_com")
        self.assertFalse(call.kwargs["local"])
        torch.testing.assert_close(call.args[0][:, 0], first)
        push.before_step()
        torch.testing.assert_close(push.last_applied_force, first)
        self.assertEqual(push.solver.apply_links_external_force.call_count, 2)
        for _ in range(3):
            push.before_step()
            torch.testing.assert_close(push.last_applied_force, torch.zeros_like(first))
        self.assertEqual(push.solver.apply_links_external_force.call_count, 2)
        push.before_step()  # 第 10 拍触发下一次；不是每拍重新抽样方向。
        self.assertEqual(push.solver.apply_links_external_force.call_count, 3)

    def test_targets_are_per_environment_and_only_one_link_is_pushed(self):
        torch.manual_seed(17)
        push = self.make_push(num_envs=128)
        push.steps_until_next.fill_(1)
        push.before_step()
        selected = []
        for call in push.solver.apply_links_external_force.call_args_list:
            self.assertIn(call.kwargs["links_idx"], ([11], [27], [43]))
            selected.append(call.kwargs["envs_idx"])
        self.assertEqual(len(selected), 3)
        torch.testing.assert_close(torch.cat(selected).sort().values, push.env_ids)
        # 所有环境每次只出现一次，避免误把同一个力加到机身和两个轮子。
        self.assertEqual(torch.cat(selected).unique().numel(), 128)

    def test_partial_reset_drops_only_selected_active_pulse(self):
        push = self.make_push(targets=["right_wheel"])
        push.steps_until_next.fill_(1)
        push.before_step()
        force = push.force[1:].clone()
        countdown = push.steps_until_next[1:].clone()
        push.reset(torch.tensor([0]))
        torch.testing.assert_close(push.steps_until_next[1:], countdown)
        push.before_step()
        call = push.solver.apply_links_external_force.call_args
        torch.testing.assert_close(call.kwargs["envs_idx"], torch.tensor([1, 2, 3]))
        torch.testing.assert_close(call.args[0][:, 0], force)
        torch.testing.assert_close(push.last_applied_force[0], torch.zeros(3))

    def test_disabled_does_not_change_rng_or_call_solver(self):
        state = torch.random.get_rng_state().clone()
        push = self.make_push(enabled=False)
        push.before_step()
        push.reset(torch.tensor([0]))
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        self.assertFalse(push.solver.mock_calls)

    def test_seed_and_independent_timers(self):
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            push = self.make_push(num_envs=64, interval_s_range=[.04, .2],
                                  force_x_range=[-10., 10.], force_y_range=[-20., 20.], force_z_range=[-30., 30.])
            results.append((push.steps_until_next.clone(), []))
            self.assertGreater(push.steps_until_next.unique().numel(), 1)
            for _ in range(10):
                push.before_step()
                results[-1][1].append(push.last_applied_force.clone())
        torch.testing.assert_close(results[0], results[1])

    def test_three_axes_sample_independently_with_signed_bounds(self):
        torch.manual_seed(123)
        push = self.make_push(num_envs=128, force_x_range=[-10., 10.],
                              force_y_range=[0., 0.], force_z_range=[-30., 30.])
        push.steps_until_next.fill_(1)
        push.before_step()
        force = push.last_applied_force
        self.assertTrue(torch.all(force[:, 0].abs() <= 10.))
        self.assertTrue(torch.all(force[:, 2].abs() <= 30.))
        torch.testing.assert_close(force[:, 1], torch.zeros(128))
        for axis in (0, 2):
            self.assertTrue(torch.any(force[:, axis] > 0))
            self.assertTrue(torch.any(force[:, axis] < 0))
            self.assertGreater(force[:, axis].unique().numel(), 1)
        self.assertFalse(torch.equal(force[:, 0], force[:, 2] / 3))

    def test_invalid_config(self):
        for overrides in ({"targets": []}, {"targets": ["wheel"]}, {"targets": ["base", "base"]},
                          {"duration_s": 0.}, {"interval_s_range": [.01, .1]},
                          {"force_x_range": [3., -1.]}, {"force_y_range": [0., float("nan")]},
                          {"force_z_range": [float("-inf"), 3.]}, {"force_z_range": [0.]},
                          {"force_xy_norm_range": [0., 30.]}, {"duration_s": float("nan")}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.make_push(**overrides)
