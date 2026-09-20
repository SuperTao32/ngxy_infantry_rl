"""CPU regression checks: python -m unittest discover -s experiments/genesis/tests -v."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

import torch

from experiments.genesis.wheel_leg_infantry.core.domain_randomization import (
    DomainRandomizationManager,
    default_domain_rand_cfg,
)
from experiments.genesis.wheel_leg_infantry.tasks.locomotion.config import get_cfgs
from experiments.genesis.wheel_leg_infantry.tasks.jump.config_25cm import get_cfgs as jump25_cfgs
from experiments.genesis.wheel_leg_infantry.tasks.jump.config_45cm import get_cfgs as jump45_cfgs
from experiments.genesis.wheel_leg_infantry.core.curriculum import CurriculumManager


class DomainRandomizationTests(unittest.TestCase):
    def make_manager(self, config=None, batched=True, base_mass=18.):
        manager = DomainRandomizationManager(config)
        robot, ground, stair = (Mock(n_links=n) for n in (8, 1, 1))
        robot.get_dofs_damping.return_value = torch.full((2,), .01)
        robot.get_dofs_frictionloss.return_value = torch.full((2,), .01)
        manager.bind(
            robot=robot, friction_entities=(robot, ground, stair),
            base_link=Mock(idx_local=3, get_mass=Mock(return_value=base_mass)),
            motor_params={
                "joint_kp": torch.full((4,), 60.), "joint_kd": torch.full((4,), 3.),
                "wheel_kd": torch.full((2,), .25), "joint_force_limit": torch.full((4,), 40.),
                "wheel_force_limit": torch.full((2,), 5.),
            },
            joint_indices=torch.arange(4), wheel_indices=torch.arange(4, 6),
            num_envs=4, batched_dofs=batched,
            spring_params={"preload_force": 420., "stiffness": 1400., "damping": 50.}, num_springs=2,
            passive_joint_names=("left_front2_joint", "right_front2_joint"), passive_indices=torch.tensor([6, 7]),
            push_links={name: Mock(idx=idx) for idx, name in enumerate(("base", "left_wheel", "right_wheel"), start=10)},
        )
        return manager

    def test_disabled_and_empty_reset_preserve_rng_and_make_no_calls(self):
        for config in (None, default_domain_rand_cfg(enabled=False)):
            manager = self.make_manager(config, batched=False)
            state = torch.random.get_rng_state().clone()
            manager.reset(torch.arange(4))
            torch.testing.assert_close(torch.random.get_rng_state(), state)
            self.assertFalse(manager.robot.mock_calls)
            for name, nominal in manager.nominal_spring_params.items():
                torch.testing.assert_close(manager.gas_spring_values[name], torch.full((4, 2), nominal))
        manager = self.make_manager({"enabled": True})
        manager.reset(torch.empty(0, dtype=torch.long))
        self.assertFalse(manager.robot.mock_calls)

    def test_partial_reset_changes_only_selected_environments_and_both_contact_sides(self):
        manager = self.make_manager({"enabled": True})
        ids = torch.tensor([1, 3])
        manager.reset(ids)
        for name, scale in manager.motor_scales.items():
            torch.testing.assert_close(scale[[0, 2]], torch.ones_like(scale[[0, 2]]))
            self.assertTrue(torch.all((scale[ids] >= .9) & (scale[ids] <= 1.1)))
            self.assertFalse(torch.equal(scale[1], scale[3]))
            torch.testing.assert_close(manager.motor_values[name], manager.motor_nominals[name] * scale)
        torch.testing.assert_close(manager.friction_ratio[[0, 2]], torch.ones(2, 1))
        for entity in manager.friction_entities:
            args, kwargs = entity.set_friction_ratio.call_args
            torch.testing.assert_close(kwargs["envs_idx"], ids)
            torch.testing.assert_close(args[0], manager.friction_ratio[ids].expand(-1, entity.n_links))
        args, kwargs = manager.robot.set_dofs_kp.call_args
        torch.testing.assert_close(args[0], manager.motor_values["joint_kp"][ids])
        torch.testing.assert_close(kwargs["envs_idx"], ids)

    def test_profile_switch_and_resampling_never_compound_gains(self):
        manager = self.make_manager({"enabled": True, "motor_gains": {"joint_kp_range": [1.1, 1.1]}})
        manager.reset(torch.arange(4))
        original_scales = {name: value.clone() for name, value in manager.motor_scales.items()}
        friction = manager.friction_ratio.clone()
        for _ in range(3):
            manager.apply_motor_params({"joint_kp": torch.full((4,), 80.)})
            torch.testing.assert_close(manager.motor_values["joint_kp"], torch.full((4, 4), 88.))
            manager.apply_motor_params({"joint_kp": manager.nominal_motor_params["joint_kp"]})
            torch.testing.assert_close(manager.motor_values["joint_kp"], torch.full((4, 4), 66.))
        for name, scale in original_scales.items():
            torch.testing.assert_close(manager.motor_scales[name], scale)
        torch.testing.assert_close(manager.friction_ratio, friction)
        # 部分环境交回 teacher，其余继续 jump；reset 沿用各自当前控制模式。
        manager.apply_motor_params({"joint_kp": torch.full((4,), 80.)})
        manager.apply_motor_params({"joint_kp": torch.full((4,), 60.)}, torch.tensor([1]))
        manager.reset(torch.tensor([0, 1]))
        torch.testing.assert_close(manager.motor_values["joint_kp"][:, 0], torch.tensor([88., 66., 88., 88.]))
        torch.testing.assert_close(manager.nominal_motor_params["joint_kp"], torch.full((4,), 60.))

    def test_unbatched_profiles_and_force_limits_use_global_setter(self):
        manager = self.make_manager(batched=False)
        manager.apply_motor_params({"joint_kp": torch.full((4,), 80.), "joint_force_limit": torch.full((4,), 30.)})
        args, kwargs = manager.robot.set_dofs_kp.call_args
        self.assertEqual(args[0].shape, (4,))
        self.assertEqual(kwargs, {})
        lower, upper, _ = manager.robot.set_dofs_force_range.call_args.args
        torch.testing.assert_close(lower, -upper)
        torch.testing.assert_close(manager.motor_values["joint_force_limit"], torch.full((4, 4), 30.))

    def test_seed_reproduces_samples(self):
        first, second = (self.make_manager(default_domain_rand_cfg(enabled=True)) for _ in range(2))
        for manager in (first, second):
            torch.manual_seed(17)
            manager.reset(torch.arange(4))
        torch.testing.assert_close(first.friction_ratio, second.friction_ratio)
        for name in first.motor_scales:
            torch.testing.assert_close(first.motor_scales[name], second.motor_scales[name])
        for name in ("added_mass", "com_displacement", "motor_strengths", "motor_offsets"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))
        for name in first.gas_spring_scales:
            torch.testing.assert_close(first.gas_spring_scales[name], second.gas_spring_scales[name])

    def test_springs_sample_each_side_and_only_reset_selected_envs(self):
        manager = self.make_manager({"enabled": True, "friction": {"enabled": False},
                                     "motor_gains": {"enabled": False}, "gas_spring": {"enabled": True}}, batched=False)
        self.assertFalse(manager.requires_batched_dofs)
        manager.reset(torch.tensor([1, 3]))
        for name, scales in manager.gas_spring_scales.items():
            torch.testing.assert_close(scales[[0, 2]], torch.ones(2, 2))
            self.assertTrue(torch.all((scales >= .9) & (scales <= 1.1)))
            self.assertFalse(torch.equal(scales[[1, 3], 0], scales[[1, 3], 1]))
            torch.testing.assert_close(manager.gas_spring_values[name], manager.nominal_spring_params[name] * scales)
        before = {name: value.clone() for name, value in manager.gas_spring_values.items()}
        manager.apply_motor_params({"joint_kp": torch.full((4,), 80.)})
        for name, value in before.items():
            torch.testing.assert_close(manager.gas_spring_values[name], value)

    def test_spring_resets_use_nominal_values_and_preserve_buffer_references(self):
        manager = self.make_manager({"enabled": True, "gas_spring": {"enabled": True,
            "preload_force_range": [1.1, 1.1], "stiffness_range": [.9, .9], "damping_range": [1.2, 1.2]}})
        buffers = dict(manager.gas_spring_values)
        for _ in range(3):
            manager.reset()
            for name, expected in (("preload_force", 462.), ("stiffness", 1260.), ("damping", 60.)):
                self.assertIs(manager.gas_spring_values[name], buffers[name])
                torch.testing.assert_close(buffers[name], torch.full((4, 2), expected))
        legacy = self.make_manager({"enabled": True})
        legacy.reset()
        self.assertFalse(legacy.gas_spring_enabled)
        torch.testing.assert_close(legacy.gas_spring_values["preload_force"], torch.full((4, 2), 420.))

    def test_spring_ranges_reject_invalid_values(self):
        for limits in ([1., .9], [-.1, 1.], [0., 1.], [1., float("nan")]):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                DomainRandomizationManager({"gas_spring": {"damping_range": limits}})

    def test_passive_joint_overrides_and_partial_reset(self):
        manager = self.make_manager({"enabled": True, "motor_gains": {"enabled": False},
            "passive_joints": {"enabled": True, "damping_range": [.02, .02], "frictionloss_range": [.03, .03],
                "overrides": {"left_front2_joint": {"damping_range": [.07, .07], "frictionloss_range": [0., 0.]}}}})
        self.assertTrue(manager.requires_batched_dofs)
        for _ in range(2):
            manager.reset(torch.tensor([1, 3]))
            torch.testing.assert_close(manager.passive_joint_values["damping"],
                                       torch.tensor([[.01, .01], [.07, .02], [.01, .01], [.07, .02]]))
            torch.testing.assert_close(manager.passive_joint_values["frictionloss"],
                                       torch.tensor([[.01, .01], [0., .03], [.01, .01], [0., .03]]))
        for name in ("damping", "frictionloss"):
            args, kwargs = getattr(manager.robot, f"set_dofs_{name}").call_args
            torch.testing.assert_close(args[0], manager.passive_joint_values[name][[1, 3]])
            torch.testing.assert_close(args[1], torch.tensor([6, 7]))
            torch.testing.assert_close(kwargs["envs_idx"], torch.tensor([1, 3]))

    def test_passive_joint_config_validation_and_legacy_compatibility(self):
        for config in ({"damping_range": [-.01, .01]}, {"frictionloss_range": [1., .5]},
                       {"overrides": {"left_front2_joint": {"damping_range": [0., float("inf")]}}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                DomainRandomizationManager({"passive_joints": config})
        with self.assertRaisesRegex(ValueError, "unknown or controlled"):
            self.make_manager({"enabled": True, "passive_joints": {"enabled": True,
                               "overrides": {"left_front1_joint": {"damping_range": [.01, .02]}}}})
        with self.assertRaisesRegex(ValueError, "batch_dofs_info"):
            self.make_manager({"enabled": True, "motor_gains": {"enabled": False},
                               "passive_joints": {"enabled": True}}, batched=False)
        manager = self.make_manager({"enabled": True})
        manager.reset()
        self.assertFalse(manager.passive_joints_enabled)
        manager.robot.set_dofs_damping.assert_not_called()
        manager.robot.set_dofs_frictionloss.assert_not_called()

    def test_extended_randomization_is_local_and_uses_named_base(self):
        manager = self.make_manager(default_domain_rand_cfg(enabled=True))
        manager.reset(torch.tensor([1, 3]))
        torch.testing.assert_close(manager.added_mass[[0, 2]], torch.zeros(2, 1))
        torch.testing.assert_close(manager.com_displacement[[0, 2]], torch.zeros(2, 1, 3))
        torch.testing.assert_close(manager.motor_offsets[[0, 2]], torch.zeros(2, 4))
        torch.testing.assert_close(manager.motor_strengths[[0, 2]], torch.ones(2, 1))
        self.assertTrue(torch.all(manager.added_mass.abs() <= 1.))
        self.assertTrue(torch.all(manager.com_displacement.abs() <= .01))
        self.assertTrue(torch.all(manager.motor_offsets.abs() <= .02))
        self.assertFalse(torch.equal(manager.motor_offsets[1], manager.motor_offsets[3]))
        for setter, values in ((manager.robot.set_mass_shift, manager.added_mass),
                               (manager.robot.set_COM_shift, manager.com_displacement)):
            args, kwargs = setter.call_args
            self.assertEqual(args[1], [3])
            torch.testing.assert_close(args[0], values[[1, 3]])
            torch.testing.assert_close(kwargs["envs_idx"], torch.tensor([1, 3]))

    def test_strength_scales_gains_and_limits_without_compounding(self):
        manager = self.make_manager({"enabled": True, "motor_gains": {"enabled": False},
                                     "motor_strength": {"enabled": True, "ratio_range": [.8, .8]},
                                     "motor_offset": {"enabled": True, "offset_range": [.02, .02]}})
        manager.reset()
        for _ in range(3):
            manager.apply_motor_params({"joint_kp": torch.full((4,), 80.)})
            manager.reset(torch.tensor([1, 3]))
            torch.testing.assert_close(manager.motor_values["joint_kp"], torch.full((4, 4), 64.))
            torch.testing.assert_close(manager.motor_values["joint_force_limit"], torch.full((4, 4), 32.))
            torch.testing.assert_close(manager.motor_values["wheel_kd"], torch.full((4, 2), .2))
            torch.testing.assert_close(manager.motor_values["wheel_force_limit"], torch.full((4, 2), 4.))
        torch.testing.assert_close(manager.offset_joint_targets(torch.full((4, 4), .1)), torch.full((4, 4), .12))
        self.assertTrue(manager.requires_batched_dofs)
        with self.assertRaisesRegex(ValueError, "batch_dofs_info"):
            self.make_manager({"enabled": True, "motor_gains": {"enabled": False},
                               "motor_strength": {"enabled": True}}, batched=False)

    def test_individual_gain_switches_and_legacy_defaults(self):
        manager = self.make_manager({"enabled": True, "motor_gains": {"joint_kp_enabled": False,
                                                                                    "wheel_kd_enabled": False}})
        manager.reset()
        torch.testing.assert_close(manager.motor_scales["joint_kp"], torch.ones(4, 4))
        torch.testing.assert_close(manager.motor_scales["wheel_kd"], torch.ones(4, 2))
        self.assertFalse(torch.equal(manager.motor_scales["joint_kd"], torch.ones(4, 4)))
        self.assertFalse(manager.base_mass_enabled)
        self.assertFalse(manager.motor_strength_enabled)
        torch.testing.assert_close(manager.motor_offsets, torch.zeros(4, 4))

    def test_mass_range_rejects_nonpositive_result_and_offsets_allow_negative(self):
        with self.assertRaisesRegex(ValueError, "mass.*positive"):
            self.make_manager({"enabled": True, "base_mass": {"enabled": True, "added_mass_range": [-18., 1.]}})
        manager = self.make_manager({"enabled": True, "base_mass": {"enabled": True, "added_mass_range": [-1., -1.]},
                                     "motor_offset": {"enabled": True, "offset_range": [-.02, -.02]}})
        manager.reset()
        manager.reset()
        torch.testing.assert_close(manager.added_mass, -torch.ones(4, 1))
        torch.testing.assert_close(manager.motor_offsets, torch.full((4, 4), -.02))

    def test_config_validation_and_independent_task_defaults(self):
        for bad in ({"typo": True}, {"enabled": 1}, {"friction": {"ratio_range": [0, 1]}},
                    {"motor_gains": {"joint_kp_range": [2, 1]}},
                    {"friction": {"ratio_range": [1, float("nan")]}}, {"friction": None}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DomainRandomizationManager(bad)
        locomotion = get_cfgs()
        self.assertTrue(locomotion[0]["domain_rand"]["enabled"])
        for cfgs in (jump25_cfgs, jump45_cfgs):
            jump = cfgs(locomotion)
            self.assertFalse(jump[0]["domain_rand"]["enabled"])
            jump[0]["domain_rand"]["friction"]["ratio_range"][0] = .5
            self.assertEqual(locomotion[0]["domain_rand"]["friction"]["ratio_range"], [.8, 1.2])

    def test_curriculum_zero_is_nominal_and_does_not_sample(self):
        manager = self.make_manager(default_domain_rand_cfg(enabled=True))
        curriculum = CurriculumManager({"enabled": True, "stages": [
            {"name": "stand", "start_iteration": 0, "targets": {"domain_rand": {"strength": 0.}}},
            {"name": "randomize", "start_iteration": 10, "targets": {"domain_rand": {"strength": 1.}}},
        ]})
        curriculum.register_target("domain_rand", manager.apply_curriculum)
        curriculum.update(0)
        rng = torch.random.get_rng_state().clone()
        manager.reset()
        manager.before_step()
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        torch.testing.assert_close(manager.friction_ratio, torch.ones(4, 1))
        for name in ("added_mass", "com_displacement", "motor_offsets"):
            value = getattr(manager, name)
            torch.testing.assert_close(value, torch.zeros_like(value))
        for name, value in manager.motor_values.items():
            torch.testing.assert_close(value, manager.nominal_motor_params[name].expand_as(value))
        for name, value in manager.gas_spring_values.items():
            torch.testing.assert_close(value, torch.full_like(value, manager.nominal_spring_params[name]))
        for name, value in manager.passive_joint_values.items():
            torch.testing.assert_close(value, manager.nominal_passive_joint_values[name])
        manager.robot.solver.apply_links_external_force.assert_not_called()
        self.assertTrue(curriculum.update(10))
        # 目标已变，正在运行的回合仍是标称参数。
        torch.testing.assert_close(manager.episode_strength, torch.zeros(4))
        manager.reset(torch.tensor([1]))
        torch.testing.assert_close(manager.episode_strength, torch.tensor([0., 1., 0., 0.]))
        self.assertFalse(torch.equal(manager.motor_scales["joint_kp"][1], torch.ones(4)))
        torch.testing.assert_close(manager.motor_scales["joint_kp"][[0, 2, 3]], torch.ones(3, 4))

    def test_half_strength_interpolates_each_parameter_about_its_nominal(self):
        cfg = default_domain_rand_cfg(enabled=True)
        cfg["friction"]["ratio_range"] = [1.2, 1.2]
        cfg["base_mass"]["added_mass_range"] = [1., 1.]
        cfg["com_displacement"]["displacement_range"] = [.02, .02]
        cfg["motor_strength"]["ratio_range"] = [.8, .8]
        cfg["motor_offset"]["offset_range"] = [.04, .04]
        cfg["motor_gains"]["joint_kp_range"] = [1.2, 1.2]
        cfg["gas_spring"]["preload_force_range"] = [1.1, 1.1]
        cfg["passive_joints"].update(damping_range=[.03, .03], frictionloss_range=[.05, .05])
        cfg["push"].update(force_x_range=[10., 10.], force_y_range=[-20., -20.], force_z_range=[30., 30.])
        manager = self.make_manager(cfg)
        original = deepcopy(manager.config)
        manager.apply_curriculum({"strength": .5})
        manager.reset()
        torch.testing.assert_close(manager.friction_ratio, torch.full((4, 1), 1.1))
        torch.testing.assert_close(manager.added_mass, torch.full((4, 1), .5))
        torch.testing.assert_close(manager.com_displacement, torch.full((4, 1, 3), .01))
        torch.testing.assert_close(manager.motor_strengths, torch.full((4, 1), .9))
        torch.testing.assert_close(manager.motor_offsets, torch.full((4, 4), .02))
        torch.testing.assert_close(manager.motor_values["joint_kp"], torch.full((4, 4), 59.4))
        torch.testing.assert_close(manager.gas_spring_values["preload_force"], torch.full((4, 2), 441.))
        torch.testing.assert_close(manager.passive_joint_values["damping"], torch.full((4, 2), .02))
        torch.testing.assert_close(manager.passive_joint_values["frictionloss"], torch.full((4, 2), .03))
        manager.push.steps_until_next.fill_(1)
        manager.before_step()
        torch.testing.assert_close(manager.push.last_applied_force, torch.tensor([[5., -10., 15.]]).expand(4, -1))
        manager.apply_curriculum({"strength": 1.})
        manager.reset(torch.tensor([0]))
        torch.testing.assert_close(manager.push.strength, torch.tensor([1., .5, .5, .5]))
        self.assertEqual(manager.config, original)
        # 降回较低强度仍从原始范围计算，不会把已经收缩的范围再缩一次。
        manager.apply_curriculum({"strength": .5})
        manager.reset()
        torch.testing.assert_close(manager.motor_values["joint_kp"], torch.full((4, 4), 59.4))

    def test_curriculum_strength_validation_and_disabled_flags(self):
        for value in (-.1, 1.1, float("nan"), True, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DomainRandomizationManager({"strength": value})
        manager = self.make_manager(default_domain_rand_cfg(enabled=False), batched=False)
        with self.assertRaises(ValueError):
            manager.apply_curriculum({"enabled": True})
        manager.apply_curriculum({"strength": .5})
        manager.reset()
        manager.before_step()
        self.assertFalse(manager.robot.mock_calls)
        torch.testing.assert_close(manager.episode_strength, torch.zeros(4))


if __name__ == "__main__":
    unittest.main()
