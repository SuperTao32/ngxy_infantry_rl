"""Opt-in real Genesis CPU checks: NGXY_GENESIS_SMOKE=1 python -m unittest discover -s experiments/genesis/tests -v."""

import os
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from experiments.genesis.wheel_leg_infantry.tasks.locomotion.config import get_cfgs
from experiments.genesis.wheel_leg_infantry.tasks.jump.config_25cm import get_cfgs as jump_cfgs


@unittest.skipUnless(os.environ.get("NGXY_GENESIS_SMOKE") == "1", "set NGXY_GENESIS_SMOKE=1 for real Genesis CPU simulation")
class GenesisDomainRandomizationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import genesis as gs
        cls.gs = gs
        gs.init(backend=gs.cpu, precision="32", logging_level="warning", seed=17)

    @classmethod
    def tearDownClass(cls):
        cls.gs.destroy()

    def make_env(self, cfg, jump=False, use_domain_curriculum=False):
        from experiments.genesis.wheel_leg_infantry.tasks.locomotion.env import LocomotionEnv
        from experiments.genesis.wheel_leg_infantry.tasks.jump.env import JumpEnv
        cfg[0]["show_FPS"] = False
        curriculum_cfg = deepcopy(cfg[4])
        if not use_domain_curriculum:
            # 单项参数测试验证完整范围；课程渐进语义由专门的集成测试覆盖。
            for stage in curriculum_cfg.get("stages", []):
                stage.get("targets", {}).pop("domain_rand", None)
        env = (JumpEnv if jump else LocomotionEnv)(2, *cfg[:4], curriculum_cfg=curriculum_cfg)
        self.addCleanup(env.scene.destroy)
        return env

    def assert_solver_gains(self, env):
        kp = env.robot.get_dofs_kp(env.joints_dof_idx)
        # Genesis 的速度执行器不能使用要求位置 PD 形式的 get_dofs_kv。
        wheel_kd = -env.robot.get_dofs_act_bias(env.wheels_dof_idx)[2]
        torch.testing.assert_close(kp, env.joint_kp if kp.ndim == 2 else env.joint_kp[0])
        torch.testing.assert_close(wheel_kd, env.wheel_kd if wheel_kd.ndim == 2 else env.wheel_kd[0])

    def test_locomotion_step_and_partial_reset(self):
        env = self.make_env(get_cfgs())
        for _ in range(2):
            obs, *_ = env.step(torch.zeros((2, 6)))
            self.assertTrue(torch.isfinite(obs["policy"]).all())
        old_gain, old_friction = env.joint_kp[1].clone(), env.domain_rand.friction_ratio[1].clone()
        env._reset_idx(torch.tensor([True, False]))
        torch.testing.assert_close(env.joint_kp[1], old_gain)
        torch.testing.assert_close(env.domain_rand.friction_ratio[1], old_friction)
        self.assert_solver_gains(env)
        self.assertEqual(env.get_pd_diagnostics()["joint_kp"].shape, (4,))

    def test_jump_handoff_preserves_randomization_and_action_history(self):
        cfg = jump_cfgs()
        cfg[0]["domain_rand"]["enabled"] = True
        cfg[0]["handoff_on_landing"] = True
        cfg[4]["stages"][0]["targets"]["domain_rand"] = {"strength": .3}
        env = self.make_env(cfg, jump=True, use_domain_curriculum=True)
        torch.testing.assert_close(env.domain_rand.episode_strength, torch.full((2,), .3))
        scales = (env.domain_rand.motor_scales["joint_kp"] * env.domain_rand.motor_strengths).clone()
        offsets = env.domain_rand.motor_offsets.clone()
        added_mass = env.domain_rand.added_mass.clone()
        com = env.domain_rand.com_displacement.clone()
        friction = env.domain_rand.friction_ratio.clone()
        springs = {name: value.clone() for name, value in env.domain_rand.gas_spring_values.items()}
        passive = {name: value.clone() for name, value in env.domain_rand.passive_joint_values.items()}
        push_countdown = env.domain_rand.push.steps_until_next.clone()
        env.last_actions.fill_(.2)
        env.begin_jump_rollout()
        torch.testing.assert_close(env.domain_rand.push.steps_until_next, push_countdown)
        torch.testing.assert_close(env.joint_kp, 80 * scales)
        torch.testing.assert_close(env.domain_rand.friction_ratio, friction)
        torch.testing.assert_close(env.domain_rand.motor_offsets, offsets)
        torch.testing.assert_close(env.domain_rand.added_mass, added_mass)
        torch.testing.assert_close(env.domain_rand.com_displacement, com)
        torch.testing.assert_close(env.last_actions, torch.full((2, 6), .2))
        for name, value in springs.items():
            torch.testing.assert_close(env.domain_rand.gas_spring_values[name], value)
        for name, value in passive.items():
            torch.testing.assert_close(getattr(env.robot, f"get_dofs_{name}")(env.passive_dof_idx), value)
        env._apply_motor_params(env._locomotion_motor_params, torch.tensor([0]))
        torch.testing.assert_close(env.joint_kp[0], 60 * scales[0])
        torch.testing.assert_close(env.joint_kp[1], 80 * scales[1])
        self.assert_solver_gains(env)
        obs, *_ = env.step(torch.zeros((2, 6)))
        self.assertTrue(torch.isfinite(obs["policy"]).all())
        env.set_collect_jump_data(False)
        torch.testing.assert_close(env.joint_kp, 60 * scales)
        for name, value in springs.items():
            torch.testing.assert_close(env.domain_rand.gas_spring_values[name], value)

    def test_gas_spring_parameters_drive_actual_force_and_partial_reset(self):
        cfg = get_cfgs()
        cfg[0]["domain_rand"]["gas_spring"].update(
            preload_force_range=[1.1, 1.1], stiffness_range=[.9, .9], damping_range=[1.2, 1.2])
        env = self.make_env(cfg)
        env.robot.set_dofs_position(torch.tensor([[.03, .01], [.03, .01]]), env.springs_dof_idx)
        env.robot.set_dofs_velocity(torch.tensor([[.1, -.2], [.1, -.2]]), env.springs_dof_idx)
        env._apply_gas_spring_compensation()
        # F = 462 + 1260 * (0.06 - x) - 60 * v；与电机强度随机倍率无关。
        expected = torch.tensor([[493.8, 537.], [493.8, 537.]])
        torch.testing.assert_close(env.gas_spring_force, expected)
        torch.testing.assert_close(env.robot.get_dofs_control_force(env.springs_dof_idx), expected)
        env.domain_rand.config["gas_spring"]["preload_force_range"] = [.9, .9]
        env._reset_idx(torch.tensor([True, False]))
        torch.testing.assert_close(env.gas_spring_preload_force, torch.tensor([[378., 378.], [462., 462.]]))
        obs, *_ = env.step(torch.zeros(2, 6))
        self.assertTrue(torch.isfinite(obs["policy"]).all())

    def test_passive_joint_solver_values_and_protected_dofs(self):
        cfg = get_cfgs()
        cfg[0]["domain_rand"] = {"enabled": True, "friction": {"enabled": False},
            "motor_gains": {"enabled": False}, "passive_joints": {"enabled": True,
                "damping_range": [.005, .015], "frictionloss_range": [.005, .015],
                "overrides": {"left_front2_joint": {"damping_range": [.07, .07], "frictionloss_range": [0., 0.]}}}}
        env = self.make_env(cfg)
        expected_names = {f"{side}_{part}_joint" for side in ("left", "right")
                          for part in ("front2", "front3", "front4", "rear2", "spring1")}
        self.assertEqual(set(env.passive_joint_names), expected_names)
        self.assertTrue(env.batch_dofs_info)
        left = env.passive_joint_names.index("left_front2_joint")
        protected = torch.cat((env.joints_dof_idx, env.wheels_dof_idx, env.springs_dof_idx))
        before = {}
        for name in ("damping", "frictionloss"):
            getter = getattr(env.robot, f"get_dofs_{name}")
            before[name] = (getter(env.passive_dof_idx).clone(), getter(protected).clone())
            torch.testing.assert_close(before[name][0], env.domain_rand.passive_joint_values[name])
        torch.testing.assert_close(before["damping"][0][:, left], torch.full((2,), .07))
        torch.testing.assert_close(before["frictionloss"][0][:, left], torch.zeros(2))
        # 初始随机化没有覆盖电机、轮子或气弹簧滑动关节的模型参数。
        torch.testing.assert_close(before["damping"][1], torch.zeros(2, 8))
        torch.testing.assert_close(before["frictionloss"][1], torch.tensor([[1.5, 1.5, 0., 0., .023, .023, 0., 0.]]).expand(2, -1))
        env._reset_idx(torch.tensor([True, False]))
        for name, (old_passive, old_protected) in before.items():
            getter = getattr(env.robot, f"get_dofs_{name}")
            torch.testing.assert_close(getter(env.passive_dof_idx)[1], old_passive[1])
            self.assertFalse(torch.equal(getter(env.passive_dof_idx)[0], old_passive[0]))
            torch.testing.assert_close(getter(protected), old_protected)
        obs, *_ = env.step(torch.zeros(2, 6))
        self.assertTrue(torch.isfinite(obs["policy"]).all())

    def test_mass_com_offsets_and_actual_motor_torque(self):
        cfg = get_cfgs()
        dr = cfg[0]["domain_rand"]
        dr["base_mass"]["added_mass_range"] = [.5, .5]
        dr["com_displacement"]["displacement_range"] = [.005, .005]
        dr["motor_strength"]["ratio_range"] = [.8, .8]
        dr["motor_offset"]["offset_range"] = [.02, .02]
        dr["motor_gains"]["enabled"] = False
        env = self.make_env(cfg)
        base_id = env.robot.get_link(cfg[0]["base_link_name"]).idx
        solver = env.robot.solver
        torch.testing.assert_close(solver.get_links_mass_shift([base_id]), torch.full((2, 1), .5))
        torch.testing.assert_close(solver.get_links_COM_shift([base_id]), torch.full((2, 1, 3), .005))
        env.robot.control_dofs_position(env.joint_pos + .1, env.joints_dof_idx)
        env.robot.control_dofs_velocity(torch.ones(2, 2), env.wheels_dof_idx)
        torch.testing.assert_close(env.robot.get_dofs_control_force(env.joints_dof_idx), torch.full((2, 4), 4.8))
        torch.testing.assert_close(env.robot.get_dofs_control_force(env.wheels_dof_idx), torch.full((2, 2), .2))
        env.robot.control_dofs_position(env.joint_pos + 100., env.joints_dof_idx)
        env.robot.control_dofs_velocity(torch.full((2, 2), 100.), env.wheels_dof_idx)
        torch.testing.assert_close(env.robot.get_dofs_control_force(env.joints_dof_idx), torch.full((2, 4), 32.))
        torch.testing.assert_close(env.robot.get_dofs_control_force(env.wheels_dof_idx), torch.full((2, 2), 4.))
        env.step(torch.zeros(2, 6))
        torch.testing.assert_close(env.target_joint_pos, env.default_joint_pos.expand(2, -1) + .02)
        torch.testing.assert_close(env.target_wheel_vel, torch.zeros(2, 2))
        # 使用非连续 solver link 编号，验证只更新基座与所选环境，重复 reset 不累加质量。
        env.domain_rand.config["base_mass"]["added_mass_range"] = [-.5, -.5]
        env._reset_idx(torch.tensor([True, False]))
        torch.testing.assert_close(solver.get_links_mass_shift([base_id]), torch.tensor([[-.5], [.5]]))

    def test_legacy_config_and_unbatched_jump_switch(self):
        cfg = jump_cfgs()
        cfg[0].pop("domain_rand")
        env = self.make_env(cfg, jump=True)
        self.assertFalse(env.batch_dofs_info)
        env.begin_jump_rollout()
        self.assert_solver_gains(env)
        torch.testing.assert_close(env.joint_kp, torch.full((2, 4), 80.))
        env.set_collect_jump_data(False)
        torch.testing.assert_close(env.joint_kp, torch.full((2, 4), 60.))
        obs, *_ = env.step(torch.zeros((2, 6)))
        self.assertTrue(torch.isfinite(obs["policy"]).all())

    def test_wheel_push_reaches_solver_for_one_step_only(self):
        from genesis.utils.misc import qd_to_torch
        cfg = get_cfgs()
        cfg[0]["domain_rand"] = {"enabled": True, "friction": {"enabled": False},
            "motor_gains": {"enabled": False}, "push": {"enabled": True, "targets": ["left_wheel", "right_wheel"],
                "interval_s_range": [.1, .1], "duration_s": .02,
                "force_x_range": [-10., 10.], "force_y_range": [-10., 10.], "force_z_range": [-15., 15.]}}
        env = self.make_env(cfg)
        push = env.domain_rand.push
        # 两个环境分别处于左轮/右轮脉冲中，验证每个目标的全局索引。
        push.steps_until_next.fill_(5)
        push.steps_remaining.fill_(1)
        push.target_index[:] = torch.tensor([0, 1])
        push.force[:] = torch.tensor([[0., 10., 15.], [10., 0., -15.]])
        solver = env.robot.solver
        step = env.scene.step
        snapshots = []

        def checked_step():
            # Genesis 的 cfrc_applied_vel 保存外力的负值。
            snapshots.append(qd_to_torch(solver.dyn_state.links.cfrc_applied_vel, transpose=True, copy=True))
            step()
            cleared = qd_to_torch(solver.dyn_state.links.cfrc_applied_vel, transpose=True, copy=True)
            torch.testing.assert_close(cleared, torch.zeros_like(cleared))

        with patch.object(env.scene, "step", side_effect=checked_step):
            obs, *_ = env.step(torch.zeros(2, 6))
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            env.step(torch.zeros(2, 6))
        expected = torch.zeros_like(snapshots[0])
        for i, link_name in enumerate(cfg[0]["wheel_link_names"]):
            expected[i, env.robot.get_link(link_name).idx] = -push.force[i]
        torch.testing.assert_close(snapshots[0], expected)
        torch.testing.assert_close(snapshots[1], torch.zeros_like(snapshots[1]))

    def test_curriculum_starts_nominal_and_applies_on_reset_and_resume(self):
        env = self.make_env(get_cfgs(), use_domain_curriculum=True)
        self.assertEqual(env.domain_rand.strength, 0.)
        torch.testing.assert_close(env.joint_kp, torch.full((2, 4), 60.))
        torch.testing.assert_close(env.robot.get_dofs_damping(env.passive_dof_idx), torch.full((2, 10), .01))
        self.assertTrue(env.batch_dofs_info)  # 开始虽为零，场景仍须支持后续的逐环境随机化。
        env.curriculum.update(1000)
        self.assertEqual(env.domain_rand.strength, .2)
        torch.testing.assert_close(env.domain_rand.episode_strength, torch.zeros(2))
        env._reset_idx(torch.tensor([True, False]))
        torch.testing.assert_close(env.domain_rand.episode_strength, torch.tensor([.2, 0.]))
        torch.testing.assert_close(env.joint_kp[1], torch.full((4,), 60.))
        self.assertTrue(torch.all((env.domain_rand.motor_scales["joint_kp"][0] >= .98)
                                 & (env.domain_rand.motor_scales["joint_kp"][0] <= 1.02)))
        env.set_training_iteration(4000)
        self.assertEqual(env.domain_rand.strength, .6)
        torch.testing.assert_close(env.domain_rand.episode_strength, torch.full((2,), .6))
        torch.testing.assert_close(env.domain_rand.push.strength, torch.full((2,), .6))
        self.assert_solver_gains(env)
