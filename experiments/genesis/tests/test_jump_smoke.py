"""NGXY_JUMP_SMOKE=1 python -m unittest experiments.genesis.tests.test_jump_smoke -v"""

import os
import unittest

import torch

from experiments.genesis.wheel_leg_infantry.tasks.jump.config import get_cfgs


@unittest.skipUnless(os.environ.get("NGXY_JUMP_SMOKE") == "1", "opt-in Genesis simulation")
class JumpSmokeTests(unittest.TestCase):
    def test_three_modes_placement_observations_and_rollout(self):
        import genesis as gs
        from experiments.genesis.wheel_leg_infantry.tasks.jump.env import JumpEnv

        backend = gs.gpu if os.environ.get("NGXY_SMOKE_GPU") == "1" else gs.cpu
        gs.init(backend=backend, precision="32", logging_level="warning", seed=17)
        self.addCleanup(gs.destroy)
        cfg = get_cfgs()
        # 显式设置测试课程，使几何/生命周期检查不依赖用户正在调试的训练速度和采样比例。
        cfg[4]["stages"][0]["targets"]["command_ranges"]["lin_vel_range"] = [0., 1.]
        cfg[4]["stages"][0]["targets"]["terrain"]["mode_probabilities"] = [1., 0., 0.]
        cfg[4]["stages"][1]["targets"]["terrain"]["mode_probabilities"] = [0., 0., 1.]
        cfg[0]["jump_modes"].update(assignment="cyclic", distance_jitter_m=0.)
        cfg[0]["show_FPS"] = False
        env = JumpEnv(3, *cfg[:4], curriculum_cfg=cfg[4], steps_per_iteration=70)
        self.addCleanup(env.scene.destroy)
        obs = env.prepare_locomotion_warmup()
        self.assertEqual(obs["policy"].shape, (3, 33))
        torch.testing.assert_close(env.jump_mode, torch.arange(3, device=gs.device))
        for _ in range(2):
            env.step(torch.zeros(3, 6, device=gs.device))
        env.commands[:, 0] = 1.
        old_qvel = env.robot.get_dofs_velocity().clone()
        old_quat = env.robot.get_quat().clone()
        old_z = env.robot.get_pos()[:, 2].clone()
        env.last_actions.fill_(.1)
        obs = env.begin_jump_rollout()
        torch.testing.assert_close(env.robot.get_dofs_velocity(), old_qvel)
        torch.testing.assert_close(env.robot.get_quat(), old_quat)
        torch.testing.assert_close(env.robot.get_pos()[:, 2], old_z)
        torch.testing.assert_close(env.last_actions, torch.full((3, 6), .1, device=gs.device))
        torch.testing.assert_close(env.base_pos[:, 0], torch.tensor([0., -.3, -.4], device=gs.device))
        torch.testing.assert_close(obs["policy"][:, -3:], torch.eye(3, device=gs.device))
        self.assertEqual(obs["policy"].shape, (3, 48))
        self.assertEqual(obs["critic"].shape, (3, 67))
        torch.testing.assert_close(env.jump_stage, torch.zeros(3, dtype=torch.long, device=gs.device))
        torch.testing.assert_close(env.phase_step_boundaries, torch.tensor([8, 30], device=gs.device))
        self.assertFalse(hasattr(env, "crouch_gate"))
        self.assertFalse(hasattr(env, "crouch_airborne_gate"))
        self.assertEqual(env._get_jump_privileged_components()["privileged_jump_stage"].shape, (3, 3))
        self.assertEqual(env.get_locomotion_observations()["policy"].shape, (3, 33))
        for _ in range(70):
            obs, rewards, _, _ = env.step(torch.zeros(3, 6, device=gs.device))
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            self.assertTrue(torch.isfinite(obs["critic"]).all())
            self.assertTrue(torch.isfinite(rewards).all())
            expected_stage = torch.where(env.has_landed, 2, env.has_taken_off.long())
            torch.testing.assert_close(env.jump_stage, expected_stage)
            torch.testing.assert_close(env.jump_mode, torch.arange(3, device=gs.device))
        self.assertFalse(env.task_success.any())  # 无有效跳跃不得通过台阶任务。
        stats = env.finish_jump_rollout()["episode"]
        self.assertIn("success_step_40cm", stats)
        self.assertTrue(torch.all(stats["reward_death"] >= -300.01))  # 不累计终止后的补齐步。
        env.prepare_locomotion_warmup()
        torch.testing.assert_close(env.base_pos[:, 0], torch.full((3,), -30., device=gs.device))
        self.assertFalse(env.failure_seen.any())
        # 第 0 阶段已在场景构造前应用；切换阶段不改当前模式，下一次 reset 才采样。
        self.assertEqual(env.task_cfg["mode_probabilities"], [1., 0., 0.])
        env.task_cfg["assignment"] = "random"
        env.curriculum.update(500)
        torch.testing.assert_close(env.jump_mode, torch.arange(3, device=gs.device))
        env.reset()
        torch.testing.assert_close(env.jump_mode, torch.full((3,), 2, device=gs.device))
        # 恢复/回退课程后，同样从对应阶段重新分配。
        env.curriculum.update(0)
        env.reset()
        torch.testing.assert_close(env.jump_mode, torch.zeros(3, dtype=torch.long, device=gs.device))


if __name__ == "__main__":
    unittest.main()
