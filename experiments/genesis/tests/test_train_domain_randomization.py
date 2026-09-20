"""训练恢复配置覆盖与初始化顺序，不启动训练或写日志。"""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class TrainDomainRandomizationTests(unittest.TestCase):
    def test_saved_config_override_and_resume_after_runner_creation(self):
        from experiments.genesis.wheel_leg_infantry.tasks.locomotion import train

        cfg = train.get_cfgs()
        cfg[0].pop("domain_rand")  # 旧 checkpoint
        saved = dict(zip(("env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "curriculum_cfg"), cfg))
        saved["train_cfg"] = {"num_steps_per_env": 24}
        plan = SimpleNamespace(source_run_dir=Path("/tmp/source"), checkpoint_path=Path("/tmp/model_9.pt"), next_iteration=10)
        runner, env = Mock(), Mock()
        runner_cls = Mock(return_value=runner)
        restore = Mock(return_value=10)
        mocks = {
            "load_runner_class": Mock(return_value=runner_cls),
            "load_run_configs": Mock(return_value=saved),
            "resolve_resume_plan": Mock(return_value=plan),
            "create_versioned_run_dir": Mock(return_value=Path("/tmp/new_run")),
            "save_run_artifacts": Mock(),
            "LocomotionEnv": Mock(return_value=env),
            "restore_training_state": restore,
        }
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(train, **mocks))
            stack.enter_context(patch.object(train.gs, "init"))
            stack.enter_context(patch.object(train.gs, "device", "cpu", create=True))
            stack.enter_context(patch("sys.argv", ["train", "--resume", "latest", "--domain-rand", "--max_iterations", "20"]))
            train.main()
        env_cfg = mocks["LocomotionEnv"].call_args.kwargs["env_cfg"]
        self.assertTrue(env_cfg["domain_rand"]["enabled"])
        self.assertNotIn("domain_rand", saved["env_cfg"])
        restore.assert_called_once_with(runner, env, plan, 20)
        runner.learn.assert_called_once_with(num_learning_iterations=10, init_at_random_ep_len=True)
        artifacts = mocks["save_run_artifacts"].call_args.args
        self.assertTrue(artifacts[1]["env_cfg"]["domain_rand"]["enabled"])
        self.assertEqual(artifacts[2]["resume_from"], "/tmp/model_9.pt")
