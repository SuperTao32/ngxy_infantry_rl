# 标准库
import argparse
from copy import deepcopy
from pathlib import Path

# 第三方库
import genesis as gs

# 项目内部模块
from ...core.train_config import get_train_cfg
from ...terrains import TERRAIN_PRESETS, default_terrain_cfg
from ...terrains.curriculum import terrain_course_configs
from ...core.randomization_config import add_randomization_arguments, apply_randomization_arguments
from ...tools.run_utils import (
    add_resume_arguments,
    create_versioned_run_dir,
    load_runner_class,
    load_run_configs,
    resolve_resume_plan,
    restore_training_state,
    save_run_artifacts,
)
from .config_locomotion import get_cfgs as get_default_cfgs
from .env import LocomotionEnv
from .config_mix_terrain import get_cfgs as get_mixed_cfgs
from .runner import terrain_runner_class
from ...terrains.registry import level_catalog


def resolve_num_envs(requested, terrain_configs):
    if requested is not None:
        if requested < 1:
            raise ValueError("num_envs must be positive")
        return requested
    return 64 if any(cfg.get("mixture") is not None for cfg in terrain_configs) else 8192


def describe_terrain(config):
    if config.get("mixture") is not None:
        entries = ", ".join(
            f"{entry['preset']} (weight={entry['weight']}, "
            f"levels={entry['min_difficulty']}..{entry['max_difficulty']})"
            for entry in config["mixture"]
        )
        return f"mixture [{entries}]"
    return config.get("preset", "plane")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-e", "--exp_name", type=str, default="locomotion_v4")
    parser.add_argument("-B", "--num_envs", type=int, default=None, help="parallel environments (default: 64 for mixed terrain courses, 8192 otherwise)")
    parser.add_argument("--max_iterations", type=int, default=10001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--terrain", choices=TERRAIN_PRESETS, default=None, help="override terrain preset in base config and every curriculum stage, including saved runs")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument(
        "--config",
        choices=("config_locomotion", "config_mix_terrain", "config", "config_mixed"),
        default="config_locomotion",
        help="select a training config (default: config_locomotion); config/config_mixed are legacy aliases; when resuming, requires --resume-config current to take effect",
    )
    parser.add_argument(
        "--load-weights",
        type=Path,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="load actor/critic weights only; use --config and start a new optimizer and curriculum at iteration 0",
    )
    add_randomization_arguments(parser, evaluation=False)
    add_resume_arguments(parser)
    args = parser.parse_args()
    args.config = {"config": "config_locomotion", "config_mixed": "config_mix_terrain"}.get(args.config, args.config)

    if args.load_weights is not None and args.resume is not None:
        parser.error("--load-weights cannot be combined with --resume")

    # resume相关配置
    if args.resume is None:
        if args.checkpoint is not None:
            parser.error("--checkpoint requires --resume")
        if args.resume_config != "saved":
            parser.error("--resume-config requires --resume")

    if args.load_weights is not None:
        args.load_weights = args.load_weights.expanduser().resolve()
        if not args.load_weights.is_file():
            parser.error(f"Weights checkpoint does not exist: {args.load_weights}")

    OnPolicyRunner = terrain_runner_class(load_runner_class())
    resume_plan = None
    if args.resume is None or args.resume_config == "current":
        get_cfgs = {"config_locomotion": get_default_cfgs, "config_mix_terrain": get_mixed_cfgs}[args.config]
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name)
        print(f"[train] config: {args.config}.py")

    if args.resume is not None:
        resume_plan = resolve_resume_plan(args.log_root, args.exp_name, args.resume, args.checkpoint)
        if args.resume_config == "saved":
            saved_configs = load_run_configs(resume_plan.source_run_dir)
            env_cfg = deepcopy(saved_configs["env_cfg"])
            obs_cfg = deepcopy(saved_configs["obs_cfg"])
            reward_cfg = deepcopy(saved_configs["reward_cfg"])
            command_cfg = deepcopy(saved_configs["command_cfg"])
            curriculum_cfg = deepcopy(saved_configs["curriculum_cfg"])
            train_cfg = deepcopy(saved_configs["train_cfg"])
            print("[train] config: saved run (--config only applies with --resume-config current)")

    env_cfg.setdefault("terrain", default_terrain_cfg())
    # 配置快照是恢复依据；只在新课程/旧日志缺少快照时读本地参数文件。
    env_cfg["terrain"].setdefault("level_catalog", level_catalog())
    if args.terrain is not None:
        targets = [env_cfg["terrain"]] + [
            stage["targets"]["terrain"] for stage in curriculum_cfg.get("stages", [])
            if "terrain" in stage.get("targets", {})
        ]
        for target in targets:
            target["preset"] = args.terrain
            for key in ("mixture", "adaptive", "difficulty"):
                target.pop(key, None)
    terrain_configs = terrain_course_configs(env_cfg.get("terrain"), curriculum_cfg)
    try:
        args.num_envs = resolve_num_envs(args.num_envs, terrain_configs)
    except ValueError as exc:
        parser.error(str(exc))
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    remaining_iterations = args.max_iterations

    print(f"[train] terrain: {describe_terrain(terrain_configs[0])}")
    print(f"[train] num_envs: {args.num_envs}")

    # 每次启动都新建版本，旧日志和 checkpoint 不会被覆盖。
    run_dir = create_versioned_run_dir(args.log_root, args.exp_name)
    train_cfg["run_name"] = f"{args.exp_name}/{run_dir.name}"
    print(f"[train] saving this run to: {run_dir}")

    configs = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "curriculum_cfg": curriculum_cfg,
        "train_cfg": train_cfg,
    }
    # 保存运行参数和配置文件到日志目录中，便于后续分析和复现
    run_arguments = vars(args).copy()
    if resume_plan is not None:
        run_arguments["resume_from"] = str(resume_plan.checkpoint_path.resolve())

    if args.load_weights is not None:
        run_arguments["weights_from"] = str(args.load_weights)

    save_run_artifacts(run_dir, configs, run_arguments)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = LocomotionEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        steps_per_iteration=train_cfg["num_steps_per_env"],
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
    if resume_plan is not None:
        print(f"[train] resuming from: {resume_plan.checkpoint_path}")
        remaining_iterations = restore_training_state(runner, env, resume_plan, args.max_iterations)
        print(
            f"[train] continuing at iteration {resume_plan.next_iteration}; "
            f"{remaining_iterations} iterations remain (target={args.max_iterations})"
        )
    elif args.load_weights is not None:
        # 环境已按所选配置的第 0 阶段初始化；不恢复旧优化器和训练进度。
        runner.load(
            str(args.load_weights),
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            strict=True,
            map_location=gs.device,
        )
        runner.current_learning_iteration = 0
        print(f"[train] loaded actor/critic weights from: {args.load_weights}")
        print(f"[train] starting at iteration 0 with fresh optimizer and {args.config}.py curriculum")
    runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
