"""python -m src.genesis.wheel_leg_infantry.train"""

import argparse
from copy import deepcopy
from pathlib import Path

from .config import env_kwargs, get_cfgs, select_tasks
from .tools.distributed import training_workers
from .robust.randomization_config import add_randomization_arguments, apply_randomization_arguments
from .tasks.manager import TASK_NAMES, initial_task_counts
from .tools.run_utils import (
    add_resume_arguments,
    create_versioned_run_dir,
    load_run_configs,
    resolve_resume_plan,
    restore_training_state,
    save_run_artifacts,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train one recurrent policy on all infantry tasks")
    parser.add_argument("-e", "--exp-name", "--exp_name", default="wheel_leg_infantry")
    parser.add_argument("-B", "--num-envs", "--num_envs", type=int, default=8192)
    parser.add_argument("--max-iterations", "--max_iterations", type=int, default=10000)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES)
    parser.add_argument("--task-weights", nargs="+", metavar="NAME=WEIGHT", help="initial task weights; zero pauses a task")
    parser.add_argument(
        "--recovery-poses", nargs="+", metavar="PATH", help="recovery JSON files or collection directories; default: configured pose list"
    )
    parser.add_argument("--recovery-dynamic-probability", type=float, help="dynamic recovery reset probability in [0, 1] (new runs default: 0.2)")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-root", default="logs")
    add_resume_arguments(parser)
    parser.add_argument(
        "--load-weight", type=Path, metavar="CHECKPOINT",
        help="load actor/critic weights into a new run using current config; reset optimizer, iteration and curricula",
    )
    add_randomization_arguments(parser, evaluation=False)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.max_iterations < 1:
        parser.error("--num-envs and --max-iterations must be positive")
    if args.load_weight is not None:
        if args.resume is not None:
            parser.error("--load-weight and --resume are mutually exclusive")
        args.load_weight = args.load_weight.expanduser().resolve()
        if not args.load_weight.is_file():
            parser.error(f"Checkpoint does not exist: {args.load_weight}")
    if args.resume is None and (args.checkpoint is not None or args.resume_config != "saved"):
        parser.error("--checkpoint and --resume-config require --resume")
    if args.resume is not None and args.resume_config == "saved" and (args.tasks or args.task_weights):
        parser.error("--tasks and --task-weights require --resume-config current when resuming")
    with training_workers() as workers:
        if args.cpu and workers.distributed:
            parser.error("Distributed training requires GPUs")
        _train(args, parser, workers)


def configure_recovery_reset(configs, sources=None, dynamic_probability=None):
    """CLI 显式覆盖采集来源或比例；普通 saved 续训保留已存档的数值姿态池。"""
    if sources is None and dynamic_probability is None:
        return
    import math
    from .tasks.recovery.config import get_recovery_training_reset_cfg

    tasks = configs["env_cfg"]["tasks"]
    if "recovery" not in tasks:
        raise ValueError("Recovery reset options require the recovery task")
    env_cfg = tasks["recovery"]["env_cfg"]
    if "recovery_reset" not in env_cfg:
        env_cfg["recovery_reset"] = get_recovery_training_reset_cfg(sources)
    elif sources is not None:
        loaded = get_recovery_training_reset_cfg(sources)
        # 换数据时保留当前/存档中的动静态比例和动态难度，只替换快照池。
        for name in ("pose_pool", "side_lying_angle_deg"):
            env_cfg["recovery_reset"][name] = loaded[name]
    if dynamic_probability is not None:
        if isinstance(dynamic_probability, bool) or not math.isfinite(dynamic_probability) or not 0 <= dynamic_probability <= 1:
            raise ValueError("--recovery-dynamic-probability must be finite and in [0, 1]")
        env_cfg["recovery_reset"]["dynamic_probability"] = dynamic_probability


def _train(args, parser, workers):
    # Multi-GPU device isolation must happen before importing Genesis.
    import genesis as gs
    from .env import WheelLegInfantryEnv
    from .policy_runner import PolicyRunner

    plan = None
    if args.resume is not None:
        plan = workers.primary(resolve_resume_plan, args.log_root, args.exp_name, args.resume, args.checkpoint)
    configs = (
        deepcopy(workers.primary(load_run_configs, plan.source_run_dir))
        if plan is not None and args.resume_config == "saved"
        else get_cfgs(args.exp_name)
    )
    if configs.get("framework") != "wheel_leg_infantry_multitask_v1":
        parser.error("This entry requires a unified multi-task GRU configuration")
    try:
        if args.tasks:
            select_tasks(configs, args.tasks)
        configure_recovery_reset(configs, args.recovery_poses, args.recovery_dynamic_probability)
        tasks = configs["env_cfg"]["tasks"]
        for entry in args.task_weights or ():
            name, weight = entry.split("=", 1)
            tasks[name]["weight"] = float(weight)
            for stage in configs["curriculum_cfg"].get("stages", []):
                if stage["start_iteration"] == 0:
                    stage.setdefault("targets", {}).setdefault("task_weights", {})[name] = float(weight)
        counts = initial_task_counts(tasks, args.num_envs, configs["curriculum_cfg"])
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    configs["train_cfg"]["seed"] = args.seed
    apply_randomization_arguments(configs["env_cfg"], configs["obs_cfg"], args)
    run_dir = workers.primary(create_versioned_run_dir, args.log_root, args.exp_name)
    configs["train_cfg"]["run_name"] = f"{args.exp_name}/{run_dir.name}"
    arguments = {**vars(args), **workers.metadata(args.num_envs, args.seed)}
    if plan:
        arguments["resume_from"] = str(plan.checkpoint_path.resolve())
    if args.load_weight is not None:
        arguments["load_weight_from"] = str(args.load_weight)
    workers.primary(save_run_artifacts, run_dir, configs, arguments)
    print(f"[train] one GRU policy; tasks={dict(zip(tasks, counts))}; run={run_dir}")
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning", seed=args.seed + workers.rank, performance_mode=True)
    try:
        env = WheelLegInfantryEnv(
            args.num_envs,
            **env_kwargs(configs),
            steps_per_iteration=configs["train_cfg"]["num_steps_per_env"],
            show_viewer=args.vis and workers.rank == 0,
        )
        runner = PolicyRunner(
            env, configs["train_cfg"], str(run_dir), device=str(gs.device), workers=workers,
            allow_new_tasks=plan is not None and args.resume_config == "current",
        )
        remaining = args.max_iterations
        if plan:
            remaining = restore_training_state(runner, env, plan, args.max_iterations)
            resumed_counts = {name: len(task.env_ids) for name, task in env.task_manager.tasks.items()}
            print(f"[resume] iteration={runner.current_learning_iteration}; tasks={resumed_counts}")
        elif args.load_weight is not None:
            runner.load(
                str(args.load_weight),
                load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False),
                map_location=runner.device,
            )
            print(f"[load-weight] checkpoint={args.load_weight}; iteration=0; optimizer and task curricula start fresh")
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
