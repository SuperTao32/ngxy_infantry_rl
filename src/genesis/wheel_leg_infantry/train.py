"""python -m src.genesis.wheel_leg_infantry.train"""

import argparse
from copy import deepcopy

from .config import env_kwargs, get_cfgs, select_tasks
from .tools.distributed import training_workers
from .robust.randomization_config import add_randomization_arguments, apply_randomization_arguments
from .tasks.manager import TASK_NAMES, allocate_task_counts
from .tools.run_utils import (add_resume_arguments, create_versioned_run_dir, load_run_configs,
                              resolve_resume_plan, restore_training_state, save_run_artifacts)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train one recurrent policy on all infantry tasks")
    parser.add_argument("-e", "--exp-name", "--exp_name", default="wheel_leg_infantry")
    parser.add_argument("-B", "--num-envs", "--num_envs", type=int, default=8192)
    parser.add_argument("--max-iterations", "--max_iterations", type=int, default=10000)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES)
    parser.add_argument("--task-weights", nargs="+", metavar="NAME=WEIGHT")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-root", default="logs")
    add_resume_arguments(parser)
    add_randomization_arguments(parser, evaluation=False)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.max_iterations < 1:
        parser.error("--num-envs and --max-iterations must be positive")
    if args.resume is None and (args.checkpoint is not None or args.resume_config != "saved"):
        parser.error("--checkpoint and --resume-config require --resume")
    if args.resume is not None and (args.tasks or args.task_weights):
        parser.error("Full resume keeps its saved task set and weights; edit config.py for a new run")
    with training_workers() as workers:
        if args.cpu and workers.distributed:
            parser.error("Distributed training requires GPUs")
        _train(args, parser, workers)


def _train(args, parser, workers):
    # Multi-GPU device isolation must happen before importing Genesis.
    import genesis as gs
    from .env import WheelLegInfantryEnv
    from .policy_runner import PolicyRunner

    plan = None
    configs = get_cfgs(args.exp_name)
    if args.resume is not None:
        plan = workers.primary(resolve_resume_plan, args.log_root, args.exp_name, args.resume, args.checkpoint)
        if args.resume_config == "saved":
            configs = deepcopy(workers.primary(load_run_configs, plan.source_run_dir))
    if configs.get("framework") != "wheel_leg_infantry_multitask_v1":
        parser.error("This entry requires a unified multi-task GRU configuration")
    try:
        if args.tasks:
            select_tasks(configs, args.tasks)
        tasks = configs["env_cfg"]["tasks"]
        for entry in args.task_weights or ():
            name, weight = entry.split("=", 1)
            tasks[name]["weight"] = float(weight)
        counts = allocate_task_counts(tasks, args.num_envs)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    configs["train_cfg"]["seed"] = args.seed
    apply_randomization_arguments(configs["env_cfg"], configs["obs_cfg"], args)
    run_dir = workers.primary(create_versioned_run_dir, args.log_root, args.exp_name)
    configs["train_cfg"]["run_name"] = f"{args.exp_name}/{run_dir.name}"
    arguments = {**vars(args), **workers.metadata(args.num_envs, args.seed)}
    if plan:
        arguments["resume_from"] = str(plan.checkpoint_path.resolve())
    workers.primary(save_run_artifacts, run_dir, configs, arguments)
    print(f"[train] one GRU policy; tasks={dict(zip(tasks, counts))}; run={run_dir}")
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning",
            seed=args.seed + workers.rank, performance_mode=True)
    try:
        env = WheelLegInfantryEnv(args.num_envs, **env_kwargs(configs),
                                 steps_per_iteration=configs["train_cfg"]["num_steps_per_env"],
                                 show_viewer=args.vis and workers.rank == 0)
        runner = PolicyRunner(env, configs["train_cfg"], str(run_dir), device=str(gs.device), workers=workers)
        remaining = args.max_iterations
        if plan:
            remaining = restore_training_state(runner, env, plan, args.max_iterations)
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
