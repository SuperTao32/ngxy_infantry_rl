"""python -m src.genesis.train"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

from .config import FRAMEWORK, env_kwargs, get_cfgs
from .tools.distributed import launch_training, training_workers
from .robust.randomization_config import add_randomization_arguments, apply_randomization_arguments
from .tools.run_utils import (
    add_resume_arguments,
    create_versioned_run_dir,
    load_run_configs,
    resolve_resume_plan,
    restore_training_state,
    save_run_artifacts,
)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train the infantry locomotion environment")
    parser.add_argument("-e", "--exp-name", "--exp_name", default="wheel_leg_infantry")
    parser.add_argument("-B", "--num-envs", "--num_envs", type=int, default=8192, help="number of environments per GPU (default: 8192)")
    parser.add_argument(
        "--num-gpus", "--num_gpus", type=int, metavar="N",
        help="launch N GPU workers, e.g. 2 for dual-GPU training (default: 1, or torchrun worker count)",
    )
    parser.add_argument("--max-iterations", "--max_iterations", type=int, default=10001)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument(
        "--load-weight", type=Path, metavar="CHECKPOINT",
        help="load actor/critic weights into a new run using current config; reset optimizer, iteration and curricula",
    )
    add_resume_arguments(parser)
    add_randomization_arguments(parser, evaluation=False)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.max_iterations < 1:
        parser.error("--num-envs and --max-iterations must be positive")
    if args.num_gpus is not None and args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    if args.load_weight is not None:
        if args.resume is not None:
            parser.error("--load-weight and --resume are mutually exclusive")
        args.load_weight = args.load_weight.expanduser().resolve()
        if not args.load_weight.is_file():
            parser.error(f"Checkpoint does not exist: {args.load_weight}")
    if args.resume is None and (args.checkpoint is not None or args.resume_config != "saved"):
        parser.error("--checkpoint and --resume-config require --resume")
    try:
        if launch_training(args.num_gpus, argv, cpu=args.cpu):
            return
    except ValueError as exc:
        parser.error(str(exc))
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
    if args.resume is not None:
        plan = workers.primary(resolve_resume_plan, args.log_root, args.exp_name, args.resume, args.checkpoint)
    configs = (
        deepcopy(workers.primary(load_run_configs, plan.source_run_dir))
        if plan is not None and args.resume_config == "saved"
        else get_cfgs(args.exp_name)
    )
    if configs.get("framework") != FRAMEWORK:
        parser.error("This entry requires a v2 environment config; use --load-weight without --resume for older policies")
    configs["train_cfg"]["seed"] = args.seed
    apply_randomization_arguments(configs["env_cfg"], configs["obs_cfg"], args)
    run_dir = workers.primary(create_versioned_run_dir, args.log_root, args.exp_name)
    configs["train_cfg"]["run_name"] = f"{args.exp_name}/{run_dir.name}"
    arguments = {**vars(args), "num_gpus": workers.world_size, **workers.metadata(args.num_envs, args.seed)}
    if plan:
        arguments["resume_from"] = str(plan.checkpoint_path.resolve())
    if args.load_weight is not None:
        arguments["load_weight_from"] = str(args.load_weight)
    workers.primary(save_run_artifacts, run_dir, configs, arguments)
    if workers.rank == 0:
        print(f"[train] one GRU policy; GPUs={workers.world_size}; envs/GPU={args.num_envs}; "
              f"total_envs={args.num_envs * workers.world_size}; run={run_dir}")
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning", seed=args.seed + workers.rank, performance_mode=True)
    completed = False
    try:
        env = WheelLegInfantryEnv(
            args.num_envs,
            **env_kwargs(configs),
            steps_per_iteration=configs["train_cfg"]["num_steps_per_env"],
            show_viewer=args.vis and workers.rank == 0,
            initial_training_iteration=plan.next_iteration if plan else 0,
        )
        runner = PolicyRunner(
            env, configs["train_cfg"], str(run_dir), device=str(gs.device), workers=workers,
        )
        remaining = args.max_iterations
        if plan:
            remaining = restore_training_state(
                runner, env, plan, args.max_iterations, restore_reward_scales=args.resume_config == "saved",
            )
            print(f"[resume] iteration={runner.current_learning_iteration}")
        elif args.load_weight is not None:
            runner.load(
                str(args.load_weight),
                load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False),
                map_location=runner.device,
            )
            print(f"[load-weight] checkpoint={args.load_weight}; iteration=0; optimizer and environment curriculum start fresh")
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
        completed = True
    finally:
        # Genesis may spend time caching kernels and resetting Quadrants/CUDA.
        # NCCL must be gone first, including on Ctrl+C and torchrun SIGTERM.
        try:
            workers.close(synchronize=completed)
        finally:
            gs.destroy()


if __name__ == "__main__":
    main()
