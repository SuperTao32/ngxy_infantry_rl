"""Jump PPO 入口：从 locomotion 热启动，或从 jump checkpoint 续训。"""

from __future__ import annotations

import argparse
from copy import deepcopy

from ...core.train_config import get_train_cfg
from ...core.randomization import add_randomization_arguments, apply_randomization_arguments
from ...tools.run_utils import (
    add_resume_arguments,
    create_versioned_run_dir,
    load_run_configs,
    load_runner_class,
    resolve_checkpoint,
    resolve_run_dir,
    resolve_recorded_run,
    resolve_resume_plan,
    restore_training_state,
    save_run_artifacts,
)
from .config import MODE_NAMES, get_cfgs, validate_configs
from .phase import validate_phase_durations
from .staged_runner import validate_warmup_cfg
from .warm_start import validate_locomotion_source, warm_start_actor


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="平地 / 20 cm / 40 cm one-hot 混合跳跃；env 指定起跳距离")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-e", "--exp-name", default="jump_v2")
    parser.add_argument("-B", "--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=1001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--locomotion-log-root", default=None)
    parser.add_argument("--locomotion-exp-name", default="locomotion_v2")
    parser.add_argument("--locomotion-version", default=None)
    parser.add_argument("--locomotion-ckpt", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    add_randomization_arguments(parser, evaluation=False)
    add_resume_arguments(parser)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.max_iterations <= 0:
        parser.error("num-envs and max-iterations must be positive")
    if args.resume is None and (args.checkpoint is not None or args.resume_config != "saved"):
        parser.error("--checkpoint and --resume-config require --resume")
    return args


def main():
    args = _parse_args()
    print(f"[jump] one-hot order={MODE_NAMES}; trigger distance comes from env, no ToF")
    if args.resume is not None and args.resume_config == "saved" and not args.dry_run:
        print("[jump] config=saved jump run")
    else:
        print(f"[jump] config={get_cfgs.__module__}")
    if args.dry_run:
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg = get_cfgs()
        apply_randomization_arguments(env_cfg, obs_cfg, args)
        validate_configs(env_cfg, obs_cfg, curriculum_cfg)
        validate_phase_durations(env_cfg["jump_phase_durations_s"])
        jump_horizon = round(env_cfg["episode_length_s"] / 0.02)
        validate_warmup_cfg(env_cfg["locomotion_warmup"])
        print(
            "[jump dry-run] "
            f"actions={env_cfg['num_actions']} "
            f"policy_obs={obs_cfg['num_policy_obs']} critic_obs={obs_cfg['num_critic_obs']} "
            f"commands=[vx, wz, base_height] "
            f"clearance_targets={env_cfg['jump_modes']['clearance_targets_m']} "
            f"ppo_horizon={jump_horizon} "
            f"phases={env_cfg['jump_phase_durations_s']}"
        )
        print(
            "[jump dry-run] warmup="
            f"{env_cfg['locomotion_warmup']['min_steps']}.."
            f"{env_cfg['locomotion_warmup']['max_steps']} steps, "
            f"fixed_commands={env_cfg['locomotion_warmup']['command_ranges']}"
        )
        print(f"[jump dry-run] rewards={sorted(reward_cfg['reward_scales'])}")
        print(f"[jump dry-run] disabled_locomotion_rewards={reward_cfg['disabled_locomotion_rewards']}")
        print(f"[jump dry-run] curriculum={[stage['name'] for stage in curriculum_cfg['stages']]}")
        for stage in curriculum_cfg["stages"]:
            probabilities = stage["targets"].get("terrain", {}).get("mode_probabilities", "inherit")
            print(f"[jump dry-run] {stage['name']}: mode_probabilities={probabilities}")
        for mode, table in zip(MODE_NAMES, env_cfg["jump_modes"]["distance_tables"]):
            print(f"[jump dry-run] {mode}: [speed, distance]={table}")
        print("[jump dry-run] real training requires a locomotion checkpoint; --resume also requires a jump checkpoint")
        return

    OnPolicyRunner = load_runner_class()

    locomotion_log_root = args.locomotion_log_root or args.log_root
    resume_plan = None
    saved_configs = None
    if args.resume is not None:
        resume_plan = resolve_resume_plan(args.log_root, args.exp_name, args.resume, args.checkpoint)
        if args.max_iterations <= resume_plan.next_iteration:
            raise ValueError(
                f"--max-iterations must be greater than {resume_plan.next_iteration} "
                f"when resuming from checkpoint {resume_plan.checkpoint_iteration}"
            )
        saved_configs = load_run_configs(resume_plan.source_run_dir)
        source_info = saved_configs.get("source_locomotion")
        if not source_info:
            raise ValueError("jump run does not record its source_locomotion checkpoint")
        source_run_dir, source_checkpoint = resolve_recorded_run(source_info, locomotion_log_root)
    else:
        source_run_dir = resolve_run_dir(
            locomotion_log_root,
            args.locomotion_exp_name,
            args.locomotion_version,
            require_checkpoint=True,
        )
        source_checkpoint = resolve_checkpoint(source_run_dir, args.locomotion_ckpt)
    source_configs = load_run_configs(source_run_dir)
    validate_locomotion_source(source_configs)

    locomotion_cfgs = (
        source_configs["env_cfg"],
        source_configs["obs_cfg"],
        source_configs["reward_cfg"],
        source_configs["command_cfg"],
        source_configs["curriculum_cfg"],
    )
    if saved_configs is not None and args.resume_config == "saved":
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, train_cfg = (
            deepcopy(saved_configs[key])
            for key in ("env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "curriculum_cfg", "train_cfg")
        )
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg = get_cfgs(locomotion_cfgs)
        train_cfg = get_train_cfg(args.exp_name)
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    validate_configs(env_cfg, obs_cfg, curriculum_cfg)
    jump_horizon = round(env_cfg["episode_length_s"] / 0.01)
    if not abs(jump_horizon * 0.01 - env_cfg["episode_length_s"]) < 1e-9:
        raise ValueError("jump phase cycle must contain an integer number of 10 ms control steps")
    train_cfg["num_steps_per_env"] = jump_horizon
    validate_warmup_cfg(env_cfg["locomotion_warmup"])

    run_dir = create_versioned_run_dir(args.log_root, args.exp_name)
    train_cfg["run_name"] = f"{args.exp_name}/{run_dir.name}"
    configs = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "curriculum_cfg": curriculum_cfg,
        "train_cfg": train_cfg,
        "source_locomotion": {
            "run_dir": str(source_run_dir.resolve()),
            "checkpoint": str(source_checkpoint.resolve()),
        },
    }
    run_arguments = vars(args).copy()
    if resume_plan is not None:
        run_arguments["resume_from"] = str(resume_plan.checkpoint_path.resolve())
    save_run_artifacts(run_dir, configs, run_arguments)
    print(f"[jump train] saving this run to: {run_dir}")
    print(f"[jump train] locomotion source: {source_checkpoint}")

    # 延迟导入保证 --help/--dry-run 不初始化 Genesis 渲染后端。
    import genesis as gs

    from .env import JumpEnv
    from .staged_runner import (
        build_frozen_locomotion_actor,
        learn_staged_jump,
    )

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = JumpEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        steps_per_iteration=train_cfg["num_steps_per_env"],
        show_viewer=args.vis,
    )
    locomotion_actor = build_frozen_locomotion_actor(
        source_configs,
        source_checkpoint,
        env.get_locomotion_observations(),
        num_actions=env.num_actions,
        device=gs.device,
    )
    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
    remaining_iterations = args.max_iterations
    if resume_plan is not None:
        remaining_iterations = restore_training_state(runner, env, resume_plan, args.max_iterations)
        print(f"[jump train] resuming from: {resume_plan.checkpoint_path}")
        print(
            f"[jump train] continuing at iteration {resume_plan.next_iteration}; "
            f"{remaining_iterations} iterations remain (target={args.max_iterations})"
        )
    else:
        report = warm_start_actor(
            runner.alg.get_policy(),
            source_checkpoint,
            locomotion_obs_dim=obs_cfg["locomotion_policy_obs_dim"],
        )
        print(
            "[jump train] actor warm-started: "
            f"{report.source_input_dim}D -> {report.target_input_dim}D, "
            f"mapped_input_columns={report.mapped_input_columns}, "
            f"expanded={report.expanded_input_tensor}, copied={len(report.copied_tensors)} tensors"
        )

    learn_staged_jump(
        runner,
        locomotion_actor,
        num_learning_iterations=remaining_iterations,
        warmup_cfg=env_cfg["locomotion_warmup"],
    )


if __name__ == "__main__":
    main()
