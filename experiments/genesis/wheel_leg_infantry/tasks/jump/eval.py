"""自动重复评估三个模式；all 按批次轮换可视化模式，支持指定模式/速度。"""

import argparse
from copy import deepcopy
import math

import torch

from ...core.randomization import add_randomization_arguments, apply_randomization_arguments
from ...tools.run_utils import load_run_configs, load_runner_class, resolve_checkpoint, resolve_recorded_run, resolve_run_dir
from .staged_runner import build_frozen_locomotion_actor, stabilize_with_locomotion
from .warm_start import validate_locomotion_source
from .config import MODE_NAMES, validate_configs


def get_eval_lin_vel_limits(env_cfg, curriculum_cfg):
    """汇总训练课程里出现过的前向速度，作为交互 eval 的安全范围。"""
    ranges = [env_cfg["locomotion_warmup"]["command_ranges"]["lin_vel_range"]]
    ranges.extend(
        stage["targets"]["command_ranges"]["lin_vel_range"]
        for stage in curriculum_cfg.get("stages", [])
        if "lin_vel_range" in stage.get("targets", {}).get("command_ranges", {})
    )
    values = [float(value) for limits in ranges for value in limits]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("jump locomotion lin_vel_range values must be finite")
    return min(values), max(values)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--exp-name", default="infantry_jump")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--locomotion-log-root", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--mode", choices=("all", *MODE_NAMES), default="all",
                        help="all 每批轮换 flat → step_20cm → step_40cm，画面显示环境 0")
    parser.add_argument("--speed", type=float, default=None, help="固定前向速度 m/s；默认按保存的配置采样")
    parser.add_argument("-B", "--num-envs", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=10, help="批次，每批 num-envs 次跳跃")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    add_randomization_arguments(parser, evaluation=True)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.episodes <= 0:
        parser.error("num-envs and episodes must be positive")
    if args.speed is not None and (not math.isfinite(args.speed) or args.speed < 0):
        parser.error("--speed must be finite and non-negative")
    return args


def main():
    args = _parse_args()
    run_dir = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
    checkpoint = resolve_checkpoint(run_dir, args.ckpt)
    configs = load_run_configs(run_dir)
    env_cfg = deepcopy(configs["env_cfg"])
    validate_configs(env_cfg, configs["obs_cfg"])
    obs_cfg = deepcopy(configs["obs_cfg"])
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    env_cfg.update(show_FPS=False, viewer_realtime_factor=None)
    task = env_cfg["jump_modes"]
    task["distance_jitter_m"] = 0.0
    if args.mode == "all":
        task["assignment"] = "cyclic"
    else:
        task["assignment"] = "random"
        task["mode_probabilities"] = [float(name == args.mode) for name in MODE_NAMES]
    if args.speed is not None:
        env_cfg["locomotion_warmup"]["command_ranges"]["lin_vel_range"] = [args.speed, args.speed]
        task["flat_stationary_probability"] = 0.0
    source_run, source_checkpoint = resolve_recorded_run(
        configs["source_locomotion"], args.locomotion_log_root or args.log_root,
    )
    source_configs = load_run_configs(source_run)
    validate_locomotion_source(source_configs)

    import genesis as gs
    from .env import JumpEnv

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    env = JumpEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=deepcopy(configs["reward_cfg"]), command_cfg=deepcopy(configs["command_cfg"]),
        curriculum_cfg={"enabled": False, "stages": []},
        steps_per_iteration=configs["train_cfg"]["num_steps_per_env"], show_viewer=not args.headless,
    )
    try:
        teacher = build_frozen_locomotion_actor(
            source_configs, source_checkpoint, env.get_locomotion_observations(), env.num_actions, gs.device,
        )
        runner = load_runner_class()(env, deepcopy(configs["train_cfg"]), None, device=gs.device)
        runner.load(str(checkpoint), map_location=gs.device)
        policy = runner.get_inference_policy(device=gs.device)
        totals = torch.zeros(3, dtype=torch.long, device=gs.device)
        successes = torch.zeros_like(totals)
        print(f"[jump eval] checkpoint={checkpoint}")
        with torch.inference_mode():
            for episode in range(args.episodes):
                if args.mode == "all":
                    env.terrain.cyclic_mode_offset = episode % len(MODE_NAMES)
                stabilize_with_locomotion(env, teacher, env_cfg["locomotion_warmup"])
                obs = env.begin_jump_rollout()
                modes = env.jump_mode.clone()
                if not args.headless:
                    env.focus_viewer()
                print(f"[jump eval] batch={episode + 1}: env[0]={MODE_NAMES[int(modes[0])]}", flush=True)
                for _ in range(env.max_episode_length):
                    obs, _, _, _ = env.step(policy(obs))
                    if env.scene.rigid_solver.get_error_envs_mask().any():
                        raise RuntimeError("Genesis solver error during evaluation")
                success = env.task_success.clone()
                for mode in range(3):
                    selected = modes == mode
                    totals[mode] += selected.sum()
                    successes[mode] += (selected & success).sum()
                env.finish_jump_rollout()
                print(f"[jump eval] batch={episode + 1}: " + ", ".join(
                    f"{name}={int(successes[i])}/{int(totals[i])}" for i, name in enumerate(MODE_NAMES)
                ), flush=True)
    finally:
        env.scene.destroy()
        gs.destroy()


if __name__ == "__main__":
    main()
