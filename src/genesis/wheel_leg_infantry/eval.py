"""统一评估入口；任意任务子集使用同一模型和固定观测布局。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .config import env_kwargs, select_tasks
from .robust.randomization_config import add_randomization_arguments, apply_randomization_arguments
from .tasks.manager import TASK_NAMES
from .tools.run_utils import load_run_configs, resolve_checkpoint, resolve_run_dir


def prepare_evaluation(configs, iteration):
    """冻结所选 checkpoint 时刻的课程；测评不推进训练/地形难度。"""
    for course in [configs["curriculum_cfg"], *(task["curriculum_cfg"] for task in configs["env_cfg"]["tasks"].values())]:
        stages = [stage for stage in course.get("stages", []) if stage["start_iteration"] <= iteration]
        if course.get("enabled") and stages:
            # Collapse the curriculum at its saved checkpoint stage.
            from .curriculum import _deep_update
            targets = {}
            for stage in stages:
                _deep_update(targets, stage.get("targets", {}))
            course["stages"] = [dict(name="evaluation", start_iteration=0, targets=targets)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the shared infantry GRU policy")
    parser.add_argument("-e", "--exp-name", "--exp_name", default="wheel_leg_infantry")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--version")
    parser.add_argument("--checkpoint", "--ckpt", type=int)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES)
    parser.add_argument("-B", "--num-envs", "--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keyboard", action="store_true",
                        help="control commands with viewer keys (requires --vis)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    add_randomization_arguments(parser, evaluation=True)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    if args.keyboard and not args.vis:
        parser.error("--keyboard requires --vis")
    run_dir = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    configs = deepcopy(load_run_configs(run_dir))
    if configs.get("framework") != "wheel_leg_infantry_multitask_v1":
        parser.error("Evaluation requires a unified multi-task GRU checkpoint")
    try:
        if args.tasks:
            select_tasks(configs, args.tasks)
    except ValueError as exc:
        parser.error(str(exc))
    iteration = int(checkpoint.stem.split("_")[-1])
    prepare_evaluation(configs, iteration)
    apply_randomization_arguments(configs["env_cfg"], configs["obs_cfg"], args)
    import genesis as gs
    import torch
    from .env import WheelLegInfantryEnv
    from .policy_runner import PolicyRunner
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", seed=args.seed, logging_level="warning")
    try:
        env = WheelLegInfantryEnv(args.num_envs, **env_kwargs(configs), evaluation=True,
                                 steps_per_iteration=configs["train_cfg"]["num_steps_per_env"], show_viewer=args.vis)
        runner = PolicyRunner(env, configs["train_cfg"], None, device=str(gs.device))
        runner.load(str(checkpoint), load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False),
                    map_location=str(gs.device))
        # Restore terrain levels/windows only for tasks present in this evaluation.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False).get("infos") or {}
        state = saved.get("infantry_tasks", {})
        for name, task in env.task_manager.tasks.items():
            if name in state:
                task.load_state_dict(state[name])
        policy = runner.get_inference_policy(device=str(gs.device))
        metrics = {}
        returns = torch.zeros(env.num_envs, device=gs.device)
        with torch.inference_mode():
            policy.reset()
            obs = env.reset()
            keyboard = None
            if args.keyboard:
                from .tools.interactive_viewer import KeyboardCommand
                keyboard = KeyboardCommand(env)
                print("[eval] keyboard: Up/Down=vx (+/-0.5 m/s), Left/Right=wz (+/-0.5 rad/s), "
                      "PageUp/PageDown=height (+/-0.05 m), Space=stop, Enter=focus robot")
                print(f"[eval] command limits (env 0 task: {keyboard.task.name}): {keyboard.command_ranges}")
            for _ in range(args.steps):
                if args.vis and not env.scene.viewer.is_alive():
                    break
                if keyboard is not None:
                    # Resampling/reset may replace commands inside step(). Refresh
                    # both actor and critic observations before every inference.
                    keyboard.write_to_env()
                    env._update_observations()
                    obs = env.get_observations()
                    keyboard.update_caption()
                obs, rewards, dones, extras = env.step(policy(obs))
                policy.reset(dones)
                returns += rewards
                for name, task in env.task_manager.tasks.items():
                    ended = dones & task.mask
                    if ended.any():
                        row = metrics.setdefault(name, dict(episodes=0, return_sum=0.0, failures=0, successes=0))
                        row["episodes"] += int(ended.sum())
                        row["return_sum"] += float(returns[ended].sum())
                        row["failures"] += int(env.terminated_buf[ended].sum())
                        row["successes"] += int(extras["task_successes"][ended].sum())
                returns[dones] = 0
        summary = {name: dict(episodes=row["episodes"], mean_return=row["return_sum"] / row["episodes"],
                             failure_rate=row["failures"] / row["episodes"],
                             success_rate=row["successes"] / row["episodes"])
                   for name, row in metrics.items()}
        # Velocity tracking has no terminal success event; do not label it a failure.
        if "locomotion" in summary:
            summary["locomotion"].pop("success_rate", None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(checkpoint=str(checkpoint), tasks=summary), indent=2) + "\n")
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
