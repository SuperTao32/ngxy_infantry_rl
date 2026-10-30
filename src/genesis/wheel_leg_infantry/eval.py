"""连续交互仿真入口：选择地形后，用键盘操纵同一机器人。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .config import env_kwargs
from .robust.randomization_config import add_randomization_arguments, apply_randomization_arguments
from .tasks.manager import task_configs
from .terrains import TERRAIN_PRESETS, default_terrain_cfg, resolve_terrain_cfg
from .tools.run_utils import load_run_configs, resolve_checkpoint, resolve_run_dir


def prepare_evaluation(configs, iteration):
    """冻结 checkpoint 对应课程，不随仿真步数推进。"""
    for course in [configs["curriculum_cfg"], *(task["curriculum_cfg"] for task in configs["env_cfg"]["tasks"].values())]:
        stages = [stage for stage in course.get("stages", []) if stage["start_iteration"] <= iteration]
        if course.get("enabled") and stages:
            from .curriculum import _deep_update
            targets = {}
            for stage in stages:
                _deep_update(targets, stage.get("targets", {}))
            course["stages"] = [dict(name="evaluation", start_iteration=0, targets=targets)]


def prepare_interactive_configs(configs, terrain):
    """提取部署命令参数；场景和复位不受训练任务配额/课程控制。"""
    saved = configs["env_cfg"]["tasks"]
    controls = {name: deepcopy(saved[name] if name in saved else task_configs(name))
                for name in ("locomotion", "recovery", "jump")}
    command_cfg = {**deepcopy(configs["command_cfg"]), **deepcopy(controls["locomotion"]["command_cfg"])}
    course = controls["locomotion"]["curriculum_cfg"]
    if course.get("enabled"):
        for stage in course.get("stages", []):
            command_cfg.update(deepcopy(stage.get("targets", {}).get("command_ranges", {})))
    zeros = [[0.0, 0.0] for _ in range(3)]
    carrier = dict(
        weight=1.0,
        env_cfg=dict(terrain=deepcopy(terrain), base_init_pos_range=[[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                     base_init_rpy_offset_range_deg=deepcopy(zeros), base_init_lin_vel_range=deepcopy(zeros),
                     base_init_ang_vel_range=deepcopy(zeros), reset_joint_pos=deepcopy(configs["env_cfg"]["default_joint_pos"])),
        reward_cfg=dict(reward_scales={}), command_cfg=command_cfg,
        curriculum_cfg=dict(enabled=False, stages=[]),
    )
    configs["env_cfg"]["terrain"] = deepcopy(terrain)
    configs["env_cfg"]["tasks"] = {"locomotion": carrier}
    configs["reward_cfg"]["reward_scales"] = {}
    return controls


def main(argv=None):
    parser = argparse.ArgumentParser(description="Drive one infantry robot continuously with keyboard commands")
    parser.add_argument("-e", "--exp-name", "--exp_name", default="wheel_leg_infantry")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--version")
    parser.add_argument("--checkpoint", "--ckpt", type=int)
    parser.add_argument("--terrain", "-terrain", choices=TERRAIN_PRESETS, default="plane")
    parser.add_argument("--difficulty", type=int, help="existing terrain difficulty level")
    parser.add_argument("--terrain-size", type=float, nargs=2, metavar=("LENGTH", "WIDTH"))
    parser.add_argument("--steps", type=int, help="optional step limit; default: run until the viewer closes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, help="save runtime summary")
    add_randomization_arguments(parser, evaluation=True)
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if not args.vis and args.steps is None:
        parser.error("--no-vis requires --steps")
    terrain = default_terrain_cfg(args.terrain)
    if args.terrain_size is not None:
        terrain["tile_size"] = args.terrain_size
    if args.difficulty is not None:
        terrain["difficulty"] = args.difficulty
    try:
        terrain = resolve_terrain_cfg(terrain)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    run_dir = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    configs = deepcopy(load_run_configs(run_dir))
    if configs.get("framework") != "wheel_leg_infantry_multitask_v1":
        parser.error("Evaluation requires a unified multi-task GRU checkpoint")
    prepare_evaluation(configs, int(checkpoint.stem.split("_")[-1]))
    controls = prepare_interactive_configs(configs, terrain)
    apply_randomization_arguments(configs["env_cfg"], configs["obs_cfg"], args)
    import genesis as gs
    import torch
    from .policy_runner import PolicyRunner
    from .tools.interactive_env import InteractiveEnv
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", seed=args.seed, logging_level="warning")
    try:
        env = InteractiveEnv(1, **env_kwargs(configs), control_configs=controls, evaluation=True,
                             steps_per_iteration=configs["train_cfg"]["num_steps_per_env"], show_viewer=args.vis)
        runner = PolicyRunner(env, configs["train_cfg"], None, device=str(gs.device))
        runner.load(str(checkpoint), load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False),
                    map_location=str(gs.device))
        policy = runner.get_inference_policy(device=str(gs.device))
        print(f"[eval] checkpoint: {checkpoint}")
        print(f"[eval] terrain: {args.terrain}; drive limits: {env.command_ranges}")
        print("[eval] arrows=vx/wz, PgUp/PgDn=height, Space=stop, J=jump, 1/2/3=jump mode, "
              "R=recover here, L=drive, Backspace=manual reset, Enter=focus")
        with torch.inference_mode():
            policy.reset()
            obs = env.reset()
            keyboard = None
            if args.vis:
                from .tools.interactive_viewer import KeyboardCommand
                keyboard = KeyboardCommand(env)
            step = 0
            while args.steps is None or step < args.steps:
                if args.vis and not env.scene.viewer.is_alive():
                    break
                if keyboard is not None:
                    if keyboard.process_requests():
                        policy.reset()
                    keyboard.write_to_env()
                    env._update_observations()
                    obs = env.get_observations()
                    keyboard.update_caption()
                obs, _, _, _ = env.step(policy(obs))
                step += 1
        summary = dict(checkpoint=str(checkpoint), terrain=terrain, steps=step, simulated_seconds=step * env.dt)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
