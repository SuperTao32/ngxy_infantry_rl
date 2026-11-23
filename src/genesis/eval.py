"""评估入口：加载 v2 配置，支持键盘控制 locomotion、jump 和 recovery。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .config import FRAMEWORK, env_kwargs
from .robust.randomization_config import add_randomization_arguments, apply_randomization_arguments
from .terrains import TERRAIN_PRESETS, default_terrain_cfg, resolve_terrain_cfg
from .tools.run_utils import load_run_configs, resolve_checkpoint, resolve_run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an infantry policy with optional keyboard locomotion, jump and recovery commands")
    parser.add_argument("-e", "--exp-name", "--exp_name", default="wheel_leg_infantry")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--version")
    parser.add_argument("--checkpoint", "--ckpt", type=int)
    parser.add_argument("--terrain", "-terrain", choices=TERRAIN_PRESETS, default="plane")
    parser.add_argument("--difficulty", type=int)
    parser.add_argument("--terrain-size", type=float, nargs=2, metavar=("LENGTH", "WIDTH"))
    parser.add_argument("--steps", type=int, help="run until viewer closes if omitted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--control", choices=("manual", "auto"),
                        help="default: manual with viewer, auto without viewer")
    parser.add_argument("--output", type=Path)
    add_randomization_arguments(parser, evaluation=True)
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if not args.vis and args.steps is None:
        parser.error("--no-vis requires --steps")
    control = args.control or ("manual" if args.vis else "auto")
    if control == "manual" and not args.vis:
        parser.error("--control manual requires --vis for keyboard input")
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
    if configs.get("framework") != FRAMEWORK:
        parser.error("This entry requires a v2 environment configuration")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = (saved.get("infos") or {}).get("infantry_env", {})
    iteration = int(state.get("iteration", int(saved["iter"]) + 1))
    del saved
    configs["env_cfg"]["terrain"] = terrain
    apply_randomization_arguments(configs["env_cfg"], configs["obs_cfg"], args)
    import genesis as gs
    from .env import WheelLegInfantryEnv
    from .tools.interactive_env import InteractiveEnv
    from .policy_runner import PolicyRunner
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", seed=args.seed, logging_level="warning")
    try:
        env_class = InteractiveEnv if control == "manual" else WheelLegInfantryEnv
        env = env_class(
            1, **env_kwargs(configs), evaluation=True, initial_training_iteration=iteration,
            steps_per_iteration=configs["train_cfg"]["num_steps_per_env"], show_viewer=args.vis,
        )
        runner = PolicyRunner(env, configs["train_cfg"], None, device=str(gs.device))
        runner.load(str(checkpoint), load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False), map_location=str(gs.device))
        policy = runner.get_inference_policy(device=str(gs.device))
        print(f"[eval] checkpoint: {checkpoint}; terrain: {args.terrain}; control: {control}")
        with torch.inference_mode():
            policy.reset()
            obs = env.reset()
            keyboard = None
            if args.vis:
                from .tools.keyboard import KeyboardCommand
                keyboard = KeyboardCommand(env)
                print("[eval] arrows=vx/wz, PgUp/PgDn=height, Space=stop, Backspace=reset, Enter=focus")
                print("[eval] J=restart jump immediately, 1/2=JUMP_20/JUMP_40, R=random pose reset/recover (Backspace skips pose pool)")
                if control == "manual":
                    print("[eval] L=drive; continuous simulation, use Backspace to reset")
                else:
                    print("[eval] special terrains own commands; keyboard driving is available on plane")
            step = 0
            while args.steps is None or step < args.steps:
                if args.vis and not env.scene.viewer.is_alive():
                    break
                if keyboard is not None:
                    if keyboard.update():
                        policy.reset()
                    env._update_observations()
                    obs = env.get_observations()
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                step += 1
        summary = dict(checkpoint=str(checkpoint), terrain=terrain, control=control, steps=step, simulated_seconds=step * env.dt)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
