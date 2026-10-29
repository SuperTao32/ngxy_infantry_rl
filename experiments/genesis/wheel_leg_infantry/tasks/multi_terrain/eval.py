"""评估单一任务；默认采用该任务最终几何，保留自动命令和终点重置。"""

import argparse
from copy import deepcopy

import genesis as gs
import torch

from ...core.randomization_config import add_randomization_arguments, apply_randomization_arguments
from ...tools.run_utils import load_run_configs, load_runner_class, resolve_checkpoint, resolve_run_dir
from ...tools.tof_viewer import prepare_tof_for_viewer
from .config import TASK_NAMES, get_cfgs
from .curriculum import resolve_tasks
from .env import MultiTerrainEnv


def _override_intermittent(task, name, enabled):
    """显式开关优先于存档和所有难度覆盖；旧存档补齐对应地形的参数。"""
    if enabled is None:
        return
    if enabled and name == "locomotion":
        raise ValueError("--intermittent requires a traversal task, not locomotion")
    if enabled and "intermittent" not in task:
        task["intermittent"] = deepcopy(get_cfgs()[0]["multi_terrain"]["tasks"][name]["intermittent"])
    task.setdefault("intermittent", {})["probability"] = float(enabled)
    for override in task["curriculum"].get("overrides", {}).values():
        override.setdefault("intermittent", {})["probability"] = float(enabled)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", default="multi_terrain_v5")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--version", default=None)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--task", choices=TASK_NAMES, default="stairs_up")
    parser.add_argument("--difficulty", "--stage", dest="difficulty", type=int, default=None,
                        help="terrain LEVELS difficulty; default: maximum configured difficulty")
    parser.add_argument("--intermittent", action=argparse.BooleanOptionalAction, default=None,
                        help="force intermittent traversal on/off (default: use saved episode probability)")
    parser.add_argument("--parking-only", action="store_true", help="evaluate pre-terrain parking without crossing terrain")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    add_randomization_arguments(parser, evaluation=True)
    args = parser.parse_args(argv)
    run_dir = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
    configs = deepcopy(load_run_configs(run_dir))
    env_cfg, obs_cfg = configs["env_cfg"], configs["obs_cfg"]
    if "multi_terrain" not in env_cfg or args.task not in env_cfg["multi_terrain"]["tasks"]:
        parser.error("saved run has no requested multi_terrain task")
    if args.parking_only:
        if args.task == "locomotion" or args.intermittent is False:
            parser.error("--parking-only requires a traversal task and cannot combine with --no-intermittent")
        env_cfg["multi_terrain"].setdefault("parking_curriculum", {})["enabled"] = True
        args.intermittent = True
    task = env_cfg["multi_terrain"]["tasks"][args.task]
    task.update(enabled=True, weight=1.0)
    env_cfg["multi_terrain"]["tasks"] = {args.task: task}
    try:
        _override_intermittent(task, args.task, args.intermittent)
        resolved = resolve_tasks(env_cfg["multi_terrain"], require_locomotion=False)[args.task]
    except ValueError as exc:
        parser.error(str(exc))
    course = task["curriculum"]
    maximum = resolved["curriculum"]["max_difficulty"]
    difficulty = maximum if args.difficulty is None else args.difficulty
    if not course["min_difficulty"] <= difficulty <= maximum:
        parser.error("--difficulty is outside the task curriculum")
    if args.steps < 1:
        parser.error("--steps must be positive")
    course.update(enabled=False, difficulty=difficulty)
    stage = resolved["resolved_stages"][difficulty - course["min_difficulty"]]
    probability = stage.get("intermittent", {}).get("probability", 0.0)
    print(f"[eval] {args.task}/level_{difficulty}: intermittent_probability={probability:g}")
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    if not args.headless:
        prepare_tof_for_viewer(obs_cfg)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    env = MultiTerrainEnv(1, env_cfg, obs_cfg, configs["reward_cfg"], configs["command_cfg"],
                          evaluation=True, parking_evaluation=args.parking_only,
                          curriculum_cfg=dict(enabled=False, stages=[]), show_viewer=not args.headless)
    runner = load_runner_class()(env, configs["train_cfg"], None, device=gs.device)
    runner.load(str(resolve_checkpoint(run_dir, args.ckpt)),
                load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False), map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)
    observations = env.reset()
    wins = episodes = 0
    mode_wins = dict(ordinary=0, intermittent=0)
    mode_episodes = dict(ordinary=0, intermittent=0)
    parking_wins = parking_episodes = 0
    with torch.no_grad():
        for _ in range(args.steps):
            if not args.headless and not env.scene.viewer.is_alive():
                break
            observations, _, done, info = env.step(policy(observations))
            if done.any():
                metrics = info["episode"]
                prefix = f"task/{args.task}/"
                if args.parking_only:
                    won = int(metrics[prefix + "parking_pre_success"].item())
                    parking_wins += won
                    parking_episodes += 1
                    print(f"[eval] {args.task}/level_{difficulty}: parking_pre success={won} "
                          f"pre={parking_wins}/{parking_episodes}")
                    continue
                # step 已自动 reset；从本回合的结算指标读取模式，不能读取下一回合的状态。
                mode = "intermittent" if prefix + "intermittent_success" in metrics else "ordinary"
                won = int(metrics[prefix + "success"].item())
                episodes += 1
                wins += won
                mode_episodes[mode] += 1
                mode_wins[mode] += won
                detail = ""
                if mode == "intermittent":
                    detail = " " + " ".join(
                        f"{key}={metrics[prefix + 'intermittent_' + key].item():g}"
                        for key in ("pre_stop_completed", "stop_failure", "max_stop_drift_m"))
                print(f"[eval] {args.task}/level_{difficulty}: mode={mode} success={wins}/{episodes} "
                      f"ordinary={mode_wins['ordinary']}/{mode_episodes['ordinary']} "
                      f"intermittent={mode_wins['intermittent']}/{mode_episodes['intermittent']}{detail}")


if __name__ == "__main__":
    main()
