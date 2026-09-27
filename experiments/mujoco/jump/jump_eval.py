"""python -m experiments.mujoco.jump.jump_eval：locomotion 后按空格跳跃。"""

import argparse
import csv
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import numpy as np
import torch

from experiments.genesis.wheel_leg_infantry.tasks.jump.eval import get_eval_lin_vel_limits
from experiments.genesis.wheel_leg_infantry.tools.run_utils import (
    load_run_configs, resolve_checkpoint, resolve_recorded_run, resolve_run_dir,
)
from experiments.mujoco.locomotion.locomotion_eval import load_actor
from .jump_env import MujocoJumpEnv


def handle_key(env, key, velocity_limits):
    """Viewer 回调仅入队；所有物理状态修改在仿真线程执行。"""
    if key == ord("R"):
        env.reset()
        return "reset → locomotion"
    if key == 32:
        return "jump triggered" if env.trigger_jump() else "jump already active; ignored"
    if env.mode != "locomotion":
        return None
    if key in (ord("I"), 265, ord("K"), 264):
        delta = .1 if key in (ord("I"), 265) else -.1
        env.command_vx = float(np.clip(env.command_vx + delta, *velocity_limits))
    elif key == 259:  # Backspace; Space is reserved for jump.
        env.command_vx = 0.0
    else:
        return None
    return f"vx={env.command_vx:+.2f} m/s"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MuJoCo locomotion → Space → jump → locomotion")
    p.add_argument("-e", "--exp-name", default="infantry_jump_v6")
    p.add_argument("--log-root", default="log_shared")
    p.add_argument("--version")
    p.add_argument("--ckpt", type=int)
    p.add_argument("--checkpoint", type=Path, help="jump checkpoint；同目录必须有 cfgs.pkl")
    p.add_argument("--locomotion-log-root", help="迁移日志后，在该根目录查找记录的 teacher")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--duration", type=float, default=30, help="仿真秒数，默认 30")
    p.add_argument("--vx", type=float, default=0)
    p.add_argument("--jump-at", type=float, nargs="+", default=[], metavar="SECONDS",
                   help="在指定仿真时间触发，用于无窗口验证；跳跃中的触发忽略")
    p.add_argument("--handoff", choices=("landing", "horizon"), default="landing",
                   help="双轮落地交回 locomotion（默认），或完整执行训练时间窗")
    p.add_argument("--csv", type=Path)
    p.add_argument("--print-interval", type=int, default=50)
    args = p.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0 or not np.isfinite(args.vx):
        p.error("duration 必须为有限正数，vx 必须有限")
    if any(not np.isfinite(t) or t < 0 or t >= args.duration for t in args.jump_at):
        p.error("jump-at 必须在 [0, duration) 范围内")
    if args.print_interval < 0:
        p.error("print-interval 不能为负")
    if args.headless and not args.jump_at:
        p.error("无窗口模式请用 --jump-at 指定跳跃触发时间")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    if args.checkpoint:
        checkpoint, run = args.checkpoint, args.checkpoint.parent
    else:
        run = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
        checkpoint = resolve_checkpoint(run, args.ckpt)
    configs = load_run_configs(run)
    if not configs.get("source_locomotion"):
        raise ValueError("Jump configuration must record source_locomotion")
    source_run, source_checkpoint = resolve_recorded_run(configs["source_locomotion"], args.locomotion_log_root or args.log_root)
    source_configs = load_run_configs(source_run)
    limits = get_eval_lin_vel_limits(configs["env_cfg"], configs.get("curriculum_cfg", {}))
    if not limits[0] <= args.vx <= limits[1]:
        raise ValueError(f"vx must lie in the jump training command range {limits}")
    env = MujocoJumpEnv(configs, source_configs, args.vx, args.handoff)
    actors = {"locomotion": load_actor(source_configs, source_checkpoint, env.observations()),
              "jump": load_actor(configs, checkpoint, env.jump_observations())}
    print(f"[sim2sim] jump={checkpoint}\n[sim2sim] locomotion={source_checkpoint}")
    print(f"[sim2sim] obs=32/44; jump horizon={env.cycle_s:.2f}s; handoff={args.handoff}; flat ground")
    print("[keys] Space: jump; I/K or Up/Down: vx ±0.1; Backspace: stop; R: reset")
    keys = SimpleQueue()
    scheduled = iter(sorted(args.jump_at))
    next_trigger = next(scheduled, None)
    viewer = output = writer = None
    completed = 0
    try:
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.sim.model, env.sim.data, key_callback=keys.put)
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.5, -20, 135
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            output = args.csv.open("w", newline="")
        with torch.inference_mode():
            for step in range(int(np.ceil(args.duration / env.dt))):
                start = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    break
                while next_trigger is not None and step * env.dt + 1e-9 >= next_trigger:
                    keys.put(32)
                    next_trigger = next(scheduled, None)
                while True:
                    try:
                        key = keys.get_nowait()
                    except Empty:
                        break
                    message = handle_key(env, key, limits)
                    if message:
                        print(f"[control] t={step * env.dt:.2f} {message}", flush=True)
                obs = env.observations()
                mode = env.mode
                command = env.sim.commands.tolist()
                action = actors[mode](obs)
                env.step(action)
                row = {"step": step, "policy_mode": mode, **env.diagnostics()}
                if mode == "jump" and env.mode == "locomotion":
                    completed += 1
                    print(f"[handoff] {env.last_result}", flush=True)
                if args.print_interval and step % args.print_interval == 0:
                    print(f"[state] t={row['time']:.2f} mode={row['mode']} vx={row['vx']:+.3f} "
                          f"height={row['height']:.3f} clearance={row['clearance']:.3f} "
                          f"tilt={row['tilt_deg']:.2f} jump_t={row['jump_time']:.2f}", flush=True)
                if output:
                    row.update(zip(("command_vx", "command_wz", "command_height"), command))
                    values = obs["policy"][0].tolist()
                    row.update({f"obs_{i}": values[i] if i < len(values) else "" for i in range(44)})
                    row.update({f"action_{i}": x for i, x in enumerate(env.sim.actions.tolist())})
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                if row["tilt_deg"] > 60 or row["height"] < .08:
                    raise RuntimeError(f"Robot fell at t={row['time']:.2f}; stopped, CSV retained")
                if viewer is not None:
                    viewer.cam.lookat[:] = env.sim.data.xpos[env.sim.base_id]
                    viewer.sync()
                    time.sleep(max(0, env.dt - (time.monotonic() - start)))
    finally:
        if viewer is not None:
            viewer.close()
        if output is not None:
            output.close()
        print(f"[summary] triggered={env.jump_count}, handed_back={completed}, mode={env.mode}")


if __name__ == "__main__":
    main()
