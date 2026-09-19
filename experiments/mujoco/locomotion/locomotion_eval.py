"""运行：python -m experiments.mujoco.locomotion_eval --help。"""

import argparse
from copy import deepcopy
import csv
from pathlib import Path
import threading
import time

import numpy as np
import torch
from rsl_rl.models import MLPModel

from experiments.genesis.wheel_leg_infantry.tools.run_utils import load_run_configs, resolve_checkpoint, resolve_run_dir
from .locomotion_env import MujocoLocomotionEnv


def load_actor(configs, checkpoint, obs):
    train = deepcopy(configs["train_cfg"])
    actor_cfg = train["actor"]
    if actor_cfg.pop("class_name") != "MLPModel" or train["obs_groups"]["actor"] != ["policy"]:
        raise ValueError("Only MLPModel with the locomotion policy observation group is supported")
    actor = MLPModel(obs, train["obs_groups"], "actor", configs["env_cfg"]["num_actions"], **actor_cfg)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    return actor.eval()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Genesis → MuJoCo locomotion 平地评估")
    p.add_argument("-e", "--exp-name", default="infantry_locomotion_v3")
    p.add_argument("--log-root", default="log_shared")
    p.add_argument("--version", default=None)
    p.add_argument("--ckpt", type=int)
    p.add_argument("--checkpoint", type=Path, help="完整模型路径；同目录必须有 cfgs.pkl")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--duration", type=float, default=30, help="仿真秒数")
    p.add_argument("--vx", type=float, default=0)
    p.add_argument("--wz", type=float, default=0)
    p.add_argument("--height", type=float, default=.22)
    p.add_argument("--csv", type=Path, help="保存每个控制周期的状态、观测和动作")
    p.add_argument("--print-interval", type=int, default=50)
    args = p.parse_args(argv)
    if not all(np.isfinite(x) for x in (args.duration, args.vx, args.wz, args.height)) or args.duration <= 0 or args.height <= 0:
        p.error("duration/height 必须为有限正数，速度命令必须有限")
    if args.print_interval < 0:
        p.error("print-interval 不能为负")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    if args.checkpoint:
        checkpoint = args.checkpoint
        run = checkpoint.parent
    else:
        run = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
        checkpoint = resolve_checkpoint(run, args.ckpt)
    configs = load_run_configs(run)
    if "jump" in configs or configs["env_cfg"].get("handoff_on_landing"):
        raise ValueError("Use a locomotion checkpoint")
    env = MujocoLocomotionEnv(configs, (args.vx, args.wz, args.height))
    obs = env.observations()
    actor = load_actor(configs, checkpoint, obs)
    print(f"[sim2sim] checkpoint={checkpoint}\n[sim2sim] obs={obs['policy'].shape[-1]}, control=50 Hz, physics=1000 Hz, plane terrain")
    print(f"[sim2sim] commands vx/wz/height={env.commands.tolist()}")
    pending = []
    lock = threading.Lock()

    def key_callback(key):
        with lock:
            pending.append(key)

    viewer = None
    output = None
    rows = []
    try:
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback)
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            print("[keys] I/K: vx ±2.0, J/L: wz ±1.0, U/O: height ±0.1, Space: stop, R: reset")
        writer = None
        steps = int(np.ceil(args.duration / env.dt))
        with torch.inference_mode():
            for step in range(steps):
                start = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    break
                with lock:
                    keys, pending[:] = pending[:], []
                for key in keys:
                    if key == ord("R"):
                        obs = env.reset()
                    elif key == 32:
                        env.commands[:2] = 0
                    else:
                        changes = {ord("I"): (0, 2.0), ord("K"): (0, -2.0), ord("J"): (1, 1.0),
                                   ord("L"): (1, -1.0), ord("U"): (2, .1), ord("O"): (2, -.1)}
                        if key in changes:
                            idx, delta = changes[key]
                            env.commands[idx] += delta
                            env.commands[2].clamp_(.1, .4)
                obs = env.observations()
                action = actor(obs)
                input_obs = obs["policy"][0].tolist()
                env.step(action)
                row = env.diagnostics()
                rows.append(row)
                if args.csv:
                    record = {"step": step, **row}
                    record.update(zip(("command_vx", "command_wz", "command_height"), env.commands.tolist()))
                    record.update({f"obs_{i}": x for i, x in enumerate(input_obs)})
                    record.update({f"action_{i}": x for i, x in enumerate(env.actions.tolist())})
                    if writer is None:
                        args.csv.parent.mkdir(parents=True, exist_ok=True)
                        output = args.csv.open("w", newline="")
                        writer = csv.DictWriter(output, fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                if args.print_interval and step % args.print_interval == 0:
                    print("[state] " + " ".join(f"{k}={v:.3f}" for k, v in row.items()), flush=True)
                if row["tilt_deg"] > 60 or row["height"] < .08:
                    raise RuntimeError(f"Robot fell; stopped without automatic reset: {row}")
                if viewer is not None:
                    viewer.cam.lookat[:] = env.data.xpos[env.base_id]
                    viewer.sync()
                    time.sleep(max(0, env.dt - (time.monotonic() - start)))
    finally:
        if viewer is not None:
            viewer.close()
        if output is not None:
            output.close()
        if rows:
            print(f"[summary] steps={len(rows)} max_tilt={max(r['tilt_deg'] for r in rows):.2f} deg "
                  f"height_range=[{min(r['height'] for r in rows):.3f}, {max(r['height'] for r in rows):.3f}] m")


if __name__ == "__main__":
    main()
