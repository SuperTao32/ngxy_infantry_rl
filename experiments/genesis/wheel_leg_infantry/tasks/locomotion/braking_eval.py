"""平地速度扫描：策略达到目标速度后发送零速命令，测量连续停稳距离。"""

import argparse
import csv
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from ...core.randomization_config import add_randomization_arguments, apply_randomization_arguments
from ...tools.run_utils import load_run_configs, load_runner_class, resolve_checkpoint, resolve_run_dir


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--exp_name", default="locomotion_v5")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--version", help="version_0003 or 3; omitted: latest valid run")
    parser.add_argument("--ckpt", type=int, help="checkpoint number; omitted: latest")
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0], help="signed target speeds, m/s")
    parser.add_argument("--trials", type=int, default=3, help="trials per target speed")
    parser.add_argument("--height", type=float, default=None, help="base height, m; omitted: saved command range midpoint")
    parser.add_argument("--settle-time", type=float, default=2.0, help="initial zero-command settling time, s")
    parser.add_argument("--accel-timeout", type=float, default=10.0, help="maximum time to reach target speed, s")
    parser.add_argument("--speed-tolerance", type=float, default=0.1, help="absolute target speed tolerance, m/s")
    parser.add_argument("--speed-hold-time", type=float, default=0.5, help="continuous target tracking required before braking, s")
    parser.add_argument("--stop-timeout", type=float, default=10.0, help="maximum braking time, s")
    parser.add_argument("--stop-speed", type=float, default=0.05, help="maximum 3D base linear speed when stopped, m/s")
    parser.add_argument("--stop-angular-speed", type=float, default=0.1, help="maximum 3D base angular speed when stopped, rad/s")
    parser.add_argument("--stop-hold-time", type=float, default=0.5, help="continuous stable stop confirmation time, s")
    parser.add_argument("--max-tilt", type=float, default=10.0, help="maximum absolute roll/pitch for stable tracking/stop, degrees")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--vis", action="store_true", help="show Genesis viewer")
    parser.add_argument("--output-dir", type=Path, help="new result directory; omitted: run/braking_eval/timestamp")
    add_randomization_arguments(parser, evaluation=True)
    parser.set_defaults(sensor_noise=False)
    args = parser.parse_args(argv)
    for name in ("settle_time", "accel_timeout", "speed_tolerance", "speed_hold_time", "stop_timeout",
                 "stop_speed", "stop_angular_speed", "stop_hold_time", "max_tilt"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.height is not None and (not math.isfinite(args.height) or args.height <= 0):
        parser.error("--height must be positive and finite")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if any(not math.isfinite(v) for v in args.speeds):
        parser.error("--speeds must be finite")
    if len(set(args.speeds)) != len(args.speeds):
        parser.error("--speeds must not contain duplicates")
    if args.speed_hold_time > args.accel_timeout or args.stop_hold_time > args.stop_timeout:
        parser.error("hold times must not exceed their corresponding timeouts")
    return args


def _state(env):
    # 真值用于计量，不改策略输入；轮速不能代表打滑时的实际车速。
    position = env.base_pos[0].detach().cpu().tolist()
    velocity = env.base_lin_vel[0].detach().cpu().tolist()
    angular = env.base_ang_vel[0].detach().cpu().tolist()
    euler = env.base_euler[0].detach().cpu().tolist()
    return dict(x=position[0], y=position[1], z=position[2], vx=velocity[0], vy=velocity[1],
                speed=math.hypot(*velocity), angular_speed=math.hypot(*angular),
                tilt=max(abs(euler[0]), abs(euler[1])), yaw=euler[2],
                base_contact=float(env.base_contact[0].item()))


def _healthy(state, args):
    return state["tilt"] <= args.max_tilt and state["base_contact"] < 0.5


class StopMeasurement:
    """累积世界水平面路程；低速过零或反弹会重新开始停稳确认。"""

    def __init__(self, state, target_speed, dt, args):
        self.origin = self.previous = (state["x"], state["y"])
        yaw = math.radians(state["yaw"])
        sign = -1 if target_speed < 0 else 1
        self.direction = (sign * math.cos(yaw), sign * math.sin(yaw))
        self.path = self.max_forward = self.max_excursion = 0.0
        self.displacement = self.forward = 0.0
        self.steps = self.stable_steps = 0
        self.dt, self.args = dt, args
        self.required_steps = math.ceil(args.stop_hold_time / dt)

    def update(self, state):
        point = (state["x"], state["y"])
        self.path += math.dist(self.previous, point)
        self.previous = point
        self.displacement = math.dist(self.origin, point)
        self.forward = sum((p - o) * d for p, o, d in zip(point, self.origin, self.direction))
        self.max_forward = max(self.max_forward, self.forward)
        self.max_excursion = max(self.max_excursion, self.displacement)
        self.steps += 1
        stable = (_healthy(state, self.args) and state["speed"] <= self.args.stop_speed
                  and state["angular_speed"] <= self.args.stop_angular_speed)
        self.stable_steps = self.stable_steps + 1 if stable else 0
        return self.stable_steps >= self.required_steps

    def result(self):
        return dict(brake_elapsed_s=self.steps * self.dt, path_distance_m=self.path,
                    displacement_m=self.displacement, forward_distance_m=self.forward,
                    max_forward_distance_m=self.max_forward, max_excursion_m=self.max_excursion)


def run_trial(env, policy, speed, trial, height, args, trace):
    """单次试验；done 后环境已自动 reset，必须在读取位置前退出。"""
    env.reset()
    if callable(getattr(policy, "reset", None)):
        policy.reset()
    result = dict(target_speed_mps=speed, trial=trial, status="", actual_brake_vx_mps=None,
                  actual_brake_speed_mps=None, brake_elapsed_s=None, stop_time_s=None,
                  path_distance_m=None, displacement_m=None, forward_distance_m=None,
                  max_forward_distance_m=None, max_excursion_m=None)
    total_steps = 0

    def step(command, phase, phase_step):
        nonlocal total_steps
        env.commands[:, 0] = command
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = height
        env._update_observations()
        actions = policy(env.get_observations())
        _, _, dones, infos = env.step(actions)
        total_steps += 1
        if bool(dones[0].item()):
            timed_out = bool(infos["time_outs"][0].item())
            return None, f"{phase}_{'env_timeout' if timed_out else 'terminated'}"
        state = _state(env)
        if not all(math.isfinite(value) for value in state.values()):
            return None, f"{phase}_nonfinite"
        trace.writerow(dict(target_speed_mps=speed, trial=trial, phase=phase,
                            time_s=total_steps * env.dt, phase_time_s=phase_step * env.dt,
                            command_vx_mps=command, **state))
        return state, None

    for i in range(math.ceil(args.settle_time / env.dt)):
        state, failure = step(0.0, "settle", i + 1)
        if failure:
            result["status"] = failure
            return result

    held = 0
    for i in range(math.ceil(args.accel_timeout / env.dt)):
        state, failure = step(speed, "accel", i + 1)
        if failure:
            result["status"] = failure
            return result
        tracking = (abs(state["vx"] - speed) <= args.speed_tolerance
                    and abs(state["vy"]) <= args.speed_tolerance
                    and state["angular_speed"] <= args.stop_angular_speed and _healthy(state, args))
        held = held + 1 if tracking else 0
        if held >= math.ceil(args.speed_hold_time / env.dt):
            break
    else:
        result["status"] = "speed_not_reached"
        return result

    result.update(actual_brake_vx_mps=state["vx"], actual_brake_speed_mps=state["speed"])
    measurement = StopMeasurement(state, speed, env.dt, args)
    # t=0 就在下一次推理前：保留控制器原有动作延迟及所有策略/传感器历史。
    trace.writerow(dict(target_speed_mps=speed, trial=trial, phase="brake", time_s=total_steps * env.dt,
                        phase_time_s=0.0, command_vx_mps=0.0, **state))
    for i in range(math.ceil(args.stop_timeout / env.dt)):
        state, failure = step(0.0, "brake", i + 1)
        if failure:
            result["status"] = failure
            # 失败不报告不完整的路程为刹停距离。
            return result
        if measurement.update(state):
            result.update(status="stopped", stop_time_s=measurement.steps * env.dt, **measurement.result())
            return result
    result.update(status="stop_timeout", **measurement.result())
    return result


def summarize(results, speeds):
    summaries = []
    for speed in speeds:
        rows = [row for row in results if row["target_speed_mps"] == speed]
        stopped = [row for row in rows if row["status"] == "stopped"]
        summary = dict(target_speed_mps=speed, trials=len(rows), stopped=len(stopped), failed=len(rows) - len(stopped))
        for name in ("actual_brake_vx_mps", "path_distance_m", "max_forward_distance_m", "stop_time_s"):
            values = [row[name] for row in stopped]
            for stat, fn in (("mean", mean), ("std", pstdev), ("max", max)):
                summary[f"{name}_{stat}"] = fn(values) if values else None
        summaries.append(summary)
    return summaries


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = _parse_args(argv)
    # 延迟重依赖导入，--help 和距离计算单测无需启动仿真。
    import genesis as gs
    import torch
    from ...terrains import default_terrain_cfg
    from .env import LocomotionEnv
    from .eval import _prepare_task_configs

    run_dir = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
    checkpoint = resolve_checkpoint(run_dir, args.ckpt)
    configs = load_run_configs(run_dir)
    env_cfg, command_cfg = _prepare_task_configs(configs)
    height = args.height if args.height is not None else mean(command_cfg["base_height_range"])
    env_cfg["terrain"] = default_terrain_cfg("plane")
    duration = args.settle_time + args.accel_timeout + args.stop_timeout + 2.0
    env_cfg.update(episode_length_s=duration, resampling_time_s=duration, show_FPS=False)
    # 每次从原点、正立静止出发，保留保存配置中的初始高度中点。
    legacy_pos = env_cfg.get("base_init_pos", [0.0, 0.0, 0.22])
    z_range = env_cfg.get("base_init_pos_z_range", [legacy_pos[2]] * 2)
    z = mean(env_cfg.get("base_init_pos_range", [[0, 0], [0, 0], z_range])[2])
    env_cfg["base_init_pos_range"] = [[0.0, 0.0], [0.0, 0.0], [z, z]]
    for key in ("base_init_rpy_offset_range_deg", "base_init_lin_vel_range", "base_init_ang_vel_range"):
        env_cfg[key] = [[0.0, 0.0] for _ in range(3)]
    obs_cfg = deepcopy(configs["obs_cfg"])
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    reward_cfg = deepcopy(configs["reward_cfg"])
    reward_cfg.update(reward_scales={}, terrain_reward_scales={})
    output = args.output_dir or run_dir / "braking_eval" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(checkpoint=str(checkpoint.resolve()), arguments=vars(args), height_m=height,
                    env_cfg=env_cfg, obs_cfg=obs_cfg, command_cfg=command_cfg)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[braking] checkpoint: {checkpoint}\n[braking] output: {output}\n[braking] plane, height={height:.3f} m", flush=True)
    print("[braking] distance/time include the continuous stop confirmation window; failed trials excluded from summary", flush=True)
    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, seed=args.seed, logging_level="warning")
    env = None
    try:
        env = LocomotionEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg,
                            curriculum_cfg={"enabled": False, "stages": []},
                            steps_per_iteration=configs["train_cfg"]["num_steps_per_env"], show_viewer=args.vis)
        runner = load_runner_class()(env, deepcopy(configs["train_cfg"]), None, device=gs.device)
        runner.load(str(checkpoint), map_location=gs.device,
                    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        policy = runner.get_inference_policy(device=gs.device)
        if args.vis:
            env.focus_viewer()
        metadata["dt_s"] = env.dt
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        results = []
        with (output / "trace.csv").open("w", newline="", encoding="utf-8") as file, torch.no_grad():
            trace = csv.DictWriter(file, fieldnames=["target_speed_mps", "trial", "phase", "time_s", "phase_time_s",
                                                   "command_vx_mps", *_state(env)])
            trace.writeheader()
            for speed in args.speeds:
                for trial in range(1, args.trials + 1):
                    result = run_trial(env, policy, speed, trial, height, args, trace)
                    results.append(result)
                    file.flush()
                    _write_csv(output / "trials.csv", results)
                    _write_csv(output / "summary.csv", summarize(results, args.speeds))
                    detail = ""
                    if result["status"] == "stopped":
                        detail = (f" actual_vx={result['actual_brake_vx_mps']:.3f} m/s"
                                  f" distance={result['path_distance_m']:.3f} m time={result['stop_time_s']:.2f} s")
                    print(f"[braking] vx={speed:+.2f} trial={trial}/{args.trials}: {result['status']}{detail}", flush=True)
        print(f"[braking] results saved: {output}")
    finally:
        if env is not None:
            env.scene.destroy()
        gs.destroy()


if __name__ == "__main__":
    main()
