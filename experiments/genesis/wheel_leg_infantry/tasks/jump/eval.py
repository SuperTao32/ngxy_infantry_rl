"""Jump checkpoint 的重复单次跳跃可视化评估。"""

from __future__ import annotations

import argparse
import math
import threading
from copy import deepcopy

import torch

from ...tools.run_utils import (
    load_run_configs,
    load_runner_class,
    resolve_checkpoint,
    resolve_recorded_run,
    resolve_run_dir,
)


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


class KeyboardJumpCommand:
    """在线程间安全地传递前向速度命令和一次性跳跃请求。"""

    def __init__(self, env, lin_vel_limits, *, lin_step=0.5):
        from genesis.vis.keybindings import Key, KeyAction, Keybind

        self.env = env
        self.lin_vel_limits = tuple(float(value) for value in lin_vel_limits)
        if len(self.lin_vel_limits) != 2 or self.lin_vel_limits[0] > self.lin_vel_limits[1]:
            raise ValueError("lin_vel_limits must be an ordered pair")
        self.lin_step = float(lin_step)
        self._lin_vel = min(max(0.0, self.lin_vel_limits[0]), self.lin_vel_limits[1])
        self._jump_requested = False
        self._lock = threading.Lock()
        viewer = env.scene.viewer
        viewer.register_keybinds(
            self._keybind("command_forward", Key.UP, self._change_lin, self.lin_step),
            self._keybind("command_backward", Key.DOWN, self._change_lin, -self.lin_step),
            Keybind(
                "command_stop",
                Key.BACKSPACE,
                key_action=KeyAction.PRESS,
                callback=self.stop,
            ),
            Keybind(
                "trigger_jump",
                Key.SPACE,
                key_action=KeyAction.PRESS,
                callback=self.request_jump,
            ),
        )

    @staticmethod
    def _keybind(name, key, callback, amount):
        from genesis.vis.keybindings import KeyAction, Keybind

        return Keybind(
            name,
            key,
            key_action=KeyAction.PRESS,
            callback=callback,
            args=(amount,),
        )

    def _change_lin(self, amount):
        with self._lock:
            lower, upper = self.lin_vel_limits
            self._lin_vel = min(max(self._lin_vel + amount, lower), upper)
            lin_vel = self._lin_vel
        print(f"[jump eval] locomotion command vx={lin_vel:+.2f} m/s", flush=True)

    def stop(self):
        with self._lock:
            self._lin_vel = min(max(0.0, self.lin_vel_limits[0]), self.lin_vel_limits[1])
            lin_vel = self._lin_vel
        print(f"[jump eval] locomotion command vx={lin_vel:+.2f} m/s", flush=True)

    def current_lin_vel(self):
        with self._lock:
            return self._lin_vel

    def write_to_env(self):
        lin_vel = self.current_lin_vel()
        self.env.commands[:, 0] = lin_vel
        self.env.commands[:, 1] = 0.0

    def request_jump(self):
        with self._lock:
            self._jump_requested = True

    def consume_jump(self):
        with self._lock:
            requested = self._jump_requested
            self._jump_requested = False
        return requested

    def clear_jump(self):
        with self._lock:
            self._jump_requested = False


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-rand", action=argparse.BooleanOptionalAction, default=False,
                        help="enable dynamics randomization; evaluation defaults to nominal dynamics")
    parser.add_argument("-e", "--exp-name", type=str, default="infantry_jump_v6")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--print-interval", type=int, default=10,
                        help="每 N 个物理步打印 PD/电机扭矩；0 关闭，1 每步打印（影响帧率）")
    parser.add_argument("--terrain", choices=("flat", "step"), default="flat", help="评估场景：平地或单级台阶")
    parser.add_argument("--step-height", type=float, default=0.20, help="台阶高度，单位 m（默认 0.20）")
    parser.add_argument("--step-distance", type=float, default=1.0, help="初始机身中心到台阶前沿的 +x 距离，单位 m")
    parser.add_argument("--step-length", type=float, default=2.0, help="台面沿 x 的长度，单位 m")
    parser.add_argument("--step-width", type=float, default=2.0, help="台面沿 y 的宽度，单位 m")
    args = parser.parse_args(argv)
    if args.print_interval < 0:
        parser.error("--print-interval cannot be negative")
    for name in ("step_height", "step_distance", "step_length", "step_width"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def main():
    args = _parse_args()
    OnPolicyRunner = load_runner_class()

    run_dir = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
    checkpoint_path = resolve_checkpoint(run_dir, args.ckpt)
    configs = load_run_configs(run_dir)
    env_cfg = deepcopy(configs["env_cfg"])
    env_cfg.setdefault("domain_rand", {})["enabled"] = args.domain_rand
    env_cfg.update(show_FPS=False, viewer_realtime_factor=None)
    env_cfg["handoff_on_landing"] = True
    obs_cfg = configs["obs_cfg"]
    reward_cfg = deepcopy(configs["reward_cfg"])
    reward_cfg["reward_scales"] = {}
    command_cfg = deepcopy(configs["command_cfg"])
    train_cfg = configs["train_cfg"]

    source_info = configs.get("source_locomotion")
    if not source_info:
        raise ValueError("jump run does not record its source_locomotion checkpoint")
    source_run_dir, source_checkpoint = resolve_recorded_run(source_info, args.log_root)
    source_configs = load_run_configs(source_run_dir)

    print(f"[jump eval] run: {run_dir}")
    print(f"[jump eval] checkpoint: {checkpoint_path}")
    print(f"[jump eval] wheel_clearance_target={env_cfg['wheel_clearance_target_m']:.3f} m")
    print(f"[jump eval] locomotion source: {source_checkpoint}")

    import genesis as gs

    from .eval_env import JumpEvalEnv
    from .staged_runner import build_frozen_locomotion_actor
    from .warm_start import validate_locomotion_source

    gs.init(backend=gs.gpu)
    env = JumpEvalEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg={"enabled": False, "stages": []},
        steps_per_iteration=train_cfg["num_steps_per_env"],
        show_viewer=True,
        step_terrain=(
            {"height": args.step_height, "distance": args.step_distance,
             "length": args.step_length, "width": args.step_width}
            if args.terrain == "step" else None
        ),
    )
    if args.terrain == "step":
        print(
            f"[jump eval] terrain=step; height={args.step_height:.3f} m, "
            f"front_x={env.step_front_x:.3f} m, length={args.step_length:.3f} m, "
            f"width={args.step_width:.3f} m; approach with UP/DOWN, SPACE to jump",
            flush=True,
        )
    else:
        print("[jump eval] terrain=flat", flush=True)
    validate_locomotion_source(source_configs)
    locomotion_actor = build_frozen_locomotion_actor(
        source_configs,
        source_checkpoint,
        env.get_locomotion_observations(),
        num_actions=env.num_actions,
        device=gs.device,
    )
    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
    runner.load(str(checkpoint_path), map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)

    jump_horizon = int(round(env.phase_cycle_s / env.dt))
    locomotion_max_episode_length = 2**30
    lin_vel_limits = get_eval_lin_vel_limits(env_cfg, configs.get("curriculum_cfg", {}))
    jump_obs = None
    jump_step = 0
    total_jump_steps = 0

    # 默认进入 locomotion。等待空格期间放宽 episode horizon，避免每 1.6 秒重置。
    locomotion_obs = env.prepare_locomotion_warmup(env_cfg["locomotion_warmup"]["command_ranges"])
    controls = KeyboardJumpCommand(env, lin_vel_limits)
    controls.write_to_env()
    locomotion_obs = env.get_locomotion_observations()
    env.max_episode_length = locomotion_max_episode_length
    env.start_motor_logging(args.print_interval)
    print(
        f"[jump eval] motor logging every {args.print_interval} steps (0=off); "
        "time=local wall clock, elapsed=wall seconds, sim=simulation seconds; "
        "tau_motor=current clamped control torque snapshot (Nm), P_est/D_est=PD components; "
        "simulation pacing=uncapped",
        flush=True,
    )
    print(
        "[jump eval] locomotion active; UP/DOWN: vx +/-0.1 m/s, "
        "BACKSPACE: vx=0, SPACE: jump",
        flush=True,
    )
    print(
        f"[jump eval] locomotion vx range={lin_vel_limits[0]:+.2f}..{lin_vel_limits[1]:+.2f} m/s",
        flush=True,
    )

    with torch.inference_mode():
        while env.scene.viewer.is_alive():
            if jump_obs is None:
                controls.write_to_env()
                locomotion_obs = env.get_locomotion_observations()
                if controls.consume_jump():
                    command_vx = controls.current_lin_vel()
                    env.max_episode_length = jump_horizon
                    jump_obs = env.begin_jump_rollout()
                    jump_step = 0
                    wheel_clearance = env.wheel_clearance_target
                    locked_vx = float(env.commands[0, 0].item())
                    print(
                        f"[jump eval] jump triggered; wheel_clearance={wheel_clearance:.3f} m, "
                        f"command_vx={command_vx:+.2f} m/s, locked_command_vx={locked_vx:+.2f} m/s",
                        flush=True,
                    )
                    continue

                actions = locomotion_actor(locomotion_obs.to(gs.device))
                _, _, dones, _ = env.step(actions.to(env.device))
                if torch.any(dones):
                    print("[jump eval] locomotion reset after unexpected termination", flush=True)
                    locomotion_obs = env.prepare_locomotion_warmup(
                        env_cfg["locomotion_warmup"]["command_ranges"]
                    )
                    controls.write_to_env()
                    locomotion_obs = env.get_locomotion_observations()
                    env.max_episode_length = locomotion_max_episode_length
                continue

            actions = policy(jump_obs)
            jump_obs, _, dones, _ = env.step(actions)
            jump_step += 1
            total_jump_steps += 1
            if args.print_interval and total_jump_steps % args.print_interval == 0:
                diag = env.get_jump_diagnostics()
                print(
                    f"[step {total_jump_steps:05d}] phase={diag['phase_name']:<7} "
                    f"clearance={diag['wheel_clearance'].item():+.3f}/{wheel_clearance:.3f} m "
                    f"max={diag['max_wheel_clearance'].item():.3f} m "
                    f"command=[{diag['command_vx'].item():+.2f}, "
                    f"{diag['command_wz'].item():+.2f}, {diag['command_base_height'].item():.2f}] "
                    f"base_to_wheel_bottom={diag['base_to_wheel_bottom_distance'].item():.3f} m "
                    f"vz={diag['world_vertical_velocity'].item():+.3f} m/s "
                    f"az={diag['world_vertical_acceleration'].item():+.2f} m/s^2 "
                    f"takeoff_vz={diag['takeoff_contact_vertical_velocity'].item():+.2f} m/s "
                    f"takeoff={diag['has_taken_off']} landed={diag['has_landed']}",
                    flush=True,
                )

            if torch.any(dones):
                print("[jump eval] jump terminated early; returning to locomotion", flush=True)
            elif env_cfg.get("handoff_on_landing", False) and torch.all(env.ready_for_locomotion()):
                env.finish_jump_rollout()
                print("[jump eval] both wheels landed; returning to locomotion", flush=True)
            elif jump_step < jump_horizon:
                continue
            else:
                env.finish_jump_rollout()
                print("[jump eval] jump finished; returning to locomotion", flush=True)

            if args.terrain == "step":
                print(
                    f"[jump eval] end_on_step={env.wheels_on_step()[0].item()} "
                    "(both wheels contacting the step top at rollout end)",
                    flush=True,
                )

            # 跳跃期间再次按下的空格不排队，回到 locomotion 后需重新按键。
            controls.clear_jump()
            # 正常交接保持物理状态；异常终止才重新初始化机器人。
            resume = env.prepare_locomotion_warmup if torch.any(dones) else env.resume_locomotion
            locomotion_obs = resume(
                env_cfg["locomotion_warmup"]["command_ranges"]
            )
            controls.write_to_env()
            locomotion_obs = env.get_locomotion_observations()
            env.max_episode_length = locomotion_max_episode_length
            jump_obs = None


if __name__ == "__main__":
    main()
