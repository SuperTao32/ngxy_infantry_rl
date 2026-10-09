"""Locomotion checkpoint 的交互式可视化评估。"""

import argparse
from copy import deepcopy

import genesis as gs
import torch

from ...terrains import TERRAIN_PRESETS, default_terrain_cfg, resolve_terrain_cfg
from ...core.randomization import add_randomization_arguments, apply_randomization_arguments
from ...tools.run_utils import (
    load_run_configs,
    load_runner_class,
    resolve_checkpoint,
    resolve_run_dir,
)
from .config_common import get_final_command_cfg
from .env import LocomotionEnv
from .interactive_viewer import KeyboardCommand, format_tensor



def _parse_args(argv=None):
    """解析并校验评估命令行参数。"""
    parser = argparse.ArgumentParser()
    add_randomization_arguments(parser, evaluation=True)
    parser.add_argument("-e", "--exp_name", type=str, default="infantry_locomotion_v3")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--version", type=str, default=None, help="version_0003 or 3; default: latest valid run")
    parser.add_argument("--ckpt", type=int, default=None, help="checkpoint number; default: latest in selected run")
    parser.add_argument("--seed", type=int, default=1, help="terrain/reset random seed")
    parser.add_argument(
        "--terrain",
        choices=TERRAIN_PRESETS,
        default=None,
        help="override the saved terrain to test the same policy on another surface",
    )
    parser.add_argument("--difficulty", type=int, default=None, help="terrain level from the corresponding terrains/<preset>.py module")
    parser.add_argument(
        "--terrain-size",
        type=float,
        nargs=2,
        metavar=("LENGTH", "WIDTH"),
        default=None,
        help="terrain tile size in meters; default: default_terrain_cfg tile_size",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=50,
        help="print named observations and PD/Kd diagnostics every N steps; 0 disables periodic output",
    )
    args = parser.parse_args(argv)

    if args.print_interval < 0:
        parser.error("--print-interval cannot be negative")
    if args.terrain_size is not None and any(value <= 0.0 for value in args.terrain_size):
        parser.error("--terrain-size values must be positive")
    return args


def _apply_terrain_overrides(env_cfg, args):
    """保留所选 preset，几何使用当前代码默认值，再应用显式 CLI 覆盖。"""
    preset = args.terrain or env_cfg.get("terrain", {}).get("preset", "plane")
    if preset == "flat":
        preset = "plane"
    env_cfg["terrain"] = default_terrain_cfg(preset)
    if args.terrain_size is not None:
        env_cfg["terrain"]["tile_size"] = list(args.terrain_size)
    if args.difficulty is not None:
        env_cfg["terrain"]["difficulty"] = args.difficulty
        env_cfg["terrain"] = resolve_terrain_cfg(env_cfg["terrain"])


def print_evaluation_diagnostics(env, step):
    """按观测分组打印策略输入，并输出关节与轮子的 PD/Kd 诊断。"""
    print(
        f"\n[eval step {step}] scaled observations "
        f"(policy={env.obs_buf.shape[-1]}, critic={env.critic_obs_buf.shape[-1]})"
    )
    for name, value in env.get_observation_components().items():
        print(f"  obs.{name:<20} {format_tensor(value)}")

    velocity = env.get_velocity_estimator_diagnostics()
    terrain_type = env.terrain.tile_type(env.terrain_tile_index[0].item())
    print(
        f"  terrain                  {env.terrain.preset}/{terrain_type}; "
        f"ground={env.terrain_height[0].item():+.4f} m, base_height={env.base_height[0].item():+.4f} m"
    )
    print(f"  velocity source          {velocity['source']}")
    print(f"  velocity true            {velocity['true_forward_velocity'].item():+.4f} m/s")
    print(f"  velocity estimated       {velocity['estimated_forward_velocity'].item():+.4f} m/s")
    print(f"  velocity from wheels     {velocity['wheel_forward_velocity'].item():+.4f} m/s")
    print(f"  IMU specific force       {format_tensor(velocity['imu_specific_force'])} m/s^2")
    print(
        "  acceleration corrected  "
        f"{velocity['gravity_compensated_forward_acceleration'].item():+.4f} m/s^2"
    )

    randomization = env.domain_rand.diagnostics()
    print(f"  randomization strength   target={randomization['domain_rand_strength']:.2f}, "
          f"episode={randomization['domain_rand_episode_strength'].item():.2f}")
    print(f"  friction ratio           {randomization['friction_ratio'].item():.3f}")
    print(f"  added base mass          {randomization['added_mass_kg'].item():+.3f} kg")
    print(f"  COM displacement         {format_tensor(randomization['com_displacement_m'])} m")
    print(f"  motor strength           {randomization['motor_strength'].item():.3f}")
    print(f"  joint motor offsets      {format_tensor(randomization['joint_motor_offsets_rad'])} rad")
    print(f"  spring preload force     {format_tensor(randomization['gas_spring_preload_force'])} N")
    print(f"  spring stiffness         {format_tensor(randomization['gas_spring_stiffness'])} N/m")
    print(f"  spring damping           {format_tensor(randomization['gas_spring_damping'])} N·s/m")
    print(f"  push target/force        {randomization['push_target']}: {format_tensor(randomization['push_force_world_N'])} N (world)")
    if "passive_joint_damping" in randomization:
        print("  passive joint names      " + ", ".join(randomization["passive_joint_names"]))
        print(f"  passive joint damping    {format_tensor(randomization['passive_joint_damping'])} N·m·s/rad")
        print(f"  passive joint friction   {format_tensor(randomization['passive_joint_frictionloss'])} N·m")
    pd = env.get_pd_diagnostics()
    print("  joint names             " + ", ".join(pd["joint_names"]))
    print(f"  joint Kp                {format_tensor(pd['joint_kp'])}")
    print(f"  joint Kd                {format_tensor(pd['joint_kd'])}")
    print(f"  joint target position   {format_tensor(pd['joint_target_pos'])}")
    print(f"  joint position          {format_tensor(pd['joint_pos'])}")
    print(f"  joint velocity qdot     {format_tensor(pd['joint_vel'])}")
    print(f"  joint -Kd*qdot (estimate) {format_tensor(pd['joint_kd_damping'])}")
    print("  wheel names             " + ", ".join(pd["wheel_names"]))
    print(f"  wheel Kd                {format_tensor(pd['wheel_kd'])}")
    print(f"  wheel target velocity   {format_tensor(pd['wheel_target_vel'])}")
    print(f"  wheel velocity qdot     {format_tensor(pd['wheel_vel'])}")
    print(f"  wheel -Kd*qdot (estimate) {format_tensor(pd['wheel_kd_damping'])}", flush=True)


def main():
    args = _parse_args()
    OnPolicyRunner = load_runner_class()

    # 默认选择“含配置且含 checkpoint”的最大版本号，自动跳过中断训练。
    run_dir = resolve_run_dir(
        args.log_root,
        args.exp_name,
        args.version,
        require_checkpoint=True,
    )
    checkpoint_path = resolve_checkpoint(run_dir, args.ckpt)
    configs = load_run_configs(run_dir)
    env_cfg = deepcopy(configs["env_cfg"])
    _apply_terrain_overrides(env_cfg, args)
    obs_cfg = deepcopy(configs["obs_cfg"])
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    reward_cfg = deepcopy(configs["reward_cfg"])
    command_cfg = get_final_command_cfg(configs["command_cfg"], configs["curriculum_cfg"])
    train_cfg = configs["train_cfg"]
    reward_cfg["reward_scales"] = {}

    print(f"[eval] run:        {run_dir}")
    print(f"[eval] checkpoint: {checkpoint_path}")
    print(f"[eval] terrain:    {env_cfg.get('terrain', {}).get('preset', 'plane')}")
    if env_cfg.get("terrain", {}).get("preset", "plane") != "plane":
        print(f"[eval] terrain size: {env_cfg['terrain']['tile_size']} m")
    print(
        "[eval] commands:   "
        f"vx={command_cfg['lin_vel_range']} "
        f"wz={command_cfg['ang_vel_range']} "
        f"base_height={command_cfg['base_height_range']}"
    )

    gs.init(backend=gs.gpu, seed=args.seed)

    env = LocomotionEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        # Locomotion eval 直接使用累计到最终阶段的命令范围，不在评估时重跑课程。
        curriculum_cfg={"enabled": False, "stages": []},
        steps_per_iteration=train_cfg["num_steps_per_env"],
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
    runner.load(str(checkpoint_path), map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    env.focus_viewer()
    keyboard = KeyboardCommand(env)
    print("[eval] camera: left drag=rotate, middle/Shift+left drag=pan, wheel/right drag=zoom, Home=focus robot")
    print_evaluation_diagnostics(env, step=0)
    step = 0
    with torch.no_grad():
        while env.scene.viewer.is_alive():
            # step() 内部可能重采样指令，因此每次策略推理前都覆盖为键盘指令。
            keyboard.write_to_env()
            env._update_observations()
            obs_dict = env.get_observations()
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)
            keyboard.update_caption()
            step += 1
            if args.print_interval and step % args.print_interval == 0:
                print_evaluation_diagnostics(env, step)


if __name__ == "__main__":
    main()
