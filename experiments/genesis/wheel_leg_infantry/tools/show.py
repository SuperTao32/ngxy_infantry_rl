"""用与训练一致的控制参数检查步兵车模型。"""

import argparse

import genesis as gs
import numpy as np
import torch

from ..core.terrain import TERRAIN_PRESETS, TerrainManager
from ..tasks.locomotion.config import get_cfgs


def main():
    parser = argparse.ArgumentParser(description="检查 wheelbipeV14_2 的关节、PD 和气弹簧")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--terrain", choices=TERRAIN_PRESETS, default="plane")
    parser.add_argument(
        "--joint-target",
        type=float,
        default=0.0,
        help="左右腿对称张开量：front=+target, rear=-target，单位 rad",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    env_cfg, _, _, _, _ = get_cfgs()
    env_cfg["terrain"]["preset"] = args.terrain
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    terrain = TerrainManager(env_cfg["terrain"])
    spawn_center, tile_index = terrain.sample_spawn_tiles(torch.tensor([0], device=gs.device))
    spawn_x, spawn_y = spawn_center[0].tolist()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=20),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            tolerance=1e-5,
            max_collision_pairs=20,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(spawn_x + 1.5, spawn_y - 2.5, 1.5),
            camera_lookat=(spawn_x, spawn_y, 0.3),
            camera_fov=35,
            # 启用 ImGui 调试面板，可在 viewer 中暂停、单步、复位并调节关节。
            enable_gui=True,
        ),
        show_viewer=args.vis,
    )
    ground = terrain.add_to_scene(scene)
    terrain.bind_entity(ground, device=gs.device, dtype=gs.tc_float)
    terrain_height = terrain.height_at(spawn_center)[0].item()
    print(f"[show] terrain={args.terrain}/{terrain.tile_type(tile_index[0].item())}")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=env_cfg["robot_mjcf"],
            pos=(spawn_x, spawn_y, terrain_height + env_cfg["base_init_pos_range"][2][0]),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    )
    scene.build()

    joint_idx = _dof_indices(robot, env_cfg["joint_names"])
    wheel_idx = _dof_indices(robot, env_cfg["wheel_names"])
    spring_idx = _dof_indices(robot, env_cfg["spring_names"])
    _configure_actuators(robot, joint_idx, wheel_idx, env_cfg)

    joint_target = torch.tensor(
        [[args.joint_target, args.joint_target, -args.joint_target, -args.joint_target]],
        dtype=gs.tc_float,
        device=gs.device,
    )
    wheel_target = torch.zeros((1, 2), dtype=gs.tc_float, device=gs.device)
    robot.set_dofs_position(np.array([-0.2, -0.2, -0.2, -0.2]), joint_idx)

    for _ in range(args.steps):
        # if(_ == 1):
        #     robot.set_dofs_position(joint_target, joint_idx)
        #     robot.set_dofs_velocity(wheel_target, wheel_idx)
        robot.control_dofs_force(_gas_spring_force(robot, spring_idx, env_cfg), spring_idx)
        scene.step()


def _dof_indices(robot, names):
    return torch.tensor(
        [robot.get_joint(name).dof_start for name in names],
        dtype=gs.tc_int,
        device=gs.device,
    )


def _configure_actuators(robot, joint_idx, wheel_idx, cfg):
    joint_count = len(joint_idx)
    wheel_count = len(wheel_idx)
    joint_limit = torch.full((joint_count,), cfg["joint_force_limit"], device=gs.device)
    wheel_limit = torch.full((wheel_count,), cfg["wheel_force_limit"], device=gs.device)
    robot.set_dofs_kp([cfg["joint_kp"]] * joint_count, joint_idx)
    robot.set_dofs_kv([cfg["joint_kd"]] * joint_count, joint_idx)
    robot.set_dofs_kv([cfg["wheel_kd"]] * wheel_count, wheel_idx)
    robot.set_dofs_force_range((-joint_limit).tolist(), joint_limit.tolist(), joint_idx)
    robot.set_dofs_force_range((-wheel_limit).tolist(), wheel_limit.tolist(), wheel_idx)


def _gas_spring_force(robot, spring_idx, cfg):
    spring_pos = robot.get_dofs_position(spring_idx)
    spring_vel = robot.get_dofs_velocity(spring_idx)
    max_compression = cfg["gas_spring_max_compression"]
    compression = torch.clamp(max_compression - spring_pos, 0.0, max_compression)
    return (
        cfg["gas_spring_preload_force"]
        + cfg["gas_spring_stiffness"] * compression
        - cfg["gas_spring_damping"] * spring_vel
    )


if __name__ == "__main__":
    main()
