"""从不同高度和朝向自由落下，收集 recovery 可直接读取的完整静止姿态。

从仓库根目录运行：
    python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses --num-poses 32
    python -m src.genesis.wheel_leg_infantry.tools.collect_reset_poses --num-poses 4 --vis

电机始终为零力矩，仅保留模型本身的摩擦、阻尼、闭链约束和训练配置的气弹簧。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import genesis as gs
from genesis.utils.geom import xyz_to_quat
import numpy as np
import torch

from ..config import get_env_cfg
from ..tasks.recovery.reset_pose import get_reset_pose_cfg

RECOVERY_DIR = Path(__file__).resolve().parents[1] / "tasks" / "recovery"
BASE_NAMES = tuple(f"floating_base_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz"))
DT = 0.01


def coordinate_layout(robot, wheel_names):
    """使用实体局部 q 索引，包含闭链被动关节；轮角不参与姿态去重。"""
    names = [None] * robot.n_qs
    linear_dofs = [0, 1, 2]
    revolute_names, prismatic_names = [], []
    for joint in robot.joints:
        indices = joint.qs_idx_local
        if not indices:
            continue
        if joint.n_qs == 7 and indices == list(range(7)) and joint.name == "floating_base":
            names[:7] = BASE_NAMES
        elif joint.n_qs == 1:
            names[indices[0]] = joint.name
            if joint.type == gs.JOINT_TYPE.PRISMATIC:
                linear_dofs.extend(joint.dofs_idx_local)
                prismatic_names.append(joint.name)
            elif joint.name not in wheel_names:
                revolute_names.append(joint.name)
        else:
            raise ValueError(f"Unsupported reset joint: {joint.name} ({joint.n_qs} coordinates)")
    if tuple(names[:7]) != BASE_NAMES or None in names or len(set(names)) != len(names):
        raise ValueError("Expected floating_base followed by named scalar joints")
    return names, linear_dofs, revolute_names, prismatic_names


def pose_is_similar(pose, other, revolute_names, prismatic_names):
    """忽略平移、全局 yaw 和轮子转角，避免把同一种倒地姿态反复收录。"""

    def body_up(p):
        w, x, y, z = (p[name] for name in BASE_NAMES[3:])
        norm = math.hypot(w, x, y, z)
        w, x, y, z = (v / norm for v in (w, x, y, z))
        return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])

    if abs(pose[BASE_NAMES[2]] - other[BASE_NAMES[2]]) > 0.01:
        return False
    if np.dot(body_up(pose), body_up(other)) < math.cos(math.radians(5.0)):
        return False
    for name in revolute_names:
        delta = pose[name] - other[name]
        if abs(math.atan2(math.sin(delta), math.cos(delta))) > 0.05:
            return False
    return all(abs(pose[name] - other[name]) <= 0.005 for name in prismatic_names)


def write_json(path, value):
    """原子替换报告；中断时已完成的样本仍然可用。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def settle(scene, robot, ground, spring_indices, cfg, args, velocity_limits):
    """自然落地，所有 DOF 连续低速且接触地面后才返回；不在过程中清零速度。"""
    force = torch.zeros((1, robot.n_dofs), dtype=gs.tc_float, device=gs.device)
    hold_steps = math.ceil(args.hold_time / DT)
    stable_steps = 0
    for step in range(1, math.ceil(args.max_time / DT) + 1):
        if args.vis and not scene.viewer.is_alive():
            raise KeyboardInterrupt
        spring_pos = robot.get_dofs_position(spring_indices)
        spring_vel = robot.get_dofs_velocity(spring_indices)
        compression = torch.clamp(cfg["gas_spring_max_compression"] - spring_pos, 0.0, cfg["gas_spring_max_compression"])
        force[:, spring_indices] = (
            cfg["gas_spring_preload_force"] + cfg["gas_spring_stiffness"] * compression - cfg["gas_spring_damping"] * spring_vel
        )
        # 对全部 DOF 显式使用力控制，避免 Viewer/上一次试验遗留 PD 控制。
        robot.control_dofs_force(force)
        scene.step()
        qpos, velocity = robot.get_qpos(), robot.get_dofs_velocity()
        elapsed = step * DT
        if scene.rigid_solver.get_error_envs_mask().any() or not (torch.isfinite(qpos).all() and torch.isfinite(velocity).all()):
            return None, dict(status="invalid_state", elapsed_s=elapsed)
        velocity_ratio = (velocity.abs() / velocity_limits).max().item()
        still = elapsed >= args.min_time and velocity_ratio <= 1.0
        if still:
            still = bool(robot.get_contacts(with_entity=ground)["valid_mask"].any())
        stable_steps = stable_steps + 1 if still else 0
        if stable_steps >= hold_steps:
            return qpos.flatten().tolist(), dict(
                status="settled",
                elapsed_s=elapsed,
                max_velocity_ratio=velocity_ratio,
                # 仅报告原始速度，姿态 JSON 中不混入诊断字段。
                dofs_velocity=velocity.flatten().tolist(),
            )
    return None, dict(status="timeout", elapsed_s=elapsed, max_velocity_ratio=velocity_ratio)


def collect(args):
    cfg = get_env_cfg()
    sources = [(source, get_reset_pose_cfg(source)) for source in args.reset_poses]
    rng = np.random.default_rng(args.seed)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=20),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False, tolerance=1e-5, max_collision_pairs=20),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -2.8, 1.5),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )
    run_dir = args.output_dir.expanduser().resolve() / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        dt_s=DT,
        substeps=20,
        robot_mjcf=cfg["robot_mjcf"],
        gas_spring={k: v for k, v in cfg.items() if k.startswith("gas_spring_")},
        poses=[],
        trials=[],
    )
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    try:
        ground = scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(gs.morphs.MJCF(file=cfg["robot_mjcf"], pos=(0.0, 0.0, 1.0)))
        scene.build(n_envs=1)
        names, linear_dofs, revolute_names, prismatic_names = coordinate_layout(robot, cfg["wheel_names"])
        spring_indices = [robot.get_joint(name).dof_start - robot.dof_start for name in cfg["spring_names"]]
        velocity_limits = torch.full((robot.n_dofs,), args.angular_tol, dtype=gs.tc_float, device=gs.device)
        velocity_limits[linear_dofs] = args.linear_tol
        for source, source_cfg in sources:
            unknown = source_cfg["reset_joint_pos"].keys() - set(names[7:])
            if unknown:
                raise ValueError(f"{source}: unknown reset joints: {sorted(unknown)}")
        report["coordinate_names"] = names
        report["dof_velocity_limits"] = velocity_limits.tolist()
        accepted = []
        print(f"[output] {run_dir}", flush=True)
        for trial in range(args.max_attempts):
            # 完整清理物理状态、接触和求解器缓存，再注入本次完整 qpos。
            scene.reset()
            source, source_cfg = sources[trial % len(sources)]
            rpy = rng.uniform([args.roll[0], args.pitch[0], args.yaw[0]], [args.roll[1], args.pitch[1], args.yaw[1]])
            quat = xyz_to_quat(torch.tensor(rpy, dtype=gs.tc_float, device=gs.device), rpy=True, degrees=True)
            initial = [0.0, 0.0, float(rng.uniform(*args.height)), *quat.tolist()]
            initial.extend(source_cfg["reset_joint_pos"].get(name, 0.0) for name in names[7:])
            robot.set_qpos([initial], zero_velocity=True)
            # 任意朝向下都保证从地面上方释放，而非从穿地状态弹出。
            lift = max(0.0, 0.02 - robot.get_AABB()[0, 0, 2].item())
            if lift:
                initial[2] += lift
                robot.set_qpos([initial], zero_velocity=True)
            record = dict(
                status="running", trial=trial + 1, source=source, rpy_deg=rpy.tolist(), clearance_lift_m=lift, initial_pose=dict(zip(names, initial))
            )
            report["trials"].append(record)
            values, result = settle(scene, robot, ground, spring_indices, cfg, args, velocity_limits)
            record.update(result)
            if values is not None:
                pose = dict(zip(names, values))
                duplicate = not args.keep_duplicates and any(pose_is_similar(pose, other, revolute_names, prismatic_names) for other in accepted)
                if duplicate:
                    record["status"] = "duplicate"
                else:
                    filename = f"pose_{len(accepted):04d}.json"
                    write_json(run_dir / filename, pose)
                    accepted.append(pose)
                    report["poses"].append(filename)
                    record.update(status="saved", file=filename)
            write_json(report_path, report)
            print(
                f"[trial {trial + 1}/{args.max_attempts}] {record['status']} t={record['elapsed_s']:.2f}s " f"saved={len(accepted)}/{args.num_poses}",
                flush=True,
            )
            if len(accepted) >= args.num_poses:
                break
        report["status"] = "complete" if len(accepted) >= args.num_poses else "partial"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception as exc:
        report.update(status="error", error=str(exc))
        raise
    finally:
        write_json(report_path, report)
        scene.destroy()
        print(f"[saved] {len(report['poses'])} poses; report: {report_path}", flush=True)
    return 0 if report["status"] == "complete" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-poses", type=int, default=32, help="期望保存的不同静止姿态数")
    parser.add_argument("--max-attempts", type=int, help="最多释放次数，默认 num-poses * 10")
    parser.add_argument(
        "--reset-poses",
        nargs="+",
        default=["default", str(RECOVERY_DIR / "reset_pos.json")],
        help="轮流使用这些 JSON 的完整关节初态或 default；基座高度/朝向由范围重新采样",
    )
    parser.add_argument("--output-dir", type=Path, default=RECOVERY_DIR / "reset_pos")
    parser.add_argument("--height", type=float, nargs=2, default=(0.6, 1.0), metavar=("MIN", "MAX"), help="释放基座高度范围 [m]")
    for angle in ("roll", "pitch", "yaw"):
        parser.add_argument(f"--{angle}", type=float, nargs=2, default=(-180.0, 180.0), metavar=("MIN", "MAX"), help="绝对角度范围 [deg]")
    parser.add_argument("--min-time", type=float, default=2.0, help="至少模拟此时间再开始静止计时 [s]")
    parser.add_argument("--max-time", type=float, default=20.0, help="每次释放的最大模拟时间 [s]，超时跳过")
    parser.add_argument("--hold-time", type=float, default=0.5, help="全部速度连续达标且接触地面的时间 [s]")
    parser.add_argument("--linear-tol", type=float, default=0.01, help="基座线速度、滑动关节速度各分量阈值 [m/s]")
    parser.add_argument("--angular-tol", type=float, default=0.05, help="基座角速度、所有转动关节（含轮）速度各分量阈值 [rad/s]")
    parser.add_argument("--keep-duplicates", action="store_true", help="保留相似终态，默认忽略平移/yaw/轮角去重")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--vis", action="store_true", help="显示采集过程；关闭窗口或 Ctrl-C 保留已保存样本")
    args = parser.parse_args(argv)
    if args.max_attempts is None:
        args.max_attempts = args.num_poses * 10
    if args.num_poses <= 0 or args.max_attempts <= 0:
        parser.error("--num-poses and --max-attempts must be positive")
    for name in ("height", "roll", "pitch", "yaw"):
        lower, upper = getattr(args, name)
        if not all(math.isfinite(v) for v in (lower, upper)) or lower > upper:
            parser.error(f"--{name} requires finite MIN <= MAX")
    if args.height[0] <= 0:
        parser.error("--height must be positive")
    for name in ("min_time", "max_time", "hold_time", "linear_tol", "angular_tol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.max_time < args.min_time + args.hold_time:
        parser.error("--max-time must be at least --min-time + --hold-time")
    return args


if __name__ == "__main__":
    raise SystemExit(collect(parse_args()))
