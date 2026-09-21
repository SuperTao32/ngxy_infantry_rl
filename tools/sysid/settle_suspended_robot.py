#!/usr/bin/env python3
"""固定车身，仅施加两侧气弹簧力，得到悬空静置后的关节初始位置。

运行：
    python tools/sysid/settle_suspended_robot.py
    python tools/sysid/settle_suspended_robot.py --view

仿真过程中不强制清零速度，也不增加阻尼或位置控制。达到静止判据后，
导出完整关节位置以及全零初始速度。只依赖 MuJoCo 和 NumPy。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import time

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPRING_ACTUATORS = ("left_spring2_joint_ctrl", "right_spring2_joint_ctrl")
GAS_SPRING_FORCE = 420.0  # 每个气弹簧沿伸长方向施加 +420 N。


@dataclass(frozen=True)
class SettleConfig:
    model: Path = PROJECT_ROOT / "assets/robot/wheelbipeV14_2/mjcf/wheelbipeV14_2.xml"
    out: Path = PROJECT_ROOT / "tools/sysid/results/suspended_initial_state.npz"
    base_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    max_time: float = 60.0       # 最大仿真时间 [s]；超时不导出未收敛状态。
    min_time: float = 2.0       # 至少运行这么长时间 [s]。
    hold_time: float = 1.0      # 静止判据必须连续满足这么长时间 [s]。
    velocity_tol: float = 1e-3  # 转动关节：rad/s；滑动关节：m/s。
    accel_tol: float = 1e-3     # 转动关节：rad/s²；滑动关节：m/s²。


CFG = SettleConfig()


def make_suspended_model(cfg: SettleConfig) -> mujoco.MjModel:
    """在内存中固定基座；保留原模型的重力、摩擦、阻尼和闭链约束。"""
    spec = mujoco.MjSpec.from_file(str(cfg.model.resolve()))
    spec.delete(spec.joint("floating_base"))
    spec.body("base_link").pos = cfg.base_pos
    model = spec.compile()

    # 导出格式针对当前模型的一维转动/滑动关节。
    scalar_types = [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]
    if not np.all(np.isin(model.jnt_type, scalar_types)):
        raise ValueError("固定基座后，模型必须只包含 hinge/slide 关节。")
    return model


def make_spring_control(model: mujoco.MjModel) -> np.ndarray:
    """所有电机力矩为零，仅两个气弹簧输出恒力。"""
    ctrl = np.zeros(model.nu)
    for name in SPRING_ACTUATORS:
        ctrl[model.actuator(name).id] = GAS_SPRING_FORCE

    # 验证 MJCF 中气弹簧的增益、传动比和限幅没有改变实际输出力。
    probe = mujoco.MjData(model)
    probe.ctrl[:] = ctrl
    mujoco.mj_forward(model, probe)
    for name in SPRING_ACTUATORS:
        aid = model.actuator(name).id
        jid = model.actuator_trnid[aid, 0]
        dof = model.jnt_dofadr[jid]
        if not np.isclose(probe.qfrc_actuator[dof], GAS_SPRING_FORCE):
            raise ValueError(f"{name} 的实际关节力不是 420 N，请检查执行器定义。")
    return ctrl


def settle(
    model: mujoco.MjModel,
    cfg: SettleConfig,
    view: bool = False,
) -> mujoco.MjData:
    """从 XML 的默认姿态和零速度开始，等待系统自然静止。"""
    data = mujoco.MjData(model)
    ctrl = make_spring_control(model)
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    viewer_context = nullcontext(None)
    if view:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(model, data)

    stable_time = 0.0
    dt = model.opt.timestep
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = cfg.base_pos
            viewer.cam.distance = 2.0

        for _ in range(int(np.ceil(cfg.max_time / dt))):
            started = time.monotonic()
            if viewer is not None and not viewer.is_running():
                raise RuntimeError("窗口已关闭，尚未得到静止状态；未保存结果。")

            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            # 更新到当前 qpos/qvel，使加速度和约束误差与保存的姿态对应。
            mujoco.mj_forward(model, data)
            if (not np.all(np.isfinite(data.qpos))
                    or not np.all(np.isfinite(data.qvel))
                    or not np.all(np.isfinite(data.qacc))
                    or np.any(data.warning.number)):
                raise RuntimeError("仿真出现非有限状态或 MuJoCo 警告；未保存结果。")

            max_velocity = float(np.max(np.abs(data.qvel)))
            max_accel = float(np.max(np.abs(data.qacc)))
            is_still = max_velocity < cfg.velocity_tol and max_accel < cfg.accel_tol
            stable_time = stable_time + dt if is_still else 0.0

            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, dt - (time.monotonic() - started)))

            if data.time >= cfg.min_time and stable_time >= cfg.hold_time:
                return data

    raise RuntimeError(
        f"{cfg.max_time:g} s 内未静止：max|dq|={max_velocity:.3e}，"
        f"max|ddq|={max_accel:.3e}；未保存结果。"
    )


def save_initial_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: SettleConfig,
) -> None:
    """保存完整状态和可直接按名字核对的关节表；导出速度始终为零。"""
    names = np.asarray([model.joint(i).name for i in range(model.njnt)])
    qpos = data.qpos.copy()
    qvel = np.zeros(model.nv)
    joint_q = qpos[model.jnt_qposadr]
    left_ids = np.flatnonzero(np.char.startswith(names, "left_"))

    equality_mask = data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    closure_error = float(np.max(np.abs(data.efc_pos[equality_mask]), initial=0.0))
    limit_error = np.maximum(
        np.maximum(model.jnt_range[:, 0] - joint_q, joint_q - model.jnt_range[:, 1]),
        0.0,
    )
    limit_error[~model.jnt_limited.astype(bool)] = 0.0

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg.out,
        joint_names=names,
        qpos=qpos,
        qvel=qvel,
        ctrl=data.ctrl.copy(),
        qvel_before_zero=data.qvel.copy(),
        settled_time=np.asarray(data.time),
        left_joint_names=names[left_ids],
        left_qpos=joint_q[left_ids],
        left_qvel=np.zeros(len(left_ids)),
    )

    report = {
        "model": str(cfg.model.resolve()),
        "base_pos": list(cfg.base_pos),
        "gas_spring_force_N_each": GAS_SPRING_FORCE,
        "settled_time_s": float(data.time),
        "velocity_tol": cfg.velocity_tol,
        "accel_tol": cfg.accel_tol,
        "hold_time_s": cfg.hold_time,
        "max_velocity_before_zero": float(np.max(np.abs(data.qvel))),
        "max_acceleration": float(np.max(np.abs(data.qacc))),
        "max_equality_error_m": closure_error,
        "joints": {
            name: {
                "q": float(joint_q[i]),
                "dq": 0.0,
                "unit": "m" if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE else "rad",
                "limit_violation": float(limit_error[i]),
            }
            for i, name in enumerate(names)
        },
    }
    report_path = cfg.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"[settled] t={data.time:.3f} s, max|dq|={report['max_velocity_before_zero']:.3e}")
    print(f"[closure] max equality error={closure_error:.3e} m")
    for i, name in enumerate(names):
        unit = report["joints"][name]["unit"]
        print(f"  {name:26s} q={joint_q[i]: .9f} {unit}, dq=0")
        if limit_error[i] > 1e-6:
            print(f"    [soft limit] exceeded by {limit_error[i]:.3e} {unit}")
    print(f"[saved] {cfg.out}\n[saved] {report_path}")


def main(cfg: SettleConfig = CFG, view: bool = False) -> None:
    if not all(np.isfinite(x) and x > 0 for x in (
        cfg.max_time, cfg.min_time, cfg.hold_time, cfg.velocity_tol, cfg.accel_tol
    )):
        raise ValueError("时间和静止阈值必须为有限正数。")
    if cfg.max_time < max(cfg.min_time, cfg.hold_time):
        raise ValueError("max_time 必须不小于 min_time 和 hold_time。")
    if cfg.out.suffix != ".npz":
        raise ValueError("输出文件后缀必须为 .npz。")
    model = make_suspended_model(cfg)
    print(f"[model] fixed base={cfg.base_pos}, nq={model.nq}, nv={model.nv}")
    print("[control] motors=0 Nm, gas springs=420 N each")
    data = settle(model, cfg, view=view)
    save_initial_state(model, data, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="实时显示悬空静置过程")
    main(view=parser.parse_args().view)
