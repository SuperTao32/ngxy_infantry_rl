#!/usr/bin/env python3
"""
wheelbipeV14_2 右腿闭链机构的 MuJoCo 系统辨识脚本。

辨识四个主动关节等效参数：
    damping_front, damping_rear      黏性阻尼 [Nms/rad]
    friction_front, friction_rear    库仑摩擦 [Nm]

测量输入：
    实际电机电流 Iq [A]，通过固定 K [Nm/A] 换算为力矩。

测量输出：
    主动关节位置 q 和速度 dq。

说明：
- 从动关节的阻尼和摩擦参数保持 MJCF 中的定义。
- 保留模型中的闭链等式约束。
- right_leg.xml 仅包含右腿，基座已固定在 z=1 m。
- 气弹簧执行器沿滑动关节轴向施加恒定 +420 N；CSV 无需记录气弹簧力。
- 每条轨迹均从保存的悬空姿态开始，所有初始速度为零。
  CSV 中的 q/dq 是测量输出，不用于设置初始状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mujoco
from mujoco import sysid


# ---------------------------------------------------------------------------
# 用户配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SysIDConfig:
    # 输入与输出路径。将所有轨迹加入 data，进行联合拟合。
    model: Path
    data: tuple[Path, ...]
    out: Path
    initial_state: Path  # settle_suspended_robot.py 生成的 NPZ 初态文件。

    # 实机电流方向与 MuJoCo 关节轴方向相反时，用此系数修正符号。
    front_current_sign: float = 1.0
    rear_current_sign: float = 1.0

    # 固定电机力矩常数 [Nm/A]，包含减速器传动。
    # 这些值由配置指定，不参与辨识。
    k_front: float = 1.0
    k_rear: float = 1.0

    # 主动关节黏性阻尼：初始猜测值与上界 [Nms/rad]。
    damping_front0: float = 0.02
    damping_rear0: float = 0.02
    damping_max: float = 3.0

    # 主动关节库仑摩擦：初始猜测值与上界 [Nm]。
    # front 的初值 1.5 来自 MJCF；rear 的初值取大于零下界的数值。
    friction_front0: float = 1.5
    friction_rear0: float = 0.2
    friction_max: float = 6.0

    max_iters: int = 100
    threads: int = 0  # <= 0 时由 SysID 库选择线程数。


# 运行前编辑此配置块。路径可使用绝对路径，或相对于当前工作目录的路径。
# 只有一个 CSV 文件时，也要保留路径后面的逗号。
CFG = SysIDConfig(
    model=PROJECT_ROOT / "assets/robot/wheelbipeV14_2/mjcf/right_leg.xml",
    data=(
        # PROJECT_ROOT / "logs/sysid/slow_forward.csv",
        # PROJECT_ROOT / "logs/sysid/slow_reverse.csv",
    ),
    out=PROJECT_ROOT / "sysid_results/right_leg",
    initial_state=PROJECT_ROOT / "sysid_results/suspended_initial_state.npz",
)


# ---------------------------------------------------------------------------
# right_leg.xml 中的机器人关节与执行器名称
# ---------------------------------------------------------------------------

ACTIVE_JOINTS = {
    "front": "right_front1_joint",
    "rear": "right_rear1_joint",
}

ACTIVE_ACTUATORS = {
    "front": "right_front1_joint_ctrl",
    "rear": "right_rear1_joint_ctrl",
}

# 气弹簧执行器的增益和传动比均为 1，因此 ctrl 表示力，单位为 N。
# 正向力使气弹簧沿 right_spring2_joint 的轴向 (-1, 0, 0) 伸长。
SPRING_ACTUATOR = "right_spring2_joint_ctrl"
GAS_SPRING_FORCE = 420.0

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def actuator_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx < 0:
        raise KeyError(f"Actuator not found: {name}")
    return idx


def make_initial_state(model: mujoco.MjModel, path: Path) -> np.ndarray:
    """按关节名加载右腿静置后的 q，并将所有初始速度设为零。

    原样使用 settle_suspended_robot.py 保存的姿态，包括带载时的软约束变形。
    不对关节位置进行投影或限位裁剪。
    """
    # 共用静置文件中的 joint_names/qpos 包含左右腿的关节数据。
    with np.load(path, allow_pickle=False) as saved:
        required = ("joint_names", "qpos")
        missing = [key for key in required if key not in saved]
        if missing:
            raise ValueError(f"{path}: missing initial-state arrays: {missing}")
        names = np.asarray(saved["joint_names"], dtype=str)
        positions = np.asarray(saved["qpos"], dtype=float)

    if names.ndim != 1 or positions.shape != names.shape:
        raise ValueError(f"{path}: joint_names and qpos must be matching 1D arrays")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate joint names in initial state")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{path}: initial joint positions must be finite")

    q_by_name = dict(zip(names, positions))
    q0 = np.zeros(model.nq)
    for jid in range(model.njnt):
        name = model.joint(jid).name
        if model.jnt_type[jid] not in (
            mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
        ):
            raise ValueError(f"Initial-state loading requires hinge/slide joints: {name}")
        if name not in q_by_name:
            raise ValueError(f"{path}: initial state is missing joint {name}")
        q0[model.jnt_qposadr[jid]] = q_by_name[name]

    return sysid.create_initial_state(model, q0, np.zeros(model.nv))


def csv_has_column(data: np.ndarray, name: str) -> bool:
    return data.dtype.names is not None and name in data.dtype.names


def load_log(
    path: Path,
    model: mujoco.MjModel,
    front_current_sign: float,
    rear_current_sign: float,
):
    """
    CSV 必需列：
        time       时间 [s]
        iq_front   前电机实际电流 [A]
        iq_rear    后电机实际电流 [A]
        q_front    前主动关节角度 [rad]
        q_rear     后主动关节角度 [rad]

    可选列：
        dq_front   前主动关节角速度 [rad/s]
        dq_rear    后主动关节角速度 [rad/s]

    缺少 dq 列时，使用 numpy.gradient 根据角度估算速度。
    """
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if raw.shape == ():
        raw = raw.reshape(1)

    required = ("time", "iq_front", "iq_rear", "q_front", "q_rear")
    missing = [c for c in required if not csv_has_column(raw, c)]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    t = np.asarray(raw["time"], dtype=float)
    if len(t) < 10:
        raise ValueError(f"{path}: trajectory is too short ({len(t)} rows).")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: time must be strictly increasing.")
    t = t - t[0]

    q_front = np.asarray(raw["q_front"], dtype=float)
    q_rear = np.asarray(raw["q_rear"], dtype=float)

    if csv_has_column(raw, "dq_front"):
        dq_front = np.asarray(raw["dq_front"], dtype=float)
    else:
        dq_front = np.gradient(q_front, t)

    if csv_has_column(raw, "dq_rear"):
        dq_rear = np.asarray(raw["dq_rear"], dtype=float)
    else:
        dq_rear = np.gradient(q_rear, t)

    iq_front = front_current_sign * np.asarray(raw["iq_front"], dtype=float)
    iq_rear = rear_current_sign * np.asarray(raw["iq_rear"], dtype=float)

    # 两个电机通道输入实测电流 [A]，由固定增益换算成力矩。
    # 气弹簧通道输入恒定力 [N]。
    ctrl = np.zeros((len(t), model.nu), dtype=float)
    ctrl[:, actuator_id(model, ACTIVE_ACTUATORS["front"])] = iq_front
    ctrl[:, actuator_id(model, ACTIVE_ACTUATORS["rear"])] = iq_rear

    ctrl[:, actuator_id(model, SPRING_ACTUATOR)] = GAS_SPRING_FORCE

    control_ts = sysid.TimeSeries.from_control_names(t, ctrl, model)

    # 使用主动关节的 q 和 dq 作为测量输出。
    # 它们是模型状态信号，不是 MJCF 中定义的传感器信号。
    y = np.column_stack([q_front, q_rear, dq_front, dq_rear])
    measurement_ts = sysid.TimeSeries.from_names(
        t,
        y,
        model,
        names=[
            (f'{ACTIVE_JOINTS["front"]}_qpos', sysid.SignalType.MjStateQPos),
            (f'{ACTIVE_JOINTS["rear"]}_qpos', sysid.SignalType.MjStateQPos),
            (f'{ACTIVE_JOINTS["front"]}_qvel', sysid.SignalType.MjStateQVel),
            (f'{ACTIVE_JOINTS["rear"]}_qvel', sysid.SignalType.MjStateQVel),
        ],
    )

    return control_ts, measurement_ts


# ---------------------------------------------------------------------------
# 系统辨识参数
# ---------------------------------------------------------------------------

def set_fixed_motor_gains(spec: mujoco.MjSpec, cfg: SysIDConfig) -> None:
    """一次性设置 tau = K * Iq 的固定增益，保留 MJCF 中的力矩限幅和传动比。"""
    spec.actuator(ACTIVE_ACTUATORS["front"]).gainprm[0] = cfg.k_front
    spec.actuator(ACTIVE_ACTUATORS["rear"]).gainprm[0] = cfg.k_rear


def set_joint_scalar(joint_name: str, attr: str):
    def modifier(spec: mujoco.MjSpec, p: sysid.Parameter):
        joint = spec.joint(joint_name)
        value = float(p.value[0])
        if attr == "damping" and isinstance(joint.damping, np.ndarray):
            # MuJoCo 3.10 使用数组存储多项式阻尼系数；
            # 第一个系数就是这里需要拟合的线性黏性阻尼。
            joint.damping[0] = value
        else:
            setattr(joint, attr, value)
    return modifier


def build_params(cfg: SysIDConfig) -> sysid.ParameterDict:
    params = sysid.ParameterDict()

    # 主动关节的等效黏性阻尼。
    params.add(sysid.Parameter(
        "damping_front",
        nominal=cfg.damping_front0,
        min_value=0.0,
        max_value=cfg.damping_max,
        modifier=set_joint_scalar(ACTIVE_JOINTS["front"], "damping"),
    ))
    params.add(sysid.Parameter(
        "damping_rear",
        nominal=cfg.damping_rear0,
        min_value=0.0,
        max_value=cfg.damping_max,
        modifier=set_joint_scalar(ACTIVE_JOINTS["rear"], "damping"),
    ))

    # 主动关节的等效库仑摩擦。
    params.add(sysid.Parameter(
        "friction_front",
        nominal=cfg.friction_front0,
        min_value=0.0,
        max_value=cfg.friction_max,
        modifier=set_joint_scalar(ACTIVE_JOINTS["front"], "frictionloss"),
    ))
    params.add(sysid.Parameter(
        "friction_rear",
        nominal=cfg.friction_rear0,
        min_value=0.0,
        max_value=cfg.friction_max,
        modifier=set_joint_scalar(ACTIVE_JOINTS["rear"], "frictionloss"),
    ))

    return params


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def validate_config(cfg: SysIDConfig) -> None:
    if not cfg.data:
        raise ValueError("CFG.data is empty; add at least one CSV trajectory in the CFG block")
    if not cfg.model.is_file():
        raise FileNotFoundError(f"CFG.model does not exist: {cfg.model}")
    if not cfg.initial_state.is_file():
        raise FileNotFoundError(
            f"CFG.initial_state does not exist: {cfg.initial_state}; "
            "run tools/sysid/settle_suspended_robot.py first"
        )
    missing_data = [path for path in cfg.data if not path.is_file()]
    if missing_data:
        raise FileNotFoundError(f"CFG.data contains missing files: {missing_data}")
    if cfg.max_iters <= 0:
        raise ValueError("CFG.max_iters must be positive")
    if not all(np.isfinite(k) and k > 0 for k in (cfg.k_front, cfg.k_rear)):
        raise ValueError("CFG.k_front and CFG.k_rear must be finite and positive")
    if not 0 <= cfg.damping_front0 <= cfg.damping_max or not 0 <= cfg.damping_rear0 <= cfg.damping_max:
        raise ValueError("CFG damping initial guesses must be inside [0, damping_max]")
    if not 0 <= cfg.friction_front0 <= cfg.friction_max or not 0 <= cfg.friction_rear0 <= cfg.friction_max:
        raise ValueError("CFG friction initial guesses must be inside [0, friction_max]")


def main(cfg: SysIDConfig = CFG):
    validate_config(cfg)

    # 输入 MJCF 已经是固定基座、悬空的右腿模型。
    print(f"[model] SysID MJCF: {cfg.model}")
    spec = mujoco.MjSpec.from_file(str(cfg.model.resolve()))
    set_fixed_motor_gains(spec, cfg)
    model = spec.compile()

    print(
        f"[model] nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"neq={model.neq}, nsensordata={model.nsensordata}"
    )
    print(f"[model] constant gas spring force: {GAS_SPRING_FORCE:g} N")
    print(f"[model] fixed K_front={cfg.k_front:g}, K_rear={cfg.k_rear:g} Nm/A")

    initial_state = make_initial_state(model, cfg.initial_state)
    print(f"[initial] suspended q from {cfg.initial_state}; all qvel=0")

    controls = []
    measurements = []
    initial_states = []
    sequence_names = []

    for path in cfg.data:
        control_ts, measurement_ts = load_log(
            path=path,
            model=model,
            front_current_sign=cfg.front_current_sign,
            rear_current_sign=cfg.rear_current_sign,
        )
        controls.append(control_ts)
        measurements.append(measurement_ts)
        # 每次实验都从相同的悬空静置姿态开始。
        initial_states.append(initial_state.copy())
        sequence_names.append(path.stem)
        print(f"[data] loaded {path}: {len(control_ts.times)} samples")

    ms = sysid.ModelSequences(
        name="right_leg_closed_chain",
        spec=spec,
        sequence_name=sequence_names,
        initial_state=initial_states,
        control=controls,
        sensordata=measurements,
        allow_missing_sensors=True,
    )

    params = build_params(cfg)

    # 选择 q 和 dq 作为观测量；默认残差会按测量信号幅值归一化，
    # 因此通常可以先使用相同权重。
    enabled_observations = [
        (f'{ACTIVE_JOINTS["front"]}_qpos', sysid.SignalType.MjStateQPos),
        (f'{ACTIVE_JOINTS["rear"]}_qpos', sysid.SignalType.MjStateQPos),
        (f'{ACTIVE_JOINTS["front"]}_qvel', sysid.SignalType.MjStateQVel),
        (f'{ACTIVE_JOINTS["rear"]}_qvel', sysid.SignalType.MjStateQVel),
    ]

    n_threads = None if cfg.threads <= 0 else cfg.threads
    residual_kwargs = dict(
        models_sequences=[ms],
        enabled_observations=enabled_observations,
        resample_true=True,
    )
    if n_threads is not None:
        residual_kwargs["n_threads"] = n_threads

    residual_fn = sysid.build_residual_fn(**residual_kwargs)

    print("\n[sysid] optimizing 4 parameters:")
    for name, p in params.parameters.items():
        print(
            f"  {name:18s} x0={float(p.value[0]):.6g} "
            f"[{float(p.min_value[0]):.6g}, {float(p.max_value[0]):.6g}]"
        )

    opt_params, opt_result = sysid.optimize(
        initial_params=params,
        residual_fn=residual_fn,
        optimizer="scipy_parallel_fd",
        loss="soft_l1",
        x_scale="jac",
        max_iters=cfg.max_iters,
        check_conditioning=True,
        verbose=True,
    )

    print("\n[result]")
    for name, p in opt_params.parameters.items():
        print(f"  {name:18s} = {float(p.value[0]):.9g}")

    cfg.out.mkdir(parents=True, exist_ok=True)
    sysid.save_results(
        cfg.out,
        [ms],
        params,
        opt_params,
        opt_result,
        residual_fn,
    )

    np.savez(
        cfg.out / "identified_params.npz",
        **{
            name: np.asarray(p.value).copy()
            for name, p in opt_params.parameters.items()
        },
    )
    print(f"\n[done] results saved to: {cfg.out}")


if __name__ == "__main__":
    main()
