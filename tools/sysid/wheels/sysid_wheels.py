"""辨识两个悬空轮的黏性阻尼和库仑摩擦。

直接加载 wheels.xml，两个轮轴由固定支架支撑，只有轮子可以转动。
仅优化 damping_left/right 和 friction_left/right；
电机力矩常数 K、轮子惯量和关节附加转动惯量 armature 保持固定。

每次实验的轮轴姿态由 XML 固定，初始速度为零。
轮角度使用 CSV 首帧初始化，并采用模型关节坐标系下的连续角度 [rad]，
不能直接使用编码器计数或每圈跳变的角度。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
from mujoco import sysid
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WHEEL_JOINTS = {side: f"{side}_wheel_joint" for side in ("left", "right")}
WHEEL_ACTUATORS = {side: f"{joint}_ctrl" for side, joint in WHEEL_JOINTS.items()}
OBSERVATIONS = [
    (f"{joint}_{suffix}", signal_type)
    for suffix, signal_type in (
        ("qpos", sysid.SignalType.MjStateQPos),
        ("qvel", sysid.SignalType.MjStateQVel),
    )
    for joint in WHEEL_JOINTS.values()
]


@dataclass(frozen=True)
class SysIDConfig:
    model: Path
    data: tuple[Path, ...]
    out: Path

    k_left: float = 1.0   # 固定轮轴力矩与电流的换算系数 [Nm/A]。
    k_right: float = 1.0
    left_current_sign: float = 1.0
    right_current_sign: float = 1.0

    damping_left0: float = 0.001  # 单位为 Nms/rad；初值取大于零下界的数值。
    damping_right0: float = 0.001
    damping_max: float = 0.1
    friction_left0: float = 0.023  # 单位为 Nm，初值来自 MJCF 中的轮关节。
    friction_right0: float = 0.023
    friction_max: float = 1.0

    max_iters: int = 100
    threads: int = 0


# 在此填写实测轨迹路径和已知的轮电机力矩常数。
CFG = SysIDConfig(
    model=PROJECT_ROOT / "assets/robot/wheelbipeV14_2/mjcf/wheels.xml",
    data=(
        # PROJECT_ROOT / "logs/sysid/wheels_forward.csv",
        # PROJECT_ROOT / "logs/sysid/wheels_reverse.csv",
    ),
    out=PROJECT_ROOT / "sysid_results_wheels",
)


def load_wheel_spec(cfg: SysIDConfig) -> mujoco.MjSpec:
    """直接加载双轮专用模型，并设置不参与辨识的电机增益 K。"""
    spec = mujoco.MjSpec.from_file(str(cfg.model.resolve()))
    # 确保导出的模型在结果目录中也能找到网格资源。
    spec.meshdir = str((cfg.model.resolve().parent / spec.meshdir).resolve())
    spec.actuator(WHEEL_ACTUATORS["left"]).gainprm[0] = cfg.k_left
    spec.actuator(WHEEL_ACTUATORS["right"]).gainprm[0] = cfg.k_right
    return spec


def load_log(path: Path, model: mujoco.MjModel, cfg: SysIDConfig):
    """读取轮电流和连续角度，缺少 dq 列时通过差分估算速度。"""
    raw = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=float))
    columns = raw.dtype.names or ()
    required = ("time", "iq_left", "iq_right", "q_left", "q_right")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    if len(raw) < 10:
        raise ValueError(f"{path}: trajectory needs at least 10 rows")
    used = (*required, *(name for name in ("dq_left", "dq_right") if name in columns))
    if any(not np.all(np.isfinite(raw[name])) for name in used):
        raise ValueError(f"{path}: input columns contain NaN or infinity")
    t = raw["time"] - raw["time"][0]
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: time must be strictly increasing")

    q = {side: raw[f"q_{side}"] for side in WHEEL_JOINTS}
    dq = {
        side: raw[f"dq_{side}"] if f"dq_{side}" in columns else np.gradient(q[side], t)
        for side in WHEEL_JOINTS
    }
    ctrl = np.zeros((len(t), model.nu))
    for side, sign in (("left", cfg.left_current_sign), ("right", cfg.right_current_sign)):
        ctrl[:, model.actuator(WHEEL_ACTUATORS[side]).id] = sign * raw[f"iq_{side}"]
    control = sysid.TimeSeries.from_control_names(t, ctrl, model)
    measured = sysid.TimeSeries.from_names(
        t, np.column_stack([q["left"], q["right"], dq["left"], dq["right"]]),
        model, names=OBSERVATIONS,
    )

    q0 = model.qpos0.copy()
    for side, name in WHEEL_JOINTS.items():
        q0[model.joint(name).qposadr[0]] = q[side][0]
    # 与实机流程一致：两个轮子开始驱动前就开始记录，初始速度为零。
    state = sysid.create_initial_state(model, q0, np.zeros(model.nv))
    return control, measured, state


def set_joint_parameter(joint_name: str, attr: str):
    def modifier(spec: mujoco.MjSpec, parameter: sysid.Parameter):
        joint = spec.joint(joint_name)
        value = float(parameter.value[0])
        if attr == "damping" and isinstance(joint.damping, np.ndarray):
            joint.damping[0] = value  # MuJoCo 3.10 中的线性阻尼系数。
        else:
            setattr(joint, attr, value)
    return modifier


def build_params(cfg: SysIDConfig) -> sysid.ParameterDict:
    params = sysid.ParameterDict()
    for quantity, attr, upper in (
        ("damping", "damping", cfg.damping_max),
        ("friction", "frictionloss", cfg.friction_max),
    ):
        for side, joint in WHEEL_JOINTS.items():
            name = f"{quantity}_{side}"
            params.add(sysid.Parameter(
                name,
                nominal=getattr(cfg, f"{name}0"),
                min_value=0.0,
                max_value=upper,
                modifier=set_joint_parameter(joint, attr),
            ))
    return params


def validate_config(cfg: SysIDConfig) -> None:
    if not cfg.data:
        raise ValueError("CFG.data is empty; add measured wheel CSV paths")
    for path in (cfg.model, *cfg.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not all(np.isfinite(k) and k > 0 for k in (cfg.k_left, cfg.k_right)):
        raise ValueError("Fixed K values must be finite and positive")
    if cfg.left_current_sign not in (-1, 1) or cfg.right_current_sign not in (-1, 1):
        raise ValueError("Current signs must be +1 or -1")
    if cfg.max_iters <= 0:
        raise ValueError("max_iters must be positive")
    for quantity in ("damping", "friction"):
        upper = getattr(cfg, f"{quantity}_max")
        if not np.isfinite(upper) or upper <= 0:
            raise ValueError(f"{quantity}_max must be finite and positive")
        for side in WHEEL_JOINTS:
            if not 0 <= getattr(cfg, f"{quantity}_{side}0") <= upper:
                raise ValueError(f"{quantity}_{side}0 must be inside [0, {upper}]")


def main(cfg: SysIDConfig = CFG) -> None:
    validate_config(cfg)
    spec = load_wheel_spec(cfg)
    model = spec.compile()
    if (model.nq, model.nv, model.nu) != (2, 2, 2):
        raise ValueError("The fixed-leg model must have exactly two wheel joints and actuators")
    print(f"[model] 固定轮轴模型：{cfg.model}，nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"[model] fixed K_left={cfg.k_left:g}, K_right={cfg.k_right:g} Nm/A")
    print("[initial] 轮轴姿态由 XML 固定，轮角取 CSV 首帧，qvel=0")

    controls, measurements, initial_states = [], [], []
    for path in cfg.data:
        control, measured, state = load_log(path, model, cfg)
        controls.append(control)
        measurements.append(measured)
        initial_states.append(state)
        print(f"[data] {path}: {len(control.times)} samples")

    sequences = sysid.ModelSequences(
        name="suspended_wheels",
        spec=spec,
        sequence_name=[path.stem for path in cfg.data],
        initial_state=initial_states,
        control=controls,
        sensordata=measurements,
        allow_missing_sensors=True,
    )
    residual_kwargs = dict(
        models_sequences=[sequences], enabled_observations=OBSERVATIONS, resample_true=True,
    )
    if cfg.threads > 0:
        residual_kwargs["n_threads"] = cfg.threads
    residual_fn = sysid.build_residual_fn(**residual_kwargs)
    params = build_params(cfg)
    print("[sysid] optimizing damping_left/right and friction_left/right")
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

    cfg.out.mkdir(parents=True, exist_ok=True)
    sysid.save_results(cfg.out, [sequences], params, opt_params, opt_result, residual_fn)
    identified_spec = sysid.apply_param_modifiers_spec(opt_params, spec)
    identified_spec.to_file(str(cfg.out / "identified_wheels.xml"))
    np.savez(cfg.out / "identified_params.npz", **{
        name: np.asarray(parameter.value).copy()
        for name, parameter in opt_params.parameters.items()
    })
    for name, parameter in opt_params.parameters.items():
        print(f"  {name:20s} = {float(parameter.value[0]):.9g}")
    print(f"[done] {cfg.out}")


if __name__ == "__main__":
    main()
