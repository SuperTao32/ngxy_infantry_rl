# 右腿闭链系统辨识

`sysid_right_leg.py` 与左腿版本使用相同的辨识流程，只辨识四个参数：

- `damping_front`、`damping_rear`：前后主动关节的黏性阻尼。
- `friction_front`、`friction_rear`：前后主动关节的库仑摩擦。

`CFG.k_front`、`CFG.k_rear` 为固定的电流到关节力矩换算系数，单位 Nm/A，
包含减速器传动，默认均为 1.0。运行前填写实际值。气弹簧恒力为 +420 N。

## 模型与初态

模型为 `assets/robot/wheelbipeV14_2/mjcf/right_leg.xml`，从整车 XML 的实际
右腿结构提取，保留右腿质量、惯量、关节参数、闭链约束和接触排除关系。
基座固定在 z=1 m，仅保留前电机、后电机和气弹簧三个执行器，轮关节无驱动。
从动关节的阻尼、摩擦及惯量参数不参与辨识。

初态复用 `settle_suspended_robot.py` 生成的整车静置结果。默认路径沿用现有的
`sysid_results_left_leg/suspended_initial_state.npz`；虽然目录名含 left，文件中的
`joint_names` / `qpos` 包含左右腿。脚本按右腿关节名提取 q，所有初始速度设为 0，
保留静置结果，不进行闭链投影或限位裁剪。已有该文件时可以直接使用。

每条 CSV 都从相同的静置初态开始。CSV 中的 q/dq 用于与仿真结果比较，
不覆盖初态；实机编码器角度应与模型关节坐标一致。

## CSV 格式

使用同目录的 `sysid_log_template.csv`，填写右腿数据：

- 必需：`time` [s]、`iq_front` / `iq_rear` [A]、`q_front` / `q_rear` [rad]。
- 可选：`dq_front` / `dq_rear` [rad/s]；缺失时由角度差分估算。

电流应为实际测得的 Iq。若电流方向与模型相反，可通过
`front_current_sign` / `rear_current_sign` 配置符号。

## 运行

环境依赖与左腿版本相同：`mujoco[sysid]`、NumPy、SciPy。
编辑脚本顶部 `CFG`，填写 CSV 路径与固定 K：

```python
CFG = SysIDConfig(
    model=PROJECT_ROOT / "assets/robot/wheelbipeV14_2/mjcf/right_leg.xml",
    data=(PROJECT_ROOT / "logs/sysid/right_slow_forward.csv",),
    out=PROJECT_ROOT / "sysid_results_right_leg",
    initial_state=PROJECT_ROOT / "sysid_results_left_leg/suspended_initial_state.npz",
    k_front=1.0,
    k_rear=1.0,
)
```

```bash
# 没有静置结果时先生成一次，左右腿共用。
python tools/sysid/settle_suspended_robot.py
python tools/sysid/right_leg/sysid_right_leg.py
```

辨识报告和 `identified_params.npz` 保存至 `sysid_results_right_leg/`。
