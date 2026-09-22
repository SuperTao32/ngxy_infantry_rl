# 右腿闭链系统辨识

`sysid_right_leg.py` 与左腿版本使用相同的辨识流程，只辨识四个参数：

- `damping_front`、`damping_rear`：前后主动关节的黏性阻尼。
- `friction_front`、`friction_rear`：前后主动关节的库仑摩擦。

输入直接使用经过减速器后的关节输出端力矩 torque [Nm]，执行器增益为 1，
保留 ±54 Nm 力矩限幅。脚本不再配置 K 或进行电流换算，气弹簧恒力为 +420 N。

## 模型与初态

模型为 `assets/robot/wheelbipeV14_2/mjcf/right_leg.xml`，从整车 XML 的实际
右腿结构提取，保留右腿质量、惯量、关节参数、闭链约束和接触排除关系。
基座固定在 z=1 m，仅保留前电机、后电机和气弹簧三个执行器，轮关节无驱动。
从动关节的阻尼、摩擦及惯量参数不参与辨识。

初态复用 `settle_suspended_robot.py` 生成的整车静置结果。默认路径沿用现有的
`tools/sysid/results/suspended_initial_state.npz`，文件中的
`joint_names` / `qpos` 包含左右腿。脚本按右腿关节名提取 q，所有初始速度设为 0，
保留静置结果，不进行闭链投影或限位裁剪。已有该文件时可以直接使用。

每条 CSV 都从相同的静置初态开始。CSV 中的 q/dq 用于与仿真结果比较，
不覆盖初态；实机编码器角度应与模型关节坐标一致。

## CSV 格式

使用同目录的 `sysid_log_template.csv`，填写右腿数据：

- 必需：`time` [s]、`torque_front` / `torque_rear` [Nm]、`q_front` / `q_rear` [rad]。
- 可选：`dq_front` / `dq_rear` [rad/s]；缺失时由角度差分估算。

输入应为实际施加的关节输出端力矩。若力矩方向与模型相反，可通过
`front_torque_sign` / `rear_torque_sign` 配置符号。
旧 CSV 中的电流数值不能直接改列名当作力矩使用。

## 运行

环境依赖与左腿版本相同：`mujoco[sysid]`、NumPy、SciPy。
编辑脚本顶部 `CFG`，填写 CSV 路径：

```python
CFG = SysIDConfig(
    model=PROJECT_ROOT / "assets/robot/wheelbipeV14_2/mjcf/right_leg.xml",
    data=(PROJECT_ROOT / "logs/sysid/right_slow_forward.csv",),
    out=PROJECT_ROOT / "tools/sysid/results/right_leg",
    initial_state=PROJECT_ROOT / "tools/sysid/results/suspended_initial_state.npz",
)
```

```bash
# 没有静置结果时先生成一次，左右腿共用。
python tools/sysid/settle_suspended_robot.py
python tools/sysid/right_leg/sysid_right_leg.py
```

辨识报告和 `identified_params.npz` 保存至 `tools/sysid/results/right_leg/`。

## 离线滤波

默认以 `sample_rate_hz=1000.0`、`filter_cutoff_hz=50.0` 对力矩和已有速度列
进行四阶 Butterworth 双向零相位低通滤波（SciPy `sosfiltfilt`）。
缺少速度列时，先平滑角度再用 `numpy.gradient` 求速度；位置观测保留原始角度。
原始 CSV 不会被修改。上述参数直接在 `CFG = SysIDConfig(...)` 中配置，
`filter_cutoff_hz=None` 可关闭滤波，恢复原始数据及原始角度差分。

50 Hz 是起始设置，并非针对实测数据调好的值；激励频段应低于截止频率。
双向滤波的幅频响应是单次滤波的平方，端点可能有瞬态，应保留采集前后的静止段。
启用时至少需要 16 个样本，时间戳间隔应为 0.001 s（允许 5% 抖动）；
丢帧或采样率不符会报错，需先整理为等间隔数据。
