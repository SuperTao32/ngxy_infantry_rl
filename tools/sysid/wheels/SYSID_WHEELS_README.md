# 悬空双轮阻尼与摩擦辨识

实验条件：车身和腿部机械固定，轮子离地，只有左右轮可以转动。
使用轮轴输出端力矩 torque 和轮角度/角速度，联合辨识四个参数：

| 参数 | 含义 | 单位 |
| --- | --- | --- |
| `damping_left` | 左轮黏性阻尼 | N·m·s/rad |
| `damping_right` | 右轮黏性阻尼 | N·m·s/rad |
| `friction_left` | 左轮库仑摩擦 | N·m |
| `friction_right` | 右轮库仑摩擦 | N·m |

输入直接使用经过传动后的轮轴输出端力矩 [Nm]，执行器增益为 1，
保留原 XML 的 ±5 Nm 限幅。脚本不再配置 K 或进行电流换算。

## 模型和初态

脚本直接加载双轮专用模型 `assets/robot/wheelbipeV14_2/mjcf/wheels.xml`。
模型仅包含两个固定支架、两个轮轴转动关节及其电机执行器，没有车身、腿部、
气弹簧或闭链约束，也不与地面接触。这对应机械夹具固定轮轴的悬空实验。

轮子质量、完整惯量张量、质心、轮轴关节附加转动惯量（armature）和重力沿用
整车模型。支架的 `pos`、`quat` 已按此前悬空静置结果写入 XML，
加载时不再依赖整车模型或静置 NPZ 文件。原 XML 文件不会被脚本修改；
输出报告中的模型保存了固定姿态和单位力矩增益。

每条轨迹的轮角度取 CSV 首帧，初始速度设为 0。请从轮子静止时开始记录，
然后施加激励；不应直接使用已经转动中的片段作为轨迹开头。
如果实际夹具姿态不同，可修改 XML 中 `left_wheel_mount`、`right_wheel_mount`
的 `pos` 和 `quat`。保留轮角的模型坐标定义，避免改变编码器零位对应关系。

## CSV 数据格式

同目录的 `sysid_log_template.csv` 提供表头：

```text
time,torque_left,torque_right,q_left,q_right,dq_left,dq_right
```

- `time`：时间 [s]，严格递增，每段至少 10 个样本。
- `torque_left`、`torque_right`：左右轮轴输出端力矩 [Nm]。
- `q_left`、`q_right`：连续累计轮角度 [rad]，坐标方向与 MJCF 一致。
- `dq_left`、`dq_right`：可选角速度 [rad/s]；缺失时对相应角度差分估算。

多圈轮角不能直接使用每圈跳变的编码器读数；先解包角度再写入 CSV。
若原始速度为 rpm，先乘 `2*pi/60` 转成 rad/s。力矩方向可通过
`left_torque_sign` / `right_torque_sign` 设置为 ±1；角度和速度也需事先
转换到模型的关节方向。缺失值或非有限值会报错。
旧 CSV 中的电流数值不能直接改列名当作力矩使用。

建议从静止开始做正转、反转和多个速度档的激励，并记录断电滑行段，
让数据同时覆盖低速摩擦和随速度变化的阻尼。多个 CSV 会联合拟合。
左右轮都需要有足够的运动数据；结果不包含轮地摩擦。

## 运行

依赖与腿部脚本相同：MuJoCo SysID、NumPy、SciPy。
编辑 `sysid_wheels.py` 顶部配置：

```python
CFG = SysIDConfig(
    model=PROJECT_ROOT / "assets/robot/wheelbipeV14_2/mjcf/wheels.xml",
    data=(
        PROJECT_ROOT / "logs/sysid/wheels_forward.csv",
        PROJECT_ROOT / "logs/sysid/wheels_reverse.csv",
    ),
    out=PROJECT_ROOT / "tools/sysid/results/wheels",
)
```

```bash
python tools/sysid/wheels/sysid_wheels.py
```

结果保存到 `tools/sysid/results/wheels/`，包括辨识报告、包含四个参数的
`identified_params.npz` 和写入辨识参数的 `identified_wheels.xml`。
`damping_max` / `friction_max` 是辨识上界，
分别默认为 0.1 和 1.0，可按电机实际情况调整。
