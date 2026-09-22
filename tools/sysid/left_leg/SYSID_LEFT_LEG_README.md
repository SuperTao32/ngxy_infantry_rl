# 左腿闭链系统辨识

脚本联合辨识四个参数：

- `damping_front`、`damping_rear`：前后主动关节的黏性阻尼。
- `friction_front`、`friction_rear`：前后主动关节的库仑摩擦。

从动关节参数不参与优化。
输入直接使用主动关节输出端力矩 torque，单位为 Nm，执行器增益为 1。
力矩应是经过减速器后的关节力矩，脚本不再配置 K 或进行电流换算。
保留原 XML 的 ±54 Nm 力矩限幅。

## CSV 数据列

必需列：

- `time`：时间 [s]。
- `torque_front`：前主动关节输出端力矩 [Nm]。
- `torque_rear`：后主动关节输出端力矩 [Nm]。
- `q_front`：前主动关节角度 [rad]。
- `q_rear`：后主动关节角度 [rad]。

建议提供的可选列：

- `dq_front`：前主动关节角速度 [rad/s]。
- `dq_rear`：后主动关节角速度 [rad/s]。

缺少 dq 列时，脚本使用 `numpy.gradient` 根据角度估算速度。
气弹簧执行器沿滑动关节轴向施加恒定 +420 N，与位置和速度无关。
CSV 无需提供气弹簧力。
力矩方向可通过 `front_torque_sign` / `rear_torque_sign` 设置为 ±1。
旧 CSV 中的电流数值不能直接改列名当作力矩使用。

## 安装依赖

```bash
uv add "mujoco[sysid]" scipy numpy
```

## 运行

编辑 `sysid_left_leg.py` 顶部的 `CFG = SysIDConfig(...)` 配置块，例如：

```python
CFG = SysIDConfig(
    model=PROJECT_ROOT / "assets/robot/wheelbipeV14_2/mjcf/left_leg.xml",
    data=(
        PROJECT_ROOT / "logs/sysid/slow_forward.csv",
        PROJECT_ROOT / "logs/sysid/slow_reverse.csv",
        PROJECT_ROOT / "logs/sysid/chirp_front.csv",
        PROJECT_ROOT / "logs/sysid/chirp_rear.csv",
        PROJECT_ROOT / "logs/sysid/chirp_both.csv",
    ),
    out=PROJECT_ROOT / "tools/sysid/results/left_leg",
    initial_state=PROJECT_ROOT / "tools/sysid/results/suspended_initial_state.npz",
)
```

然后直接运行，无需命令行参数：

```bash
# 首次运行时先生成悬空初态，详见下文；已有结果时可跳过。
python tools/sysid/settle_suspended_robot.py
python tools/sysid/left_leg/sysid_left_leg.py
```

脚本直接加载 `left_leg.xml`。该模型仅包含左腿，车身已固定在 z=1 m。
两个主动电机使用关节力矩作为输入；`left_spring2_joint_ctrl` 在每个采样点
接收 420 N 恒力。轮关节没有执行器。从动关节参数、等式约束和接触设置
保持 MJCF 中的定义，不会另外生成修改后的输入 MJCF。

每条 CSV 轨迹都使用 `CFG.initial_state` 中的同一份初始姿态，所有初始速度
设为零。每次实机实验应从该悬空静止姿态开始记录，编码器角度需转换到模型
关节坐标系。CSV 中的 q/dq 仅作为测量输出，不覆盖初态。
若初态文件缺失，脚本会报错，不会退回使用 XML 默认姿态。

## 实验说明

固定车身并使左腿悬空，记录实际施加的关节输出端力矩。
采集正反向运动、多种力矩幅值，以及运动过程中不同腿部构型的数据。
此阶段不辨识轮地摩擦。

## 悬空静置，生成初始 q

运行独立脚本复现「固定车身 → 电机不输出力矩 → 气弹簧加载 → 静置」流程：

```bash
python tools/sysid/settle_suspended_robot.py
# 可选：实时查看运动过程
python tools/sysid/settle_suspended_robot.py --view
```

脚本在内存中加载整车 `wheelbipeV14_2.xml`，移除浮动基座自由度，
将车身固定在 z=1 m。两个气弹簧各施加 +420 N，其余执行器输入均为零；
重力、关节摩擦、阻尼和闭链约束保留。原 XML 不会被修改。

从 XML 默认 q 和零速度开始仿真，所有关节速度与加速度绝对值均低于
`1e-3` 并持续 1 s 后，保存此时的 q，并将导出的 dq 全部设为 0。
转动关节使用 rad，滑动关节使用 m，时间单位为 s。
仿真过程中不会强制清零速度。可在脚本顶部 `CFG` 中调整阈值和等待时间；
若 60 s 内未达到静止判据则报错，不写入新的结果。

共用静置结果位于 `tools/sysid/results/`；左腿辨识结果写入其 `left_leg/` 子目录。
静置脚本输出：

- `suspended_initial_state.npz`：整车固定基座的 `joint_names`、`qpos`、
  `qvel`、`ctrl`，以及左腿的 `left_joint_names`、`left_qpos`、`left_qvel`。
  `qvel_before_zero` 保留实际停止时的微小速度供核对。
- `suspended_initial_state.json`：按关节名列出 q、零速度、单位，并记录
  仿真时长、闭链误差和软限位越界量。

```python
with np.load("tools/sysid/results/suspended_initial_state.npz") as state:
    left_q_by_name = dict(zip(state["left_joint_names"], state["left_qpos"]))
    q0 = np.array([left_q_by_name[model.joint(i).name] for i in range(model.njnt)])
    dq0 = np.zeros(model.nv)
    # 此处 model 应为 left_leg.xml 编译得到的模型。
```

整车与左腿模型的状态维数不同，复用时按关节名匹配。
辨识脚本通过 `CFG.initial_state` 直接读取此文件，按关节名填入左腿 q，
所有初始 dq 设为 0。每条 CSV 都使用该静置初态；不再用首帧测量值初始化，
也不再进行闭链投影或限位裁剪，保留静置结果中的软约束变形。
这是当前模型及其默认起始姿态下的静置结果，使用前应与实机静置角度核对。
特别注意输出的软限位越界量：MuJoCo 的软约束允许带载越界，
不能将这种越界当作实机硬限位后的真实姿态。

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
