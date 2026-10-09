# Genesis → MuJoCo locomotion

默认加入前左、前右、下左、下右四个 ToF，当前策略输入为 36 维，需要使用新配置重新训练。
支持保存配置中的 `tof.history_frames`（默认 `1`，包含当前帧），输入维度为 `32 + 4 * history_frames`。
同一个 ToF 的多帧连续排列，组内旧帧在前、当前帧在后；组间依次为前左、前右、下左、下右。
`tof.update_hz` 配置更新频率，默认 50 Hz；必须为正有限数，且不超过 MuJoCo 的 50 Hz 控制频率。
只有新测距帧到来时更新历史，采样间隔内保持上一帧；reset 立即测距、用首帧填满并重置采样相位。
读取 `observations()` 不推进历史。非整数步采样周期对齐下一控制步，并保留相位余量。
历史名义时间跨度为 `(history_frames - 1) / update_hz`，50 Hz、5 帧在两套环境中均对应 80 ms。
`tof.include_in_observation=False` 时保留测距但策略输入为 32 维，可用于训练 jump teacher；评估沿用存档配置。
旧 32 维存档未声明 ToF 时仍使用原接口；旧 33 维模型无法直接加载。
安装、量程和归一化规则详见 [ToF 配置](../../genesis/wheel_leg_infantry/core/TOF.md)。
GUI 右上角按 ToF 更新频率显示米制距离：`Front L/R` 为前左/前右，`Down L/R` 为下左/下右。
`[no hit/range]` 表示未命中或量程外。旧权重未启用测距时，GUI 自动开启诊断测距，策略输入维度保持不变。

在项目根目录运行，默认读取 `log_shared/infantry_locomotion_v3` 最新可用版本及 checkpoint：

```bash
.venv/bin/python -m experiments.mujoco.locomotion_eval
```

窗口内 W/S 调整前进速度，A/D 调整转向速度，Q/E 调整高度，空格将速度命令归零，R 重置。
默认运行 30 秒，可用 `--duration 300` 延长。鼠标操作沿用 MuJoCo viewer。

指定模型、速度及记录文件：

```bash
.venv/bin/python -m experiments.mujoco.locomotion_eval \
  --checkpoint log_shared/infantry_locomotion_v3/version_0001/model_6000.pt \
  --vx 0.5 --wz 0 --height 0.22 --duration 30 \
  --headless --csv /tmp/sim2sim_forward.csv
```

也可用 `--log-root logs -e <实验名> --version 1 --ckpt 6000` 选择训练结果。
CSV 每行记录该周期的输入观测（当前为 `obs_0..35`）、裁剪后动作 `action_0..5`、命令和步进后状态。
`vx/wz` 是本体坐标系下基座原点的速度，`x/y` 是世界坐标位置；`estimated_vx` 是仅供诊断的 IMU/轮速融合估计，不进入策略输入。

## 对齐内容

- 从同目录 `cfgs.pkl` 读取参数，严格加载 RSL-RL `MLPModel` actor，保留 Beta 确定性输出和观测归一化。
- 36 维观测顺序：陀螺仪、加速度计比力、投影重力、3 个命令、关节位置偏移、关节速度、轮速、虚拟腿长、虚拟腿角、上次裁剪后的动作、前左/前右/下左/下右归一化 ToF 距离。
- 按训练配置中的关节名称映射动作，不依赖 MJCF 的关节排列。
- 控制周期 20 ms，20 个 1 ms 物理子步；腿部位置 PD、轮速控制、力矩饱和、虚拟腿目标约束及一拍动作延迟使用保存参数。
- 气弹簧力每个控制周期更新并保持，PD 力矩每个物理子步重新计算。
- 加载训练所用 robot MJCF，在内存中添加平面和灯光，禁用自碰撞。原始模型文件不变。
- IMU 使用模型自带的 gyro / accelerometer，reset 首帧加速度和速度估计清零，与 Genesis 环境一致。

MuJoCo 的 `mjOBJ_BODY` 速度采用惯性主轴坐标系；这里用 `mjOBJ_XBODY` 得到本体坐标系速度。
参考 [MuJoCo 对象类型](https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtobj)。

## 验证结果与范围

以下是删除速度估计输入之前的历史结果，不代表当前 36 维接口的验证结果。
使用本仓库 `version_0001/model_6000.pt`，高度命令 0.22 m，各运行 30 秒：

| 场景 | 命令 | 结果 |
| --- | --- | --- |
| 站立 | vx=0, wz=0 | 最大倾角 1.18°，高度范围 0.207–0.231 m |
| 前进 | vx=0.5 m/s | 10–30 秒平均实际速度 0.5373 m/s，估计速度 0.5375 m/s，最大倾角 1.72° |
| 转向 | wz=0.3 rad/s | 10–30 秒平均实际角速度 0.2857 rad/s，最大倾角 1.21° |

均未触发跌倒停止。以上为无窗口测试；图形窗口和键盘交互仍需在本机实际查看。

这是平地、确定性复位基线：位置取保存 reset 范围中点，初始姿态正立、速度为零；不启用训练课程、随机扰动、地形或 IMU 噪声/延迟。
非零 IMU 噪声/延迟配置会明确报错。当前支持 locomotion MLP，尚未迁移 jump。
倾角超过 60°、基座高度低于 0.08 m 或发生数值警告时停止并保留 CSV，不自动重置掩盖失败。
本基线跌倒条件用于诊断，不等同于 Genesis 训练中的 episode 终止规则；两引擎接触和约束求解也不保证完全相同。


覆盖观测初值与关节映射、动作延迟/裁剪/力矩限制、IMU 坐标系、气弹簧保持、控制时钟、非法动作，以及实际速度与位移差分的一致性。

ToF 测距噪声和安装偏移使用存档中的 `env_cfg.randomization.sensors.tof` 配置，
可用 `--sensor-noise` / `--no-sensor-noise` 覆盖总开关（仅作用于 ToF，其他传感器仍为理想测量）。
安装位置、角度及测距偏置在 reset 时重采样，白噪声随新测距帧更新；
详见 [ToF 配置与随机化](../../genesis/wheel_leg_infantry/core/TOF.md)。
