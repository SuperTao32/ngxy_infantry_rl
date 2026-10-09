# MuJoCo jump sim2sim

速度估计输入已删除，旧 33/45 维 checkpoint 无法直接加载。此适配器仍使用旧版平地 jump 结构（44 维），
不加入 ToF 观测，也不支持当前 Genesis 的 47 维多模式 jump。
teacher 应使用关闭 `tof.include_in_observation` 后训练的 32 维 locomotion 权重，或已有兼容权重。

在项目根目录启动：

```bash
.venv/bin/python -m experiments.mujoco.jump.jump_eval --duration 300
```

启动后先运行 locomotion，按 **空格**触发单次 jump，确认离地后双轮重新着地即恢复 locomotion。
交接保持位置、姿态、速度、速度估计器和动作延迟状态；可再次按空格跳跃。
跳跃中重复按空格会忽略，不会排队触发下一次。

| 按键 | 行为 |
| --- | --- |
| Space | 从 locomotion 触发 jump |
| I / K 或 ↑ / ↓ | locomotion 前向速度 ±0.1 m/s，限制在 jump 配置记录的训练命令范围 |
| Backspace | locomotion 前向速度归零 |
| R | 重置机器人，恢复 locomotion |

Jump 期间锁定触发时的 vx、保持 wz=0，高度命令由保存的 `height_reference` 随时间生成。
该高度参考是机身到轮底的距离，不是机身到地面的高度。
运行默认 30 仿真秒，可用 `--duration` 延长；关闭窗口退出。
检测到倾角 >60°、机身高度 <0.08 m 或数值异常时停止，不自动重置掩盖失败。

## 模型选择

默认读取 `log_shared/infantry_jump_v6` 最新可用版本与 checkpoint，并自动加载该版本
`source_locomotion` 记录的 teacher，不会将 teacher 自动替换为最新模型。
当前仓库默认组合为 jump `version_0006/model_2000.pt` 和 locomotion `version_0001/model_6000.pt`。

```bash
.venv/bin/python -m experiments.mujoco.jump.jump_eval \
  --checkpoint log_shared/infantry_jump_v6/version_0006/model_2000.pt \
  --vx 0.5 --duration 300
```

也可用 `--log-root logs -e infantry_jump_v6 --version 6 --ckpt 2000`。
若 teacher 日志在另一根目录，用 `--locomotion-log-root` 指定；保留原实验名、版本及文件名。
两种 checkpoint 的同目录都需要 `cfgs.pkl`。

## 观测、控制与交接

- 共用 locomotion 的 MuJoCo 物理实现：50 Hz 控制、1 ms 子步、位置/轮速 PD、气弹簧、限幅及一拍动作延迟。
- Locomotion teacher 使用 32 维观测，jump 使用相同的 32 维前缀，加 6 维 `last_actions` 和 6 维多尺度时间编码。
- 动作历史按 Genesis 返回观测时的顺序保存：`actions` 是刚提交动作，`last_actions` 是前一拍动作；触发首帧二者相同。
- Jump 开始时使用 `jump_motor_params` 非空覆盖值，结束恢复原参数。本模型腿部 Kp/Kd 从 60/3 切到 80/1。
- 默认 `--handoff landing` 在确认双轮离地、随后双轮接触时交回 teacher，最迟到训练时间窗结束交回。
- 为避免 MuJoCo 蹬地期间接触力瞬时变零导致提前交接，离地确认同时要求较低轮底至少高于平面 5 mm；该判据只影响交接/诊断，不加入 actor 观测。
- `--handoff horizon` 完整执行保存的 jump 时间窗（当前 1.1 秒），用于对照训练中的完整落地段。
- 未确认起跳但时间窗结束时同样恢复 locomotion，并在结果中输出 `taken_off=False`，不当作成功跳跃。

当前只支持平地；`clearance` 为两侧轮底到地面距离的较小值，`peak_clearance` 是原始物理峰值，未套用 Genesis 奖励中的失稳归零规则。
模型配置不兼容或观测维度不匹配会报错，不静默更换模型或截断观测。


CSV 记录每拍使用的 `policy_mode`、输入观测、裁剪后动作、输入命令，以及步进后的状态与模式。
Locomotion 行的 `obs_32..43` 留空，jump 行包含全部 44 维。
时间触发按整次运行时间计时，不受 R 重置后的物理时间归零影响。

以下为删除速度估计输入之前的历史 checkpoint 验证结果：

| 测试 | 结果 |
| --- | --- |
| 2 秒、6 秒原地各跳一次 | 两次峰值轮底离地约 0.338 m，均在触发后 0.54 秒交回 locomotion；10 秒内最大倾角 3.64° |
| vx=0.5 m/s，2 秒触发 | 峰值轮底离地 0.338 m，0.54 秒交接，8 秒内最大倾角 3.85° |
| 完整时间窗执行 | 1.10 秒交接，之后恢复站立，8 秒内最大倾角 1.46° |

三项均未触发跌倒停止，11 项 locomotion/jump 回归测试通过。
以上为 CPU 无窗口验证；真实窗口显示与物理键盘操作尚未验证。
