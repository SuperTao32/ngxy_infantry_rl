# Locomotion

统一环境中的速度、转向和站立跟踪任务。独立配置在 `config.py`，`task.py` 继承通用任务基类及根目录 `rewards.py` 中的公共奖励；指数静止正奖励与 recovery 共用计算公式。

`standing_still` 仅在静止指令下生效，采用公共的静止命令阈值（默认 `abs(vx_cmd) < 0.02 m/s` 且 `abs(wz_cmd) < 0.02 rad/s`），直接追踪实际零速度，不减去命令值：

```text
standing_still = height_gate * attitude_gate
                * exp(-(vx² + vy²) / lin_scale² - wz² / ang_scale²)
```

速度为机身坐标系下的真实水平线速度与偏航角速度。没有速度死区，站稳且速度严格为零时原始奖励为 1；单一方向速度偏差达到一个尺度时降为 `exp(-1) ≈ 0.368`，达到两个尺度时降为 `exp(-4) ≈ 0.018`。高度/姿态门不使用保底值，门关闭时奖励为 0。

当前 `standing_still_lin_vel_scale=0.01 m/s`、`standing_still_ang_vel_scale=0.02 rad/s`，越小越严格。`reward_scales.standing_still` 初始为 0.0，课程在第 1000/2000/3000 轮分别调整为 0.5/1.0/2.0，第 4000 轮继承 2.0；按控制步长缩放，每步最大贡献为当前权重乘以 `dt`。可通过阶段的 `targets.reward_scales.standing_still` 覆盖。它与已有速度跟踪奖励、`standing_drift` 漂移惩罚共同生效，日志为 `locomotion/rewards/standing_still`。旧存档续训要启用此项，使用 `--resume-config current`。

训练和评估均使用包根目录的 `train.py`、`eval.py`；默认与 recovery、jump、multi_terrain 共同训练单一策略。训练聚焦采样可传 `--tasks locomotion`，混合比例可传 `--task-weights locomotion=4`。eval 使用 `--terrain` 选择场景，通过按键连续操纵同一机器人，无需选择任务。

制动测量函数位于 `tools/braking.py`，公共 viewer 工具位于 `tools/interactive_viewer.py`。整体入口、观测和恢复约定见 [框架说明](../../README.md)。
