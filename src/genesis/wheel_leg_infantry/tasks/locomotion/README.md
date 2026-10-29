# Locomotion

统一环境中的速度、转向和站立跟踪任务。独立配置在 `config.py`，`task.py` 直接继承通用任务基类及根目录 `rewards.py` 中的公共奖励。

训练和评估均使用包根目录的 `train.py`、`eval.py`；默认与 recovery、jump、multi_terrain 共同训练单一策略。聚焦采样可传 `--tasks locomotion`，混合比例可传 `--task-weights locomotion=4`。

制动测量函数位于 `tools/braking.py`，公共 viewer 工具位于 `tools/interactive_viewer.py`。整体入口、观测和恢复约定见 [框架说明](../../README.md)。
