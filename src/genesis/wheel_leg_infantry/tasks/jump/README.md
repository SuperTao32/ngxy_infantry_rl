# Jump

统一策略中的条件跳跃任务，保留 flat / step_20cm / step_40cm 三种模式。

- `config.py`：模式比例、阶段时长、高度轨迹、起跳距离表、奖励和课程。
- `buffers.py` / `reward_state.py`：回合状态、连续腾空确认、峰值与落地结算。
- `task.py`：起跳状态采样、阶段更新、命令、终止和公共环境 hook。
- `rewards.py`：任务奖励。
- `terrain.py`：任务专用平台通道和模式分配，复用 `terrains` 的公共几何接口。
- `geometry.py` / `height_reference.py` / `phase.py`：几何、轨迹与阶段编码。

jump 与其他任务共享观测、动作和 PPO。每个回合直接采样起跳位置及前向初速度，结束后正常 reset；原 teacher、warm-start、staged runner 和终止后补齐流程已移除。配置中的电机参数仅应用到 jump 对应的环境行。

通过统一 `eval.py --tasks jump` 评估同一个策略。入口和网络约定见 [框架说明](../../README.md)。
