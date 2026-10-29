# Multi Terrain

统一策略中的多地形任务。内部包含上下台阶、上下坡、方波与平台凸台等通行子任务，共用主训练分给 multi_terrain 的环境配额。平地运动由独立的 `tasks/locomotion` 提供。

- `config.py`：子任务比例、地形、命令、奖励、成功条件和课程。
- `task.py`：跨越逻辑与统一环境的任务 hook。
- `buffers.py` / `rewards.py`：子任务状态与奖励。
- `terrain.py` / `curriculum.py`：几何分配和成功率难度课程。
- `intermittent.py`：间断通行状态机与制动标定、预测。
- `parking.py`：停车课程。

内部的 `task_weights` 调节子任务比例；包根配置的 `weight` 调节 multi_terrain 相对另外三个任务的比例。课程改变只在 reset 更新任务/地形分配，checkpoint 保存难度、成功率窗口和停车状态。

使用包根 `train.py` / `eval.py`；多卡配置已归入 `core/distributed.py`，runner 统一为 `policy_runner.py`。完整示例见 [框架说明](../../README.md)。
