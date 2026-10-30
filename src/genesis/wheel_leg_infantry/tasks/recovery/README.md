# Recovery

从原 stand_up 功能独立出的起身/恢复任务，为统一策略提供恢复回合。

`config.py` 配置初始姿态、连续站稳条件、奖励和终止；`reset_pos.json` 为当前初始姿态，`reset_pose.py` 将其解析为可保存的数值配置。`buffers.py` 管理站稳计时与成功事件，`rewards.py` 定义恢复奖励，`task.py` 接入共享环境。

交互 eval 的 R 键从 `config.py::get_recovery_reset_cfgs()` 提供的数值姿态池随机采样并复位起身。默认池只有 `reset_pos.json`；扩展时在 `RECOVERY_RESET_POSES` 中追加其他 JSON 快照路径或 `"default"`，无需修改 eval 或按键代码。训练默认复位仍由 `RESET_POSE` 决定。

恢复成功需要连续满足高度、姿态、腿角、接触和速度条件；成功回合作为正常终局结束。恢复任务和另外三个任务共用动作缩放、观测与 GRU policy。

统一评估入口支持 `--tasks recovery`。整体用法见 [框架说明](../../README.md)。
