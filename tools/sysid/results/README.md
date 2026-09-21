# 初始状态与辨识结果

本目录集中存放系统辨识所需的静置初态和各项辨识输出。

- `suspended_initial_state.npz`：整车悬空静置后的关节位置和零初始速度，左右腿共用。
- `suspended_initial_state.json`：对应的可读关节表和静置诊断数据。
- `left_leg/`：左腿辨识输出，运行左腿脚本时创建。
- `right_leg/`：右腿辨识输出，运行右腿脚本时创建。
- `wheels/`：双轮辨识输出，运行双轮脚本时创建。

双轮辨识直接加载 `wheels.xml`，无需读取静置初态。
重新运行 `tools/sysid/settle_suspended_robot.py` 会更新本目录的静置初态文件。
