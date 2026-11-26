# 串口保存 CSV

接收单片机 `UartSend_Send` 发送的 500 Hz 固定二进制包。
按小端 MCU、32 位 IEEE 754 float 解析，使用 `--data-len` 匹配固件的
`config.data_len`，默认 `sizeof(Sysid_log) = 28`。
前缀由 `s` 和配置的长度字节生成，帧长自动计算为 `data_len + 5`，
CRC 位于偏移 `data_len + 2`，帧尾位于偏移 `data_len + 4`。
以下是默认 28 字节数据区对应的布局：

| 字节偏移 | 长度 | 内容 |
| --- | --- | --- |
| 0 | 1 | 帧头 `s`（0x73） |
| 1 | 1 | 数据长度 28（0x1C） |
| 2 | 4 | `uint32_t time_ms` |
| 6 | 8 | `float torque[2]` |
| 14 | 8 | `float q[2]` |
| 22 | 8 | `float dq[2]` |
| 30 | 2 | 固件 CRC16，低字节在前 |
| 32 | 1 | 帧尾 `e`（0x65） |

CRC 覆盖偏移 1～29（`txbuf + 1`，`data_len + 1 = 29` 字节），
按 `soft_crc.c` 的实际配置计算：多项式 0x1021，反射实现使用 0x8408，
初值 0xFFFF，无最终异或。虽然固件函数名为 `CRC16_Modbus_calc`，
其默认配置并非标准 Modbus 的 0x8005（反射后 0xA001）。
默认配置生成前缀 `73 1C`；更改 `--data-len` 后长度字节及帧尾位置随之变化，
数据中的 `s` 或 `e` 不作为分隔符。CRC 不包含帧头、CRC 本身和帧尾。
CRC 初值固定为 `soft_crc_Init()` 中使用的 0xFFFF。
用提供的 C 实现交叉验证：`123456789` 的 CRC 为 0x6F91。

在项目根目录运行 `uv sync` 安装项目依赖，再用 `--target` 选择本次采集对象。

```bash
uv sync
.venv/bin/python tools/sysid/record_serial.py --target left_leg --port /dev/ttyUSB0
.venv/bin/python tools/sysid/record_serial.py --target right_leg --port /dev/ttyUSB0
.venv/bin/python tools/sysid/record_serial.py --target wheels --port /dev/ttyUSB0
```

| `--target` | 默认输出文件 | 28 字节帧的 CSV 表头 |
| --- | --- | --- |
| `left_leg` | `logs/sysid/left_leg_YYYYMMDD_HHMMSS.csv` | `time,torque_front,torque_rear,q_front,q_rear,dq_front,dq_rear` |
| `right_leg` | `logs/sysid/right_leg_YYYYMMDD_HHMMSS.csv` | `time,torque_front,torque_rear,q_front,q_rear,dq_front,dq_rear` |
| `wheels` | `logs/sysid/wheels_YYYYMMDD_HHMMSS.csv` | `time,torque_left,torque_right,q_left,q_right,dq_left,dq_right` |

固件数组下标 0 和 1 分别对应前、后关节，或左、右轮。省略 `--target` 时使用原有的
`serial_YYYYMMDD_HHMMSS.csv` 文件名和 `front`、`rear` 表头。
可用 `--out logs/sysid/wheels_forward.csv` 指定路径。Windows 将串口改为
`--port COM3`。按 Ctrl+C 保存退出，已有文件不会被覆盖。

其他数据长度按 `uint32_t time_ms + float32 数组` 解析，支持 4～252 字节的 4 的倍数，
列名自动生成为 `time,value_0,value_1,...`，共 `data_len / 4` 列。
若数据包含其他类型或填充，需要同步修改解析布局，不能只调整长度。
每次采集使用一个固定的 `data_len`。`time_ms` 除以 1000 保存为秒，其余数值不滤波、不重采样。
供现有 SysID 使用时，固件力矩单位应为 Nm、角度为 rad、速度为 rad/s。

接收端处理拆包和粘包，CRC 或帧尾校验失败后重新寻找前缀，丢弃非有限浮点数。
每次采集只生成一个数据 CSV，每秒刷新文件，并在终端汇总保存帧数、CRC 错误、帧尾错误、
非法数值、丢弃字节和时间回退次数。
不严格检查 2 ms 间隔，接受重复时间戳、周期抖动和缺帧造成的正向时间跳变。
仅统计时间回退，仍保留对应数据；正常 uint32 回绕不计为回退。CSV 保留原始 MCU 时间，
回绕或重启后时间会回退，需分段用于辨识。

500 Hz × 33 字节 = 16500 字节/秒，8N1 至少需要 165000 波特率；
建议使用 230400 或更高，默认 921600，并与固件保持一致。
500 Hz 由固件每 2 ms 发送一次保证，接收端不主动限速。

## 查看离线辨识的拟合曲线

将采集结果填入对应辨识脚本的 `CFG.data`，并按采集频率设置 `CFG.sample_rate_hz`。
例如，用 `record_serial.py` 采集的双轮 CSV 对应 500 Hz，配置后运行：

```bash
.venv/bin/python tools/sysid/wheels/sysid_wheels.py
```

每条输入 CSV 会在 `tools/sysid/results/<target>/` 下生成一张 `fit_curves_*.png`，
终端会打印图像路径。图中叠加参与拟合的实测曲线、初始参数的仿真曲线和拟合后参数的仿真曲线。
左右腿图包含前后关节的角度和角速度，双轮图包含左右轮角速度。

## 实时验证辨识参数

先运行相应的辨识脚本生成 `identified_params.npz`，按辨识时的悬空工况准备实车。
在电机开始动作前打开实时验证窗口。串口由一个进程独占，先退出同一串口上的采集程序。

```bash
uv sync
.venv/bin/python -m tools.sysid.validate_live --target left_leg --port /dev/ttyUSB0
```

`--target` 支持 `left_leg`、`right_leg`、`wheels`，须与固件正在发送的对象一致。
默认从对应辨识脚本的输出目录加载参数，也可通过 `--params /path/to/identified_params.npz` 指定。
串口沿用采集工具的 921600 波特率、CRC 和默认 28 字节数据区。用 `--baud` 调整波特率，
用 `--window` 调整滚动窗口的秒数，默认 8 秒。关闭图窗退出。

图中叠加实测、原始 MJCF 和辨识后的角度及速度，下方显示送入两个模型的同一组力矩。
工具从串口接收数据。它沿用辨识脚本的力矩方向、模型约束和初态规则。
腿从悬空快照和零速度开始，气弹簧使用与左右腿离线辨识共用的标称曲线。轮从首帧角度和零转速开始，连续轮角用 rad。
后续实测角度和速度用于比较，仿真状态按实测力矩独立推进。横轴按 MCU 时间对齐，
每个间隔保持上一帧力矩，在线曲线直接使用串口数据。
MCU 时间回退或相邻有效帧间隔超过 100 ms 时开始新段，清空曲线并重置仿真；
通过 `--max-gap-ms` 可调整间隔阈值。采集前让实车处于对应的初始姿态并静止，
可以减少初始条件差异对比较的影响。
