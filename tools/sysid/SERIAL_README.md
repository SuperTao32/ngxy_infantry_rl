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

安装依赖并运行（项目根目录）：

```bash
uv pip install pyserial
.venv/bin/python tools/sysid/record_serial.py --port /dev/ttyUSB0 --baud 921600 --data-len 28 --out logs/sysid/left_forward.csv
```

Windows 将串口改为 `--port COM3`。输出路径可省略，默认保存到
`logs/sysid/serial_日期_时间.csv`。按 Ctrl+C 保存退出，已有文件不会被覆盖。

错误数据自动保存到同目录的 `<文件名>_errors.csv`，例如 `data.csv` 对应
`data_errors.csv`，无需额外参数。两个文件每秒刷新，退出时也会刷新，均不覆盖已有文件。
错误日志包含：

- `host_time`：电脑记录该错误的本地时间，带时区，不是 MCU 采样时间。
- `stream_offset`：从本次接收起始位置计数的字节偏移（从 0 开始）。
- `error`：`crc_error`、`trailer_error`、`invalid_values`、`discarded_bytes` 或 `incomplete_frame`。
- `byte_count`、`raw_hex`：记录的字节数和原始十六进制数据。
- `crc_received`、`crc_calculated`：候选帧中的 CRC 与接收端计算的 CRC。
- `trailer_received`、`trailer_expected`：实际帧尾与预期的 `0x65`。

CRC/帧尾错误保存从候选前缀开始、长度为 `data_len + 5` 的完整字节窗口。
发生丢字节或失步时，该窗口可能跨越实际发送帧；帧尾错误行的 CRC 值也只是
按候选位置计算，不能当作已确认帧的 CRC。重新同步时丢弃的字节单独记录，
可能与前面的错误窗口重叠，可通过 `stream_offset` 对照，日志行数不等于坏帧数。
找不到前缀的数据也会作为 `discarded_bytes` 保存；退出时的残留数据记为
`incomplete_frame`。这些字节片段没有 CRC/帧尾字段，相关列留空。


CSV 表头为 `time,torque_front,torque_rear,q_front,q_rear,dq_front,dq_rear`。
数组下标 0 对应 front，1 对应 rear，列名自动生成。
其他数据长度按 `uint32_t time_ms + float32 数组` 解析，支持 4～252 字节的 4 的倍数，
列名自动生成为 `time,value_0,value_1,...`，共 `data_len / 4` 列。
若数据包含其他类型或填充，需要同步修改解析布局，不能只调整长度。
每次采集使用一个固定的 `data_len`。`time_ms` 除以 1000 保存为秒，其余数值不滤波、不重采样。
供现有 SysID 使用时，固件力矩单位应为 Nm、角度为 rad、速度为 rad/s。

接收端处理拆包和粘包，CRC 或帧尾校验失败后重新寻找前缀，丢弃非有限浮点数。
每秒刷新文件并报告保存帧数、CRC 错误、帧尾错误、非法数值、丢弃字节、时间回退次数。
不严格检查 2 ms 间隔，接受重复时间戳、周期抖动和缺帧造成的正向时间跳变。
仅统计时间回退，仍保留对应数据；正常 uint32 回绕不计为回退。CSV 保留原始 MCU 时间，
回绕或重启后时间会回退，需分段用于辨识。

500 Hz × 33 字节 = 16500 字节/秒，8N1 至少需要 165000 波特率；
建议使用 230400 或更高，默认 921600，并与固件保持一致。
500 Hz 由固件每 2 ms 发送一次保证，接收端不主动限速。
