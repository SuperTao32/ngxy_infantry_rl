#!/usr/bin/env python3
"""接收 500 Hz Sysid_log 固定二进制包，校验 CRC 后保存 CSV。"""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import struct
import time

PAYLOAD = struct.Struct("<I6f")
DEFAULT_DATA_LEN = PAYLOAD.size
LEG_CHANNELS = ("front", "rear")
TARGET_CHANNELS = {
    "left_leg": LEG_CHANNELS,
    "right_leg": LEG_CHANNELS,
    "wheels": ("left", "right"),
}


def payload_for_length(data_len):
    # 当前数据布局：一个 uint32 毫秒时间戳，后续均为 float32。
    if not isinstance(data_len, int) or not 4 <= data_len <= 255 or data_len % 4:
        raise ValueError("data_len 必须是 4～252 之间的 4 的倍数（uint32 + float32 数组）")
    return struct.Struct(f"<I{data_len // 4 - 1}f")


def crc16_firmware(data):
    """匹配 soft_crc.c：0x1021 反射为 0x8408，初值 0xFFFF，无最终异或。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
    return crc


class CsvRecorder:
    def __init__(self, output, data_len=DEFAULT_DATA_LEN, *, channels=LEG_CHANNELS):
        self.payload = payload_for_length(data_len)
        self.frame_size = data_len + 5
        if data_len == DEFAULT_DATA_LEN:
            columns = [f"{quantity}_{channel}" for quantity in ("torque", "q", "dq") for channel in channels]
        else:
            columns = [f"value_{i}" for i in range(data_len // 4 - 1)]
        self.writer = csv.writer(output)
        self.writer.writerow(["time", *columns])
        self.header = b"s" + bytes([data_len])
        self.pending = bytearray()
        self.rows = 0
        self.crc_errors = 0
        self.trailer_errors = 0
        self.invalid_values = 0
        self.discarded_bytes = 0
        self.timestamp_backtracks = 0
        self.previous_ms = None

    def accept(self, chunk):
        self.pending.extend(chunk)
        while True:
            start = self.pending.find(self.header)
            if start < 0:
                # 保留可能被拆开的前缀首字节。
                keep = 1 if self.pending.endswith(self.header[:1]) else 0
                discard = len(self.pending) - keep
                self.discarded_bytes += discard
                del self.pending[:discard]
                return
            if start:
                self.discarded_bytes += start
                del self.pending[:start]
            if len(self.pending) < self.frame_size:
                return
            frame = self.pending[:self.frame_size]
            if frame[-1] != ord("e"):
                self.trailer_errors += 1
                self.discarded_bytes += 1
                del self.pending[:1]
                continue
            expected = int.from_bytes(frame[-3:-1], "little")
            # 对应固件 txbuf + 1，长度 data_len + 1；不包含 txbuf[0]。
            if crc16_firmware(frame[1:-3]) != expected:
                self.crc_errors += 1
                self.discarded_bytes += 1
                del self.pending[:1]
                continue
            time_ms, *values = self.payload.unpack(frame[2:-3])
            del self.pending[:self.frame_size]
            if self.previous_ms is not None:
                delta = (time_ms - self.previous_ms) & 0xFFFFFFFF
                # 容忍重复、抖动和缺帧；模运算允许正常 uint32 回绕。
                if delta > 0x7FFFFFFF:
                    self.timestamp_backtracks += 1
            self.previous_ms = time_ms
            if not all(math.isfinite(value) for value in values):
                self.invalid_values += 1
                continue
            # 毫秒转秒，保留 MCU 时间戳，不重采样、不滤波。
            self.writer.writerow([f"{time_ms // 1000}.{time_ms % 1000:03d}", *values])
            self.rows += 1

    def status(self):
        return (f"保存 {self.rows} 帧，CRC 错误 {self.crc_errors}，"
                f"帧尾错误 {self.trailer_errors}，"
                f"非法数值 {self.invalid_values}，丢弃 {self.discarded_bytes} 字节，"
                f"时间回退 {self.timestamp_backtracks}")


def receive(port, output, recorder):
    last_report = time.monotonic()
    try:
        while True:
            recorder.accept(port.read(4096))
            now = time.monotonic()
            if now - last_report >= 1.0:
                output.flush()
                print(recorder.status(), flush=True)
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        output.flush()
        print(f"采集结束：{recorder.status()}，残留 {len(recorder.pending)} 字节")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGET_CHANNELS, help="采集对象，决定默认文件名前缀和 28 字节格式的 CSV 表头")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="串口，例如 /dev/ttyUSB0 或 COM3")
    parser.add_argument("--baud", type=int, default=921600, help="波特率，默认 921600")
    parser.add_argument("--data-len", type=int, default=DEFAULT_DATA_LEN, help="固件 config.data_len，默认 28；数据为 uint32 时间戳 + float 数组")
    parser.add_argument("--out", type=Path, help="输出 CSV 路径，默认写入 logs/sysid/采集对象_日期_时间.csv")
    args = parser.parse_args()
    out = args.out
    if out is None:
        prefix = args.target or "serial"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("logs/sysid") / f"{prefix}_{timestamp}.csv"
    channels = TARGET_CHANNELS.get(args.target, LEG_CHANNELS)
    try:
        payload_for_length(args.data_len)
    except ValueError as exc:
        parser.error(str(exc))
    frame_size = args.data_len + 5
    if args.baud < frame_size * 500 * 10:
        parser.error(f"500 Hz、{frame_size} 字节帧、8N1 至少需要 {frame_size * 500 * 10} 波特率")
    try:
        import serial
    except ImportError:
        parser.exit(1, "缺少 pyserial，请先运行：uv pip install pyserial\n")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", newline="", encoding="utf-8", buffering=65536) as output:
            with serial.Serial(args.port, args.baud, timeout=0.1) as port:
                print(f"接收 {args.port} @ {args.baud}，500 Hz / {frame_size} 字节 → {out}，按 Ctrl+C 保存退出", flush=True)
                receive(port, output, CsvRecorder(output, args.data_len, channels=channels))
    except (OSError, ValueError, serial.SerialException) as exc:
        parser.exit(1, f"采集停止：{exc}\n")


if __name__ == "__main__":
    main()
