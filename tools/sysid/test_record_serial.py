import csv
import io
import struct
import unittest
from contextlib import redirect_stdout

from record_serial import COLUMNS_7, CsvRecorder, crc16_firmware, receive

HEADER = b's\x1c'
VALUES = (1.25, -2.5, 0.125, -0.25, 10.0, -20.0)


def frame(time_ms, values=VALUES):
    data = HEADER + struct.pack('<I6f', time_ms, *values)
    return data + crc16_firmware(data[1:]).to_bytes(2, 'little') + b'e'


class FakeSerial:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def read(self, size):
        try:
            return next(self.chunks)
        except StopIteration:
            raise KeyboardInterrupt


class SerialTests(unittest.TestCase):
    def recorder(self, **kwargs):
        output = io.StringIO()
        return output, CsvRecorder(output, **kwargs)

    def test_crc_known_vector(self):
        self.assertEqual(crc16_firmware(b'123456789'), 0x6F91)

    def test_layout_and_time_units(self):
        output, recorder = self.recorder()
        # CRC 0x51C1 已用固件 soft_crc.c 编译验证，低字节在前。
        packet = bytes.fromhex('731c393000000000a03f000020c00000003e000080be000020410000a0c1c15165')
        self.assertEqual(len(packet), recorder.frame_size)
        recorder.accept(packet)
        rows = list(csv.reader(io.StringIO(output.getvalue())))
        self.assertEqual(rows[0], COLUMNS_7)
        self.assertEqual(rows[1][0], '12.345')
        self.assertEqual(tuple(map(float, rows[1][1:])), VALUES)

    def test_every_split_and_timeouts(self):
        packet = frame(2)
        for split in range(len(packet) + 1):
            with self.subTest(split=split):
                output, recorder = self.recorder()
                with redirect_stdout(io.StringIO()):
                    receive(FakeSerial([packet[:split], b'', packet[split:]]), output, recorder)
                self.assertEqual(recorder.rows, 1)
                self.assertEqual(recorder.pending, b'')

    def test_corruption_noise_and_resync(self):
        _, recorder = self.recorder()
        broken = bytearray(frame(2))
        broken[8] ^= 1
        recorder.accept(b'noise' + broken + frame(4)[:-1] + frame(6) + frame(8)[:13])
        self.assertEqual(recorder.rows, 1)
        self.assertEqual(recorder.crc_errors, 1)
        self.assertEqual(recorder.trailer_errors, 1)
        self.assertEqual(recorder.pending, frame(8)[:13])
        recorder.accept(frame(8)[13:])
        self.assertEqual(recorder.rows, 2)
        self.assertEqual(recorder.timestamp_backtracks, 0)

    def test_500_hz_stream(self):
        output, recorder = self.recorder()
        data = b''.join(frame(i * 2) for i in range(1000))
        with redirect_stdout(io.StringIO()):
            receive(FakeSerial([data[i:i+4096] for i in range(0, len(data), 4096)]), output, recorder)
        self.assertEqual((recorder.rows, recorder.crc_errors, recorder.timestamp_backtracks), (1000, 0, 0))
        self.assertEqual(len(output.getvalue().splitlines()), 1001)

    def test_bad_trailer_and_wrong_length_resync(self):
        _, recorder = self.recorder()
        bad_length = bytearray(frame(0))
        bad_length[1] = 27
        recorder.accept(bad_length + frame(2)[:-1] + b'x' + frame(4))
        self.assertEqual(recorder.rows, 1)
        self.assertEqual(recorder.trailer_errors, 1)
        self.assertEqual(recorder.crc_errors, 0)

    def test_configured_data_length(self):
        for data_len in (4, 16, 32, 252):
            with self.subTest(data_len=data_len):
                output, recorder = self.recorder(data_len=data_len)
                values = [float(i) for i in range(data_len // 4 - 1)]
                data = b's' + bytes([data_len]) + struct.pack(f'<I{len(values)}f', 2, *values)
                packet = data + crc16_firmware(data[1:]).to_bytes(2, 'little') + b'e'
                recorder.accept(packet[:3])
                recorder.accept(packet[3:] + packet)
                self.assertEqual(recorder.frame_size, data_len + 5)
                self.assertEqual(recorder.rows, 2)
                rows = list(csv.reader(io.StringIO(output.getvalue())))
                self.assertEqual(rows[1], ['0.002', *map(str, values)])

    def test_invalid_data_length(self):
        for data_len in (0, 3, 27, 256):
            with self.subTest(data_len=data_len), self.assertRaises(ValueError):
                self.recorder(data_len=data_len)

    def test_markers_inside_payload(self):
        _, recorder = self.recorder()
        recorder.accept(frame(0x651c73) + frame(0x651c75))
        self.assertEqual(recorder.rows, 2)
        self.assertEqual(recorder.timestamp_backtracks, 0)

    def test_invalid_values_and_timestamp_anomalies(self):
        _, recorder = self.recorder()
        recorder.accept(frame(0) + frame(2, (float('nan'), *VALUES[1:])) + frame(6) + frame(6))
        self.assertEqual((recorder.rows, recorder.invalid_values, recorder.timestamp_backtracks), (3, 1, 0))

    def test_relaxed_timing_and_restart(self):
        _, recorder = self.recorder()
        times = (0, 5, 15, 30, 50, 50, 1000, 0, 5)
        recorder.accept(b''.join(frame(t) for t in times))
        self.assertEqual(recorder.rows, len(times))
        self.assertEqual(recorder.timestamp_backtracks, 1)

    def test_timestamp_rollover(self):
        _, recorder = self.recorder()
        recorder.accept(frame(0xFFFFFFFE) + frame(0))
        self.assertEqual((recorder.rows, recorder.timestamp_backtracks), (2, 0))

    def test_noise_buffer_bounded(self):
        _, recorder = self.recorder()
        recorder.accept(b'x' * 100000 + HEADER[:1])
        self.assertEqual(recorder.pending, HEADER[:1])
        recorder.accept(frame(0)[1:])
        self.assertEqual(recorder.rows, 1)

    def test_error_log_preserves_bad_frames_and_offsets(self):
        errors = io.StringIO()
        output, recorder = self.recorder(error_output=errors)
        bad_crc = bytearray(frame(2))
        bad_crc[8] ^= 1
        bad_trailer = frame(4)[:-1] + b'x'
        invalid = frame(6, (float('nan'), *VALUES[1:]))
        partial = frame(10)[:12]
        data = b'noise' + bad_crc + bad_trailer + invalid + frame(8) + partial
        with redirect_stdout(io.StringIO()):
            receive(FakeSerial([data[:20], data[20:]]), output, recorder)
        rows = list(csv.DictReader(io.StringIO(errors.getvalue())))
        failures = [r for r in rows if r['error'] != 'discarded_bytes']
        self.assertEqual([r['error'] for r in failures],
                         ['crc_error', 'trailer_error', 'invalid_values', 'incomplete_frame'])
        for row, raw, offset in zip(failures, [bad_crc, bad_trailer, invalid, partial], [5, 38, 71, 137]):
            self.assertEqual(bytes.fromhex(row['raw_hex']), raw)
            self.assertEqual(int(row['stream_offset']), offset)
            self.assertEqual(int(row['byte_count']), len(raw))
        self.assertNotEqual(failures[0]['crc_received'], failures[0]['crc_calculated'])
        self.assertEqual(failures[1]['crc_received'], failures[1]['crc_calculated'])
        self.assertEqual(failures[1]['trailer_received'], '0x78')
        self.assertEqual(failures[1]['trailer_expected'], '0x65')
        self.assertEqual(failures[-1]['crc_received'], '')
        self.assertEqual(recorder.rows, 1)
        self.assertEqual(len(output.getvalue().splitlines()), 2)

    def test_clean_stream_error_log_has_header_only(self):
        errors = io.StringIO()
        output, recorder = self.recorder(error_output=errors)
        with redirect_stdout(io.StringIO()):
            receive(FakeSerial([frame(0), frame(2)]), output, recorder)
        self.assertEqual(list(csv.DictReader(io.StringIO(errors.getvalue()))), [])


if __name__ == '__main__':
    unittest.main()
