"""Serial-stream regression tests. No physical device execution is claimed."""
import contextlib
from collections import deque
import importlib.util
import io
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

DEVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEVICE / 'tools'))
import serial_common as runner
from clock_support import install_logger
install_logger(runner.SerialLink)

PRED = {'kind': 'pred', 'seq': 1, 'variant': 0,
        'bits': ['00000000', '3f800000']}
FRAME = (json.dumps(PRED) + '\n').encode('ascii')


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now


class ScriptedSerial:
    def __init__(self, chunks, clock):
        self.chunks = deque(chunks)
        self.clock = clock
        self.writes = []
        self.closed = False
        self.port = None
        self.dtr = self.rts = True
        self.open_states = []
    def open(self):
        self.open_states.append((self.port, self.dtr, self.rts))
    @property
    def in_waiting(self):
        if self.chunks and isinstance(self.chunks[0], bytes):
            return len(self.chunks[0])
        return 0
    def read(self, size):
        self.clock.now += .30
        if not self.chunks: return b''
        item = self.chunks.popleft()
        if isinstance(item, Exception): raise item
        if len(item) > size:
            self.chunks.appendleft(item[size:])
            item = item[:size]
        return item
    def reset_input_buffer(self): pass
    def write(self, data):
        self.writes.append(data)
        return len(data)
    def flush(self): pass
    def close(self): self.closed = True


class SerialTransportTests(unittest.TestCase):
    def test_control_lines_are_inactive_when_port_opens(self):
        clock = Clock()
        fake = ScriptedSerial([], clock)
        trace = io.StringIO()
        constructor = unittest.mock.Mock(return_value=fake)
        with patch.dict(sys.modules, {'serial': types.SimpleNamespace(Serial=constructor)}), \
             patch.object(runner.time, 'sleep'):
            link = runner.SerialLink('SIMULATED', trace)
            try:
                constructor.assert_called_once_with(port=None, baudrate=115200,
                                                    timeout=.25, write_timeout=5)
                self.assertEqual(fake.open_states, [('SIMULATED', False, False)])
                event = json.loads(trace.getvalue().splitlines()[0])
                self.assertTrue(event['control_lines_configured_before_open'])
            finally:
                link.close()

    def test_failed_open_closes_handle_and_logs_error(self):
        fake = ScriptedSerial([], Clock())
        trace = io.StringIO()
        with patch.dict(sys.modules, {'serial': types.SimpleNamespace(Serial=lambda **k: fake)}), \
             patch.object(fake, 'open', side_effect=OSError('port unavailable')):
            with self.assertRaisesRegex(OSError, 'port unavailable'):
                runner.SerialLink('SIMULATED', trace)
        self.assertTrue(fake.closed)
        event = json.loads(trace.getvalue().splitlines()[-1])
        self.assertEqual(event['event'], 'port_open_error')

    @contextlib.contextmanager
    def stream(self, chunks):
        clock = Clock()
        fake = ScriptedSerial(chunks, clock)
        trace = io.StringIO()
        module = types.SimpleNamespace(Serial=lambda *a, **k: fake)
        with patch.dict(sys.modules, {'serial': module}), \
             patch.object(runner.time, 'sleep'), \
             patch.object(runner.time, 'monotonic', clock):
            link = runner.SerialLink('SIMULATED', trace)
            try: yield link, trace, fake
            finally: link.close()

    def test_fragmented_response_survives_gaps_longer_than_old_read_timeout(self):
        with self.stream([b'{"ki', b'', b'', FRAME[4:17], b'', FRAME[17:]]) as (link, trace, _):
            self.assertEqual(link.receive(), PRED)
            self.assertGreater(len(trace.getvalue().splitlines()), 2)

    def test_every_two_fragment_boundary_including_newline(self):
        for cut in range(1, len(FRAME)):
            with self.subTest(cut=cut), self.stream([FRAME[:cut], b'', FRAME[cut:]]) as (link, _, _):
                self.assertEqual(link.receive(), PRED)

    def test_single_byte_fragmentation(self):
        with self.stream([bytes([b]) for b in FRAME]) as (link, _, _):
            self.assertEqual(link.receive(timeout=60), PRED)

    def test_complete_json_without_newline_does_not_pass(self):
        with self.stream([FRAME[:-1]]) as (link, trace, _):
            with self.assertRaisesRegex(TimeoutError, 'pending'):
                link.receive(timeout=1)
            last = json.loads(trace.getvalue().splitlines()[-1])
            self.assertEqual(bytes.fromhex(last['partial_hex']), FRAME[:-1])

    def test_two_responses_in_one_read_are_not_merged_or_discarded(self):
        second = {**PRED, 'seq': 2}
        with self.stream([FRAME + (json.dumps(second)+'\n').encode()]) as (link, _, _):
            self.assertEqual(link.receive(), PRED)
            self.assertEqual(link.receive(), second)

    def test_crlf_is_accepted(self):
        with self.stream([FRAME[:-1] + b'\r\n']) as (link, _, _):
            self.assertEqual(link.receive(), PRED)

    def test_startup_boot_text_is_retained_in_log(self):
        with self.stream([b'ESP-ROM:esp32s3\r\n', FRAME]) as (link, trace, _):
            self.assertEqual(link.receive(startup=True), PRED)
            self.assertIn('startup_text', trace.getvalue())

    def test_bad_complete_protocol_frame_is_never_skipped(self):
        for startup in [False, True]:
            with self.subTest(startup=startup), self.stream([b'{"ki\n', FRAME]) as (link, trace, _):
                with self.assertRaisesRegex(ValueError, 'Malformed complete'):
                    link.receive(startup=startup)
                self.assertIn(b'{"ki\n'.hex(), trace.getvalue())

    def test_non_ascii_and_non_object_frames_fail(self):
        for data in [b'{"kind":"\xff"}\n', b'[]\n', b'42\n']:
            with self.subTest(data=data), self.stream([data]) as (link, trace, _):
                with self.assertRaises(ValueError): link.receive()
                self.assertIn(data.hex(), trace.getvalue())

    def test_oversized_frame_with_and_without_newline_fails(self):
        for suffix in [b'', b'\n']:
            data = b'{' + b'a' * runner.MAX_SERIAL_FRAME_BYTES + suffix
            with self.subTest(suffix=suffix), self.stream([data]) as (link, _, _):
                with self.assertRaisesRegex(ValueError, 'size limit'): link.receive()

    def test_silent_timeout_keeps_command_for_diagnosis(self):
        with self.stream([]) as (link, trace, _):
            link.send('H')
            with self.assertRaisesRegex(TimeoutError, 'pending 0 bytes'): link.receive(timeout=1)
            last = json.loads(trace.getvalue().splitlines()[-1])
            self.assertEqual(last['last_command'], 'H')

    def test_disconnect_logs_partial_bytes(self):
        with self.stream([b'{"ki', OSError('disconnected')]) as (link, trace, _):
            with self.assertRaisesRegex(OSError, 'disconnected'): link.receive()
            last = json.loads(trace.getvalue().splitlines()[-1])
            self.assertEqual(last['event'], 'read_error')
            self.assertEqual(bytes.fromhex(last['partial_hex']), b'{"ki')

    def test_firmware_error_is_not_ignored(self):
        with self.stream([b'{"kind":"error","reason":"invalid_vector"}\n']) as (link, _, _):
            with self.assertRaisesRegex(ValueError, 'Firmware error'): link.receive()

    def test_duplicate_sequence_is_still_rejected(self):
        with self.stream([FRAME + FRAME]) as (link, _, _):
            runner.prediction_check(link.receive(), 1, 0, PRED['bits'], {'atol': 2e-5, 'rtol': 2e-5})
            with self.assertRaisesRegex(ValueError, 'Unexpected/duplicate'):
                runner.prediction_check(link.receive(), 2, 0, PRED['bits'], {'atol': 2e-5, 'rtol': 2e-5})

    def test_short_write_is_not_retried_silently(self):
        with self.stream([]) as (link, trace, fake), patch.object(fake, 'write', return_value=1):
            with self.assertRaisesRegex(IOError, 'Incomplete'): link.send('H')
            self.assertIn('write_error', trace.getvalue())




if __name__ == "__main__": unittest.main()
