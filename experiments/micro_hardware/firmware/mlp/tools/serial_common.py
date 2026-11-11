import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import time
RUNNER_VERSION = '2026-11-07.usb-idle-rearm-v4'
MAX_SERIAL_FRAME_BYTES = 4096
PROBE_LENGTHS = [1, 63, 64, 65, 127, 255, 384]

def decode(words):
    if len(words) != 2 or any((not isinstance(w, str) or len(w) != 8 for w in words)):
        raise ValueError('Expected exactly two binary32 hex words')
    values = [struct.unpack('<f', struct.pack('<I', int(w, 16)))[0] for w in words]
    if not all((math.isfinite(v) for v in values)):
        raise ValueError('Nonfinite device logits')
    return values

def prediction_check(response, seq, variant, expected, tol):
    if response.get('kind') != 'pred' or response.get('seq') != seq or response.get('variant') != variant:
        raise ValueError('Unexpected/duplicate prediction frame')
    y = decode(response['bits'])
    ref = decode(expected)
    if variant == 1 and [int(w,16) for w in response['bits']] != [int(w,16) for w in expected]:
        raise ValueError(f'INT8 output bits differ from the portable golden: seq={seq}')
    error = max((abs(a - b) for a, b in zip(y, ref)))
    close = all((abs(a - b) <= tol['atol'] + tol['rtol'] * abs(b) for a, b in zip(y, ref)))
    changed = int((y[1] > y[0]) != (ref[1] > ref[0]))
    if not close or changed:
        raise ValueError(f'Device fidelity failed: seq={seq}, variant={variant}, error={error}, changed={changed}')
    return (y, error)

class SerialLink:
    transport = 'physical_serial'

    def __init__(self, port, trace):
        import serial
        self.trace = trace
        self.pending = bytearray()
        self.last_command = None
        self.serial = serial.Serial(port=None, baudrate=115200, timeout=0.25, write_timeout=5)
        try:
            self.serial.dtr = False
            self.serial.rts = False
            self.serial.port = port
            self.serial.open()
            self.log('port_open', port=port, dtr=False, rts=False, control_lines_configured_before_open=True)
            time.sleep(1)
            self.serial.reset_input_buffer()
        except Exception as error:
            self.serial.close()
            self.log('port_open_error', error=str(error), port=port)
            raise

    def log(self, event, **fields):
        record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event': event, **fields}
        self.trace.write(json.dumps(record) + '\n')
        self.trace.flush()

    def fail_frame(self, reason, raw):
        self.log('frame_error', reason=reason, raw_hex=raw.hex(), last_command=self.last_command)
        raise ValueError(f'{reason}; received {len(raw)} bytes: {raw[:160]!r}. See transport.jsonl for the complete bytes and last command.')

    def send(self, line):
        self.last_command = line
        self.log('tx', command=line)
        data = (line + '\n').encode('ascii')
        written = self.serial.write(data)
        if written != len(data):
            self.log('write_error', expected_bytes=len(data), written_bytes=written)
            raise IOError('Incomplete serial command write; see transport.jsonl')
        self.serial.flush()

    def receive(self, timeout=10, startup=False):
        until = time.monotonic() + timeout
        while True:
            end = self.pending.find(b'\n')
            if end >= 0:
                raw = bytes(self.pending[:end + 1])
                del self.pending[:end + 1]
                if len(raw) > MAX_SERIAL_FRAME_BYTES:
                    self.fail_frame('Serial frame exceeds size limit', raw)
                try:
                    obj = json.loads(raw.decode('ascii'))
                except (UnicodeError, json.JSONDecodeError):
                    if startup and (not raw.lstrip().startswith(b'{')):
                        self.log('startup_text', raw_hex=raw.hex())
                        continue
                    self.fail_frame('Malformed complete serial frame', raw)
                if not isinstance(obj, dict):
                    self.fail_frame('Expected a JSON object in serial frame', raw)
                self.log('response', response=obj)
                if obj.get('kind') == 'error':
                    raise ValueError('Firmware error: ' + str(obj))
                return obj
            if len(self.pending) > MAX_SERIAL_FRAME_BYTES:
                self.fail_frame('Unterminated serial frame exceeds size limit', bytes(self.pending))
            if time.monotonic() >= until:
                raw = bytes(self.pending)
                self.log('receive_timeout', partial_hex=raw.hex(), last_command=self.last_command, timeout_seconds=timeout)
                raise TimeoutError(f'No newline-terminated firmware response within {timeout}s; pending {len(raw)} bytes: {raw[:160]!r}. See transport.jsonl.')
            try:
                data = self.serial.read(min(512, max(1, self.serial.in_waiting)))
            except Exception as error:
                self.log('read_error', error=str(error), partial_hex=bytes(self.pending).hex(), last_command=self.last_command)
                raise
            if data:
                self.log('rx_chunk', n_bytes=len(data), raw_hex=data.hex())
                self.pending.extend(data)

    def close(self):
        self.serial.close()

def transport_probe(link):
    count = 0
    for repeat in range(3):
        for length in PROBE_LENGTHS:
            count += 1
            seq = 1000000 + count
            payload = (f'{seq:08x}' + '0123456789abcdef' * 24)[:length]
            link.send(f'P {seq} {payload}')
            response = link.receive()
            if response.get('kind') != 'probe' or response.get('seq') != seq or response.get('payload') != payload:
                raise ValueError('USB round-trip mismatch; see transport.jsonl')
    return {'status': 'PASS', 'round_trips': count, 'payload_lengths': PROBE_LENGTHS, 'scope': 'Communication check only; not model correctness or latency'}

def load_batch(link, data):
    indices = data['benchmark_indices']
    if len(indices) != 32 or len(set(indices)) != 32:
        raise ValueError('Invalid benchmark sample IDs')
    labels = [data['records'][i]['label'] for i in indices]
    if labels.count(0) != 16 or labels.count(1) != 16:
        raise ValueError('Benchmark batch is not balanced')
    for slot, index in enumerate(indices):
        link.send(f'L {slot} ' + ' '.join(data['records'][index]['input_bits']))
        r = link.receive()
        if r.get('kind') != 'loaded' or r.get('slot') != slot:
            raise ValueError('Batch acknowledgment mismatch')
