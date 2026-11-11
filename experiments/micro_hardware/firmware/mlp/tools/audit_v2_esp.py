"""Offline mlp-v2 ESP capture audit; Python standard library only.

Reads immutable capture/package ZIPs or an unpacked package. Never accesses
hardware and never imports the acquisition runner. Fails closed on mismatch.
"""
import argparse
from clock_support import inspect_clock
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics
import struct
import zipfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def zip_files(path, root_file):
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None, 'ZIP CRC failure')
        files = {}
        for info in z.infolist():
            if info.is_dir():
                continue
            p = PurePosixPath(info.filename.replace('\\', '/'))
            require(not p.is_absolute() and '..' not in p.parts, 'Unsafe ZIP path')
            require(str(p) not in files, 'Duplicate ZIP member')
            files[str(p)] = z.read(info)
    roots = [n[:-len(root_file)] for n in files if PurePosixPath(n).name == root_file]
    require(len(roots) == 1, 'Ambiguous ZIP root: ' + root_file)
    root = roots[0]
    require(all(n.startswith(root) for n in files), 'File outside ZIP root')
    return {n[len(root):]: data for n, data in files.items()}


def decode(bits):
    require(len(bits) == 2 and all(isinstance(w, str) and len(w) == 8 for w in bits), 'Output word format')
    values = [struct.unpack('!f', bytes.fromhex(w))[0] for w in bits]
    require(all(math.isfinite(v) for v in values), 'Nonfinite logits')
    return values


def klass(values):
    return int(values[1] > values[0])


def percentile(values, p):
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def session_audit(files, prefix, batch, package_read, golden, session, require_physical=True):
    def read(name):
        return files[prefix + name]
    summary = json.loads(read('summary.json'))
    predictions = [json.loads(line) for line in read('predictions.jsonl').splitlines()]
    trace = [json.loads(line) for line in read('transport.jsonl').splitlines()]
    blocks = json.loads(read('latency_blocks.json'))
    clock = inspect_clock(trace, summary)
    target = batch['target']
    variants = [0, 1]
    order = variants + [2]
    core, mhz, chip = (1, 240, 'ESP32-S3') if target == 's3' else (0, 160, 'ESP32-C3')
    for key, expected in dict(status='PASS', physical_execution=require_physical, case=golden['case'],
                              session=session, runner_version='mlp-v2',
                              variants=variants, board_id=batch['board_id'],
                              artifact=golden['artifact']).items():
        require(summary.get(key) == expected, 'Summary identity: ' + key)
    for key, name in [('runner_sha256', 'tools/run_mlp.py'),
                      ('golden_sha256', f'esp32/generated/model{golden["case"]}_golden.json'),
                      ('firmware_sha256', f'esp32/firmware/{target}_case{golden["case"]}_merged.bin')]:
        require(summary[key] == sha(package_read(name)), 'Summary/package hash: ' + key)
    records = golden['records']
    require(len(records) == len({r['image_id'] for r in records}) == 1024, 'Golden IDs/count')
    require(Counter(r['label'] for r in records) == {0: 512, 1: 512}, 'Evaluation balance')
    require(len(predictions) == len(variants) * 1024, 'Prediction count')
    require(len(blocks) == 30 * len(order), 'Timing block count')

    # Independently reconstruct newline-delimited replies from raw byte chunks.
    pending, command, exchanges = bytearray(), None, []
    counts, rx_bytes, previous = Counter(), 0, None
    for index, event in enumerate(trace):
        now = datetime.fromisoformat(event['utc'])
        previous = now
        kind = event['event']
        counts[kind] += 1
        if kind == 'port_open':
            require(index == 0 and event['port'] == summary['port'], 'Port-open identity')
            require(event['dtr'] is False and event['rts'] is False and
                    event['control_lines_configured_before_open'] is True, 'USB control lines')
        elif kind == 'tx':
            require(command is None and not pending, 'Overlapping request/leftover RX')
            command = event['command']
        elif kind == 'rx_chunk':
            require(command is not None, 'Unsolicited RX')
            raw = bytes.fromhex(event['raw_hex'])
            require(len(raw) == event['n_bytes'] and len(raw) > 0, 'RX byte count')
            pending.extend(raw)
            rx_bytes += len(raw)
        elif kind == 'response':
            require(command is not None and b'\n' in pending, 'Missing complete RX frame')
            end = pending.index(b'\n') + 1
            require(end <= 4096, 'Oversize frame')
            response = json.loads(bytes(pending[:end]).decode('ascii'))
            del pending[:end]
            require(response == event['response'], 'Raw bytes/parsed response mismatch')
            exchanges.append((command, response))
            command = None
        else:
            raise ValueError('Unexpected transport event: ' + kind)
    require(command is None and not pending, 'Incomplete final exchange')
    require(counts['port_open'] == 1 and counts['tx'] == counts['response'] == len(exchanges), 'Exchange counts')
    cursor = 0

    def take(expected_command):
        nonlocal cursor
        require(cursor < len(exchanges), 'Missing exchange')
        cmd, response = exchanges[cursor]
        cursor += 1
        require(cmd == expected_command, 'Unexpected command at exchange ' + str(cursor))
        return response

    hello = []
    for _ in range(3):
        h = take('H')
        hello.append(h)
        for key, expected in dict(kind='hello', protocol=4, model=golden['model'],
                                  artifact=golden['artifact'], input_dim=golden['input_dim'], variants=2,
                                  chip=chip, cpu_mhz=mhz, task_core=core, task_affinity=core,
                                  radios_started=False, psram_bytes=0,
                                  serial_transport='esp-idf-usb-serial-jtag-v4').items():
            require(h.get(key) == expected, 'Hello field: ' + key)
        require(h['dram_copy_ok'] is False and h['dram_copy_bytes']==0, 'Unexpected DRAM copy')
    require(summary['device'] == hello[0] and summary['hello_round_trips'] == 3, 'Saved hello')
    lengths = [1, 63, 64, 65, 127, 255, 384]
    for i, length in enumerate(lengths * 3, 1):
        seq = 1000000 + i
        payload = (f'{seq:08x}' + '0123456789abcdef' * 24)[:length]
        response = take(f'P {seq} {payload}')
        require(response == dict(kind='probe', seq=seq, payload=payload), 'Probe echo')
    require(summary['transport_probe']['status'] == 'PASS' and
            summary['transport_probe']['round_trips'] == 21 and
            summary['transport_probe']['payload_lengths'] == lengths, 'Saved transport probe')

    seq, recomputed, outputs, sinks = 0, [], {}, {}
    last_id = records[golden['benchmark_indices'][-1]]['image_id']
    for v in variants:
        stats = dict(variant=v, name=golden['variants'][v], n=0, max_abs_error=0.0,
                     correct=0, changed_decisions_vs_golden=0,
                     changed_decisions_vs_trained_fp32=0)
        outputs[v] = []
        for rec in records:
            seq += 1
            r = take(f'V {seq} {v} ' + ' '.join(rec['input_bits']))
            require(r['kind'] == 'pred' and r['seq'] == seq and r['variant'] == v, 'Prediction identity')
            require(predictions[seq - 1] == dict(image_id=rec['image_id'], label=rec['label'], **r), 'Prediction file/trace mismatch')
            y, expected, reference = decode(r['bits']), decode(rec['expected_bits'][v]), decode(rec['reference_bits'])
            if v == 1:
                require([int(w,16) for w in r['bits']] == [int(w,16) for w in rec['expected_bits'][v]], 'INT8 output bits differ')
            tol = golden['tolerances']
            require(all(abs(a - b) <= tol['atol'] + tol['rtol'] * abs(b) for a, b in zip(y, expected)), 'Numerical tolerance exceeded')
            require(klass(y) == klass(expected), 'Decision differs from own golden')
            stats['n'] += 1
            stats['max_abs_error'] = max(stats['max_abs_error'], *(abs(a - b) for a, b in zip(y, expected)))
            stats['correct'] += int(klass(y) == rec['label'])
            stats['changed_decisions_vs_trained_fp32'] += int(klass(y) != klass(reference))
            outputs[v].append(r['bits'])
            if rec['image_id'] == last_id:
                sinks[v] = r['bits'][0]
        recomputed.append(stats)
    require(recomputed == summary['correctness'], 'Recomputed correctness/summary mismatch')
    pairs = []
    for new, old in [(1, 0)]:
        a, b = outputs[new], outputs[old]
        pairs.append(dict(new_variant=new, baseline_variant=old, n=1024,
            bit_identical_vectors=sum(x == y for x, y in zip(a, b)),
            changed_decisions=sum(klass(decode(x)) != klass(decode(y)) for x, y in zip(a, b)),
            max_abs_logit_difference=max(abs(i - j) for x, y in zip(a, b) for i, j in zip(decode(x), decode(y)))))

    indices = golden['benchmark_indices']
    require(len(indices) == len(set(indices)) == 32 and
            Counter(records[i]['label'] for i in indices) == {0: 16, 1: 16}, 'Benchmark inputs')
    for slot, index in enumerate(indices):
        r = take(f'L {slot} ' + ' '.join(records[index]['input_bits']))
        require(r == dict(kind='loaded', slot=slot), 'Load acknowledgment')
    sinks[2] = records[indices[-1]]['input_bits'][0]
    benchmark_responses = []

    def bench(v, rounds):
        nonlocal seq
        seq += 1
        r = take(f'B {seq} {v} {rounds}')
        for key, expected in dict(kind='bench', seq=seq, variant=v, rounds=rounds,
                                  n=rounds * 32, complete=True, sink=sinks[v],
                                  core_start=core, core_end=core).items():
            require(r.get(key) == expected, 'Timing frame field: ' + key)
        require(type(r['elapsed_us']) is int and 0 <= r['elapsed_us'] < 5000000, 'Timer range')
        require(type(r['elapsed_cycles']) is int and 0 < r['elapsed_cycles'] < 2**32, 'Cycle range')
        if r['elapsed_us'] >= 1000:
            require(.95 <= r['elapsed_cycles'] / (mhz * r['elapsed_us']) <= 1.05, 'Cycles/wall-time gate')
        benchmark_responses.append(r)
        return r

    rounds, gauge_count = {}, 0
    for v in order:
        gauge = 1
        while True:
            r = bench(v, gauge)
            gauge_count += 1
            if r['elapsed_us'] >= 10 or gauge == 32:
                break
            gauge *= 2
        require(r['elapsed_us'] > 0, 'Unresolved timing resolution')
        rounds[v] = max(1, min(32, 20000 * gauge // r['elapsed_us']))
        for _ in range(5):
            bench(v, rounds[v])
    rng, block_index = random.Random(20261109 + 100 * golden['case'] + session), 0
    for repeat in range(30):
        current = list(order)
        rng.shuffle(current)
        for v in current:
            r = bench(v, rounds[v])
            require(r['elapsed_us'] > 0, 'Zero measured block')
            expected = dict(r, repeat=repeat, us_per_inference=r['elapsed_us'] / r['n'],
                            cycles_per_inference=r['elapsed_cycles'] / r['n'])
            require(blocks[block_index] == expected, 'Saved block/trace/schedule mismatch')
            block_index += 1
    require(cursor == len(exchanges), 'Unconsumed exchanges')
    require(len(benchmark_responses) == gauge_count + 5 * len(order) + len(blocks), 'Benchmark counts')
    latency = []
    require(len(summary['latency']) == len(order), 'Latency summary length')
    for v, saved in zip(order, summary['latency']):
        rows = [r for r in blocks if r['variant'] == v]
        us = [r['elapsed_us'] / r['n'] for r in rows]
        cycles = [r['elapsed_cycles'] / r['n'] for r in rows]
        expected = dict(variant=v, name=golden['variants'][v] if v != 2 else 'loop_overhead',
                        n_blocks=30, median_us=statistics.median(us), median_cycles=statistics.median(cycles))
        require(saved == expected, 'Recomputed latency/summary mismatch')
        latency.append(dict(expected, p05_us=percentile(us, .05), p95_us=percentile(us, .95),
                            min_us=min(us), max_us=max(us), rounds_per_block=rounds[v],
                            inferences_per_block=32 * rounds[v]))
    medians = {r['variant']: r['median_us'] for r in latency}
    comparisons = []
    for baseline, new in [(0, 1)]:
        by_repeat = {(r['repeat'], r['variant']): r['elapsed_us'] / r['n'] for r in blocks}
        paired = [by_repeat[(i, baseline)] / by_repeat[(i, new)] for i in range(30)]
        comparisons.append(dict(baseline_variant=baseline, new_variant=new,
            ratio_of_medians=medians[baseline] / medians[new],
            latency_reduction_percent=100 * (1 - medians[new] / medians[baseline]),
            median_paired_ratio=statistics.median(paired),
            paired_ratio_p05=percentile(paired, .05), paired_ratio_p95=percentile(paired, .95),
            new_faster_in_paired_blocks=sum(x > 1 for x in paired), paired_blocks=30))
    ratios = [r['elapsed_cycles'] / r['elapsed_us'] for r in benchmark_responses if r['elapsed_us'] >= 1000]
    return dict(status='PASS', physical_execution=summary['physical_execution'], host_clock=clock, board_id=summary['board_id'], target=target, case=golden['case'], session=session,
        start_utc=summary['start_utc'], end_utc=summary['end_utc'], device=summary['device'],
        recorded_hashes={k: summary[k] for k in ('runner_sha256', 'golden_sha256', 'firmware_sha256', 'artifact')},
        transport=dict(events=dict(counts), response_count=len(exchanges), rx_bytes=rx_bytes,
                       command_counts=dict(Counter(cmd.split()[0] for cmd, _ in exchanges)),
                       raw_frames_reconstructed=True, trailing_rx_bytes=0),
        correctness=recomputed, new_kernel_pairs=pairs, latency=latency,
        timing=dict(recorded_blocks=len(blocks), warmup_blocks=5 * len(order), gauge_blocks=gauge_count,
                    total_timing_frames=len(benchmark_responses), schedule_seed=20261109 + 100 * golden['case'] + session,
                    cycles_per_microsecond_min=min(ratios), cycles_per_microsecond_max=max(ratios),
                    heap_free_values=sorted({r['heap_free'] for r in benchmark_responses}),
                    heap_min_values=sorted({r['heap_min'] for r in benchmark_responses}),
                    minimum_stack_watermark_bytes=min(r['stack_watermark_bytes'] for r in benchmark_responses)),
        comparisons=comparisons,
        scope='One physical board/session; 30 warm-cache blocks per method, not independent devices. No energy measurement. Both MLP formats reside in mapped flash in the same ESP binary; no DRAM copies.')

