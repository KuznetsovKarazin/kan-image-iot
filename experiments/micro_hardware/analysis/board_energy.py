"""One board at a time: KAN + MLP energy acquisition with audited checkpoints.

FNIRSI recording remains manual. A completed checkpoint is a verified device
trace plus a structurally checked native CFN, NOT an accepted energy estimate.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from cfn_reader import parse_cfn


def must(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checked_path(folder, relative):
    path = (folder / relative).resolve()
    must(path.is_relative_to(folder.resolve()), 'Unsafe stored path')
    return path


def release_check():
    m = json.loads((ROOT / 'MANIFEST.json').read_text())
    must(m['version'] == 'all-boards-energy-v1', 'Wrong release')
    for name, digest in m['sha256'].items():
        must(sha(checked_path(ROOT, name)) == digest, 'Changed release file: ' + name)
    return sha(ROOT / 'MANIFEST.json')


def make_plan(target, board):
    jobs = []
    prior = json.loads((ROOT / 'prior_evidence/accepted_s3_c0.json').read_text())
    skip = target == 's3' and board == prior['board_id']
    for case in ([0, 1, 2] if target == 's3' else [1, 2]):
        candidates = []
        for family in ['kan', 'mlp']:
            if skip and case == 0 and family == 'mlp':
                continue
            variants = ([0, 4, 5, 8, 9, 10, 11] if target == 's3' else [0, 4, 5, 8, 9]) if family == 'kan' else [0, 1]
            for selected in ([[v] for v in variants] if target == 'mega' else [variants]):
                key = f'{family}_C{case}' + (f'_v{selected[0]}' if target == 'mega' else '')
                candidates.append(dict(key=key, family=family, case=case, variants=selected,
                                       repetitions=3, active_seconds=60, baseline_seconds=15,
                                       nominal_recording_seconds=288 * len(selected)))
        random.Random(20261110 + case).shuffle(candidates)
        jobs.extend(candidates)
    return jobs


def print_plan(target, board, jobs):
    print(f'BOARD {board} ({target}); {len(jobs)} native CFN recordings', flush=True)
    if target == 's3' and board == 'S3-A':
        print('SKIP accepted MLP S3-A C0 FP32 + INT8 energy. Previous latency series retained.')
    for i, job in enumerate(jobs, 1):
        print(f'{i:2d}. {job["key"]}: variants {job["variants"]}, '
              f'3 repeats each; nominal recording {job["nominal_recording_seconds"]/60:.1f} min')
    print(f'Nominal recording total: {sum(j["nominal_recording_seconds"] for j in jobs)/60:.1f} min.')
    print('Add flash, 1024-vector verification per variant, warm-up, operator time and batch completion.')
    print('Mega can finish the last 32-vector batch after 60 s; use actual measured elapsed time.')


def snapshot(folder):
    archive = Path(str(folder) + '.zip')
    tmp = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix != '.tmp':
                z.write(p, p.relative_to(folder.parent).as_posix())
    tmp.replace(archive)
    return archive


def audit_capture(job, folder):
    package = ROOT / 'packages' / job['family']
    output = folder / 'device_audit.json'
    subprocess.run([sys.executable, str(package / 'tools/audit_energy.py'),
                    '--capture', str(folder), '--out', str(output)], check=True)
    s = json.loads((folder / 'energy_summary.json').read_text())
    must(s['case'] == job['case'] and s['variants'] == job['variants'], 'Capture/job mismatch')
    must(s['seconds'] == 60 and s['baseline_seconds'] == 15 and s['repetitions'] == 3,
         'Capture protocol differs from board schedule')
    return s


def normalized(value):
    return re.sub('[^a-z0-9]', '', (value or '').lower())


def resolve_port(port, target, expected):
    from serial.tools.list_ports import comports
    ports = list(comports())
    if expected and expected.get('serial_number'):
        matches = [p for p in ports if normalized(p.serial_number) == normalized(expected['serial_number'])]
    else:
        matches = [p for p in ports if p.device.casefold() == port.casefold()]
    must(len(matches) == 1, 'Board COM port missing or ambiguous; inspect --list-ports')
    p = matches[0]
    must(p.vid is not None and p.pid is not None, 'Selected port is not USB')
    if expected:
        must((p.vid, p.pid) == (expected['vid'], expected['pid']), 'USB identity changed')
    return p.device, dict(vid=p.vid, pid=p.pid, serial_number=p.serial_number, port=p.device)


def meter_check(path, summary):
    _, _, meta = parse_cfn(path)
    must(meta['nominal_rate_hz'] == 100, 'Use the frozen 100 samples/s recording setting')
    must(meta['start_current_ma'] == 0 and meta['stop_current_ma'] == 0,
         'Start and stop current must both be 0 mA')
    must(meta['max_gap_s'] <= .020001, 'CFN contains missing time samples')
    windows = summary['windows']
    duration = meta['last_time_s'] - meta['first_time_s']
    required = windows[-1]['host_estimated_end_rel_s'] - windows[0]['host_estimated_start_rel_s']
    must(duration >= required + 5, 'CFN too short to include all windows and recording margins')
    meta['scope'] = 'File/channel/duration checks only; alignment and electrical energy review are pending'
    return meta


def verify_completed(folder, state):
    jobs = {j['key']: j for j in state['plan']}
    seen = set()
    for key, entry in state['completed'].items():
        must(key in jobs, 'Unknown completed job')
        for name, digest in entry['files'].items():
            must(sha(checked_path(folder, name)) == digest, 'Changed completed file: ' + name)
        capture = checked_path(folder, entry['capture'])
        s = audit_capture(jobs[key], capture)
        must(s['board_id'] == state['board_id'] and s['target'] == state['target'], 'Completed board mismatch')
        meter = checked_path(folder, entry['meter'])
        meter_check(meter, s)
        digest = sha(meter)
        must(digest not in seen, 'Same CFN attached to different jobs')
        seen.add(digest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['s3', 'c3', 'mega'])
    p.add_argument('--board-id')
    p.add_argument('--port')
    p.add_argument('--resume', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--list-ports', action='store_true')
    a = p.parse_args()
    if a.list_ports:
        from serial.tools.list_ports import comports
        for x in comports():
            print(x.device, x.description, x.serial_number)
        return
    release = release_check()
    if a.check_only:
        print('RELEASE_FILES_PASS; no board accessed'); return
    if a.resume:
        must(a.out is None, 'Use --resume without --out')
        folder = a.resume.resolve()
        state = json.loads((folder / 'board_state.json').read_text())
        must(state['release_sha256'] == release, 'Resume requires the exact same release')
        must(state['plan'] == make_plan(state['target'], state['board_id']), 'Stored plan changed')
        must(a.target in [None, state['target']] and a.board_id in [None, state['board_id']], 'Resume target changed')
        verify_completed(folder, state)
        a.target, a.board_id = state['target'], state['board_id']
        a.port = a.port or state['port']
    else:
        must(a.target and a.board_id, 'Supply --target and --board-id')
        must(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,47}', a.board_id), 'Invalid board ID')
        jobs = make_plan(a.target, a.board_id)
        print_plan(a.target, a.board_id, jobs)
        if a.plan_only: return
        must(a.port, 'Supply --port')
        folder = (a.out or ROOT / 'captures' / f'{a.board_id}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
        must(not folder.exists() and not Path(str(folder) + '.zip').exists(), 'Output already exists; use --resume')
        folder.mkdir(parents=True)
        state = dict(version='all-boards-energy-v1', release_sha256=release,
                     target=a.target, board_id=a.board_id, port=a.port, plan=jobs,
                     completed={}, attempts=[], created_utc=utc(), status='IN_PROGRESS')
        if a.target == 's3' and a.board_id == 'S3-A':
            prior = json.loads((ROOT / 'prior_evidence/accepted_s3_c0.json').read_text())
            state['previously_accepted'] = prior
            state['usb_identity'] = prior['usb_identity']
    if a.plan_only:
        print_plan(a.target, a.board_id, state['plan']); return
    known = json.loads((ROOT / 'prior_evidence/board_identities.json').read_text())
    if a.board_id in known:
        must(known[a.board_id]['target'] == a.target, 'Known board ID belongs to another target')
        state.setdefault('usb_identity', known[a.board_id]['usb_identity'])
    save(folder / 'board_state.json', state)
    print('CHECKPOINT:', folder, flush=True)
    print('Single 5 V board supply THROUGH FNB58. No additional USB/power source to the board.', flush=True)
    try:
        for job in state['plan']:
            key = job['key']
            if key in state['completed']:
                print('ALREADY CAPTURED:', key, flush=True); continue
            # A valid count capture survives stopping before the CFN was attached.
            capture = None; s = None
            for attempt in reversed(state['attempts']):
                if attempt['key'] != key: continue
                candidate = checked_path(folder, attempt['capture'])
                try:
                    summary = json.loads((candidate / 'energy_summary.json').read_text())
                    if summary['status'] == 'DEVICE_COUNTS_CAPTURED':
                        s = audit_capture(job, candidate); capture = candidate; break
                except FileNotFoundError: pass
            if capture is None:
                answer = input(f'Next {key}. Stop/save any previous meter recording. Enter = prepare; q = pause: ').strip().lower()
                if answer == 'q': raise KeyboardInterrupt
                state['port'], identity = resolve_port(a.port, a.target, state.get('usb_identity'))
                state.setdefault('usb_identity', identity)
                a.port = state['port']
                capture = folder / 'device' / f'{key}_attempt{1+sum(x["key"]==key for x in state["attempts"]):02d}'
                state['attempts'].append(dict(key=key, capture=capture.relative_to(folder).as_posix(), start_utc=utc()))
                save(folder / 'board_state.json', state)
                cmd = [sys.executable, str(ROOT / 'packages' / job['family'] / 'tools/run_energy.py'),
                       '--target', a.target, '--case', str(job['case']), '--port', a.port,
                       '--board-id', a.board_id, '--seconds', '60', '--repetitions', '3', '--out', str(capture)]
                if a.target == 'mega': cmd += ['--variant', str(job['variants'][0])]
                print('Start a NEW CFN only when the child runner prints READY; wait 5 s before Enter.', flush=True)
                subprocess.run(cmd, check=True)
                s = audit_capture(job, capture)
            must(s['board_id'] == a.board_id and s['target'] == a.target, 'Capture board identity mismatch')
            print('Wait at least 5 seconds after the last window; STOP and SAVE the native .cfn.', flush=True)
            while True:
                raw = input(f'Full path to CFN for {key} (q = pause, or r = repeat this job if recording was lost): ').strip().strip('"')
                if raw.lower() == 'q': raise KeyboardInterrupt
                if raw.lower() == 'r':
                    # Keep the old acquisition, explicitly mark it for exclusion on resume.
                    state['discarded_captures'] = state.get('discarded_captures', []) + [capture.relative_to(folder).as_posix()]
                    state['attempts'] = [x for x in state['attempts'] if x['capture'] != capture.relative_to(folder).as_posix()]
                    save(folder / 'board_state.json', state)
                    print('Old files retained. Resume this checkpoint to repeat the job.'); raise KeyboardInterrupt
                native = Path(raw)
                try:
                    must(native.is_file() and native.suffix.lower() == '.cfn', 'Provide an existing native .cfn file')
                    meta = meter_check(native, s)
                    must(sha(native) not in [x['meter_sha256'] for x in state['completed'].values()], 'This CFN was already used')
                    break
                except (ValueError, AssertionError, OSError) as e:
                    print('CFN not accepted:', e, flush=True)
            dest = folder / 'meter' / f'{key}.cfn'; dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(native, dest)
            must(sha(dest) == meta['source_sha256'], 'CFN changed while copying')
            save(dest.with_suffix('.metadata.json'), dict(**meta, original_filename=native.name))
            paths = [x for x in capture.rglob('*') if x.is_file()] + [dest, dest.with_suffix('.metadata.json')]
            state['completed'][key] = dict(capture=capture.relative_to(folder).as_posix(), meter=dest.relative_to(folder).as_posix(),
                                          meter_sha256=sha(dest), status='DEVICE_COUNTS_AUDITED_METER_ATTACHED',
                                          files={x.relative_to(folder).as_posix(): sha(x) for x in paths})
            save(folder / 'board_state.json', state)
            print('SAVED CHECKPOINT ZIP:', snapshot(folder), flush=True)
        state['status'] = 'ALL_DEVICE_COUNTS_AUDITED_METERS_ATTACHED'
        print('Board acquisition complete. Send the checkpoint ZIP for alignment and energy analysis.', flush=True)
    except KeyboardInterrupt:
        state['status'] = 'PAUSED'
        print('Paused. Completed jobs retained; use --resume with the CHECKPOINT directory.', flush=True)
    except BaseException as e:
        state['status'] = 'STOPPED_WITH_ERROR'; state['last_error'] = str(e)
        raise
    finally:
        state['updated_utc'] = utc()
        save(folder / 'board_state.json', state)
        print('SEND THIS ZIP:', snapshot(folder), flush=True)


if __name__ == '__main__':
    main()
