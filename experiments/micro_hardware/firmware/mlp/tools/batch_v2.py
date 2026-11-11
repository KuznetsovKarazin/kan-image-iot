"""Resume only independently verified complete captures; preserve every attempt."""
import argparse
import datetime
import json
from pathlib import Path
import random
import re
import sys
import time
import zipfile
from audit_v2 import audit_capture, selected_capture, release_check
from device_support import KeepAwake, normalized_serial


def make_plan(target,cases,repetitions):
    result={}
    for case in cases:
        for rep in range(1,repetitions+1):
            variants=[None]
            if target=='mega':
                variants=[0,1];random.Random(20261109+100*case+rep).shuffle(variants)
            for variant in variants:
                key=f'case{case}_rep{rep}'+(f'_v{variant}' if variant is not None else '')
                result[key]=dict(case=case,repetition=rep,variant=variant)
    return result


def snapshot(folder):
    archive=Path(str(folder)+'.zip');temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(folder.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(folder.parent).as_posix())
    temporary.replace(archive)
    return archive


def checked_completed(folder,state,root):
    """Run before any device access; a changed completed capture blocks resume."""
    if list(state['plan'].items())!=list(make_plan(state['target'],state['cases'],state['repetitions']).items()):
        raise ValueError('Stored schedule differs from the frozen protocol')
    for key,entry in state['completed'].items():
        if key not in state['plan'] or entry['step']!=state['plan'][key] or entry['board_id']!=state['board_id']:
            raise ValueError('Stored completed slot identity differs')
        result=selected_capture(folder,entry,root)
        if result['report'].get('target')!=state['target']:
            raise ValueError('Stored completed capture targets another architecture')


def run_cli(runner,argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=['s3','c3','mega'])
    parser.add_argument('--port');parser.add_argument('--board-id')
    parser.add_argument('--cases',nargs='+',type=int,choices=[0,1,2])
    parser.add_argument('--repetitions',type=int,choices=[1,3])
    parser.add_argument('--out',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv);root=runner.ROOT
    release_hash=release_check(root)
    if args.resume:
        if args.out:parser.error('Use --resume without --out')
        folder=args.resume.resolve();state=json.loads((folder/'batch.json').read_text(encoding='utf-8'))
        runner.must(state['version']==runner.VERSION and state['release_sha256']==release_hash,'Resume requires the identical package')
        for supplied,stored,label in [(args.target,state['target'],'target'),(args.board_id,state['board_id'],'board ID'),
                                       (args.cases,state['cases'],'cases'),(args.repetitions,state['repetitions'],'repetitions')]:
            runner.must(supplied is None or supplied==stored,'Resume '+label+' differs')
        checked_completed(folder,state,root)
        args.target=state['target'];args.board_id=state['board_id'];args.cases=state['cases'];args.repetitions=state['repetitions']
        if set(state['completed'])==set(state['plan']):
            if state['status']!='PASS' and not args.check_only:
                state['history'].append(dict(event='complete_after_reaudit',utc=runner.utc(),previous_status=state['status']))
                state.update(status='PASS',end_utc=runner.utc());runner.save(folder/'batch.json',state);snapshot(folder)
            print('ALREADY_COMPLETE: all captures passed independent re-audit; hardware not accessed')
            return
    else:
        runner.must(args.target is not None,'Supply --target')
        args.cases=args.cases or ([0,1,2] if args.target=='s3' else [1,2])
        args.repetitions=args.repetitions or 3
        runner.must(len(set(args.cases))==len(args.cases),'Duplicate cases')
        folder=None;state=None
    runner.preflight(args.target)
    if args.target=='mega':
        images=json.loads((root/'mega/firmware/manifest.json').read_text())['images']
        runner.must(all(f'case{c}_v{v}' in images for c in args.cases for v in runner.VARIANTS),'Selected Mega case is not included; use cases 1 2')
    else:
        images=json.loads((root/'esp32/firmware/manifest.json').read_text())['images']
        runner.must(all(f'{args.target}_case{c}' in images for c in args.cases),'Selected case is not included')
    if args.check_only:
        print('FILES_PASS; hardware not accessed; target:',args.target,'cases:',args.cases,'repetitions:',args.repetitions)
        return
    runner.must(args.port and args.board_id,'Supply --port and --board-id (board ID is inherited when resuming)')
    runner.must(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}',args.board_id),'Invalid board ID')
    if state is None:
        folder=(args.out or root/'captures'/f'{args.board_id}_mlp_v2_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
        runner.must(not folder.exists() and not Path(str(folder)+'.zip').exists(),'Output already exists; use --resume for that batch or choose a new path')
        folder.mkdir(parents=True)
        state=dict(version=runner.VERSION,status='IN_PROGRESS',target=args.target,board_id=args.board_id,
                   cases=args.cases,repetitions=args.repetitions,release_sha256=release_hash,
                   start_utc=runner.utc(),power_cycle=False,plan=make_plan(args.target,args.cases,args.repetitions),
                   completed={},attempts=[],history=[])
    else:
        for entry in state['attempts']:
            if entry['status']=='IN_PROGRESS':entry.update(status='ABANDONED',reason='Previous process did not finalize this attempt')
        state['history'].append(dict(event='resume',utc=runner.utc(),previous_status=state['status']))
        state['status']='IN_PROGRESS'
    runner.save(folder/'batch.json',state)
    print('BATCH_DIR='+str(folder),flush=True)
    current=None
    try:
        with KeepAwake() as power:
            state['sleep_policy']=power.status
            if args.target in ['s3','c3']:
                tracker=runner.PortTracker(args.port);identity=tracker.identity
                if state.get('usb_identity'):
                    old=state['usb_identity']
                    runner.must((old['vid'],old['pid'])==(identity['vid'],identity['pid']) and
                                normalized_serial(old['serial_number'])==normalized_serial(identity['serial_number']),
                                'USB identity differs from the batch; refusing resume on another board')
                else:state['usb_identity']=identity
            else:
                identity=runner.bridge_identity(args.port)
                if state.get('usb_identity'):runner.check_bridge(state['usb_identity'],identity)
                else:state['usb_identity']=identity
            runner.save(folder/'batch.json',state)
            active_case=None;flash_log=None
            for key,step in state['plan'].items():
                if key in state['completed']:
                    print('SKIP verified:',key,flush=True);continue
                case,rep,v=step['case'],step['repetition'],step['variant']
                count=sum(a['slot']==key for a in state['attempts'])+1
                attempt=folder/'runs'/key/f'attempt_{count:03d}'
                attempt.mkdir(parents=True,exist_ok=False)
                current=dict(slot=key,step=step,board_id=args.board_id,directory=attempt.relative_to(folder).as_posix(),
                             status='IN_PROGRESS',start_utc=runner.utc(),usb_identity=identity)
                state['attempts'].append(current);runner.save(folder/'batch.json',state)
                evidence=[]
                print('START',key,'attempt',count,flush=True)
                if args.target in ['s3','c3']:
                    port=tracker.resolve();expected=identity['serial_number'].lower()
                    if active_case!=case:
                        before=attempt/'before_flash_reset.log'
                        mac=runner.reset_esp(args.target,port,before,state.get('mac'))
                        runner.must(normalized_serial(mac)==normalized_serial(expected),'ROM MAC differs from selected USB identity')
                        state['mac']=mac;time.sleep(1);port=tracker.resolve()
                        flash_log=attempt/'flash.log'
                        output=runner.execute([sys.executable,root/'esp32/tools/flash.py','--target',args.target,'--case',str(case),'--port',port],flash_log)
                        runner.must('Hash of data verified.' in output and mac in output.lower(),'Missing flash verification/MAC evidence')
                        active_case=case;evidence.append(before)
                    evidence.append(flash_log)
                    reset_log=attempt/'reset.log';port=tracker.resolve()
                    mac=runner.reset_esp(args.target,port,reset_log,state['mac']);evidence.append(reset_log)
                    time.sleep(1);port=tracker.resolve()
                    runner.capture_esp(args.target,port,args.board_id,attempt/'capture',rep,case)
                    current.update(flash_log=flash_log.relative_to(folder).as_posix(),reset_log=reset_log.relative_to(folder).as_posix(),mac=mac)
                else:
                    runner.check_bridge(identity,runner.bridge_identity(args.port))
                    avrdude_log=attempt/'avrdude.log';flash_log=attempt/'flash.log'
                    runner.execute([sys.executable,root/'mega/tools/mega_flash.py','--case',str(case),'--variant',str(v),
                                    '--port',args.port,'--log',avrdude_log],flash_log)
                    runner.execute([sys.executable,root/'mega/tools/mega_runner.py','--case',str(case),'--variant',str(v),
                                    '--port',args.port,'--board-id',args.board_id,'--out',attempt/'capture'],
                                   attempt/'capture_console.log',timeout=3600)
                    runner.check_bridge(identity,runner.bridge_identity(args.port))
                    evidence.extend([flash_log,avrdude_log]);current['avrdude_log']=avrdude_log.relative_to(folder).as_posix()
                result=audit_capture(attempt/'capture',root)
                current.update(file_sha256=result['file_sha256'],
                               evidence_sha256={p.relative_to(folder).as_posix():runner.sha(p) for p in evidence})
                selected_capture(folder,current,root)
                runner.save(attempt/'audit.json',result)
                current.update(status='PASS',end_utc=runner.utc());state['completed'][key]=dict(current)
                runner.save(folder/'batch.json',state)
                print('ACCEPTED',key,f'({len(state["completed"])}/{len(state["plan"])})',flush=True)
            state.update(status='PASS',end_utc=runner.utc())
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAIL',error=str(error),end_utc=runner.utc())
        if current and current['status']=='IN_PROGRESS':current.update(status=state['status'],error=str(error),end_utc=runner.utc())
        state['history'].append(dict(event='stopped',utc=runner.utc(),reason=str(error)))
        raise
    finally:
        runner.save(folder/'batch.json',state)
        print('Saved batch archive:',snapshot(folder),flush=True)
    print('BATCH_PASS:',folder,flush=True)
