"""Offline raw-frame audit for mlp-v2 captures and resumed batches."""
import argparse
from clock_support import inspect_clock, VERSION
import json
import random
from pathlib import Path
from audit_v2_esp import session_audit, require, sha
from audit_v2_mega import audit_one

ROOT=Path(__file__).resolve().parents[1]
FILES=['summary.json','predictions.jsonl','latency_blocks.json','transport.jsonl']


def release_check(root=ROOT):
    raw=(root/'MANIFEST.json').read_bytes();manifest=json.loads(raw)
    require(manifest['version']=='mlp-v2','Wrong package version')
    for name,digest in manifest['sha256'].items():
        require(sha((root/name).read_bytes())==digest,'Changed package file: '+name)
    return sha(raw)


def audit_capture(folder,root=ROOT):
    raw={name:(folder/name).read_bytes() for name in FILES}
    summary=json.loads(raw['summary.json'])
    require(summary['status']=='PASS','Incomplete capture')
    if summary.get('target') in ['s3','c3']:
        golden=json.loads((root/f'esp32/generated/model{summary["case"]}_golden.json').read_bytes())
        report=session_audit(raw,'',dict(target=summary['target'],board_id=summary['board_id']),
                             lambda name:(root/name).read_bytes(),golden,summary['session'])
    else:
        class Package:
            def __getitem__(self,name):return (root/name).read_bytes()
        report=audit_one(raw,Package())
        report['target']='mega'
    return dict(status='PASS',audit_version=VERSION,file_sha256={name:sha(data) for name,data in raw.items()},report=report)


def selected_capture(batch_dir,entry,root=ROOT):
    relative=Path(entry['directory'])
    require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe attempt path')
    result=audit_capture(batch_dir/relative/'capture',root)
    require(result['file_sha256']==entry['file_sha256'],'Previously accepted capture changed')
    report=result['report'];step=entry['step']
    require(report['case']==step['case'],'Case/slot mismatch')
    if report['target']=='mega':require(report['variant']==step['variant'],'Variant/slot mismatch')
    else:require(report['session']==step['repetition'],'Repetition/slot mismatch')
    require(report['board_id']==entry['board_id'],'Board/slot mismatch')
    for name,digest in entry['evidence_sha256'].items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'Unsafe evidence path')
        require(sha((batch_dir/path).read_bytes())==digest,'Flash/reset evidence changed')
    required_logs=['avrdude_log'] if report['target']=='mega' else ['flash_log','reset_log']
    require(all(entry[name] in entry['evidence_sha256'] for name in required_logs),'Unhashed flash/reset evidence')
    if report['target']=='mega':
        log=(batch_dir/entry['avrdude_log']).read_text(encoding='utf-8',errors='replace')
        require('Device signature = 0x1e9801' in log and
                f"{report['flash_bytes']} bytes of flash verified" in log and
                f"mega_case{step['case']}_v{step['variant']}.hex" in log,'Missing AVR flash verification')
    else:
        log=(batch_dir/entry['flash_log']).read_text(encoding='utf-8',errors='replace')
        reset=(batch_dir/entry['reset_log']).read_text(encoding='utf-8',errors='replace')
        mac=entry['mac'].lower()
        require('Hash of data verified.' in log and mac in log.lower(),'Missing ESP flash verification')
        require('Hard resetting via RTS pin' in reset and mac in reset.lower(),'Missing ESP reset identity')
    return result


def audit_batch(folder,root=ROOT):
    release_hash=release_check(root)
    state=json.loads((folder/'batch.json').read_text(encoding='utf-8'))
    require(state['version']=='mlp-v2' and state['release_sha256']==release_hash,'Batch/package mismatch')
    target=state['target'];cases=state['cases'];repetitions=state['repetitions']
    require(target in ['s3','c3','mega'] and repetitions in [1,3],'Invalid batch parameters')
    require(cases and len(set(cases))==len(cases) and
            all(c in [0,1,2] for c in cases),'Invalid batch cases')
    expected_plan={}
    for case in cases:
        for repetition in range(1,repetitions+1):
            variants=[None]
            if target=='mega':
                variants=[0,1];random.Random(20261109+100*case+repetition).shuffle(variants)
            for variant in variants:
                key=f'case{case}_rep{repetition}'+(f'_v{variant}' if variant is not None else '')
                expected_plan[key]=dict(case=case,repetition=repetition,variant=variant)
    require(list(state['plan'].items())==list(expected_plan.items()),'Batch schedule differs from frozen protocol')
    reports=[]
    for key,entry in state['completed'].items():
        require(entry['step']==state['plan'][key],'Unexpected accepted step')
        require(entry['board_id']==state['board_id'],'Batch board identity')
        result=selected_capture(folder,entry,root)
        require(result['report']['target']==state['target'],'Batch target identity')
        reports.append(dict(slot=key,**result))
    complete=set(state['completed'])==set(state['plan'])
    if state['status']=='PASS':require(complete,'PASS batch is incomplete')
    return dict(status='PASS' if complete else 'VALIDATED_PARTIAL',audit_version=VERSION,physical_execution=True,
                accepted_steps=len(reports),planned_steps=len(state['plan']),captures=reports,
                incomplete_attempts_retained=sum(a['status']!='PASS' for a in state['attempts']),
                scope='Selected complete captures only; incomplete attempts remain in the archive and are excluded.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();report=audit_batch(args.batch)
    if args.out:args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],report['accepted_steps'],'/',report['planned_steps'],'complete captures')
