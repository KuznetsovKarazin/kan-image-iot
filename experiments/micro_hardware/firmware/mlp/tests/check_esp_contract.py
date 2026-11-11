"""Synthetic protocol and failure-injection checks. No board timings are produced."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import zipfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_mlp as runner
import batch_v2
from audit_v2 import audit_capture
from device_support import KeepAwake


from clock_support import install_logger
from audit_v2_esp import session_audit

def audit_capture(folder,root):
    raw={p.name:p.read_bytes() for p in folder.iterdir()}
    summary=json.loads(raw['summary.json']);g=json.loads((root/f'esp32/generated/model{summary["case"]}_golden.json').read_text())
    report=session_audit(raw,'',dict(target=summary['target'],board_id=summary['board_id']),lambda n:(root/n).read_bytes(),g,summary['session'],require_physical=False)
    assert report['physical_execution'] is False
    try:session_audit(raw,'',dict(target=summary['target'],board_id=summary['board_id']),lambda n:(root/n).read_bytes(),g,summary['session'])
    except ValueError as e:assert 'physical_execution' in str(e)
    else:raise AssertionError('Synthetic capture accepted as physical')
    return dict(status='PASS',report=report)

class SyntheticLink:
    transport='synthetic_test_fixture'
    """Explicit test fixture: identities resemble a device to exercise the auditor."""
    def __init__(self,target,golden,trace):
        self.target=target;self.g=golden;self.trace=trace;self.command=None
        self.variants=[0,1]
        self.emit('port_open',port='UNIT-TEST',dtr=False,rts=False,control_lines_configured_before_open=True)
    def emit(self,event,**fields):self.log(event,**fields)
    def send(self,command):
        assert self.command is None;self.command=command;self.emit('tx',command=command)
    def close(self):pass
    def receive(self,**_):
        g=self.g;parts=self.command.split();command=parts[0]
        if command=='H':
            s3=self.target=='s3'
            response=dict(kind='hello',protocol=4,model=g['model'],artifact=g['artifact'],input_dim=g['input_dim'],variants=2,
                          chip='ESP32-S3' if s3 else 'ESP32-C3',revision=0,cores=2 if s3 else 1,cpu_mhz=240 if s3 else 160,
                          flash_bytes=4194304,psram_bytes=0,heap_free=100000,heap_min=100000,stack_watermark_bytes=6000,
                          idf='UNIT-TEST-FIXTURE',radios_started=False,serial_transport='esp-idf-usb-serial-jtag-v4',
                          rx_buffer_bytes=2048,tx_buffer_bytes=2048,task_core=1 if s3 else 0,task_affinity=1 if s3 else 0,
                          dram_copy_ok=False,dram_copy_bytes=g['dram_total_bytes'] if s3 else 0,
                          dram_fp32_bytes=g['dram_fp32_bytes'] if s3 else 0,dram_int8_bytes=g['dram_int8_bytes'] if s3 else 0,
                          dram_allocation='internal_8bit')
        elif command=='P':response=dict(kind='probe',seq=int(parts[1]),payload=parts[2])
        elif command=='V':
            seq,v=int(parts[1]),int(parts[2]);record=g['records'][(seq-1)%1024]
            assert parts[3:]==record['input_bits']
            response=dict(kind='pred',seq=seq,variant=v,bits=record['expected_bits'][v])
        elif command=='L':response=dict(kind='loaded',slot=int(parts[1]))
        elif command=='B':
            seq,v,rounds=map(int,parts[1:]);last=g['records'][g['benchmark_indices'][-1]]
            sink=last['input_bits'][0] if v==2 else last['expected_bits'][v][0]
            elapsed=rounds*(1 if v==2 else 10000);core=1 if self.target=='s3' else 0
            response=dict(kind='bench',seq=seq,variant=v,rounds=rounds,n=32*rounds,elapsed_us=elapsed,
                          elapsed_cycles=elapsed*(240 if core else 160),complete=True,sink=sink,
                          core_start=core,core_end=core,heap_free=100000,heap_min=100000,stack_watermark_bytes=6000)
        else:raise AssertionError(command)
        raw=(json.dumps(response)+'\n').encode('ascii')
        for start in range(0,len(raw),37):
            chunk=raw[start:start+37];self.emit('rx_chunk',raw_hex=chunk.hex(),n_bytes=len(chunk))
        self.emit('response',response=response);self.command=None
        return response


install_logger(SyntheticLink)

def protocol_audits():
    accepted=[];rejected=[]
    with tempfile.TemporaryDirectory() as tmp:
        for target,case in [(t,c) for t in ['s3','c3'] for c in range(3)]:
            g=json.loads((ROOT/f'esp32/generated/model{case}_golden.json').read_text())
            folder=Path(tmp)/f'{target}{case}'
            with patch.object(runner,'SerialLink',side_effect=lambda port,trace:SyntheticLink(target,g,trace)),contextlib.redirect_stdout(io.StringIO()):
                runner.capture_esp(target,'UNIT-TEST','UNIT_TEST_FIXTURE',folder,2,case)
            result=audit_capture(folder,ROOT);assert result['status']=='PASS'
            accepted.append(dict(target=target,case=case,raw_frames=result['report']['transport']['response_count']))
            if target=='s3' and case==2:
                path=folder/'transport.jsonl';original=path.read_bytes();events=[json.loads(x) for x in original.splitlines()]
                events[10]['utc']='2026-01-01T00:00:00+00:00';path.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
                assert audit_capture(folder,ROOT)['report']['host_clock']['utc_order_warning'] is True
                events[10]['monotonic_ns']=0;path.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
                try:audit_capture(folder,ROOT)
                except ValueError:rejected.append('backwards_monotonic_clock')
                else:raise AssertionError('Backwards monotonic time accepted')
                path.write_bytes(original)
                # Reject incomplete raw transport even if the saved summary still says PASS.
                path=folder/'transport.jsonl';original=path.read_bytes()
                path.write_bytes(original.rsplit(b'\n',2)[0]+b'\n')
                try:audit_capture(folder,ROOT)
                except (ValueError,KeyError):rejected.append('truncated_raw_trace')
                else:raise AssertionError('Truncated trace accepted')
                path.write_bytes(original)
                path=folder/'latency_blocks.json';original=path.read_bytes();blocks=json.loads(original)
                blocks[0]['elapsed_cycles']+=1;path.write_text(json.dumps(blocks))
                try:audit_capture(folder,ROOT)
                except ValueError:rejected.append('saved_timing_changed')
                else:raise AssertionError('Changed timing accepted')
                path.write_bytes(original)
                path=folder/'summary.json';original=path.read_bytes();summary=json.loads(original)
                summary['firmware_sha256']='0'*64;path.write_text(json.dumps(summary))
                try:audit_capture(folder,ROOT)
                except ValueError:rejected.append('wrong_firmware')
                else:raise AssertionError('Wrong firmware accepted')
                path.write_bytes(original)
    return dict(accepted_test_fixtures=accepted,rejected_mutations=rejected,physical_execution=False)


def resume_checks():
    calls=[];fail_once=[True];serial_number=['aa:bb:cc:dd:ee:ff']
    class Tracker:
        def __init__(self,_):self.identity=dict(vid=0x303a,pid=0x1001,serial_number=serial_number[0],initial_port='UNIT-TEST')
        def resolve(self):return 'UNIT-TEST'
    def check(ok,why):
        if not ok:raise ValueError(why)
    def fake_audit(folder,root):
        names=['summary.json','predictions.jsonl','latency_blocks.json','transport.jsonl']
        return dict(status='PASS',file_sha256={n:runner.sha(folder/n) for n in names},report={'target':'s3'})
    def fake_selected(folder,entry,root):
        result=fake_audit(folder/entry['directory']/'capture',root)
        check(result['file_sha256']==entry['file_sha256'],'Accepted file changed')
        return result
    def reset(target,port,log,expected=None):
        calls.append('reset');log.write_text('Hard resetting via RTS pin...\nMAC: aa:bb:cc:dd:ee:ff\n')
        return 'aa:bb:cc:dd:ee:ff'
    def execute(argv,log,**_):
        calls.append('flash');text='Hash of data verified.\nMAC: aa:bb:cc:dd:ee:ff\n';log.write_text(text);return text
    def capture(target,port,board,folder,rep,case):
        calls.append(('capture',rep));folder.mkdir()
        if rep==2 and fail_once[0]:
            fail_once[0]=False;(folder/'partial.txt').write_text('INTERRUPTED_TEST_ATTEMPT');raise TimeoutError('injected timeout')
        for name in ['summary.json','predictions.jsonl','latency_blocks.json','transport.jsonl']:
            (folder/name).write_text(json.dumps(dict(test_fixture=True,rep=rep,name=name)))
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'esp32/firmware').mkdir(parents=True)
        (root/'esp32/firmware/manifest.json').write_text(json.dumps({'images':{'s3_case0':{}}}))
        fake=types.SimpleNamespace(ROOT=root,VERSION='mlp-v2',VARIANTS=[0,1],utc=runner.utc,
             save=runner.save,sha=runner.sha,must=check,preflight=lambda _:None,PortTracker=Tracker,
             reset_esp=reset,execute=execute,capture_esp=capture)
        folder=root/'batch'
        with patch.object(batch_v2,'release_check',return_value='UNIT_TEST_RELEASE'),patch.object(batch_v2,'audit_capture',side_effect=fake_audit),\
             patch.object(batch_v2,'selected_capture',side_effect=fake_selected),patch.object(batch_v2.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            try:batch_v2.run_cli(fake,['--target','s3','--port','UNIT-TEST','--board-id','UNIT','--cases','0','--repetitions','3','--out',str(folder)])
            except TimeoutError:pass
            else:raise AssertionError('Injected failure did not stop batch')
            state=json.loads((folder/'batch.json').read_text());assert state['status']=='FAIL' and len(state['completed'])==1
            partial=folder/state['attempts'][-1]['directory']/'capture/partial.txt';partial_bytes=partial.read_bytes()
            accepted=folder/state['completed']['case0_rep1']['directory']/'capture/predictions.jsonl'
            original=accepted.read_bytes();accepted.write_bytes(b'TAMPERED');before=len(calls)
            try:batch_v2.run_cli(fake,['--resume',str(folder),'--port','UNIT-TEST'])
            except ValueError:pass
            else:raise AssertionError('Changed completed data accepted')
            assert len(calls)==before;accepted.write_bytes(original)
            serial_number[0]='00:00:00:00:00:00'
            try:batch_v2.run_cli(fake,['--resume',str(folder),'--port','UNIT-TEST'])
            except ValueError:pass
            else:raise AssertionError('Different USB board accepted')
            assert len(calls)==before;serial_number[0]='aa:bb:cc:dd:ee:ff'
            batch_v2.run_cli(fake,['--resume',str(folder),'--port','UNIT-TEST'])
            state=json.loads((folder/'batch.json').read_text())
            assert state['status']=='PASS' and len(state['completed'])==3 and len(state['attempts'])==4
            assert [x for x in calls if isinstance(x,tuple)]==[('capture',1),('capture',2),('capture',2),('capture',3)]
            assert partial.read_bytes()==partial_bytes
            before=len(calls);batch_v2.run_cli(fake,['--resume',str(folder),'--port','UNIT-TEST']);assert len(calls)==before
            with zipfile.ZipFile(str(folder)+'.zip') as z:assert z.testzip() is None and any(n.endswith('partial.txt') for n in z.namelist())
    flags=[]
    def api(value):flags.append(value);return 0x80000000
    try:
        with KeepAwake(system='win32',function=api):raise RuntimeError('injected')
    except RuntimeError:pass
    assert flags==[0x80000001,0x80000000]
    try:
        with KeepAwake(system='win32',function=lambda _:0):raise AssertionError('Unavailable API did not fail closed')
    except RuntimeError:pass
    return dict(status='PASS',completed_slots_skipped=True,failed_attempt_preserved=True,changed_capture_blocks_resume=True,
                wrong_usb_identity_blocks_flashing=True,complete_resume_accesses_no_hardware=True,
                windows_api_flags_and_cleanup_mocked=True,physical_execution=False)


if __name__=='__main__':
    result=dict(status='PASS_SYNTHETIC_CONTRACT_TESTS',physical_execution=False,
                protocol=protocol_audits(),resume=resume_checks())
    (ROOT/'evidence/contract_tests.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
