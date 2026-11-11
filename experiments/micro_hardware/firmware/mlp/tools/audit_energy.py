"""Audit device counts against raw UART/USB frames; this is not an energy measurement."""
import argparse,json
from pathlib import Path
from clock_support import inspect_clock
from audit_v2 import release_check
from serial_common import prediction_check,decode
from run_energy import validate_window,plan
import run_mlp as runner
ROOT=runner.ROOT

def audit(folder,require_physical=True):
    s=json.loads((folder/'energy_summary.json').read_text())
    runner.must(s['status']=='DEVICE_COUNTS_CAPTURED','Incomplete energy-count capture')
    runner.must(s['release_sha256']==release_check(ROOT),'Energy package identity')
    runner.must(s['physical_execution'] is require_physical,'Synthetic capture is not physical evidence')
    runner.must(s['transport']==('physical_serial' if require_physical else 'synthetic_test_fixture'),'Transport identity')
    runner.must(s['runner_sha256']==runner.sha(ROOT/'tools/run_energy.py'),'Energy runner SHA')
    for name,digest in s['file_sha256'].items():
        runner.must(Path(name).name==name,'Unsafe capture path')
        runner.must(runner.sha(folder/name)==digest,'Changed capture file: '+name)
    target=s['target'];case=s['case'];variants=s['variants']
    runner.must(target in ['mega','s3','c3'] and variants in [[0],[1],[0,1]],'Capture target/variants')
    runner.must(s['repetitions'] in [1,3] and 10<=s['seconds']<=60 and 10<=s['baseline_seconds']<=60,'Energy protocol parameters')
    runner.must(s['plan']==plan(variants,s['repetitions'],s['seconds'],s['baseline_seconds'],case),'Frozen energy schedule')
    gp=ROOT/(f'mega/reference/model{case}_golden.json' if target=='mega' else f'esp32/generated/model{case}_golden.json')
    g=json.loads(gp.read_text());runner.must(runner.sha(gp)==s['golden_sha256'] and g['artifact']==s['artifact'],'Golden identity')
    trace=[json.loads(line) for line in (folder/'transport.jsonl').read_text().splitlines()]
    clock=inspect_clock(trace,s);buffer=bytearray();command=None;remaining=0;exchanges=[];responses=[];times=[];opened=0
    for e in trace:
        kind=e['event']
        if kind=='port_open':
            opened+=1;runner.must(opened==1 and not exchanges,'Unexpected serial opening')
        elif kind=='dtr_reset':runner.must(target=='mega' and not exchanges,'Unexpected reset')
        elif kind=='tx':
            runner.must(command is None and not buffer,'Overlapping requests/leftover bytes')
            command=e['command'];remaining=2 if command.startswith('E ') else 1;responses=[];times=[]
        elif kind=='rx_chunk':
            raw=bytes.fromhex(e['raw_hex']);runner.must(command is not None and len(raw)==e['n_bytes'],'Raw RX mismatch')
            buffer.extend(raw)
        elif kind in ['response','startup_text']:
            runner.must(command is not None and b'\n' in buffer,'No raw response frame')
            end=buffer.index(b'\n')+1;raw=bytes(buffer[:end]);del buffer[:end]
            if kind=='startup_text':
                runner.must(command=='H' and raw.hex()==e['raw_hex'] and not raw.lstrip().startswith(b'{'),'Bad startup frame')
                continue
            r=json.loads(raw);runner.must(r==e['response'],'Parsed response differs from raw bytes')
            responses.append(r);times.append(e['monotonic_ns']);remaining-=1
            if not remaining:exchanges.append((command,responses,times));command=None
        else:raise ValueError('Unexpected trace event: '+kind)
    runner.must(opened==1 and command is None and not buffer,'Incomplete trace')
    index=0
    def take(expected):
        nonlocal index
        runner.must(index<len(exchanges),'Missing exchange');c,rs,ts=exchanges[index];index+=1
        runner.must(c==expected,'Unexpected command: '+c);return rs,ts
    hello=take('H')[0][0];runner.must(hello==s['device'] and hello.get('energy_protocol')==1,'Energy capability/hello')
    expected=dict(model=g['model'],input_dim=g['input_dim'],cpu_mhz={'s3':240,'c3':160,'mega':16}[target])
    if target=='mega':
        runner.must(len(variants)==1,'Multiple formats in one Mega image')
        image=json.loads((ROOT/'mega/firmware/manifest.json').read_text())['images'][f'case{case}_v{variants[0]}']
        expected.update(artifact=image['artifact'],variant=variants[0],protocol=2,chip='ATmega2560')
        runner.must(image['hex_sha256']==s['firmware_sha256'],'Mega firmware SHA')
    else:
        expected.update(artifact=g['artifact'],protocol=4,variants=2,task_core=1 if target=='s3' else 0,radios_started=False,psram_bytes=0)
        image=json.loads((ROOT/'esp32/firmware/manifest.json').read_text())['images'][f'{target}_case{case}']
        runner.must(image['sha256']==s['firmware_sha256'],'ESP firmware SHA')
    for k,v in expected.items():runner.must(hello.get(k)==v,'Hello identity: '+k)
    for i,length in enumerate([1,63,64,65,127,255,384]*3,1):
        seq=1000000+i;payload=(f'{seq:08x}'+'0123456789abcdef'*24)[:length]
        runner.must(take(f'P {seq} {payload}')[0][0]==dict(kind='probe',seq=seq,payload=payload),'Probe')
    predictions=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
    runner.must(len(predictions)==1024*len(variants),'Prediction file count')
    cursor=0;sinks={};seq=0
    for v in variants:
        for rec in g['records']:
            seq+=1;r=take(f'V {seq} {v} '+ ' '.join(rec['input_bits']))[0][0]
            prediction_check(r,seq,v,rec['expected_bits'][v],g['tolerances'])
            runner.must(predictions[cursor]==dict(image_id=rec['image_id'],label=rec['label'],**r),'Prediction file/raw mismatch');cursor+=1
            if rec['image_id']==g['records'][g['benchmark_indices'][-1]]['image_id']:sinks[v]=r['bits'][0]
    for slot,i in enumerate(g['benchmark_indices']):
        runner.must(take(f'L {slot} '+' '.join(g['records'][i]['input_bits']))[0][0]==dict(kind='loaded',slot=slot),'Loaded batch')
    for v in variants:
        step=dict(seq=1900000+v,variant=v,duration_ms=3000)
        rs,_=take(f'E {step["seq"]} {v} 3000');validate_window(*rs,step,target,sinks[v])
    windows=json.loads((folder/'windows.json').read_text())
    runner.must(windows==s['windows'] and len(windows)==len(s['plan']),'Window file/summary')
    audited=[]
    for step,row in zip(s['plan'],windows):
        runner.must(all(row[k]==v for k,v in step.items()),'Window schedule mismatch')
        rs,ts=take(f'E {step["seq"]} {step["variant"]} {step["duration_ms"]}')
        runner.must(rs==[row['armed'],row['device']],'Saved/raw energy response')
        metrics=validate_window(*rs,step,target,sinks.get(step['variant']))
        runner.must(all(row[k]==v for k,v in metrics.items()),'Computed window metrics')
        for saved,t in zip(['armed_received_rel_s','end_received_rel_s'],ts):
            raw_rel=(t-s['recording_origin_monotonic_ns'])/1e9
            runner.must(0<=row[saved]-raw_rel<=1,'Host timestamp inconsistent with raw trace')
        audited.append(dict(**step,**metrics))
    runner.must(index==len(exchanges),'Unaccounted serial exchanges')
    return dict(status='DEVICE_COUNTS_AUDITED',physical_execution=require_physical,windows=audited,
                prediction_checks=len(predictions),host_clock=clock,energy_measurement_status='EXTERNAL_POWER_TRACE_REQUIRED')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();report=audit(a.capture)
    if a.out:a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(report['windows']),'windows; electrical energy still requires a meter trace')
