"""Independently reconstruct a Mega capture from raw serial bytes and frozen references."""
import argparse
from clock_support import inspect_clock
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
from collections import Counter
from audit_v2_esp import zip_files, sha, decode, klass
TIMER_HZ=2000000
PROBE_LENGTHS=[1,63,64,65,127,255,384]

def need(condition,message):
    if not condition:raise ValueError(message)
def words(data):
    need(isinstance(data,list) and len(data)==2,'Output shape')
    result=[struct.unpack('<f',struct.pack('<I',int(x,16)))[0] for x in data]
    need(all(len(x)==8 for x in data) and all(math.isfinite(x) for x in result),'Invalid outputs')
    return result
def quantile(x,p):
    x=sorted(x);i=(len(x)-1)*p;a=int(i);b=min(a+1,len(x)-1);return x[a]+(x[b]-x[a])*(i-a)

def audit_one(raw,package,require_physical=True):
    filenames={'summary.json','predictions.jsonl','latency_blocks.json','transport.jsonl'}
    need(set(raw)==filenames,'Capture file set')
    s=json.loads(raw['summary.json']);need(s['status']=='PASS' and s['phase']=='complete','Incomplete capture')
    v=s['variant'];case=s['case'];need(case in [0,1,2] and v in [0,1],'Case/variant identity')
    image=json.loads(package['mega/firmware/manifest.json'])['images'][f'case{case}_v{v}']
    g=json.loads(package[f'mega/reference/model{case}_golden.json'])
    need(image['hex_sha256']==sha(package['mega/'+image['hex_path']]),'Firmware image SHA')
    need(image['golden_sha256']==sha(package[f'mega/reference/model{case}_golden.json']),'Reference SHA')
    need(s['runner_version']=='2026-11-09.mega-mlp-v2','Runner version')
    need(len(g['records'])==1024 and Counter(r['label'] for r in g['records'])=={0:512,1:512},'Reference size/balance')
    need(s['artifact']==image['artifact'] and s['source_artifact']==g['artifact'],'Artifact mismatch')
    need(s['firmware_hex_sha256']==image['hex_sha256'] and s['golden_sha256']==image['golden_sha256'],'Image/golden hash')
    need(s['runner_sha256']==sha(package['mega/tools/mega_runner.py']),'Runner hash')
    need(s['model']==g['model'] and s['flash_bytes']==image['flash_bytes'] and s['static_sram_bytes']==image['static_sram_bytes'],'Capture metadata')
    need(s['physical_execution']==(s['transport']=='physical_serial'),'Transport/scope mismatch')
    if require_physical:need(s['physical_execution'] is True,'Simulator data cannot count as board data')
    need(s['reset_method']=='dtr_pulse' and s['power_cycle'] is False,'Reset mode')
    events=[json.loads(line) for line in raw['transport.jsonl'].splitlines()]
    clock=inspect_clock(events)
    pending=bytearray();command=None;pairs=[];opens=0;resets=0
    previous=None
    for i,e in enumerate(events):
        now=datetime.datetime.fromisoformat(e['utc'])
        previous=now
        event=e['event']
        if event=='port_open':
            opens+=1
            need(i==0 and e['port']==s['port'] and e['dtr'] is False and e['rts'] is False
                 and e['control_lines_configured_before_open'] is True,'Port-open identity/control lines')
        elif event=='dtr_reset':
            need(i==1,'Reset order');resets+=1;need(e['power_cycle'] is False and e['pulse_seconds']==.1 and e['settle_seconds']==3.,'DTR procedure')
        elif event=='tx':
            need(command is None and not pending,'Overlapping exchanges');command=e['command']
        elif event=='rx_chunk':
            need(command is not None,'Unsolicited RX')
            data=bytes.fromhex(e['raw_hex']);need(len(data)==e['n_bytes'] and len(data)>0,'RX length');pending.extend(data)
        elif event=='startup_text':
            end=pending.find(b'\n');need(not pairs and end>=0,'Unexpected startup text')
            data=bytes(pending[:end+1]);del pending[:end+1]
            need(data==bytes.fromhex(e['raw_hex']) and not data.lstrip().startswith(b'{'),'Skipped protocol frame')
        elif event=='response':
            end=pending.find(b'\n');need(command is not None and end>=0,'Missing response bytes')
            frame=bytes(pending[:end+1]);del pending[:end+1]
            response=json.loads(frame.decode('ascii'));need(response==e['response'],'Raw/log response mismatch')
            pairs.append((command,response));command=None
        else:raise ValueError('Error or unknown transport event: '+event)
    need(opens==1 and resets==1 and command is None and not pending,'Incomplete transport/reset')
    cursor=0
    def take(expected):
        nonlocal cursor
        need(cursor<len(pairs),'Missing exchange');cmd,r=pairs[cursor];cursor+=1
        need(cmd==expected,'Command/order differs at '+str(cursor));return r
    hellos=[]
    for _ in range(3):
        r=take('H');hellos.append(r)
        expected=dict(kind='hello',protocol=2,chip='ATmega2560',cpu_mhz=16,sram_bytes=8192,flash_bytes=262144,input_dim=g['input_dim'],
                      variant=v,model=g['model'],variant_name=g['variants'][v],artifact=image['artifact'],source_artifact=g['artifact'],
                      parameter_bytes=image['parameter_bytes'],timer_hz=TIMER_HZ,serial_transport='avr-uart0-v1')
        need(all(r.get(k)==val for k,val in expected.items()),'Hello identity')
        need(type(r.get('untouched_sram_gap_bytes')) is int and r['untouched_sram_gap_bytes']>=128,'SRAM gap')
    need(s['device']==hellos[0] and s['hello_round_trips']==3,'Saved hello')
    count=0
    for _ in range(3):
        for length in PROBE_LENGTHS:
            count+=1;seq=1000000+count;payload=(f'{seq:08x}'+'0123456789abcdef'*24)[:length]
            need(take(f'P {seq} {payload}')==dict(kind='probe',seq=seq,payload=payload),'Probe mismatch')
    need(s['probe']['round_trips']==21 and s['probe']['status']=='PASS' and s['probe']['payload_lengths']==PROBE_LENGTHS,'Saved probes')
    predictions=[json.loads(line) for line in raw['predictions.jsonl'].splitlines()]
    need(len(predictions)==1024,'Prediction count')
    computed=dict(variant=v,name=g['variants'][v],n=0,max_abs_error=0.,decision_changes_vs_golden=0,
                  decision_changes_vs_trained_fp32=0,correct=0)
    bits_by_image={};tol=g['tolerances']
    for seq,rec in enumerate(g['records'],1):
        need(rec['image_id'] not in bits_by_image,'Duplicate input ID')
        r=take(f'V {seq} {v} '+' '.join(rec['input_bits']))
        need(r.get('kind')=='pred' and r.get('seq')==seq and r.get('variant')==v,'Prediction identity')
        need(predictions[seq-1]==dict(image_id=rec['image_id'],label=rec['label'],**r),'Prediction file differs')
        actual=words(r['bits']);expected=words(rec['expected_bits'][v]);trained=words(rec['reference_bits'])
        if v==1:need([int(w,16) for w in r['bits']]==[int(w,16) for w in rec['expected_bits'][v]],'INT8 output bits differ')
        need(all(abs(a-b)<=tol['atol']+tol['rtol']*abs(b) for a,b in zip(actual,expected)),'Golden tolerance exceeded')
        klass=int(actual[1]>actual[0]);need(klass==int(expected[1]>expected[0]),'Own-golden class mismatch')
        computed['n']+=1;computed['max_abs_error']=max(computed['max_abs_error'],max(abs(a-b) for a,b in zip(actual,expected)))
        computed['correct']+=int(klass==rec['label']);computed['decision_changes_vs_trained_fp32']+=int(klass!=int(trained[1]>trained[0]))
        bits_by_image[rec['image_id']]=r['bits']
    need(s['correctness']==computed and s['correctness_status']=='PASS','Correctness summary differs')
    ids=g['benchmark_indices'];need(len(ids)==len(set(ids))==32,'Timing input set')
    need(sum(g['records'][i]['label'] for i in ids)==16,'Timing class balance')
    for slot,index in enumerate(ids):
        r=take(f'L {slot} '+' '.join(g['records'][index]['input_bits']));need(r==dict(kind='loaded',slot=slot),'Batch load')
    last=g['records'][ids[-1]];sinks={v:bits_by_image[last['image_id']][0],2:last['input_bits'][0]}
    seq=1024;responses=[]
    def bench(variant,rounds):
        nonlocal seq
        seq+=1;r=take(f'B {seq} {variant} {rounds}')
        expected=dict(kind='bench',seq=seq,variant=variant,rounds=rounds,n=rounds*32,timer_hz=TIMER_HZ,complete=True,sink=sinks[variant])
        need(all(r.get(k)==val for k,val in expected.items()),'Benchmark fields/sink')
        need(type(r.get('elapsed_ticks')) is int and 0<r['elapsed_ticks']<2**32,'Benchmark timer')
        need(type(r.get('untouched_sram_gap_bytes')) is int and r['untouched_sram_gap_bytes']>=128,'Benchmark SRAM guard')
        responses.append(r)
        return dict(**r,us_per_inference=r['elapsed_ticks']/(2*r['n']))
    rounds={}
    for variant in [v,2]:
        gauge=1
        while True:
            r=bench(variant,gauge)
            if r['elapsed_ticks']>=20 or gauge==32:break
            gauge*=2
        rounds[variant]=max(1,min(32,int(40000*gauge/r['elapsed_ticks'])))
        for _ in range(5):bench(variant,rounds[variant])
    blocks=json.loads(raw['latency_blocks.json']);need(len(blocks)==60,'Measured block count')
    rng=random.Random(20261107);i=0;derived=[]
    for repeat in range(30):
        order=[v,2];rng.shuffle(order)
        for variant in order:
            r=bench(variant,rounds[variant]);r['repeat']=repeat
            need(r==blocks[i],'Measured block/trace/order differs');derived.append(r);i+=1
    need(cursor==len(pairs),'Extra protocol exchanges')
    lat=[]
    for variant in [v,2]:
        x=[r['us_per_inference'] for r in derived if r['variant']==variant]
        lat.append(dict(variant=variant,name=g['variants'][variant] if variant!=2 else 'loop_overhead',n_blocks=30,
                        median_us=statistics.median(x),p05_us=quantile(x,.05),p95_us=quantile(x,.95)))
    need(lat==s['latency'],'Latency summary differs')
    start=datetime.datetime.fromisoformat(events[0]['utc']);end=datetime.datetime.fromisoformat(events[-1]['utc'])
    return dict(status='PASS',host_clock=clock,physical_execution=s['physical_execution'],board_id=s['board_id'],case=s['case'],variant=v,
                model=g['model'],artifact=image['artifact'],correctness=computed,latency=lat,exchanges=len(pairs),
                port=s['port'],flash_bytes=s['flash_bytes'],static_sram_bytes=s['static_sram_bytes'],
                parameter_bytes=image['parameter_bytes'],firmware_hex_sha256=image['hex_sha256'],
                start_utc=events[0]['utc'],end_utc=events[-1]['utc'],
                transport_events=dict(Counter(e['event'] for e in events)),
                command_counts=dict(Counter(cmd.split()[0] for cmd,_ in pairs)),
                maximum_timing_window_seconds=max(r['elapsed_ticks']/TIMER_HZ for r in responses),
                rounds_per_block=rounds,
                prediction_checks=1024,measured_blocks=60,setup_blocks=len(responses)-60,
                min_untouched_sram_gap_bytes=min(r['untouched_sram_gap_bytes'] for r in hellos+responses),
                host_session_seconds=clock['host_duration_seconds'],capture_file_sha256={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},
                timing_scope='Separate selected-variant firmware; 30 blocks paired with loop control, no overhead subtraction')
