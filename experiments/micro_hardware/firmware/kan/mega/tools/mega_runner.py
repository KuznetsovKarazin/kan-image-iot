"""Strict one-variant Mega capture. Flash first; keep the USB cable connected."""
import argparse
import json
from pathlib import Path
import random
import statistics
import time
from mega_common import ROOT,VARIANTS,RUNNER_VERSION,TIMER_HZ,preflight,sha
from serial_common import SerialLink,prediction_check,decode,transport_probe,load_batch

class MegaLink(SerialLink):
    def __init__(self,port,trace):
        import serial
        self.trace=trace;self.pending=bytearray();self.last_command=None
        self.serial=serial.Serial(port=None,baudrate=115200,timeout=.1,write_timeout=5)
        try:
            self.serial.dtr=False;self.serial.rts=False;self.serial.port=port;self.serial.open()
            self.log('port_open',port=port,dtr=False,rts=False,control_lines_configured_before_open=True)
            self.serial.dtr=True;time.sleep(.1);self.serial.dtr=False
            self.log('dtr_reset',pulse_seconds=.1,settle_seconds=3.,power_cycle=False)
            time.sleep(3.);self.serial.reset_input_buffer()
        except BaseException:
            self.serial.close();raise
    def receive(self,timeout=180,startup=False):return super().receive(timeout,startup)

def handshake(link,image):
    link.send('H');r=link.receive(startup=True)
    expected=dict(kind='hello',protocol=2,chip='ATmega2560',cpu_mhz=16,sram_bytes=8192,flash_bytes=262144,
                  input_dim=image['input_dim'],variant=image['variant'],model=image['model'],variant_name=image['variant_name'],
                  artifact=image['artifact'],source_artifact=image['source_artifact'],parameter_bytes=image['parameter_bytes'],
                  timer_hz=TIMER_HZ,serial_transport='avr-uart0-v1')
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Mega firmware/variant/protocol mismatch')
    if type(r.get('untouched_sram_gap_bytes')) is not int or r['untouched_sram_gap_bytes']<128:raise ValueError('Insufficient SRAM gap')
    return r

def verify(link,data,variant,output):
    stats=dict(variant=variant,name=data['variants'][variant],n=0,max_abs_error=0.,decision_changes_vs_golden=0,
               decision_changes_vs_trained_fp32=0,correct=0)
    bits_by_image={}
    for seq,rec in enumerate(data['records'],1):
        if rec['image_id'] in bits_by_image:raise ValueError('Duplicate image ID')
        link.send(f'V {seq} {variant} '+' '.join(rec['input_bits']));r=link.receive()
        y,error=prediction_check(r,seq,variant,rec['expected_bits'][variant],data['tolerances'])
        pred=int(y[1]>y[0]);ref=decode(rec['reference_bits'])
        stats['n']+=1;stats['max_abs_error']=max(stats['max_abs_error'],error)
        stats['correct']+=int(pred==rec['label']);stats['decision_changes_vs_trained_fp32']+=int(pred!=int(ref[1]>ref[0]))
        bits_by_image[rec['image_id']]=r['bits']
        output.write(json.dumps(dict(image_id=rec['image_id'],label=rec['label'],**r))+'\n')
        if seq%128==0:output.flush();print(f'Verified {seq}/1024 predictions',flush=True)
    if stats['n']!=1024:raise ValueError('Incomplete prediction set')
    return stats,bits_by_image

def once(link,seq,variant,rounds,sink):
    link.send(f'B {seq} {variant} {rounds}');r=link.receive()
    expected=dict(kind='bench',seq=seq,variant=variant,rounds=rounds,n=rounds*32,complete=True,timer_hz=TIMER_HZ,sink=sink)
    if any(r.get(k)!=v for k,v in expected.items()) or type(r.get('elapsed_ticks')) is not int or r['elapsed_ticks']<=0:
        raise ValueError('Invalid/incomplete benchmark response')
    if type(r.get('untouched_sram_gap_bytes')) is not int or r['untouched_sram_gap_bytes']<128:raise ValueError('SRAM guard')
    r['us_per_inference']=r['elapsed_ticks']/(2*r['n']);return r

def benchmark(link,data,variant,bits_by_image):
    load_batch(link,data)
    last=data['records'][data['benchmark_indices'][-1]]
    sinks={variant:bits_by_image[last['image_id']][0],11:last['input_bits'][0]}
    seq=1024;rounds={};rows=[];order=[variant,11]
    for v in order:
        print(f'Preparing timing for variant {v} (5 warm-up blocks)',flush=True)
        gauge=1
        while True:
            seq+=1;r=once(link,seq,v,gauge,sinks[v])
            if r['elapsed_ticks']>=20 or gauge==32:break
            gauge*=2
        rounds[v]=max(1,min(32,int(40000*gauge/r['elapsed_ticks'])))
        for _ in range(5):seq+=1;once(link,seq,v,rounds[v],sinks[v])
    rng=random.Random(20261107)
    for repeat in range(30):
        order=[variant,11];rng.shuffle(order)
        for v in order:
            seq+=1;r=once(link,seq,v,rounds[v],sinks[v]);r['repeat']=repeat;rows.append(r)
        if (repeat+1)%3==0:print(f'Timing blocks: {repeat+1}/30 per variant/control',flush=True)
    def q(x,p):
        x=sorted(x);i=(len(x)-1)*p;a=int(i);b=min(a+1,len(x)-1);return x[a]+(x[b]-x[a])*(i-a)
    summary=[]
    for v in [variant,11]:
        values=[r['us_per_inference'] for r in rows if r['variant']==v]
        summary.append(dict(variant=v,name=data['variants'][v] if v<11 else 'loop_overhead',n_blocks=30,
                            median_us=statistics.median(values),p05_us=q(values,.05),p95_us=q(values,.95)))
    return rows,summary

def run_capture(args,link_factory=MegaLink):
    _,image,data=preflight(args.case,args.variant)
    args.out.mkdir(parents=True,exist_ok=False)
    base=dict(status='IN_PROGRESS',phase='handshake',board_id=args.board_id,port=args.port,case=args.case,variant=args.variant,
              runner_version=RUNNER_VERSION,runner_sha256=sha(Path(__file__)),golden_sha256=image['golden_sha256'],
              artifact=image['artifact'],source_artifact=image['source_artifact'],firmware_hex_sha256=image['hex_sha256'],
              model=image['model'],reset_method='dtr_pulse',power_cycle=False,
              flash_bytes=image['flash_bytes'],static_sram_bytes=image['static_sram_bytes'])
    path=args.out/'summary.json';path.write_text(json.dumps(base,indent=2));link=None
    try:
        with (args.out/'transport.jsonl').open('w') as trace,(args.out/'predictions.jsonl').open('w') as predictions:
            link=link_factory(args.port,trace)
            base['transport']=link.transport;base['physical_execution']=link.transport=='physical_serial'
            base['device']=handshake(link,image)
            for _ in range(2):handshake(link,image)
            base['hello_round_trips']=3;base['phase']='probe';base['probe']=transport_probe(link)
            print(f"Connected: ATmega2560 {image['model']} / {image['variant_name']}",flush=True)
            base['phase']='correctness';base['correctness'],predicted=verify(link,data,args.variant,predictions)
            base['correctness_status']='PASS';base['phase']='latency'
            rows,base['latency']=benchmark(link,data,args.variant,predicted)
            (args.out/'latency_blocks.json').write_text(json.dumps(rows,indent=2))
        base['status']='PASS';base['phase']='complete'
        print('Saved verified Mega capture:',args.out,flush=True)
    except BaseException as e:
        base['status']='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAIL';base['error']=str(e);raise
    finally:
        path.write_text(json.dumps(base,indent=2))
        if link:link.close()
    return base

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True);p.add_argument('--board-id',required=True)
    p.add_argument('--case',type=int,choices=[0,1,2],required=True);p.add_argument('--variant',type=int,choices=VARIANTS,required=True)
    p.add_argument('--out',type=Path,required=True);run_capture(p.parse_args())
