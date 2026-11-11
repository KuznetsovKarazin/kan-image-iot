"""Strict board capture. Standard library + pyserial; no training dependencies."""
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
PROBE_LENGTHS = [1,63,64,65,127,255,384]


def decode(words):
    if len(words)!=2 or any(not isinstance(w,str) or len(w)!=8 for w in words):
        raise ValueError('Expected exactly two binary32 hex words')
    values=[struct.unpack('<f',struct.pack('<I',int(w,16)))[0] for w in words]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite device logits')
    return values


def prediction_check(response,seq,variant,expected,tol):
    if response.get('kind')!='pred' or response.get('seq')!=seq or response.get('variant')!=variant:
        raise ValueError('Unexpected/duplicate prediction frame')
    y=decode(response['bits']);ref=decode(expected)
    error=max(abs(a-b) for a,b in zip(y,ref))
    close=all(abs(a-b)<=tol['atol']+tol['rtol']*abs(b) for a,b in zip(y,ref))
    changed=int((y[1]>y[0])!=(ref[1]>ref[0]))
    if not close or changed:raise ValueError(f'Device fidelity failed: seq={seq}, variant={variant}, error={error}, changed={changed}')
    return y,error


class SerialLink:
    transport='physical_serial'
    def __init__(self,port,trace):
        import serial
        self.trace=trace
        self.pending=bytearray()
        self.last_command=None
        # Set control lines while closed. Opening with pySerial's default
        # asserted lines and changing them afterwards may reset native USB/JTAG.
        self.serial=serial.Serial(port=None,baudrate=115200,timeout=.25,write_timeout=5)
        try:
            self.serial.dtr=False;self.serial.rts=False
            self.serial.port=port
            self.serial.open()
            self.log('port_open',port=port,dtr=False,rts=False,
                     control_lines_configured_before_open=True)
            time.sleep(1)
            self.serial.reset_input_buffer()
        except Exception as error:
            self.serial.close()
            self.log('port_open_error',error=str(error),port=port)
            raise
    def log(self,event,**fields):
        record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'event':event,**fields}
        self.trace.write(json.dumps(record)+'\n');self.trace.flush()
    def fail_frame(self,reason,raw):
        self.log('frame_error',reason=reason,raw_hex=raw.hex(),
                 last_command=self.last_command)
        raise ValueError(f'{reason}; received {len(raw)} bytes: {raw[:160]!r}. '
                         'See transport.jsonl for the complete bytes and last command.')
    def send(self,line):
        self.last_command=line
        self.log('tx',command=line)
        data=(line+'\n').encode('ascii')
        written=self.serial.write(data)
        if written!=len(data):
            self.log('write_error',expected_bytes=len(data),written_bytes=written)
            raise IOError('Incomplete serial command write; see transport.jsonl')
        self.serial.flush()
    def receive(self,timeout=10,startup=False):
        until=time.monotonic()+timeout
        while True:
            end=self.pending.find(b'\n')
            if end>=0:
                raw=bytes(self.pending[:end+1]);del self.pending[:end+1]
                if len(raw)>MAX_SERIAL_FRAME_BYTES:
                    self.fail_frame('Serial frame exceeds size limit',raw)
                try:
                    obj=json.loads(raw.decode('ascii'))
                except (UnicodeError,json.JSONDecodeError):
                    # ROM boot text may precede the initial hello. Never skip a
                    # damaged protocol object, or any bad frame after handshake.
                    if startup and not raw.lstrip().startswith(b'{'):
                        self.log('startup_text',raw_hex=raw.hex())
                        continue
                    self.fail_frame('Malformed complete serial frame',raw)
                if not isinstance(obj,dict):
                    self.fail_frame('Expected a JSON object in serial frame',raw)
                self.log('response',response=obj)
                if obj.get('kind')=='error':raise ValueError('Firmware error: '+str(obj))
                return obj
            if len(self.pending)>MAX_SERIAL_FRAME_BYTES:
                self.fail_frame('Unterminated serial frame exceeds size limit',bytes(self.pending))
            if time.monotonic()>=until:
                raw=bytes(self.pending)
                self.log('receive_timeout',partial_hex=raw.hex(),
                         last_command=self.last_command,timeout_seconds=timeout)
                raise TimeoutError(f'No newline-terminated firmware response within {timeout}s; '
                                   f'pending {len(raw)} bytes: {raw[:160]!r}. See transport.jsonl.')
            try:
                # read() may return any fragment, including a timeout-shortened
                # fragment. Only a newline terminates a protocol response.
                data=self.serial.read(min(512,max(1,self.serial.in_waiting)))
            except Exception as error:
                self.log('read_error',error=str(error),partial_hex=bytes(self.pending).hex(),
                         last_command=self.last_command)
                raise
            if data:
                self.log('rx_chunk',n_bytes=len(data),raw_hex=data.hex())
                self.pending.extend(data)
    def close(self):self.serial.close()


def handshake(link,data):
    link.send('H');meta=link.receive(startup=True)
    if meta.get('kind')!='hello' or meta.get('protocol')!=1:raise ValueError('Wrong protocol')
    for key in ['artifact','model','input_dim']:
        if meta.get(key)!=data[key]:raise ValueError('Firmware/golden mismatch: '+key)
    expected_transport='esp-idf-usb-serial-jtag-v4'
    if meta.get('serial_transport')!=expected_transport or data.get('serial_transport')!=expected_transport:
        raise ValueError('USB transport version mismatch: install the v4 firmware and matching golden file')
    if meta.get('variants')!=len(data['variants']):raise ValueError('Variant mismatch')
    if link.transport=='physical_serial':
        chip=meta.get('chip','')
        if chip.startswith('ESP32-S3'):clock=240
        elif chip.startswith('ESP32-C3'):clock=160
        else:raise ValueError('Unexpected target chip: '+chip)
        if meta.get('cpu_mhz')!=clock:raise ValueError('Unexpected CPU clock')
        if meta.get('flash_bytes',0)<4*1024*1024:raise ValueError('Firmware requires at least 4 MB flash')
        if meta.get('radios_started') is not False:raise ValueError('Unexpected radio state')
    return meta


def transport_probe(link):
    count=0
    for repeat in range(3):
        for length in PROBE_LENGTHS:
            count+=1;seq=1000000+count
            payload=(f'{seq:08x}'+'0123456789abcdef'*24)[:length]
            link.send(f'P {seq} {payload}')
            response=link.receive()
            if response.get('kind')!='probe' or response.get('seq')!=seq or response.get('payload')!=payload:
                raise ValueError('USB round-trip mismatch; see transport.jsonl')
    return {'status':'PASS','round_trips':count,'payload_lengths':PROBE_LENGTHS,
            'scope':'Communication check only; not model correctness or latency'}


def verify(link,data,output,progress=None):
    seq=0;stats=[{'variant':v,'name':name,'n':0,'max_abs_error':0.,'decision_changes_vs_golden':0,
                 'decision_changes_vs_trained_fp32':0,'correct':0} for v,name in enumerate(data['variants'])]
    seen=set()
    for rec in data['records']:
        if rec['image_id'] in seen:raise ValueError('Duplicate image ID in golden file')
        seen.add(rec['image_id'])
        for variant,s in enumerate(stats):
            seq+=1
            link.send(f"V {seq} {variant} "+' '.join(rec['input_bits']))
            response=link.receive()
            y,error=prediction_check(response,seq,variant,rec['expected_bits'][variant],data['tolerances'])
            pred=int(y[1]>y[0]);ref=decode(rec['reference_bits'])
            s['n']+=1;s['max_abs_error']=max(s['max_abs_error'],error)
            s['decision_changes_vs_trained_fp32']+=int(pred!=int(ref[1]>ref[0]))
            s['correct']+=int(pred==rec['label'])
            output.write(json.dumps({'image_id':rec['image_id'],'label':rec['label'],**response})+'\n')
        if len(seen)%128==0:
            output.flush()
            if progress is not None:progress(seq,8192)
    if len(seen)!=1024 or any(s['n']!=1024 for s in stats):raise ValueError('Incomplete evaluation set')
    return stats,seq


def load_batch(link,data):
    indices=data['benchmark_indices']
    if len(indices)!=32 or len(set(indices))!=32:raise ValueError('Invalid benchmark sample IDs')
    labels=[data['records'][i]['label'] for i in indices]
    if labels.count(0)!=16 or labels.count(1)!=16:raise ValueError('Benchmark batch is not balanced')
    for slot,index in enumerate(indices):
        link.send(f'L {slot} '+' '.join(data['records'][index]['input_bits']))
        r=link.receive()
        if r.get('kind')!='loaded' or r.get('slot')!=slot:raise ValueError('Batch acknowledgment mismatch')


def bench_once(link,seq,variant,rounds,allow_zero=False):
    link.send(f'B {seq} {variant} {rounds}');r=link.receive()
    if (r.get('kind')!='bench' or r.get('seq')!=seq or r.get('variant')!=variant or
        r.get('rounds')!=rounds or r.get('n')!=rounds*32 or r.get('complete') is not True or
        not isinstance(r.get('elapsed_us'),int) or r['elapsed_us']<0 or
        (r['elapsed_us']==0 and not allow_zero)):
        raise ValueError('Invalid/incomplete timing frame')
    r['us_per_inference']=r['elapsed_us']/r['n']
    return r


def benchmark(link,data,seq,repetitions=30):
    load_batch(link,data)
    rounds={};rows=[]
    for variant in range(9):
        gauge=1
        while True:
            seq+=1;r=bench_once(link,seq,variant,gauge,allow_zero=True)
            if r['elapsed_us']>=10 or gauge==32:break
            gauge*=2
        if r['elapsed_us']==0:raise ValueError('Timing remains below timer resolution')
        rounds[variant]=max(1,min(32,int(20000*gauge/r['elapsed_us'])))
        for _ in range(5):seq+=1;bench_once(link,seq,variant,rounds[variant])
    rng=random.Random(20261107)
    for repeat in range(repetitions):
        order=list(range(9));rng.shuffle(order)
        for variant in order:
            seq+=1;r=bench_once(link,seq,variant,rounds[variant]);r['repeat']=repeat;rows.append(r)
    def quantile(x,p):
        x=sorted(x);idx=(len(x)-1)*p;lo=int(idx);hi=math.ceil(idx)
        return x[lo]+(x[hi]-x[lo])*(idx-lo)
    summary=[]
    for v in range(9):
        x=[r['us_per_inference'] for r in rows if r['variant']==v]
        summary.append({'variant':v,'name':data['variants'][v] if v<8 else 'loop_overhead',
                        'n_blocks':len(x),'median_us':statistics.median(x),'p05_us':quantile(x,.05),'p95_us':quantile(x,.95)})
    return rows,summary,seq


def run(args):
    data=json.loads(args.golden.read_text())
    if len(data['records'])!=1024 or len(data['variants'])!=8:raise ValueError('Incomplete golden dataset')
    args.out.mkdir(parents=True,exist_ok=False)
    base={'board_id':args.board_id,'port':args.port,'transport':'physical_serial',
          'runner_version':RUNNER_VERSION,
          'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'golden_sha256':hashlib.sha256(args.golden.read_bytes()).hexdigest(),
          'artifact':data['artifact'],'model':data['model'],'status':'IN_PROGRESS','phase':'handshake'}
    summary_path=args.out/'summary.json'
    summary_path.write_text(json.dumps(base,indent=2))
    link=None
    try:
        with (args.out/'transport.jsonl').open('w') as trace,(args.out/'predictions.jsonl').open('w') as predictions:
            link=SerialLink(args.port,trace)
            base['device']=handshake(link,data)
            print('Connected:',base['device']['chip'],data['model'],flush=True)
            base['phase']='repeat_handshake'
            for _ in range(2):
                handshake(link,data)
            base['hello_round_trips']=3
            print('Handshake passed (3 replies)',flush=True)
            base['phase']='usb_probe'
            base['usb_probe']=transport_probe(link)
            print('USB round-trip check passed (21 frames)',flush=True)
            base['phase']='correctness'
            base['correctness'],seq=verify(link,data,predictions,
                progress=lambda done,total:print(f'Verified {done}/{total} predictions',flush=True))
            base['correctness_status']='PASS'
            print('All 8192 predictions passed',flush=True)
            base['phase']='latency'
            rows,base['latency'],seq=benchmark(link,data,seq)
            (args.out/'latency_blocks.json').write_text(json.dumps(rows,indent=2))
            if args.energy_variant is not None:
                base['phase']='energy'
                seq+=1;v=args.energy_variant
                link.send(f'E {seq} {v} {args.energy_seconds*1000}')
                armed=link.receive()
                if armed.get('kind')!='energy_armed' or armed.get('seq')!=seq or armed.get('variant')!=v:
                    raise ValueError('Invalid energy start marker')
                print('Energy window armed; save the FNB58 trace separately.',flush=True)
                end=link.receive(timeout=args.energy_seconds+10)
                if end.get('kind')!='energy_end' or end.get('seq')!=seq or end.get('variant')!=v:
                    raise ValueError('Invalid energy end marker')
                base['energy_window']={'armed':armed,'end':end,'meter_energy_joules':None,
                                      'note':'No energy measurement is inferred from firmware counters.'}
        base['status']='PASS';base['phase']='complete';summary_path.write_text(json.dumps(base,indent=2))
        print('Saved verified device capture:',args.out)
    except Exception as e:
        base['status']='FAIL';base['error']=str(e);summary_path.write_text(json.dumps(base,indent=2));raise
    finally:
        if link:link.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--port',required=True)
    p.add_argument('--golden',type=Path,required=True)
    p.add_argument('--board-id',required=True,help='Physical specimen ID, e.g. S3-A or C3-A')
    p.add_argument('--out',type=Path,required=True,help='New capture directory; will not overwrite')
    p.add_argument('--energy-variant',type=int,choices=range(9),help='Optional extra window; 8=idle')
    p.add_argument('--energy-seconds',type=int,choices=range(1,61),default=60)
    run(p.parse_args())
