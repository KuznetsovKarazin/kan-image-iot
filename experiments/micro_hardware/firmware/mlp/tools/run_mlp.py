"""Pretrained MLP FP32/INT8: audited resumable batches; no cable removal."""
import argparse, datetime, hashlib, importlib.metadata, json, math, random, re
from pathlib import Path
import statistics, subprocess, sys, time, zipfile
from serial_common import SerialLink, prediction_check, decode, transport_probe, load_batch
from device_support import PortTracker, validate_reset_output
from clock_support import install_logger
install_logger(SerialLink)

ROOT=Path(__file__).resolve().parents[1]
VERSION='mlp-v2'
VARIANTS=[0,1]
LOOP=2
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');q.replace(p)
def must(ok,why):
    if not ok:raise ValueError(why)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def preflight(target):
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    must(manifest['version']==VERSION,'Wrong release')
    for name,digest in manifest['sha256'].items():must(sha(ROOT/name)==digest,'SHA256 mismatch: '+name)
    must(importlib.metadata.version('pyserial')=='3.5','Install requirements.txt: pyserial 3.5 required')
    if target in ['s3','c3']:
        must(importlib.metadata.version('esptool')=='4.8.1','Install requirements.txt: esptool 4.8.1 required')
    return manifest
def execute(argv,log,timeout=300):
    print('Running:', ' '.join(map(str,argv)),flush=True)
    with log.open('x',encoding='utf-8') as f:
        p=subprocess.Popen([str(x) for x in argv],stdout=f,stderr=subprocess.STDOUT)
        start=last=time.monotonic()
        try:
            while p.poll() is None:
                now=time.monotonic()
                if now-start>timeout:raise TimeoutError('Command timeout; see '+str(log))
                if now-last>30:print('Still running; log:',log.name,flush=True);last=now
                time.sleep(.2)
        except BaseException:
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            raise
    text=log.read_text(encoding='utf-8',errors='replace')
    print(text[-1800:],flush=True);must(p.returncode==0,'Command failed; see '+str(log))
    return text
def reset_esp(target,port,log,expected_mac=None):
    chip='esp32s3' if target=='s3' else 'esp32c3'
    text=execute([sys.executable,'-m','esptool','--chip',chip,'--port',port,'--baud','460800',
                  '--after','hard_reset','read_mac'],log)
    return validate_reset_output(text,target,expected_mac)
def hello(link,g,target):
    link.send('H');r=link.receive(startup=True)
    for k,v in dict(kind='hello',protocol=4,artifact=g['artifact'],model=g['model'],input_dim=g['input_dim'],
                   variants=2,serial_transport='esp-idf-usb-serial-jtag-v4',radios_started=False,
                   cpu_mhz=240 if target=='s3' else 160).items():must(r.get(k)==v,'Handshake mismatch: '+k)
    must(r['chip'].startswith('ESP32-S3' if target=='s3' else 'ESP32-C3'),'Wrong chip')
    core=1 if target=='s3' else 0
    must(r.get('task_core')==core and r.get('task_affinity')==core,'Benchmark task is not pinned to expected core')
    must(r.get('psram_bytes')==0,'Unexpected PSRAM')
    return r
def verify(link,g,variants,handle):
    seq=0;stats=[];sinks={};last=g['records'][g['benchmark_indices'][-1]]['image_id']
    for v in variants:
        row=dict(variant=v,name=g['variants'][v],n=0,max_abs_error=0.0,correct=0,
                 changed_decisions_vs_golden=0,changed_decisions_vs_trained_fp32=0)
        seen=set()
        for rec in g['records']:
            must(rec['image_id'] not in seen,'Duplicate evaluation ID');seen.add(rec['image_id'])
            seq+=1;link.send(f'V {seq} {v} '+ ' '.join(rec['input_bits']));r=link.receive()
            y,err=prediction_check(r,seq,v,rec['expected_bits'][v],g['tolerances'])
            pred=int(y[1]>y[0]);ref=decode(rec['reference_bits'])
            row['n']+=1;row['max_abs_error']=max(row['max_abs_error'],err)
            row['correct']+=int(pred==rec['label'])
            row['changed_decisions_vs_trained_fp32']+=int(pred!=int(ref[1]>ref[0]))
            if rec['image_id']==last:sinks[v]=r['bits'][0]
            handle.write(json.dumps(dict(image_id=rec['image_id'],label=rec['label'],**r))+'\n')
            if row['n']%256==0:handle.flush();print(f'{row["name"]}: verified {row["n"]}/1024',flush=True)
        must(row['n']==1024,'Incomplete correctness pass');stats.append(row)
    sinks[LOOP]=g['records'][g['benchmark_indices'][-1]]['input_bits'][0]
    return stats,sinks,seq
def check_timing(r,seq,v,rounds,sink,mhz,core,allow_zero=False):
    expected=dict(kind='bench',seq=seq,variant=v,rounds=rounds,n=rounds*32,complete=True,sink=sink,
                  core_start=core,core_end=core)
    for k,x in expected.items():must(r.get(k)==x,'Timing response mismatch: '+k)
    for k in ['elapsed_cycles','elapsed_us']:must(type(r.get(k)) is int and r[k]>=0,'Bad timer: '+k)
    must(r['elapsed_cycles']>0,'Cycle counter did not advance')
    must(allow_zero or r['elapsed_us']>0,'Timer below resolution')
    must(r['elapsed_us']<5000000,'Window exceeds counter wrap policy')
    # Counter includes ISR/RTOS activity, as does wall time. Check clock/core, not equality at one cycle.
    if r['elapsed_us']>=1000:
        ratio=r['elapsed_cycles']/(mhz*r['elapsed_us'])
        must(.95<=ratio<=1.05,'Cycle/wall-clock ratio outside 5% gate')
    r['us_per_inference']=r['elapsed_us']/r['n'];r['cycles_per_inference']=r['elapsed_cycles']/r['n']
    return r
def benchmark(link,g,variants,sinks,seq,target,session):
    load_batch(link,g);order=variants+[LOOP];rounds={};rows=[]
    mhz,core=(240,1) if target=='s3' else (160,0)
    def once(v,n,zero=False):
        nonlocal seq
        seq+=1;link.send(f'B {seq} {v} {n}')
        return check_timing(link.receive(),seq,v,n,sinks[v],mhz,core,zero)
    for v in order:
        gauge=1
        while True:
            r=once(v,gauge,True)
            if r['elapsed_us']>=10 or gauge==32:break
            gauge*=2
        must(r['elapsed_us']>0,'Timer remains below resolution')
        rounds[v]=max(1,min(32,int(20000*gauge/r['elapsed_us'])))
        for _ in range(5):once(v,rounds[v])
    rng=random.Random(20261109+100*g['case']+session)
    for repeat in range(30):
        current=list(order);rng.shuffle(current)
        for v in current:
            r=once(v,rounds[v]);r['repeat']=repeat;rows.append(r)
        if (repeat+1)%10==0:print(f'Timing blocks: {repeat+1}/30',flush=True)
    summary=[]
    for v in order:
        block=[r for r in rows if r['variant']==v]
        summary.append(dict(variant=v,name=g['variants'][v] if v!=LOOP else 'loop_overhead',n_blocks=30,
            median_us=statistics.median(r['us_per_inference'] for r in block),
            median_cycles=statistics.median(r['cycles_per_inference'] for r in block)))
    return rows,summary
def capture_esp(target,port,board,folder,session,case=0):
    folder.mkdir(exist_ok=False);g=json.loads((ROOT/f'esp32/generated/model{case}_golden.json').read_text())
    variants=VARIANTS
    state=dict(status='IN_PROGRESS',physical_execution=False,board_id=board,port=port,target=target,case=case,session=session,
               artifact=g['artifact'],runner_version=VERSION,runner_sha256=sha(__file__),
               golden_sha256=sha(ROOT/f'esp32/generated/model{case}_golden.json'),variants=variants,
               firmware_sha256=json.loads((ROOT/'esp32/firmware/manifest.json').read_text())['images'][f'{target}_case{case}']['sha256'],
               timer_scope='per-core elapsed cycles and wall time; includes interrupts, RTOS and loop work; no overhead subtraction',
               memory_scope='FP32 and INT8 MLP parameters in mapped flash; both formats linked; no DRAM copies',
               start_utc=utc())
    save(folder/'summary.json',state);link=None
    try:
        with (folder/'transport.jsonl').open('w') as trace,(folder/'predictions.jsonl').open('w') as preds:
            link=SerialLink(port,trace);state['transport']=link.transport;state['physical_execution']=link.transport=='physical_serial'
            state['device']=hello(link,g,target)
            for _ in range(2):hello(link,g,target)
            state['hello_round_trips']=3;state['transport_probe']=transport_probe(link)
            state['correctness'],sinks,seq=verify(link,g,variants,preds)
            rows,state['latency']=benchmark(link,g,variants,sinks,seq,target,session)
            save(folder/'latency_blocks.json',rows)
        state.update(status='PASS',end_utc=utc())
    except BaseException as e:state.update(status='FAIL',error=str(e));raise
    finally:
        if link:link.close()
        save(folder/'summary.json',state)
    print('Saved verified MLP capture:',folder,flush=True)
def bridge_identity(port):
    from serial.tools.list_ports import comports
    found=[p for p in comports() if p.device.casefold()==port.casefold()]
    must(len(found)==1,'Mega COM port missing or ambiguous');p=found[0]
    must(p.vid is not None and p.pid is not None,'Selected port is not a USB bridge')
    return dict(vid=p.vid,pid=p.pid,serial_number=p.serial_number,port=p.device)
def check_bridge(old,new):
    must((old['vid'],old['pid'])==(new['vid'],new['pid']),'USB bridge VID/PID changed')
    if old['serial_number']:must(old['serial_number']==new['serial_number'],'USB bridge serial changed')
    else:must(old['port'].casefold()==new['port'].casefold(),'Unnumbered bridge moved to another COM port')
def main():
    from batch_v2 import run_cli
    run_cli(sys.modules[__name__])

if __name__=='__main__':main()
