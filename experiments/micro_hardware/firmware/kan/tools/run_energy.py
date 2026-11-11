"""Count long inference windows. Electrical measurements require an external meter.

No energy value is invented from runtime. Save the meter's native recording beside
this capture. See ENERGY_RU.txt for the wiring, baseline and alignment protocol.
"""
import argparse,datetime,json,random,re,sys,time
from pathlib import Path
import run_kan as runner
from serial_common import SerialLink,load_batch,transport_probe
from device_support import KeepAwake,normalized_serial
from capture_common import snapshot
ROOT=runner.ROOT
sys.path.insert(0,str(ROOT/'mega/tools'))
import mega_runner
from mega_common import preflight as mega_preflight


def idle_variant(target): return 11 if target=='mega' else 12

def allowed_variants(target): return [0,4,5,8,9,10,11] if target=='s3' else [0,4,5,8,9]

def plan(variants,repetitions,duration,baseline,case,target):
    rows=[]
    for rep in range(1,repetitions+1):
        order=list(variants);random.Random(20261109+100*case+rep).shuffle(order)
        for v in order:
            for role,variant,seconds in [('idle_before',idle_variant(target),baseline),('active',v,duration),('idle_after',idle_variant(target),baseline)]:
                rows.append(dict(seq=2000000+len(rows),repetition=rep,active_variant=v,
                                 role=role,variant=variant,duration_ms=seconds*1000))
    return rows


def validate_window(armed,end,step,target,sink):
    fields=dict(kind='energy_armed',seq=step['seq'],variant=step['variant'],lead_ms=2000,duration_ms=step['duration_ms'])
    if target=='mega':fields['energy_protocol']=1
    for k,value in fields.items():
        runner.must(armed.get(k)==value,'Energy armed identity: '+k)
    fields=dict(kind='energy_end',seq=step['seq'],variant=step['variant'])
    if target=='mega':fields['energy_protocol']=1
    for k,value in fields.items():
        runner.must(end.get(k)==value,'Energy end identity: '+k)
    n=end.get('n');runner.must(type(n) is int and 0<=n<2**32,'Invalid inference counter')
    runner.must(n==0 if step['variant']==idle_variant(target) else n>0 and n%32==0,'Incomplete energy batch/count')
    if target=='mega' and step['variant']!=idle_variant(target):runner.must(end.get('sink')==sink,'Wrong final inference output')
    if target=='mega':
        runner.must(end.get('timer_hz')==2000000 and type(end.get('elapsed_ticks')) is int,'Invalid Timer1 response')
        runner.must(end.get('untouched_sram_gap_bytes',0)>=128,'Energy SRAM guard')
        seconds=end['elapsed_ticks']/2000000
        runner.must(end.get('scope')=='whole_board_polling_loop','Wrong Mega energy scope')
    else:
        runner.must(type(end.get('elapsed_us')) is int,'Invalid ESP timer response')
        # Exact architecture-v2 ESP images report counts/time/yields, not end-core or sink.
        # No missing field is synthesized; hello verifies task affinity before the run.
        runner.must(type(end.get('scheduler_yields')) is int and end['scheduler_yields']>0,'Missing scheduler yields')
        runner.must(end.get('scope')=='whole_board_cooperative_loop','Wrong ESP energy scope')
        seconds=end['elapsed_us']/1000000
    requested=step['duration_ms']/1000
    runner.must(requested<=seconds<=requested+(30 if target=='mega' else 5),'Incomplete/overlong energy window')
    return dict(device_elapsed_s=seconds,inferences=n,throughput_per_s=n/seconds if n else None)


def capture_windows(link,steps,target,sinks,folder,origin_ns):
    windows=[]
    for step in steps:
        print(f"Energy window {len(windows)+1}/{len(steps)}: rep {step['repetition']}, {step['role']}, variant {step['variant']}, {step['duration_ms']//1000}s",flush=True)
        sent=time.monotonic_ns();link.send(f"E {step['seq']} {step['variant']} {step['duration_ms']}")
        armed=link.receive(timeout=10);armed_at=time.monotonic_ns()
        runner.must(armed.get('kind')=='energy_armed','Missing energy armed frame')
        end=link.receive(timeout=step['duration_ms']/1000+45);ended=time.monotonic_ns()
        metrics=validate_window(armed,end,step,target,sinks.get(step['variant']))
        row=dict(**step,armed=armed,device=end,**metrics,
                 command_sent_rel_s=(sent-origin_ns)/1e9,
                 armed_received_rel_s=(armed_at-origin_ns)/1e9,
                 end_received_rel_s=(ended-origin_ns)/1e9)
        # These are timing aids only: USB latency and independent device clocks
        # prevent treating this estimate as a synchronized power-meter trigger.
        row['host_estimated_end_rel_s']=row['end_received_rel_s']
        row['host_estimated_start_rel_s']=row['end_received_rel_s']-metrics['device_elapsed_s']
        windows.append(row);runner.save(folder/'windows.json',windows)
    return windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['s3','c3','mega'],required=True)
    p.add_argument('--case',type=int,choices=[0,1,2],required=True)
    p.add_argument('--variant',type=int,choices=[0,4,5,8,9,10,11],help='Required on Mega; all architecture variants by default on ESP')
    p.add_argument('--port');p.add_argument('--board-id')
    p.add_argument('--seconds',type=int,default=60,choices=range(10,61),metavar='10..60')
    p.add_argument('--baseline-seconds',type=int,default=15,choices=range(10,61),metavar='10..60')
    p.add_argument('--repetitions',type=int,choices=[1,3],default=3)
    p.add_argument('--out',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();runner.preflight(a.target)
    if a.target=='mega' and a.variant is None:p.error('--variant 0,4,5,8,9 is required for Mega')
    variants=[a.variant] if a.variant is not None else allowed_variants(a.target)
    runner.must(all(v in allowed_variants(a.target) for v in variants),'Variant unavailable on this target')
    runner.must(a.case in ([0,1,2] if a.target=='s3' else [1,2]),'Case outside the frozen architecture matrix')
    if a.check_only:print('ENERGY_FILES_PASS; hardware not accessed');return
    runner.must(a.port and a.board_id,'Supply --port and --board-id')
    runner.must(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}',a.board_id),'Invalid board ID')
    folder=(a.out or ROOT/'energy_captures'/f'{a.board_id}_C{a.case}_kan_energy_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    runner.must(not folder.exists() and not Path(str(folder)+'.zip').exists(),'Output already exists; use a new directory')
    folder.mkdir(parents=True)
    golden=ROOT/(f'mega/reference/model{a.case}_golden.json' if a.target=='mega' else f'esp32/generated/model{a.case}_golden.json')
    g=json.loads(golden.read_text());steps=plan(variants,a.repetitions,a.seconds,a.baseline_seconds,a.case,a.target)
    state=dict(status='IN_PROGRESS',physical_execution=False,target=a.target,board_id=a.board_id,case=a.case,
               variants=variants,start_utc=runner.utc(),runner_version=runner.VERSION,
               release_sha256=runner.sha(ROOT/'MANIFEST.json'),runner_sha256=runner.sha(__file__),
               golden_sha256=runner.sha(golden),artifact=g['artifact'],plan=steps,
               repetitions=a.repetitions,seconds=a.seconds,baseline_seconds=a.baseline_seconds,
               energy_measurement_status='POWER_RECORDING_REQUIRED',
               scope='Frozen CNN features to head logits; long windows include loop and scheduler work; whole powered board')
    link=None;runner.save(folder/'energy_summary.json',state)
    try:
        with KeepAwake() as awake:
            state['sleep_policy']=awake.status
            if a.target=='mega':
                identity=runner.bridge_identity(a.port);state['usb_identity']=identity
                runner.execute([sys.executable,ROOT/'mega/tools/mega_flash.py','--case',str(a.case),'--variant',str(a.variant),
                                '--port',a.port,'--log',folder/'avrdude.log'],folder/'flash.log')
                runner.check_bridge(identity,runner.bridge_identity(a.port))
                _,firmware,_=mega_preflight(a.case,a.variant);state['firmware_sha256']=firmware['hex_sha256']
                port=a.port
            else:
                tracker=runner.PortTracker(a.port);state['usb_identity']=tracker.identity
                port=tracker.resolve();mac=runner.reset_esp(a.target,port,folder/'before_flash_reset.log')
                runner.must(normalized_serial(mac)==normalized_serial(tracker.identity['serial_number']),'USB/MAC mismatch')
                time.sleep(1);port=tracker.resolve()
                output=runner.execute([sys.executable,ROOT/'esp32/tools/flash.py','--target',a.target,'--case',str(a.case),'--port',port],folder/'flash.log')
                runner.must('Hash of data verified.' in output and mac in output.lower(),'Missing flash verification')
                port=tracker.resolve();runner.reset_esp(a.target,port,folder/'reset.log',mac)
                time.sleep(1);port=tracker.resolve();state['mac']=mac
                firmware=json.loads((ROOT/'esp32/firmware/manifest.json').read_text())['images'][f'{a.target}_case{a.case}']
                state['firmware_sha256']=firmware['sha256']
            with (folder/'transport.jsonl').open('w') as trace,(folder/'predictions.jsonl').open('w') as preds:
                link=mega_runner.MegaLink(port,trace) if a.target=='mega' else SerialLink(port,trace)
                state['physical_execution']=link.transport=='physical_serial';state['transport']=link.transport
                state['port']=port;state['device']=mega_runner.handshake(link,firmware) if a.target=='mega' else runner.hello(link,g,a.target)
                runner.must(a.target!='mega' or state['device'].get('energy_protocol')==1,'Mega firmware has no energy-count protocol')
                state['energy_wire_protocol']='mega-energy-1' if a.target=='mega' else 'architecture-v2-legacy-E'
                state['post_window_sink_reported']=a.target=='mega'
                state['post_window_core_reported']=False
                state['probe']=transport_probe(link)
                if a.target=='mega':
                    correct,bits=mega_runner.verify(link,g,a.variant,preds);state['correctness']=[correct]
                    last=g['records'][g['benchmark_indices'][-1]]['image_id'];sinks={a.variant:bits[last][0]}
                else:state['correctness'],sinks,_=runner.verify(link,g,variants,preds)
                load_batch(link,g)
                # A short warm-up is retained in the raw trace, excluded from the experiment.
                for v in variants:
                    step=dict(seq=1900000+v,variant=v,duration_ms=3000)
                    link.send(f'E {step["seq"]} {v} 3000')
                    armed=link.receive();end=link.receive(timeout=40)
                    validate_window(armed,end,step,a.target,sinks[v])
                print('READY. In FNIRSI UsbMeter: start recording VBUS/IBUS/PBUS at 100 sps; start/stop current = 0.',flush=True)
                input('After the meter has recorded at least 5 seconds, press Enter here. ')
                origin=time.monotonic_ns();state['recording_origin_monotonic_ns']=origin
                state['recording_origin_utc']=runner.utc()
                state['windows']=capture_windows(link,steps,a.target,sinks,folder,origin)
                state.update(status='DEVICE_COUNTS_CAPTURED',end_utc=runner.utc())
                print('Wait 5 seconds, then stop and SAVE the meter recording. Keep its original file; do not replace it by a screenshot.',flush=True)
    except BaseException as e:
        state.update(status='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAIL',error=str(e),end_utc=runner.utc());raise
    finally:
        if link:link.close()
        state['file_sha256']={p.name:runner.sha(p) for p in folder.iterdir() if p.is_file() and p.name!='energy_summary.json'}
        runner.save(folder/'energy_summary.json',state)
        print('DEVICE COUNTS archive:',snapshot(folder),flush=True)
    print('Electrical energy is PENDING: attach the native power-meter recording as well as this ZIP.',flush=True)


if __name__=='__main__':main()
