"""Synthetic transport fixtures and analytic integration cases; no device results."""
from pathlib import Path
import contextlib,copy,io,json,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import run_mlp as runner
from serial_common import SerialLink,load_batch,transport_probe,prediction_check
from run_energy import plan,capture_windows,validate_window
from audit_energy import audit
from integrate_energy import integrate
from analyze_inference import metrics,paired
from unittest.mock import patch
import integrate_energy

class Fixture(SerialLink):
    transport='synthetic_test_fixture'
    def __init__(self,target,g,image,trace):
        self.target=target;self.g=g;self.image=image;self.trace=trace;self.queue=[];self.calls=0
        self.log('port_open',port='UNIT-TEST',dtr=False,rts=False,control_lines_configured_before_open=True)
    def send(self,command):
        assert not self.queue;self.log('tx',command=command);p=command.split();g=self.g
        if p[0]=='H':
            s3=self.target=='s3';mega=self.target=='mega'
            h=dict(kind='hello',protocol=2 if mega else 4,model=g['model'],artifact=self.image['artifact'],
                   input_dim=g['input_dim'],variants=2,energy_protocol=1,cpu_mhz=16 if mega else 240 if s3 else 160,
                   chip='ATmega2560' if mega else 'ESP32-S3' if s3 else 'ESP32-C3',psram_bytes=0,
                   task_core=1 if s3 else 0,task_affinity=1 if s3 else 0,radios_started=False,
                   serial_transport='avr-uart0-v1' if mega else 'esp-idf-usb-serial-jtag-v4')
            if mega:h['variant']=self.image['variant']
            self.queue=[h]
        elif p[0]=='P':self.queue=[dict(kind='probe',seq=int(p[1]),payload=p[2])]
        elif p[0]=='V':
            seq,v=int(p[1]),int(p[2]);r=g['records'][(seq-1)%1024];assert p[3:]==r['input_bits']
            self.queue=[dict(kind='pred',seq=seq,variant=v,bits=r['expected_bits'][v])]
        elif p[0]=='L':self.queue=[dict(kind='loaded',slot=int(p[1]))]
        elif p[0]=='E':
            seq,v,duration=map(int,p[1:]);last=g['records'][g['benchmark_indices'][-1]]
            end=dict(kind='energy_end',seq=seq,variant=v,energy_protocol=1,n=0 if v==2 else 32000,
                     sink='00000000' if v==2 else last['expected_bits'][v][0])
            if self.target=='mega':end.update(elapsed_ticks=duration*2000,timer_hz=2000000,untouched_sram_gap_bytes=512,scope='whole_board_polling_loop')
            else:end.update(elapsed_us=duration*1000,scheduler_yields=1000,core_end=1 if self.target=='s3' else 0,scope='whole_board_cooperative_loop')
            self.queue=[dict(kind='energy_armed',seq=seq,variant=v,energy_protocol=1,lead_ms=2000,duration_ms=duration),end]
        else:raise AssertionError(command)
    def receive(self,**kwargs):
        r=self.queue.pop(0);raw=(json.dumps(r)+'\n').encode()
        for i in range(0,len(raw),31):
            part=raw[i:i+31];self.log('rx_chunk',raw_hex=part.hex(),n_bytes=len(part))
        self.log('response',response=r);return r

def rejects(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Invalid input accepted')

def main():
    tested=[]
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        for target in ['s3','c3','mega']:
            for case in range(3):
                for variants in ([[0],[1]] if target=='mega' else [[0,1]]):
                    folder=Path(tmp)/f'{target}_{case}_{variants[0]}';folder.mkdir()
                    gp=ROOT/(f'mega/reference/model{case}_golden.json' if target=='mega' else f'esp32/generated/model{case}_golden.json')
                    g=json.loads(gp.read_text())
                    image=json.loads((ROOT/('mega/firmware/manifest.json' if target=='mega' else 'esp32/firmware/manifest.json')).read_text())['images'][f'case{case}_v{variants[0]}' if target=='mega' else f'{target}_case{case}']
                    s=dict(status='DEVICE_COUNTS_CAPTURED',physical_execution=False,transport='synthetic_test_fixture',target=target,case=case,variants=variants,
                           start_utc=runner.utc(),release_sha256=runner.sha(ROOT/'MANIFEST.json'),runner_sha256=runner.sha(ROOT/'tools/run_energy.py'),
                           artifact=g['artifact'],golden_sha256=runner.sha(gp),firmware_sha256=image['hex_sha256'] if target=='mega' else image['sha256'],
                           plan=plan(variants,1,10,10,case),repetitions=1,seconds=10,baseline_seconds=10)
                    with (folder/'transport.jsonl').open('w') as trace,(folder/'predictions.jsonl').open('w') as preds:
                        link=Fixture(target,g,image,trace);link.send('H');s['device']=link.receive();transport_probe(link)
                        _,sinks,_=runner.verify(link,g,variants,preds);load_batch(link,g)
                        for v in variants:
                            step=dict(seq=1900000+v,variant=v,duration_ms=3000);link.send(f'E {step["seq"]} {v} 3000')
                            validate_window(link.receive(),link.receive(),step,target,sinks[v])
                        origin=time.monotonic_ns();s['recording_origin_monotonic_ns']=origin
                        s['windows']=capture_windows(link,s['plan'],target,sinks,folder,origin)
                    s['end_utc']=runner.utc();s['file_sha256']={p.name:runner.sha(p) for p in folder.iterdir()}
                    runner.save(folder/'energy_summary.json',s)
                    report=audit(folder,require_physical=False);assert len(report['windows'])==len(s['plan'])
                    rejects(lambda:audit(folder))
                    if target=='s3' and case==0:
                        # End-to-end integration with known constant power and
                        # explicit synthetic audit substitution, never published.
                        native=folder/'synthetic_meter_recording.txt';native.write_text('SYNTHETIC TEST FIXTURE, not instrument data\n')
                        power=folder/'synthetic_power.csv'
                        power.write_text('time_s,voltage_v,current_a\n'+''.join(f'{j/10:.1f},5,0.4\n' for j in range(800)))
                        positions=[dict(seq=w['seq'],start_s=2+12*i,end_s=12+12*i) for i,w in enumerate(report['windows'])]
                        a=dict(physical_measurement=True,meter_model='SYNTHETIC',meter_serial='UNIT_TEST',
                               source_recording_path=native.name,source_recording_sha256=runner.sha(native),power_csv_sha256=runner.sha(power),
                               conversion_description='Synthetic constant-power analytic fixture',alignment_method='Known synthetic time coordinates',
                               calibration_note='No physical meter',max_sample_gap_s=.1,boundary_uncertainty_s=.02,windows=positions)
                        ap=folder/'alignment.json';ap.write_text(json.dumps(a))
                        with patch.object(integrate_energy,'audit',return_value=report):
                            # Floating timestamp representation can straddle 0.1;
                            # fixture uses 20 Hz for the declared 0.1-second gate.
                            power.write_text('time_s,voltage_v,current_a\n'+''.join(f'{j/20:.2f},5,0.4\n' for j in range(1600)))
                            a['power_csv_sha256']=runner.sha(power);ap.write_text(json.dumps(a))
                            result=integrate_energy.analyze(folder,power,ap)
                            assert all(abs(w['whole_board_j_per_inference']-20/32000)<1e-12 for w in result['active'])
                            assert all(abs(w['incremental_j_per_inference'])<1e-12 and w['review_required'] for w in result['active'])
                            a['windows'][0]['end_s']+=.5;ap.write_text(json.dumps(a))
                            rejects(lambda:integrate_energy.analyze(folder,power,ap))
                    bad=copy.deepcopy(s['windows'][1]['device']);bad['n']=1
                    step=s['plan'][1];rejects(lambda:validate_window(s['windows'][1]['armed'],bad,step,target,sinks[step['variant']]))
                    tested.append(dict(target=target,case=case,variants=variants,raw_transport_audit=True,synthetic_rejected_as_physical=True))
    # Analytic integrals: 2 W for 2 s = 4 J, and a linear 0..2 W ramp = 2 J.
    assert abs(integrate([(i/10,2.) for i in range(31)],.5,2.5,.101)['joules']-4)<1e-12
    assert abs(integrate([(i/10,i/10) for i in range(21)],0,2,.101)['joules']-2)<1e-12
    rejects(lambda:integrate([(0,2.),(2,2.)],0,2,.1))
    rejects(lambda:integrate([(0,2.),(.1,2.)],-.1,.1,.1))
    # One-ULP INT8 tampering must fail even when the FP32 tolerance would pass.
    rejects(lambda:prediction_check(dict(kind='pred',seq=1,variant=1,bits=['3f800001','40000000']),1,1,['3f800000','40000000'],dict(atol=2e-5,rtol=2e-5)))
    assert metrics([0,0,1,1],[0,1,1,1])['confusion_rows_true_columns_predicted']==[[1,1],[0,2]]
    assert paired([0,0,1,1],[0,1,1,0],[0,0,1,1])['candidate_only_correct']==2
    result=dict(status='PASS_SYNTHETIC_AND_ANALYTIC',physical_execution=False,fixtures=tested,
                checks=['constant-power integral','linear-power integral','whole-board and idle-subtracted energy end-to-end analytic fixture',
                        'misaligned duration rejected','gaps rejected','uncovered intervals rejected','INT8 one-ULP corruption rejected','classification accounting'])
    (ROOT/'evidence/energy_contract_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
