"""Exercise the real Mega runner and independent audit using synthetic UART replies."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mega/tools'))
import mega_runner
from serial_common import SerialLink
sys.path.insert(0,str(ROOT/'tools'))
from audit_v2_mega import audit_one


class Package:
    def __getitem__(self,name):return (ROOT/name).read_bytes()


class SyntheticMega(SerialLink):
    transport='synthetic_test_fixture'
    def __init__(self,port,trace,image,golden):
        self.trace=trace;self.image=image;self.g=golden;self.command=None
        self.log('port_open',port=port,dtr=False,rts=False,control_lines_configured_before_open=True)
        self.log('dtr_reset',pulse_seconds=.1,settle_seconds=3.,power_cycle=False)
    def close(self):pass
    def send(self,command):
        assert self.command is None;self.command=command;self.log('tx',command=command)
    def receive(self,**_):
        p=self.command.split();image=self.image;g=self.g;variant=image['variant']
        if p[0]=='H':
            r=dict(kind='hello',protocol=2,chip='ATmega2560',cpu_mhz=16,sram_bytes=8192,flash_bytes=262144,
                   input_dim=g['input_dim'],variant=variant,model=g['model'],variant_name=g['variants'][variant],
                   artifact=image['artifact'],source_artifact=g['artifact'],parameter_bytes=image['parameter_bytes'],
                   timer_hz=2000000,serial_transport='avr-uart0-v1',untouched_sram_gap_bytes=1024)
        elif p[0]=='P':r=dict(kind='probe',seq=int(p[1]),payload=p[2])
        elif p[0]=='V':
            seq=int(p[1]);assert int(p[2])==variant
            record=g['records'][seq-1];assert p[3:]==record['input_bits']
            r=dict(kind='pred',seq=seq,variant=variant,bits=record['expected_bits'][variant])
        elif p[0]=='L':
            slot=int(p[1]);assert p[2:]==g['records'][g['benchmark_indices'][slot]]['input_bits']
            r=dict(kind='loaded',slot=slot)
        elif p[0]=='B':
            seq,v,rounds=map(int,p[1:]);assert v in [variant,2]
            last=g['records'][g['benchmark_indices'][-1]]
            r=dict(kind='bench',seq=seq,variant=v,rounds=rounds,n=32*rounds,timer_hz=2000000,
                   complete=True,sink=last['input_bits'][0] if v==2 else last['expected_bits'][v][0],
                   elapsed_ticks=(16 if v==2 else 100000)*rounds,untouched_sram_gap_bytes=1024)
        else:raise AssertionError(p)
        raw=(json.dumps(r)+'\n').encode('ascii')
        for start in range(0,len(raw),31):
            chunk=raw[start:start+31];self.log('rx_chunk',raw_hex=chunk.hex(),n_bytes=len(chunk))
        self.log('response',response=r);self.command=None
        return r


def main():
    rows=[]
    images=json.loads((ROOT/'mega/firmware/manifest.json').read_bytes())['images']
    with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
        for key,image in images.items():
            g=json.loads((ROOT/f'mega/reference/model{image["case"]}_golden.json').read_bytes())
            folder=Path(temporary)/key
            args=SimpleNamespace(case=image['case'],variant=image['variant'],out=folder,port='UNIT-TEST',board_id='UNIT_TEST')
            mega_runner.run_capture(args,link_factory=lambda port,trace:SyntheticMega(port,trace,image,g))
            raw={name:(folder/name).read_bytes() for name in ['summary.json','predictions.jsonl','latency_blocks.json','transport.jsonl']}
            report=audit_one(raw,Package(),require_physical=False)
            assert report['physical_execution'] is False
            try:audit_one(raw,Package())
            except ValueError as error:assert 'Simulator data' in str(error)
            else:raise AssertionError('Synthetic data accepted as board data')
            changed=dict(raw);summary=json.loads(changed['summary.json']);summary['device']['input_dim']-=1
            changed['summary.json']=json.dumps(summary).encode()
            try:audit_one(changed,Package(),require_physical=False)
            except ValueError:pass
            else:raise AssertionError('Changed input dimension accepted')
            rows.append(dict(image=key,exchanges=report['exchanges'],prediction_checks=report['prediction_checks'],
                             synthetic_data_rejected_by_physical_audit=True,changed_input_dim_rejected=True))
    result=dict(status='PASS_SYNTHETIC_MEGA_CONTRACT',physical_execution=False,images=rows)
    (ROOT/'evidence/mega_contract_tests.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
