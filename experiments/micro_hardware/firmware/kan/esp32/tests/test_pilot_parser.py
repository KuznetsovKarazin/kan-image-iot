"""Actual firmware parser + new host runner over fragmented pipes, no device measurements."""
import contextlib, io, json, subprocess, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
DEVICE=Path(__file__).resolve().parents[1];ROOT=DEVICE.parent
sys.path.insert(0,str(ROOT/'tools'))
import run_architecture as run
from test_serial_transport import FragmentedFirmwareSerial
case=int(sys.argv[1]) if len(sys.argv)>1 else 0
g=json.loads((DEVICE/f'generated/model{case}_golden.json').read_text())
out=ROOT/'host_build';out.mkdir(exist_ok=True);obj=out/f'kernel-parser-{case}.o';exe=out/f'parser-{case}'
subprocess.run(['gcc','-O3','-ffp-contract=off',f'-DMODEL_CASE={case}','-I'+str(DEVICE/'include'),'-I'+str(DEVICE/'generated'),
                '-c',str(DEVICE/'src/kernels.c'),'-o',str(obj)],check=True)
subprocess.run(['g++','-std=c++17','-O3','-ffp-contract=off','-Wall','-Wextra','-Werror','-pthread',
                '-I'+str(DEVICE/'tests/stubs'),'-I'+str(DEVICE/'include'),str(DEVICE/'src/main.cpp'),
                str(DEVICE/'tests/stubs/host_main.cpp'),str(obj),'-lm','-o',str(exe)],check=True)
fake=FragmentedFirmwareSerial(exe)
try:
    with patch.dict(sys.modules,{'serial':types.SimpleNamespace(Serial=lambda *a,**k:fake)}),patch.object(run.time,'sleep'):
        link=run.SerialLink('HOST-PIPE',io.StringIO());link.transport='host_protocol_stub'
        link.send('H');r=link.receive(startup=True)
        assert r['chip']=='HOST-MOCK' and r['protocol']==4 and r['variants']==12
        assert r['artifact']==g['artifact'] and r['dram_copy_ok'] is True and r['dram_copy_bytes']==g['dram_total_bytes'] and r['dram_fp32_bytes']==g['dram_fp32_bytes'] and r['dram_int8_bytes']==g['dram_int8_bytes'] and r['input_dim']==g['input_dim']
        link.send('B 1 8 1');link.receive()
        raise AssertionError('Incomplete batch was accepted')
except ValueError as e:
    assert 'batch_incomplete' in str(e)
finally:
    fake.close()
fake=FragmentedFirmwareSerial(exe)
try:
    with patch.dict(sys.modules,{'serial':types.SimpleNamespace(Serial=lambda *a,**k:fake)}),patch.object(run.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
        link=run.SerialLink('HOST-PIPE',io.StringIO());link.transport='host_protocol_stub'
        assert run.transport_probe(link)['round_trips']==21
        stats,sinks,seq=run.verify(link,g,[0,4,5,8,9,10,11],io.StringIO())
        run.load_batch(link,g)
        for v in [0,4,5,8,9,10,11,12]:
            seq+=1;link.send(f'B {seq} {v} 1');r=link.receive()
            assert r['n']==32 and r['seq']==seq and r['variant']==v and r['sink']==sinks[v] and r['complete'] is True
        assert seq==7176 and all(s['n']==1024 for s in stats)
finally:fake.close()
class BadLink:
    def __init__(self,*a):self.commands=[];self.closed=False
    def send(self,s):self.commands.append(s)
    def receive(self,*a,**k):raise ValueError('Malformed complete serial frame')
    def close(self):self.closed=True
bad=BadLink()
with tempfile.TemporaryDirectory() as tmp,patch.object(run,'SerialLink',return_value=bad):
    try:run.capture_esp('s3','MOCK','MOCK',Path(tmp)/'failed',1)
    except ValueError:pass
    else:raise AssertionError('Bad stream passed')
    state=json.loads((Path(tmp)/'failed/summary.json').read_text())
    assert state['status']=='FAIL' and bad.closed and bad.commands==['H']
    assert not (Path(tmp)/'failed/latency_blocks.json').exists()
report=dict(case=case,status='PASS_ACTUAL_PARSER_ON_HOST',physical_execution=False,host_stub=True,
    predictions=7168,fragmented_reads=fake.calls,benchmark_contract_frames=8,
    bad_stream_marks_failure=True,scope='Parser/transport/numerics only; host placeholder clock is not MCU timing evidence')
(ROOT/f'evidence/v2/parser_case{case}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
