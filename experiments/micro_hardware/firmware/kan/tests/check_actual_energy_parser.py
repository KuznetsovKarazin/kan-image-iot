"""Exercise actual ESP C++ energy command implementation over fragmented pipes.
Host-only parser/numeric test, never MCU performance evidence."""
import contextlib,io,json,subprocess,sys,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'esp32'
sys.path[:0]=[str(ROOT/'tools'),str(E/'tests')]
import run_kan as run
from serial_common import load_batch
from test_serial_transport import FragmentedFirmwareSerial
from run_energy import validate_window
case=2;g=json.loads((E/f'generated/model{case}_golden.json').read_text())
out=ROOT/'host_build';out.mkdir(exist_ok=True);obj=out/'energy-kernel.o';exe=out/'energy-parser'
subprocess.run(['gcc','-O3','-ffp-contract=off',f'-DMODEL_CASE={case}','-I'+str(E/'include'),'-I'+str(E/'generated'),'-c',str(E/'src/kernels.c'),'-o',str(obj)],check=True)
subprocess.run(['g++','-std=c++17','-O3','-ffp-contract=off','-Wall','-Wextra','-Werror','-pthread','-I'+str(E/'tests/stubs'),'-I'+str(E/'include'),str(E/'src/main.cpp'),str(E/'tests/stubs/host_main.cpp'),str(obj),'-lm','-o',str(exe)],check=True)
fake=FragmentedFirmwareSerial(exe);rows=[]
try:
 with patch.dict(sys.modules,{'serial':types.SimpleNamespace(Serial=lambda *a,**k:fake)}),patch.object(run.time,'sleep'):
  link=run.SerialLink('HOST-PIPE',io.StringIO());link.transport='host_protocol_stub'
  link.send('H');h=link.receive(startup=True);assert 'energy_protocol' not in h and h['variants']==12
  link.send('E 1 4 1000')
  try:link.receive()
  except ValueError as e:assert 'batch_incomplete' in str(e)
  else:raise AssertionError('Incomplete batch accepted')
  load_batch(link,g)
  for seq,v in enumerate([0,4,5,8,9,10,11,12],2):
   step=dict(seq=seq,variant=v,duration_ms=1000)
   link.send(f'E {seq} {v} 1000');armed=link.receive();end=link.receive(timeout=8)
   sink=g['records'][g['benchmark_indices'][-1]]['expected_bits'][v][0] if v!=12 else None
   metrics=validate_window(armed,end,step,'s3',sink);rows.append(dict(variant=v,**metrics))
finally:fake.close()
report=dict(status='PASS_ORIGINAL_ESP_ENERGY_PARSER_ON_HOST',physical_execution=False,case=case,checked_variants=rows,scope='Actual C++ parser + C inference kernels; host stubs, not ESP timing or power')
(ROOT/'evidence/actual_energy_parser.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'])
