"""Flash a checked Mega image with AVRDUDE signature check and readback verification."""
import argparse
from pathlib import Path
import re
import subprocess
from mega_common import ROOT,VARIANTS,preflight
from setup_tools import ensure_avrdude

def command(exe,config,image,port):
    return [str(exe),'-C',str(config),'-p','m2560','-c','wiring','-P',port,'-b','115200','-D',
            '-U','flash:w:'+str((ROOT/image['hex_path']).resolve())+':i']

def flash(case,variant,port,log,exe=None,config=None):
    _,image,_=preflight(case,variant)
    if (exe is None)!=(config is None):raise ValueError('Provide both AVRDUDE executable and config')
    if exe is None:exe,config=ensure_avrdude()
    argv=command(exe,config,image,port)
    print(f"Flashing case{case} / {image['variant_name']}; replacing the current program.",flush=True)
    with Path(log).open('x',encoding='utf-8') as handle:
        result=subprocess.run(argv,stdout=handle,stderr=subprocess.STDOUT,timeout=180)
    text=Path(log).read_text(encoding='utf-8',errors='replace')
    if result.returncode:raise RuntimeError(f'AVRDUDE failed ({result.returncode}); see {log}')
    if not re.search(r'Device signature\s*=\s*0x1e9801\b',text,re.I):raise RuntimeError('Missing ATmega2560 signature confirmation')
    verified=re.findall(r'(\d+) bytes of flash verified',text)
    if not verified or int(verified[-1])!=image['flash_bytes']:raise RuntimeError('Missing/incorrect flash readback verification')
    return dict(status='PASS',argv=argv,signature='1e9801',verified_flash_bytes=int(verified[-1]),
                artifact=image['artifact'],hex_sha256=image['hex_sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',required=True)
    p.add_argument('--case',type=int,choices=[0,1,2],required=True);p.add_argument('--variant',type=int,choices=VARIANTS,required=True)
    p.add_argument('--log',type=Path,required=True);p.add_argument('--avrdude',type=Path);p.add_argument('--avrdude-conf',type=Path)
    args=p.parse_args();flash(args.case,args.variant,args.port,args.log,args.avrdude,args.avrdude_conf)

