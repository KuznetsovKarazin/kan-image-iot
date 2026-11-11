"""Flash a checksum-verified merged image. This replaces the board's program."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',required=True)
    p.add_argument('--target',choices=['s3','c3'],required=True)
    p.add_argument('--case',type=int,choices=[0,1,2],default=0)
    p.add_argument('--check-only',action='store_true',help='Check files only, without connecting or writing')
    a=p.parse_args()
    manifest=json.loads((ROOT/'firmware/manifest.json').read_text())
    key=f'{a.target}_case{a.case}'
    if key not in manifest['images']:
        p.error('This package does not include '+key+'; the starter ZIP contains only case 0.')
    row=manifest['images'][key];path=ROOT/row['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Firmware checksum mismatch')
    print('Verified:',key,row['model'],row['artifact'])
    if a.check_only:return
    chip='esp32s3' if a.target=='s3' else 'esp32c3'
    base=[sys.executable,'-m','esptool','--chip',chip,'--port',a.port,'--baud','460800']
    info=subprocess.run(base+['--after','no_reset','flash_id'],capture_output=True,text=True,check=True)
    print(info.stdout)
    m=re.search(r'Detected flash size:\s*(\d+)\s*MB',info.stdout,re.I)
    if not m or int(m.group(1))<4:
        raise RuntimeError('Could not verify at least 4 MB of flash. Do not flash this profile blindly.')
    print('Writing the selected benchmark firmware; the current program will be replaced.')
    subprocess.run(base+['write_flash','--flash_mode','dio','--flash_freq','80m',
        '--flash_size','4MB','0x0',str(path)],check=True)
    print('Flashed. Recheck the COM port before starting serial_runner.py.')


if __name__=='__main__':main()
