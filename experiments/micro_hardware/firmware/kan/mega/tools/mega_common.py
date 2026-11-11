"""Package integrity and the frozen Mega protocol constants."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=[0,4,5,8,9]
RUNNER_VERSION='2026-11-09.mega-architecture-v2'
PROBE_LENGTHS=[1,63,64,65,127,255,384]
TIMER_HZ=2000000

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_hex(path):
    memory={};base=0;ended=False
    for line in Path(path).read_text().splitlines():
        if not line:continue
        if ended or not line.startswith(':'):raise ValueError('Malformed HEX record/order')
        row=bytes.fromhex(line[1:])
        if len(row)<5 or len(row)!=row[0]+5 or sum(row)&255:raise ValueError('HEX checksum/length')
        length,address,kind=row[0],int.from_bytes(row[1:3],'big'),row[3];data=row[4:-1]
        if kind==0:
            for i,value in enumerate(data):
                at=base+address+i
                if at>=253952 or at in memory:raise ValueError('HEX overlaps bootloader or duplicate byte')
                memory[at]=value
        elif kind==1:
            if length or address:raise ValueError('Invalid HEX EOF')
            ended=True
        elif kind in [2,4]:
            if length!=2 or address:raise ValueError('Invalid HEX address record')
            base=int.from_bytes(data,'big')<<(4 if kind==2 else 16)
        else:raise ValueError('Unsupported HEX record type')
    if not ended or not memory or 0 not in memory:raise ValueError('Incomplete HEX')
    return memory

def preflight(case,variant,root=ROOT):
    m=json.loads((root/'firmware/manifest.json').read_text())
    row=m['images'][f'case{case}_v{variant}']
    if m['chip']!='ATmega2560' or m['signature']!='1e9801':raise ValueError('Wrong package target')
    hexfile=root/row['hex_path'];golden=root/f'reference/model{case}_golden.json'
    if sha(hexfile)!=row['hex_sha256'] or sha(golden)!=row['golden_sha256']:raise ValueError('Firmware/golden SHA mismatch')
    if len(read_hex(hexfile))!=row['flash_bytes']:raise ValueError('HEX image size mismatch')
    g=json.loads(golden.read_text())
    if (g['artifact']!=row['source_artifact'] or g['model']!=row['model'] or
        g['variants'][variant]!=row['variant_name'] or len(g['records'])!=1024):raise ValueError('Model identity/count mismatch')
    for name,digest in m.get('tool_sha256',{}).items():
        if sha(root/name)!=digest:raise ValueError('Tool SHA mismatch: '+name)
    if m.get('tool_sha256') is None:raise ValueError('Missing release tool hashes')
    return m,row,g

