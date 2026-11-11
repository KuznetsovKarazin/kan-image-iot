"""Build pinned AVR images; standard Python plus AVR-GCC 7.3.0 / avr-libc 2.0.0."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEFAULT='/root/.platformio/packages/toolchain-atmelavr/bin'
FLAGS=['-mmcu=atmega2560','-DF_CPU=16000000UL','-std=gnu11','-O3','-ffp-contract=off',
       '-fno-strict-aliasing','-ffunction-sections','-fdata-sections','-fstack-usage','-Wall','-Wextra','-Werror']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tool(name):
    configured=os.environ.get('MEGA_AVR_BIN')
    if configured:return str(Path(configured)/('avr-'+name))
    found=shutil.which('avr-'+name)
    return found or str(Path(DEFAULT)/('avr-'+name))
def run(argv):
    result=subprocess.run(argv,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr or result.stdout)
    return result.stdout
def compile_image(case,v):
    gen=ROOT/f'generated/case{case}_v{v}';out=ROOT/f'build/case{case}_v{v}';out.mkdir(parents=True,exist_ok=True)
    include=['-I'+str(ROOT/'src'),'-I'+str(gen),'-include',str(gen/'config.h')]
    objects=[]
    for name in ['kernels','main','startup']:
        flags=[f for f in FLAGS if not (name=='startup' and f=='-fstack-usage')]
        obj=out/(name+'.o');run([tool('gcc'),*flags,*include,'-c',str(ROOT/f'src/{name}.c'),'-o',str(obj)]);objects.append(str(obj))
    elf=out/'firmware.elf';run([tool('gcc'),'-mmcu=atmega2560','-Wl,--gc-sections,-Map,'+str(out/'firmware.map'),*objects,'-lm','-o',str(elf)])
    hexfile=ROOT/f'firmware/mega_case{case}_v{v}.hex'
    run([tool('objcopy'),'-O','ihex','-R','.eeprom',str(elf),str(hexfile)])
    sections={}
    text=run([tool('size'),'-A',str(elf)])
    (out/'size.txt').write_text(text)
    for line in text.splitlines():
        parts=line.split()
        if len(parts)==3 and parts[0].startswith('.'):
            sections[parts[0]]=int(parts[1])
    flash=sections.get('.text',0)+sections.get('.data',0)
    static_sram=sum(sections.get(n,0) for n in ['.data','.bss','.noinit'])
    if flash>253952 or static_sram>6144:raise RuntimeError('Image exceeds Flash or SRAM headroom policy')
    info=json.loads((gen/'export.json').read_text())
    info.update(hex_path=hexfile.relative_to(ROOT).as_posix(),hex_sha256=sha(hexfile),elf_sha256=sha(elf),
                flash_bytes=flash,static_sram_bytes=static_sram,sram_remaining_before_stack=8192-static_sram,
                sections=sections,scope='Single selected variant; static SRAM excludes call stack/ISR; no physical execution')
    return info

def main():
    version=run([tool('gcc'),'--version']).splitlines()[0]
    if '7.3.0' not in version:raise RuntimeError('Pinned AVR-GCC 7.3.0 required')
    images={}
    for case in [0,1,2]:
        for v in [0,1]:
            row=compile_image(case,v);images[f'case{case}_v{v}']=row
            print(f"case{case}/v{v}: Flash {row['flash_bytes']} B, static SRAM {row['static_sram_bytes']} B",flush=True)
    manifest=dict(version='2026-11-09.mega-mlp-v2',chip='ATmega2560',signature='1e9801',clock_hz=16000000,
                  baud=115200,protocol=2,serial_transport='avr-uart0-v1',timer_hz=2000000,
                  compiler=version,compiler_flags=FLAGS,physical_execution=False,images=images)
    manifest['tool_sha256']={p.relative_to(ROOT).as_posix():sha(p)
                            for p in sorted((ROOT/'tools').iterdir()) if p.suffix in ['.py','.json']}
    (ROOT/'firmware/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'results/build_checks.json').write_text(json.dumps(dict(status='PASS',**manifest),indent=2)+'\n')

if __name__=='__main__':main()
