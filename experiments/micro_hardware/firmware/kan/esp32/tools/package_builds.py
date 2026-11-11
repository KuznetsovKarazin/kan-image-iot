"""Merge, checksum and inspect cross-compiled ESP32 binaries; no flashing."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=ROOT/'firmware';dest.mkdir(exist_ok=True)
    images={};report=[]
    for target in ['s3','c3']:
        chip='esp32s3' if target=='s3' else 'esp32c3'
        toolchain=Path.home()/'.platformio/packages'/('toolchain-xtensa-esp32s3' if target=='s3' else 'toolchain-riscv32-esp')/'bin'
        prefix='xtensa-esp32s3-elf-' if target=='s3' else 'riscv32-esp-elf-'
        for case in range(3):
            env=f'{target}_case{case}';build=ROOT/'.pio/build'/env
            golden=json.loads((ROOT/f'generated/model{case}_golden.json').read_text())
            link_map=(build/'firmware.map').read_text()
            patched_object=f'.pio/build/{env}/src/usb_serial_jtag_compat.c.o'
            if 'LOAD '+patched_object not in link_map or 'libdriver.a(usb_serial_jtag.c.obj)' in link_map:
                raise AssertionError('Expected the patched USB driver, not the SDK archive implementation')
            if golden['artifact'].encode('ascii')+b'\0' not in (build/'firmware.bin').read_bytes():
                raise AssertionError('Firmware does not contain the expected artifact ID; rebuild it')
            merged=dest/f'{env}_merged.bin'
            cmd=[sys.executable,'-m','esptool','--chip',chip,'merge_bin','-o',str(merged),
                 '--flash_mode','dio','--flash_freq','80m','--flash_size','4MB',
                 '0x0',str(build/'bootloader.bin'),'0x8000',str(build/'partitions.bin'),
                 '0x10000',str(build/'firmware.bin')]
            subprocess.check_output(cmd,text=True)
            info=subprocess.check_output([sys.executable,'-m','esptool','--chip',chip,'image_info',str(build/'firmware.bin')],text=True)
            (ROOT/f'results/{env}_image_info.txt').write_text(info)
            if not re.search(r'^Checksum: .+ \(valid\)$',info,re.M) or not re.search(r'^Validation Hash: [0-9a-f]+ \(valid\)$',info,re.M):
                raise AssertionError('Application image checksum/hash did not validate')
            combined=merged.read_bytes()
            for offset,name in [(0,'bootloader.bin'),(0x8000,'partitions.bin'),(0x10000,'firmware.bin')]:
                raw=(build/name).read_bytes()
                if combined[offset:offset+len(raw)]!=raw:
                    raise AssertionError(f'Merged image differs from {name} at {offset:#x}')
            size_text=subprocess.check_output([str(toolchain/(prefix+'size')),str(build/'firmware.elf')],text=True)
            nums=size_text.splitlines()[1].split();text,data,bss=map(int,nums[:3])
            sections_text=subprocess.check_output([str(toolchain/(prefix+'size')),'-A',str(build/'firmware.elf')],text=True)
            sections={}
            for line in sections_text.splitlines():
                fields=line.split()
                if len(fields)==3 and fields[0].startswith('.') and fields[1].isdigit():
                    sections[fields[0]]=int(fields[1])
            (ROOT/f'results/{env}_sections.txt').write_text(sections_text)
            dram=sum(sections.get(k,0) for k in ['.dram0.data','.dram0.bss','.noinit'])
            iram=sum(size for name,size in sections.items() if name.startswith('.iram0.'))
            row={'path':str(merged.relative_to(ROOT)),'sha256':sha(merged),'bytes':merged.stat().st_size,
                 'chip':chip,'case':case,'model':golden['model'],'artifact':golden['artifact'],
                 'serial_transport':golden['serial_transport'],'usb_idle_rearm_patch_linked':True,
                 'app_sha256':sha(build/'firmware.bin'),'bootloader_sha256':sha(build/'bootloader.bin'),
                 'app_checksum_valid':True,'app_validation_hash_valid':True,'merged_segments_identical':True,
                 'partition_sha256':sha(build/'partitions.bin'),'flash_mode':'dio','flash_mhz':80,'flash_layout_mb':4,
                 'app_binary_bytes':(build/'firmware.bin').stat().st_size,
                 'elf_gnu_size_text':text,'elf_gnu_size_data':data,'elf_gnu_size_bss':bss,
                 'static_dram_data_bss_noinit_bytes':dram,'iram_sections_bytes':iram,
                 'memory_scope':'static DRAM sections only; excludes IRAM, heap, stacks, RTOS dynamic allocations; GNU data includes flash constants and is not SRAM',
                 'dram_copy_bytes_requested':golden['dram_total_bytes'] if target=='s3' else 0,
                 'scope':'all 12 variant IDs linked together; both DRAM variants available only on S3, not one-variant deployment size'}
            if row['app_binary_bytes']>0x300000:raise AssertionError('App exceeds declared partition')
            images[env]=row;report.append({'environment':env,**row})
    (dest/'manifest.json').write_text(json.dumps({'scope':'cross-compiled; physical boards not run','images':images},indent=2))
    (ROOT/'results/cross_builds.json').write_text(json.dumps({'status':'PASS','physical_execution':False,'builds':report},indent=2))
    print('Packaged',len(images),'checksum-verified merged images')


if __name__=='__main__':main()
