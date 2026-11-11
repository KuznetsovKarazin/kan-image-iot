"""Inspect actual ESP ELFs for shared kernels and the expected cycle-counter reads."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=[]
    for target in ['s3','c3']:
        toolchain='toolchain-xtensa-esp32s3' if target=='s3' else 'toolchain-riscv32-esp'
        prefix='xtensa-esp32s3-elf-' if target=='s3' else 'riscv32-esp-elf-'
        tools=Path.home()/'.platformio/packages'/toolchain/'bin'
        compiler=subprocess.check_output([str(tools/(prefix+'gcc')),'--version'],text=True).splitlines()[0]
        for case in range(3):
            env=f'{target}_case{case}';elf=ROOT/f'esp32/.pio/build/{env}/firmware.elf'
            symbols=subprocess.check_output([str(tools/(prefix+'nm')),'-S',str(elf)],text=True)
            shared=[s for s in symbols.splitlines() if s.split()[-1].startswith('table_layer')]
            assert len(shared)==1 and shared[0].split()[-1]=='table_layer',(env,shared)
            code=subprocess.check_output([str(tools/(prefix+'objdump')),'-d',str(elf)],text=True)
            counter='rsr.ccount' if target=='s3' else '0x7e2'
            assert counter in code,(env,counter)
            rows.append(dict(environment=env,compiler=compiler,elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
                             single_shared_table_layer=shared[0],cycle_counter_instruction=counter))
    report=dict(status='PASS_LINKED_ESP_CHECKS',physical_execution=False,
                platformio_core=importlib.metadata.version('platformio'),platform='espressif32@6.9.0',
                framework='Arduino ESP32 2.0.17 / IDF 4.4.7; package 3.20017.241212+sha.dcc1105b',
                elf_to_image_tool='PlatformIO tool-esptoolpy 2.40900.250804 / esptool 4.9.0',
                merge_and_image_check_tool='esptool '+importlib.metadata.version('esptool'),
                scope='Linked code inspection only; ESP32-C3 reads the IDF CPU performance counter PCCR_MACHINE (CSR 0x7e2), not mcycle.',
                images=rows)
    (ROOT/'evidence/v2/linked_esp_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
