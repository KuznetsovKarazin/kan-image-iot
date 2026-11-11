"""Check linked AVR constants and far function-pointer trampolines from ELF bytes."""
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from elftools.elf.elffile import ELFFile

ROOT=Path(__file__).resolve().parents[1]
AVR=Path.home()/'.platformio/packages/toolchain-atmelavr/bin'


def main():
    images=json.loads((ROOT/'mega/firmware/manifest.json').read_text())['images'];rows=[]
    for key,image in images.items():
        path=ROOT/'mega/build'/key/'firmware.elf'
        with path.open('rb') as handle:
            elf=ELFFile(handle)
            symbols={s.name:s for s in elf.get_section_by_name('.symtab').iter_symbols()}
            def raw(name):
                symbol=symbols[name];section=elf.get_section(symbol['st_shndx'])
                start=symbol['st_value']-section['sh_addr']
                return section.data()[start:start+symbol['st_size']]
            checked=0;highest=0
            for entry in image['arrays']:
                chunks=sorted((n for n in symbols if re.fullmatch(re.escape(entry['name'])+r'_chunk\d+',n)),
                              key=lambda n:int(n.rsplit('chunk',1)[1]))
                value=b''.join(raw(name) for name in chunks)
                assert len(value)==entry['bytes'] and hashlib.sha256(value).hexdigest()==entry['binary_sha256'],(key,entry['name'])
                highest=max(highest,max(symbols[name]['st_value']+symbols[name]['st_size'] for name in chunks))
                checked+=len(value)
            if image['variant']==8:
                cubic=(ROOT/f'mega/generated/{key}/cubic.inc').read_text()
                for name,body in re.findall(r'static const float (\w+)\[\] FLASH_STORAGE=\{(.*?)\};',cubic,re.S):
                    values=[float(s.strip().removesuffix('f')) for s in body.split(',') if s.strip()]
                    expected=struct.pack('<'+'f'*len(values),*values)
                    assert raw(name)==expected,(key,name)
                    checked+=len(expected)
            assert checked==image['parameter_bytes'],(key,checked,image['parameter_bytes'])
            indirect=0;far_indirect=0
            if image['variant'] in [4,5,9]:
                disassembly=subprocess.check_output([str(AVR/'avr-objdump'),'-d',str(path)],text=True)
                first=symbols['__trampolines_start']['st_value'];last=symbols['__trampolines_end']['st_value']
                trampolines={int(a,16):int(b,16) for a,b in re.findall(r'^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){4}jmp\s+0x([0-9a-f]+)',disassembly,re.M)
                             if first<=int(a,16)<last}
                source=(ROOT/f'mega/generated/{key}/model.inc').read_text()
                descriptors=re.search(r'static const TableLayer table_layers\[\]=\{(.*?)\};',source).group(1)
                layers=[x.split(',') for x in re.findall(r'\{([^{}]+)\}',descriptors)]
                data=raw('table_layers');assert len(data)==24*len(layers)
                for i,layer in enumerate(layers):
                    words=struct.unpack_from('<12H',data,24*i)
                    assert list(words[:3])==list(map(int,layer[:3]))
                    for pointer,name in zip(words[3:],layer[3:]):
                        if name=='NULL':assert pointer==0;continue
                        target=symbols[name]['st_value'];address=2*pointer
                        assert address==target or trampolines.get(address)==target,(key,name,address,target)
                        indirect+=1
                        if target>=0x20000:
                            assert trampolines.get(address)==target
                            far_indirect+=1
            rows.append(dict(image=key,parameter_bytes_verified=checked,highest_constant_end_address=highest,
                             indirect_reader_pointers_verified=indirect,readers_above_128KiB_with_verified_trampolines=far_indirect))
    assert any(r['readers_above_128KiB_with_verified_trampolines']>0 for r in rows)
    report=dict(status='PASS_LINKED_AVR_BYTES',physical_execution=False,images=rows,
                scope='Actual ELF constants and initialized reader-pointer relocations; not AVR emulation or physical board measurements.')
    (ROOT/'evidence/v2/linked_avr_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
