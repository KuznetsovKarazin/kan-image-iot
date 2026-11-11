"""Compare every linked parameter array with the frozen trained model export.

Requires NumPy and pyelftools. Cross-build evidence, not board execution.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from elftools.elf.elffile import ELFFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mega/tools'))
from mega_common import preflight


def check(path, case, formats, target):
    export = json.loads((ROOT / f'evidence/model_exports/model{case}.json').read_text())
    checked = []
    with path.open('rb') as handle:
        elf = ELFFile(handle)
        table = elf.get_section_by_name('.symtab')
        symbols = {s.name: s for s in table.iter_symbols()}
        for i, (f, q) in enumerate(zip(export['fp32']['layers'], export['int8']['layers'])):
            arrays = [('w', f['weight'], '<f4', 0), ('b', f['bias'], '<f4', 0),
                      ('qw', q['weight'], 'i1', 1), ('qb', q['bias'], '<i4', 1),
                      ('mult', q['multiplier'], '<i4', 1), ('shift', q['shift'], 'u1', 1)]
            for suffix, values, dtype, fmt in arrays:
                name = f'm{case}l{i}_{suffix}'
                if fmt not in formats:
                    assert name not in symbols, f'Unselected format linked: {name}'
                    continue
                expected = np.asarray(values, dtype=dtype).tobytes()
                sym = symbols[name]
                section = elf.get_section(sym['st_shndx'])
                offset = sym['st_value'] - section['sh_addr']
                actual = section.data()[offset:offset + sym['st_size']]
                assert actual == expected, f'Linked parameter mismatch: {path} / {name}'
                if target == 'mega':
                    assert section.name == '.text'
                    assert sym['st_value'] + sym['st_size'] <= 65536, 'Near-PROGMEM overflow'
                else:
                    assert section.name in ['.flash.rodata', '.flash.rodata_noload'], section.name
                checked.append(dict(symbol=name, bytes=len(actual), section=section.name,
                                    sha256=hashlib.sha256(actual).hexdigest()))
        for other in set(range(3)) - {case}:
            assert not any(n.startswith(f'm{other}l') for n in symbols), 'Other model linked'
    return dict(target=target, case=case, formats=formats, arrays=checked,
                elf_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    rows = []
    for case in range(3):
        for v in range(2):
            _, image, _ = preflight(case, v)
            assert image['flash_bytes'] < 65536, 'Near-address Flash guard'
            rows.append(check(ROOT / f'mega/build/case{case}_v{v}/firmware.elf', case, [v], 'mega'))
        for target in ['s3', 'c3']:
            rows.append(check(ROOT / f'esp32/.pio/build/{target}_case{case}/firmware.elf', case, [0, 1], target))
    result = dict(status='PASS_LINKED_MODEL_BYTES', physical_execution=False, images=rows)
    (ROOT / 'evidence/linked_models.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS:', len(rows), 'ELFs; trained arrays match byte-for-byte; no other cases linked')


if __name__ == '__main__':
    main()
