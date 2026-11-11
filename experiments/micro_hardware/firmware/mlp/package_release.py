"""Freeze a separately built MLP release; never run inside an active batch.

Standard library only. Build artifacts and tests must already have passed.
"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def included(path):
    relative = path.relative_to(ROOT)
    if not path.is_file() or path.name == 'MANIFEST.json':
        return False
    if set(relative.parts) & {'.pio', 'build', '__pycache__', '.venv', 'captures', 'vendor'}:
        return False
    if path.suffix in {'.so', '.pyc', '.zip', '.tmp'}:
        return False
    return relative.parts[0] in {
        'tools', 'tests', 'esp32', 'mega', 'evidence', 'README_RU.txt',
        'requirements.txt', 'package_release.py'
    }


def main():
    for file, status in [('evidence/linked_models.json', 'PASS_LINKED_MODEL_BYTES'),
                         ('evidence/contract_tests.json', 'PASS_SYNTHETIC_CONTRACT_TESTS'),
                         ('evidence/mega_contract_tests.json', 'PASS_SYNTHETIC_MEGA_CONTRACT'),
                         ('esp32/results/cross_builds.json', 'PASS'),
                         ('mega/results/build_checks.json', 'PASS')]:
        report = json.loads((ROOT / file).read_text())
        assert report['status'] == status and report['physical_execution'] is False, file
    files = sorted(p for p in ROOT.rglob('*') if included(p))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = dict(version='mlp-v2', release_date='2026-11-09', physical_execution=False,
                    hardware_seed=101, variants={'0': 'mlp_fp32', '1': 'mlp_int8', '2': 'loop_control'},
                    default_cases={'s3': [0, 1, 2], 'c3': [1, 2], 'mega': [1, 2]},
                    repetitions=3, sha256=hashes,
                    scope='Cross-built firmware and host verification; board captures are generated separately.')
    path = ROOT / 'MANIFEST.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    dest = ROOT.parent / 'deliverables'
    dest.mkdir(exist_ok=True)
    archive = dest / 'IEEE_Micro_MLP_MCU_RunPack_2026-11-09.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in [path, *files]:
            z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    print(archive, archive.stat().st_size, 'bytes;', len(hashes), 'verified members')


if __name__ == '__main__':
    main()
