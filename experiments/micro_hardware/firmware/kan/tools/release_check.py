import json
from pathlib import Path
from run_kan import sha,must
def release_check(root):
    p=root/'MANIFEST.json';m=json.loads(p.read_text())
    must(m['version']=='kan-energy-v1','Wrong KAN energy release')
    for n,d in m['sha256'].items():
        must((root/n).resolve().is_relative_to(root.resolve()),'Unsafe release path')
        must(sha(root/n)==d,'Changed release file: '+n)
    return sha(p)
