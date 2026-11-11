"""Fetch the checksum-pinned official Arduino AVRDUDE archive on Windows."""
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
from mega_common import ROOT,sha

def ensure_avrdude():
    if sys.platform!='win32':raise RuntimeError('Windows tool bundle: on Linux specify --avrdude and --avrdude-conf')
    spec=json.loads((ROOT/'tools/avrdude_windows.json').read_text());dest=ROOT/'vendor'
    valid=all((dest/n).is_file() and sha(dest/n)==digest for n,digest in spec['files'].items())
    if not valid:
        print('Downloading verified Arduino AVRDUDE tool (about 0.33 MB)...',flush=True)
        with urllib.request.urlopen(spec['url'],timeout=30) as response:raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=spec['archive_sha256']:raise ValueError('AVRDUDE archive SHA mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if z.testzip() is not None:raise ValueError('AVRDUDE archive CRC')
            for name,digest in spec['files'].items():
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('AVRDUDE file SHA')
                path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(path)
    return dest/'avrdude/bin/avrdude.exe',dest/'avrdude/etc/avrdude.conf'

if __name__=='__main__':
    exe,config=ensure_avrdude();print('AVRDUDE ready:',exe)

