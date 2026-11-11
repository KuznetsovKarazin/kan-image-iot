import zipfile
from pathlib import Path
def snapshot(folder):
    archive=Path(str(folder)+'.zip');temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(folder.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(folder.parent).as_posix())
    temporary.replace(archive);return archive
