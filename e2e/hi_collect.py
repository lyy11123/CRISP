"""Collect this batch's outputs and verify each archived file when restoring locally."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
from e_run import ROOT, write_json


def build(output, includes):
    files = {}
    for item in includes:
        source = ROOT / item
        for path in ([source] if source.is_file() else source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith('.part'):
                files[path.relative_to(ROOT).as_posix()] = path
    listing=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.part')
    with tarfile.open(temporary,'w:gz') as archive:
        for name,p in sorted(files.items()):
            data=p.read_bytes()
            info=tarfile.TarInfo(name); info.size=len(data)
            archive.addfile(info,io.BytesIO(data))
            listing['files'].append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        data=json.dumps(listing,indent=2).encode()
        info=tarfile.TarInfo('FILES.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    temporary.replace(output)
    meta=dict(path=str(output),files=len(files),bytes=output.stat().st_size,
              sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    write_json(output.with_suffix(output.suffix+'.json'),meta)
    print(json.dumps(meta))


def restore(archive_path):
    meta=json.loads(archive_path.with_suffix(archive_path.suffix+'.json').read_text())
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest()==meta['sha256']
    with tarfile.open(archive_path,'r:gz') as archive:
        listing=json.load(archive.extractfile('FILES.json'))
        assert set(archive.getnames())=={'FILES.json',*(r['path'] for r in listing['files'])}
        for row in listing['files']:
            dest=(ROOT/row['path']).resolve()
            assert dest.is_relative_to(ROOT/'e2e/hi')
            data=archive.extractfile(row['path']).read()
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(data)
    write_json(ROOT/'e2e/hi/evidence'/(archive_path.stem+'.verified.json'),dict(**meta,status='passed',inventory=listing))
    print(json.dumps(dict(files=meta['files'],status='verified_and_restored')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','restore'])
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--include',nargs='*')
    args=parser.parse_args()
    if args.command=='build':build(args.archive,args.include)
    else:restore(args.archive)
