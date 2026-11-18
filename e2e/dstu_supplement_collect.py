"""Archive the supplement and verify its complete file inventory on COS download."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from e_run import ROOT, write_json
from hi_collect import build

BASE = ROOT/'e2e/dstu_supplement'


def restore(filename):
    meta = json.loads(filename.with_suffix(filename.suffix+'.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(filename.read_bytes()).hexdigest() == meta['sha256']
    with tarfile.open(filename, 'r:gz') as archive:
        listing = json.load(archive.extractfile('FILES.json'))
        assert set(archive.getnames()) == {'FILES.json', *(row['path'] for row in listing['files'])}
        for row in listing['files']:
            destination = (ROOT/row['path']).resolve()
            assert destination.is_relative_to(BASE)
            payload = archive.extractfile(row['path']).read()
            assert len(payload) == row['bytes']
            assert hashlib.sha256(payload).hexdigest() == row['sha256']
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix+'.part')
            temporary.write_bytes(payload)
            temporary.replace(destination)
    evidence = dict(**meta, status='passed', inventory=listing)
    write_json(BASE/'evidence'/(filename.stem+'.verified.json'), evidence)
    print(json.dumps(dict(status='verified_and_restored', files=len(listing['files']))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('build', 'restore'))
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--include', nargs='*')
    args = p.parse_args()
    if args.command == 'restore':
        restore(args.archive)
    else:
        build(args.archive, args.include)


if __name__ == '__main__':
    main()
