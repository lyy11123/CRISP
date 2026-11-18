"""Assemble the local delivery from verified COS results and exact local inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from fg_fault import ROOT, write_json
from hi_collect import build

BASE = ROOT/'e2e/dstu_supplement'


def main():
    audit = json.loads((BASE/'evidence/input_and_reuse_manifest.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'passed'
    snapshot = BASE/'dist/source_inputs'
    sources = []
    for name, expected in audit['source_files'].items():
        original = ROOT/name
        if original.is_relative_to(BASE):
            continue
        payload = original.read_bytes()
        assert len(payload) == expected['bytes'] and hashlib.sha256(payload).hexdigest() == expected['sha256']
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        sources.append(dict(original=name, snapshot=target.relative_to(ROOT).as_posix(), **expected))
    code = BASE/'dist/local_source/e2e'
    code.mkdir(parents=True, exist_ok=True)
    for original in (ROOT/'e2e').glob('dstu_supplement*'):
        if original.is_file():
            shutil.copy2(original, code/original.name)
    write_json(BASE/'evidence/archive_input_snapshot.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_count=len(sources), sources=sources, source_ledger='evidence/input_and_reuse_manifest.json'))
    files = [p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and 'dist' not in p.relative_to(BASE).parts and not p.name.endswith('.part')
             and p.name not in ('DELIVERY.md', 'DSTU_SUPPLEMENT_20261118_complete.verified.json')]
    for folder in ('source_inputs', 'local_source', 'server_runtime', 'server_runtime_identity'):
        files.extend(p for p in (BASE/'dist'/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    archive_path = BASE/'dist/DSTU_SUPPLEMENT_20261118_complete.tar.gz'
    build(archive_path, [p.relative_to(ROOT).as_posix() for p in files])
    meta = json.loads(archive_path.with_suffix('.gz.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == meta['sha256']
    with tarfile.open(archive_path, 'r:gz') as archive:
        inventory = json.load(archive.extractfile('FILES.json'))
        assert set(archive.getnames()) == {'FILES.json', *(row['path'] for row in inventory['files'])}
        for row in inventory['files']:
            content = archive.extractfile(row['path']).read()
            assert len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256']
    write_json(BASE/'evidence/DSTU_SUPPLEMENT_20261118_complete.verified.json',
        dict(**meta, status='passed', checked_files=len(inventory['files']), exact_input_snapshots=len(sources),
             restored_over_live_files=False, verified_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='passed', **meta, exact_input_snapshots=len(sources))))


if __name__ == '__main__':
    main()
