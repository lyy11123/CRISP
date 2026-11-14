"""Package only H/I/J source additions and native model templates for this batch."""
import hashlib
import json
from pathlib import Path
import tarfile
from e_run import ROOT, write_json


def package():
    paths={}
    for pattern in ("hi_*.py","hi_*.sh","j_timing.py"):
        for p in (ROOT/"e2e").glob(pattern):
            paths[p.relative_to(ROOT).as_posix()]=p
    for p in (ROOT/"e2e/hi").rglob('*'):
        rel=p.relative_to(ROOT)
        if p.is_file() and not any(s in rel.parts for s in ('out','dist','__pycache__')):
            paths[rel.as_posix()]=p
    for model in ('llama','qwen3moe'):
        for phase in ('decode','prefill'):
            folder=f'models/intra_mapping_{model}_{phase}'
            for p in (ROOT/'third_party/isca2026-busybarn-artifact'/folder).glob('*.py'):
                paths[f'e2e/work/artifact/{folder}/{p.name}']=p
    for name in ('AGENTS.md','PROJECT_MEMORY.md','e2e/PROGRESS.md','e2e/HIJ_STAGED_PLAN.md'):
        p=ROOT/name
        paths[p.relative_to(ROOT).as_posix()]=p
    for prefix,name in (('cpcm_sa/out/s2','ffn_A3h_g0_s123'),('e2e/out/b4','pf_mha_w1_A3h_g0_s123')):
        for folder in ('snaps','runs'):
            p=ROOT/prefix/folder/(name+'.pkl.gz')
            paths[p.relative_to(ROOT).as_posix()]=p
    inventory=[dict(path=name,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in sorted(paths.items())]
    dest=ROOT/'e2e/hi/dist/HI_FIRST_INPUTS.tar.gz'
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(dest.with_suffix('.part'),'w:gz') as archive:
        for name,p in sorted(paths.items()):
            archive.add(p,arcname=name)
    dest.with_suffix('.part').replace(dest)
    write_json(dest.with_suffix('.inventory.json'),dict(files=inventory,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),size=dest.stat().st_size))
    print(json.dumps(dict(path=str(dest),files=len(paths),size=dest.stat().st_size)))


if __name__=='__main__':
    package()
