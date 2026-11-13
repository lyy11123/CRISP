"""Archive the completed cloud run, its runtime source and file hashes."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

from e_run import ROOT


def build(destination):
    output_root = ROOT / "e2e/post_fg/out"
    files = []
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "dist", ".pytest_cache"}]
        for name in names:
            source = Path(folder) / name
            if source.is_relative_to(output_root):
                archived = "out/" + source.relative_to(output_root).as_posix()
            else:
                archived = "runtime/" + source.relative_to(ROOT).as_posix()
            files.append((archived, source))
    listing = dict(created_utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT), files=[])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as archive:
            for archived, source in sorted(files):
                data = source.read_bytes()
                item = tarfile.TarInfo(archived)
                item.size = len(data)
                item.mode = 0o644
                archive.addfile(item, io.BytesIO(data))
                listing["files"].append(dict(path=archived, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            data = (json.dumps(listing, indent=2) + "\n").encode("utf-8")
            item = tarfile.TarInfo("FILES.json")
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    temporary.replace(destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = dict(path=str(destination), bytes=destination.stat().st_size,
                   files=len(files), sha256=digest, created_utc=listing["created_utc"])
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output.resolve())
