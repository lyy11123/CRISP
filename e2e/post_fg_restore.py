"""Restore this run's final archive and verify its recorded file contents."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from e_run import write_json


def restore(archive_path, sha256, destination):
    archive_path = Path(archive_path)
    destination = Path(destination).resolve()
    with archive_path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    assert actual == sha256, (actual, sha256)
    runtime = destination / "dist/server_runtime"
    verified = []
    with tarfile.open(archive_path, "r:gz") as archive:
        listing = json.load(archive.extractfile("FILES.json"))
        names = [member.name for member in archive.getmembers()]
        expected = [row["path"] for row in listing["files"]]
        assert len(names) == len(set(names)) == len(expected) + 1
        assert set(names) == set(expected) | {"FILES.json"}
        for row in listing["files"]:
            prefix, relative = row["path"].split("/", 1)
            assert prefix in ("out", "runtime")
            root = destination / "out" if prefix == "out" else runtime
            target = (root / relative).resolve()
            assert target.is_relative_to(root.resolve())
            payload = archive.extractfile(row["path"]).read()
            assert len(payload) == row["bytes"]
            assert hashlib.sha256(payload).hexdigest() == row["sha256"], row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]
            verified.append(dict(path=row["path"], bytes=row["bytes"], sha256=row["sha256"]))
    write_json(destination / "evidence/complete_file_inventory.json", listing)
    report = dict(status="passed", archive=str(archive_path.resolve()), archive_sha256=actual,
                  archive_bytes=archive_path.stat().st_size, files=len(verified),
                  content_bytes=sum(row["bytes"] for row in verified),
                  output=str(destination / "out"), runtime=str(runtime),
                  archive_created_utc=listing["created_utc"],
                  inventory="evidence/complete_file_inventory.json")
    write_json(destination / "evidence/complete_download.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", default="e2e/post_fg")
    args = parser.parse_args()
    restore(args.archive, args.sha256, args.destination)
