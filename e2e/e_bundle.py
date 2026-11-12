"""Package an E task manifest, source, templates and historical inputs for Linux.

python e2e/e_bundle.py --help
"""

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PILOT = "e2e/e/pilot.json"
SOURCE_DIRS = ("cpcm", "cpcm_sa", "e1_fidelity", "e2_attribution", "e3_warmstart",
               "e4_overlap", "eir", "dstu", "e2e")
SKIP_DIRS = {"out", "work", "work_e4", "logs", "dist", "results", "colab", "e4_out", "e4b_out",
             "dstu_w0tag", "dstu_w1", "__pycache__"}
ARTIFACT = "e2e/work/artifact"
ARTIFACT_COMMIT = "third_party/isca2026-busybarn-artifact.COMMIT"
CONFIG_SUFFIX = "_bw256_co4x4_bw256_t128x64_failpattern0"


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def source_file(name):
    path = ROOT / name
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT) or path.is_symlink() or not resolved.is_file():
        raise ValueError(f"Unsafe or non-file bundle input: {name}")
    return resolved


def tree_files(relative, suffix, excluded=()):
    base = ROOT / relative
    if not base.is_dir():
        raise FileNotFoundError(base)
    names = []
    for folder, dirs, files in os.walk(base):
        parent = Path(folder)
        dirs[:] = [name for name in dirs if name not in excluded and not (parent / name).is_symlink()]
        names.extend((parent / name).relative_to(ROOT).as_posix() for name in files if name.endswith(suffix))
    if not names:
        raise ValueError(f"No {suffix} files below {relative}")
    return names


def pilot_inputs(manifest):
    items = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if len({item["candidate_id"] for item in items}) != len(items):
        raise ValueError("Duplicate pilot candidate IDs")
    snapshots = set()
    for item in items:
        if item["kind"] != "reeval":
            continue
        for key in ("run", "snapshot"):
            relative = PurePosixPath(item[key].replace("\\", "/"))
            if (relative.is_absolute() or ".." in relative.parts
                    or relative.parts[:2] not in (("e2e", "out"), ("cpcm_sa", "out"))
                    or not relative.name.endswith(".pkl.gz")):
                raise ValueError(f"Unsafe pilot {key} path: {item[key]}")
            item[key] = relative.as_posix()
            snapshots.add(item[key])
    return items, snapshots


def model_inputs(items):
    paths = set()
    for item in items:
        phase, shape, op, method = (item[key] for key in ("phase", "shape", "op", "method"))
        model = f"{ARTIFACT}/models/intra_mapping_gpt_{phase}"
        config = f"ch{shape}{CONFIG_SUFFIX}"
        paths.add(f"{model}/cfg/config_{config}.cfg")
        if item["kind"] == "search":
            if method == "Gemini":
                template = "gemini"
            else:
                loss = int(item["x"]) if method == "A1" else 0
                template = f"greedy{int(item['g'])}_lossratio{loss}"
            paths.add(f"{model}/py/{op}_sq2048_{template}_{config}.py")
    return paths


def bundle_inputs(items, snapshots):
    files = {"e2e/E_PLAN.md", ARTIFACT_COMMIT, *snapshots, *model_inputs(items)}
    for name in SOURCE_DIRS:
        files.update(tree_files(name, ".py", SKIP_DIRS))
    for name in ("src", "utils", "tool", "fig_endtoend"):
        files.update(tree_files(f"{ARTIFACT}/{name}", ".py", SKIP_DIRS))
    return sorted(files)


def add_bytes(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(data))


def build(manifest, output):
    items, snapshots = pilot_inputs(manifest)
    files = bundle_inputs(items, snapshots)
    metadata = {
        "kind": "E_pilot",
        "jobs": len(items),
        "job_counts": dict(Counter(item["kind"] for item in items)),
        "source_git_head": subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"],
                                                   cwd=ROOT, text=True).strip(),
        "artifact_commit": source_file(ARTIFACT_COMMIT).read_text(encoding="utf-8").splitlines()[0],
        "python": sys.version.split()[0],
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "networkx", "simanneal")},
        "files_sha256": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".part",
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name in files:
                    data = source_file(name).read_bytes()
                    metadata["files_sha256"][name] = hashlib.sha256(data).hexdigest()
                    add_bytes(archive, name, data)
                data = json_bytes(items)
                metadata["files_sha256"][PILOT] = hashlib.sha256(data).hexdigest()
                add_bytes(archive, PILOT, data)
                add_bytes(archive, "BUNDLE.json", json_bytes(metadata))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    with output.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    print(f"{output}\n{len(items)} jobs, {len(snapshots)} historical inputs, "
          f"{len(metadata['files_sha256'])} files, {output.stat().st_size} bytes\nSHA256 {digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / PILOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
