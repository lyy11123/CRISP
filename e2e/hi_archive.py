"""Archive the completed H/I/J first batch, runtime sources and its exact inputs.

Run after experiment workers and report generation finish. All data members use
e2e/hi/... names, so hi_collect.py restore also restores runtime and input copies.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[1]
HI = PurePosixPath("e2e/hi")
RUNTIME = HI / "dist/server_runtime"
SKIP = {".git", ".claude", ".codex", "__pycache__", "out", "dist", "results",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv",
        "node_modules", "site-packages", "colab_results"}
SOURCE_SUFFIXES = {".py", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".json",
                   ".sh", ".md", ".txt", ".COMMIT"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def walk_files(folder, skipped):
    for current, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in skipped)
        for name in sorted(names):
            if not name.endswith((".part", ".tmp")):
                yield Path(current) / name


def source_files(root):
    """Save code/configuration at its current runtime path; never recurse into outputs."""
    for source in walk_files(root, SKIP):
        if source.is_relative_to(root / HI):
            continue  # Already included at its native e2e/hi path below.
        if source.suffix in SOURCE_SUFFIXES or source.name in {"Makefile", ".gitignore", ".gitattributes"}:
            yield source


def input_files(root, source_maps=()):
    """Enumerate the files actually read by Split, J1 preparation, J4 and J5.

    source_maps is only for reviewing cloud manifests against retained local
    copies, e.g. [("/home/ubuntu/crisp_fg_20261113", local_repository)].
    The original manifest paths remain recorded unchanged.
    """
    entries = {}
    maps = sorted(((PurePosixPath(a), Path(b)) for a, b in source_maps),
                  key=lambda row: len(row[0].parts), reverse=True)

    def local(original):
        original = PurePosixPath(original)
        for before, after in maps:
            if original.is_relative_to(before):
                return after / str(original.relative_to(before))
        return Path(str(original))

    def add(original, source_root, scope, role, job, expected=None):
        original, source_root = PurePosixPath(original), PurePosixPath(source_root)
        relative = original.relative_to(source_root)
        destination = (RUNTIME / relative if scope == "current" else
                       RUNTIME / "external" / scope / relative)
        key = str(original)
        if key not in entries:
            path = local(original)
            entries[key] = dict(path=key, original_source_path=key,
                                source_root=str(source_root), source_scope=scope,
                                archive_path=str(destination), bytes=path.stat().st_size,
                                sha256=sha(path), roles=[], jobs=[], local_path=str(path))
        row = entries[key]
        assert row["archive_path"] == str(destination)
        if expected is not None:
            assert row["sha256"] == expected, key
        if role not in row["roles"]:
            row["roles"].append(role)
        if job not in row["jobs"]:
            row["jobs"].append(job)
        return read(local(original)) if original.suffix == ".json" else None

    j = root / HI / "j"
    split_manifest = read(j / "split_supplement_cloud.json")
    for item in split_manifest["input_files"]:
        original = PurePosixPath(item["graph"])
        source_root = PurePosixPath(str(original).split("/e2e/")[0])
        add(original, source_root, "postfg", "Split_frozen_graph", original.name, item["file_sha256"])

    for job in read(j / "J1_prepare.json")["jobs"]:
        for field in ("snapshot", "source_raw"):
            original = PurePosixPath(job[field])
            marker = "/cpcm_sa/" if "/cpcm_sa/" in str(original) else "/e2e/"
            source_root = PurePosixPath(str(original).split(marker)[0])
            scope = "fg" if "/e2e/fg/out/" in str(original) else "current"
            add(original, source_root, scope, "J1_" + field, job["job_id"])

    for job in read(j / "J4.json")["jobs"]:
        source_root = PurePosixPath(job["source_root"])
        graph = add(source_root / "e2e/fg/out/main/results" / (job["graph_job"] + ".json"),
                    source_root, "fg", "J4_graph_result", job["job_id"], job["graph_sha256"])
        for stream in (0, 1):
            add(source_root / graph["files"][f"orig_{stream}"], source_root, "fg",
                f"J4_orig_{stream}", job["job_id"], job["graph_files_sha256"][str(stream)])

    tree = ast.parse((root / "e2e/post_fg_diag.py").read_text(encoding="utf-8"))
    cases = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "CASES" for t in n.targets))
    cases = {name: (config, op) for name, config, op in cases}
    for job in read(j / "J5.json")["jobs"]:
        run_out = PurePosixPath(job["run_out"])
        source_root = run_out.parents[3]
        config, op = cases[job["case"]]
        selection = add(run_out / "selection" / (config + ".json"), source_root, "fg",
                        "J5_selection", job["job_id"])
        selected = selection["operators"][op]["A3h"]
        add(source_root / selected["snapshot"], source_root, "fg", "J5_snapshot", job["job_id"])
        pre = add(run_out / "results" / f"{config}__pre_{op}_A3h.json", source_root, "fg",
                  "J5_pre_result", job["job_id"])
        add(source_root / pre["exports"]["healthy"], source_root, "fg",
            "J5_healthy_export", job["job_id"])
        add(run_out / "faults" / (config + ".json"), source_root, "fg", "J5_faults", job["job_id"])

    files = sorted(entries.values(), key=lambda row: row["path"])
    return dict(scope="Source bytes read at archival time; run-time checks remain in job results.",
                files=files, unique_files=len(files),
                role_file_counts=dict(sorted(Counter(role for row in files for role in row["roles"]).items())))


def build(destination, root=ROOT, source_maps=()):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    assert destination.is_relative_to(root / HI / "dist")
    created = datetime.now(timezone.utc).isoformat()
    inputs = input_files(root, source_maps)
    files = {p.relative_to(root).as_posix(): p
             for p in walk_files(root / HI, {"dist", "__pycache__"})}
    runtime = []
    for p in source_files(root):
        relative = p.relative_to(root).as_posix()
        archived = str(RUNTIME / relative)
        files[archived] = p
        runtime.append(dict(path=archived, original_source_path=str(p),
                            bytes=p.stat().st_size, sha256=sha(p)))
    for row in inputs["files"]:
        files[row["archive_path"]] = Path(row.pop("local_path"))

    sources = dict(created_utc=created, source_root=str(root),
                   scope="Current runtime source/config bytes; H1 execution-era source is also retained under hi/runtime/h1_20261114.",
                   files=runtime)
    for name, data in (("external_inputs.json", inputs), ("runtime_sources.json", sources)):
        path = root / RUNTIME / name
        write(path, data)
        files[path.relative_to(root).as_posix()] = path

    listing = dict(created_utc=created, source_root=str(root), files=[])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as archive:
            for name, path in sorted(files.items()):
                data = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o644
                archive.addfile(info, io.BytesIO(data))
                listing["files"].append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            data = (json.dumps(listing, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            info = tarfile.TarInfo("FILES.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    temporary.replace(destination)
    write(destination.parent / "FILES.json", listing)
    meta = dict(path=str(destination), files=len(files), bytes=destination.stat().st_size,
                sha256=sha(destination), created_utc=created, external_input_files=inputs["unique_files"],
                runtime_source_files=len(runtime))
    write(destination.with_suffix(destination.suffix + ".json"), meta)
    print(json.dumps(meta), flush=True)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / HI / "dist/HIJ_FIRST_BATCH_20261114_complete.tar.gz")
    build(parser.parse_args().output)
