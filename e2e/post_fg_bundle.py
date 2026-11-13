"""Package the post-F/G source and template inputs; reuse cloud F/G result files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import e_bundle
from e_run import ROOT


def build(output):
    files = {"AGENTS.md", "PROJECT_MEMORY.md", "e2e/POST_FG_PLAN.md",
             "e2e/CRISP_POST_FG_CODEX_HANDOFF.md", e_bundle.ARTIFACT_COMMIT}
    for directory in e_bundle.SOURCE_DIRS:
        files.update(e_bundle.tree_files(directory, ".py", e_bundle.SKIP_DIRS))
    artifact = ROOT / e_bundle.ARTIFACT
    for directory in ("src", "utils", "tool", "fig_endtoend"):
        files.update(e_bundle.tree_files(f"{e_bundle.ARTIFACT}/{directory}", ".py", e_bundle.SKIP_DIRS))
    for model in (artifact / "models").iterdir():
        if model.name.startswith(("intra_mapping_gpt_", "fg_")):
            for folder, suffix in (("cfg", ".cfg"), ("py", ".py")):
                files.update(p.relative_to(ROOT).as_posix() for p in (model / folder).glob("*" + suffix))
    for directory in ("e2e/post_fg/manifest", "e2e/f/tables", "e2e/fg/manifest"):
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / directory).glob("*")
                     if p.suffix in (".json", ".md", ".csv"))
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "e2e").glob("post_fg_*.sh"))
    metadata = dict(kind="POST_FG", source_git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        artifact_commit=(ROOT / e_bundle.ARTIFACT_COMMIT).read_text(encoding="utf-8-sig").strip(),
        previous_result_root="/home/ubuntu/crisp_fg_20261113", files_sha256={})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(files):
                data = (ROOT / name).read_bytes()
                metadata["files_sha256"][name] = hashlib.sha256(data).hexdigest()
                e_bundle.add_bytes(archive, name, data)
            e_bundle.add_bytes(archive, "BUNDLE.json", e_bundle.json_bytes(metadata))
    temporary.replace(output)
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = dict(path=str(output), bytes=output.stat().st_size, files=len(files), sha256=digest)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output.resolve())
