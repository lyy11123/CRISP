"""Package F/G code, generated model inputs, parameters and manifests."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import e_bundle
from e_run import ROOT


def build(output):
    files = {"AGENTS.md", "e2e/FG_PLAN.md", e_bundle.ARTIFACT_COMMIT,
             "e2e/e/evidence/fg_estimates.json", "e2e/e/tables/operator_table.csv"}
    for directory in e_bundle.SOURCE_DIRS:
        files.update(e_bundle.tree_files(directory, ".py", e_bundle.SKIP_DIRS))
    artifact = ROOT / e_bundle.ARTIFACT
    for directory in ("src", "utils", "tool", "fig_endtoend"):
        files.update(e_bundle.tree_files(f"{e_bundle.ARTIFACT}/{directory}", ".py", e_bundle.SKIP_DIRS))
    for directory in (ROOT / "e2e/fg/manifest", ROOT / "e2e/f/tables"):
        files.update(p.relative_to(ROOT).as_posix() for p in directory.glob("*")
                     if p.suffix in (".json", ".md", ".csv"))
    # The frozen standard templates are used to generate new parameterized inputs.
    for model in (artifact / "models").iterdir():
        if model.name.startswith(("intra_mapping_gpt_", "fg_")):
            for folder, suffix in (("cfg", ".cfg"), ("py", ".py")):
                files.update(p.relative_to(ROOT).as_posix() for p in (model / folder).glob("*" + suffix))
    metadata = dict(kind="FG_batch", source_git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        artifact_commit=(ROOT / e_bundle.ARTIFACT_COMMIT).read_text(encoding="utf-8").splitlines()[0],
        files_sha256={})
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
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size,
                          files=len(files), sha256=digest)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output.resolve())
