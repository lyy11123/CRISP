"""Archive the completed closeout and the precise external inputs it used."""
import argparse
from pathlib import Path, PurePosixPath
import json

import hi_archive as archive

ROOT = archive.ROOT
BASE = ROOT / "e2e/hi/closeout"
ORIGINAL_INPUTS = archive.input_files


def resolve_input(root, relative):
    path = root / relative
    if path.exists():
        return path
    # Earlier F/G and post-F/G runs remain in their original server repositories.
    for prefix, directory in (("e2e/fg/", "crisp_fg_20261113"),
                              ("e2e/post_fg/", "crisp_post_fg_20261113")):
        if relative.startswith(prefix):
            return root.parent / directory / relative
    return path


def input_files(root, source_maps=()):
    original = ORIGINAL_INPUTS(root, source_maps)
    files = original["files"]
    expected = {}

    def add(relative, digest, role):
        relative = str(relative).replace("\\", "/")
        if relative in expected:
            assert expected[relative]["sha256"] == digest
            expected[relative]["roles"].add(role)
        else:
            expected[relative] = dict(sha256=digest, roles={role})

    for row in archive.read(BASE / "evidence/quality_completion_audit.json")["source_inputs"]["files"]:
        add(row["path"], row["expected_sha256"], "closeout_execution")
    for case in archive.read(BASE / "manifests/j3_inputs.json")["cases"].values():
        for path, digest in case["input_sha256"].items():
            add(path, digest, "closeout_J3")
    for row in archive.read(BASE / "evidence/s_report_inputs.json")["files"]:
        add(row["path"], row["sha256"], "closeout_S_report")
    for ledger, key, role in (
        ("j_offline_cost_inputs.json", "files", "closeout_J_fourteen_candidate_costs"),
        ("j2_first_complete_review.json", "input_files", "closeout_J2_first_review"),
        ("j2_qwen_six_review.json", "new_historical_archive_inputs", "closeout_J2_Qwen_repeats_review"),
    ):
        for row in archive.read(BASE / "evidence" / ledger)[key]:
            add(row["path"], row["sha256"], role)
    for row in archive.read(BASE / "evidence/j3_cost_independent_review.json")["source_inventory"]:
        if row["path"].startswith("e2e/hi/out/"):
            add(row["path"], row["sha256"], "closeout_J3_cost_review")

    for relative, item in sorted(expected.items()):
        source = resolve_input(root, relative)
        assert archive.sha(source) == item["sha256"], relative
        native = PurePosixPath(relative)
        target = (native if native.is_relative_to(archive.HI) else
                  archive.RUNTIME / "closeout_inputs" / native)
        files.append(dict(path=str(source), original_source_path=str(source),
            source_root=str(root), source_scope="closeout", archive_path=str(target),
            bytes=source.stat().st_size, sha256=item["sha256"], roles=sorted(item["roles"]),
            jobs=[], local_path=str(source), required_repository_path=relative))
    return dict(scope="Exact inputs of the previous J batch, closeout quality, J3, S and J cost postprocessing.",
        files=files, unique_files=len({r["archive_path"] for r in files}),
        closeout_declared_files=len(expected),
        role_file_counts=dict(archive.Counter(role for row in files for role in row["roles"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--output", type=Path,
        default=ROOT / "e2e/hi/dist/HI_CLOSEOUT_20261115_complete.tar.gz")
    args = parser.parse_args()
    source_maps = ([("/home/ubuntu/" + name, ROOT) for name in
        ("crisp_hi_20261114", "crisp_fg_20261113", "crisp_post_fg_20261113")]
        if ROOT.drive else [])
    if args.inspect:
        inputs = input_files(ROOT, source_maps)
        print(json.dumps(dict(unique_files=inputs["unique_files"],
            closeout_declared_files=inputs["closeout_declared_files"],
            total_bytes=sum(r["bytes"] for r in inputs["files"]))))
        return
    for folder, count in (("closeout_20261115", 549), ("closeout_j_20261115", 80)):
        result = archive.read(ROOT / "e2e/hi/out" / folder / "controller_finish.json")
        assert result["rc"] == 0 and result["completed"] == result["expected"] == count
    archive.input_files = input_files
    archive.build(args.output, root=ROOT, source_maps=source_maps)


if __name__ == "__main__":
    main()
