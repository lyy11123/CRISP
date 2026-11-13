"""Continue the F/G queue after its current controller and workers finish."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from e_run import ROOT, write_json
from fg_batch import process_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", required=True, type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--release-file", required=True, type=Path)
    args = parser.parse_args()
    state_path = args.out / "resume_wait.status.json"
    waiting = dict(state="waiting", controller_pid=args.wait_pid,
                   manifest=str(args.manifest), output=str(args.out))
    while process_alive(args.wait_pid):
        write_json(state_path, waiting | dict(updated_utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(30)
    prior = json.loads((args.out / "full.status.json").read_text())
    # The controller may have exited before a worker; let those jobs finish too.
    while any(process_alive(row["pid"]) for row in prior["running"]):
        time.sleep(30)
    argv = [sys.executable, "-u", "-B", "e2e/fg_batch.py", "--manifest", str(args.manifest),
            "--out", str(args.out), "--workers", "128", "--reserve-gib", "16", "--name", "resume",
            "--release-file", str(args.release_file)]
    with (ROOT / "resume.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(state="started", pid=process.pid, argv=argv,
                  started_utc=datetime.now(timezone.utc).isoformat(), preceding_pid=args.wait_pid)
    write_json(ROOT / "resume.start.json", record)
    write_json(state_path, record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
