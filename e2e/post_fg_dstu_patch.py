"""Run the 60 fresh baselines, then verify three traces against them."""
import argparse
from pathlib import Path
import time

import fg_batch
from e_run import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();started=time.monotonic()
    for name,file in (("dstu_fresh_baselines","dstu_fresh_baselines60_cloud.json"),
                      ("dstu_fresh_verify","dstu_fresh_patch63_cloud.json")):
        rc=fg_batch.execute(args.manifest_dir/file,args.out,128,3,name,2)
        if rc:
            return rc
    write_json(args.out/"dstu_fresh_patch.finished.json",dict(status="ok",new_backend_calls=63,
                wall_seconds=time.monotonic()-started,stages=["dstu_fresh_baselines","dstu_fresh_verify"]))
    return 0


if __name__=="__main__":raise SystemExit(main())
