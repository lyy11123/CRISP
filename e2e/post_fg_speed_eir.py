"""Time the frozen L2Gsync search from an existing L1 state; no backend rerun."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys
import time

from post_fg_speed import ROOT, digest, plain
from e_run import write_json


class FinishedBenchmark(Exception):
    pass


def child(spec, out):
    import fg_fault
    input_root = Path(spec["input_root"]).resolve()
    legacy_root = input_root.parents[3]
    out = Path(out).resolve()
    for folder in ("results", "snapshots", "faults", "raw"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    fault_file = spec["config_id"] + ".json"
    write_json(out / "faults" / fault_file,
               json.loads((input_root / "faults" / fault_file).read_text(encoding="utf-8")))
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT / "eir"))
    import eir
    source = fg_fault.selected_source(spec, input_root)
    for key in ("record", "raw", "snapshot"):
        p = Path(source[key])
        source[key] = str(p if p.is_absolute() else legacy_root / p)
    fg_fault.selected_source = lambda _spec, _out: source
    inner_spec = dict(spec, kind="fault_rep")
    for key in ("pre_sink", "l1_sink"):
        p = Path(inner_spec[key])
        inner_spec[key] = str(p if p.is_absolute() else legacy_root / p)
    result = {}

    def run(eng, spec0, sink):
        import post_fg_bb as bb
        t0 = time.perf_counter()
        if spec["execution"] == "fast":
            bb.simulate(bb.cpcm.build([dict(tag="warm", kind="comp", deps=[], res="cpu", dur=1)]))
            bb.cpcm.simulate = bb.simulate
        cold = time.perf_counter()-t0
        bb.TIMING.clear()
        base = eir.read_sink(spec0["l1_sink"])
        eng.load(base["L1_snap"])
        assert eng.t_eval() == base["T_L1"]
        t0 = time.perf_counter()
        search = eir.search(eng, "L2Gsync", spec["steps"], spec["seed"], base["seed_nodes"],
                            spec["execution"] == "checks")
        duration = time.perf_counter()-t0
        eng.t_eval()
        quality = dict(trajectory=search["rows"], best_step=search["best_step"], best_E=search["best_E"],
                       best_T=search["best_T"], Tmax=search["Tmax"], Tmin=search["Tmin"],
                       calibration=search["cal_worse"], state=plain(eng.fingerprint()),
                       T=eng.sc.T, W=[eng.W(e) for e in range(eng.g.n)], CP=eng.analysis().critical_path())
        result.update(quality=quality, comparison_sha256=digest(quality), check_revert=search["check_revert"],
                      coverage=dict(accepted=sum(r["accepted"] for r in search["rows"]),
                                    rejected=sum(not r["accepted"] for r in search["rows"]),
                                    targeted=sum(r["targeted"] for r in search["rows"])),
                      timing=dict(calibration_seconds=search["t_calibration"], search_seconds=search["t_L2_search"],
                                  restore_best_seconds=search["t_restore_best"], search_call_seconds=duration,
                                  cold_first_call_seconds=cold, accelerated_forward=list(bb.TIMING)),
                      source_L1=spec0["l1_sink"], backend_evaluations=0)
        p = out / "snapshots" / (spec["job_id"] + ".pkl.gz")
        temp = p.with_suffix(p.suffix + ".part")
        with gzip.open(temp, "wb") as stream:
            pickle.dump(eng.snapshot(), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(p)
        raise FinishedBenchmark

    eir.child_rep = run
    try:
        fg_fault.engine_job(inner_spec, out)
    except FinishedBenchmark:
        pass
    assert "quality" in result
    import e2e_pilot_peak
    result.update(job_id=spec["job_id"], kind="speed_eir", spec=spec, status="ok",
                  wall_seconds=time.perf_counter()-started, peak_rss_gb=e2e_pilot_peak.peak()/2**30)
    write_json(out / "results" / (spec["job_id"] + ".json"), plain(result))
    print(json.dumps(dict(job_id=spec["job_id"], status="ok", seconds=result["wall_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("child",))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    child(json.loads(args.spec.read_text(encoding="utf-8")), args.out)
