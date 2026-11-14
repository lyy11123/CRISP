"""A: independent Qwen decode FFN/PF4 repair budget, from the saved L1."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import statistics
import time

import hi_fault
from hi_fault import frozen
from e_run import ROOT, write_json

BASE = ROOT / "e2e/hi/closeout"
OLD = ROOT / "e2e/hi/out/i4_light"
OUT = ROOT / "e2e/hi/out/closeout_20261115"
MANIFEST = BASE / "manifests/a.json"
INPUTS = BASE / "manifests/a_inputs.json"
CHECKPOINTS = [30, 100, 300, 500, 700, 1000]
SEEDS = (123, 1, 2, 3, 4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    source_result = next(p for p in (OLD / "results").glob("*rep_dc_ffn_PF4_s300_r123.json"))
    old = frozen.read_json(source_result)
    base = old["spec"]
    assert base["model"] == "qwen3moe" and base["hardware_profile"] == "busybarn"
    config = base["config_id"]
    selection_path = OLD / "selection" / (config + ".json")
    selection = frozen.read_json(selection_path)
    selected = selection["operators"]["dc_ffn"]["A3h"]
    faults_path = OLD / "faults" / (config + ".json")
    paths = dict(pre=frozen.path(base["pre_sink"]), l1=frozen.path(base["l1_sink"]),
        selection=selection_path, faults=faults_path,
        **{k:frozen.path(selected[k]) for k in ("record", "raw", "snapshot")})
    inputs = dict(status="passed", source_candidate=old["source_candidate"], source=selected,
        paths={k:frozen.relative(v) for k,v in paths.items()}, sha256={k:sha(v) for k,v in paths.items()},
        config_id=config, start_state="saved L1 from original I4", old_300_result=frozen.relative(source_result))
    assert selected["candidate_id"] == old["source_candidate"]
    write_json(INPUTS, inputs)
    jobs = []
    for seed in SEEDS:
        spec = {k:v for k,v in base.items() if k not in ("previous_stop", "depends_on")}
        name = f"A__qwen3moe_busybarn_dc_ffn_PF4_s1000_r{seed}"
        spec.update(job_id=name, module="hi_closeout_a", steps=1000, seed=seed,
            launch_status="authorized_HI_CLOSEOUT_A", protocol_version="hi_closeout_budget_diagnostic_v1",
            checkpoints=CHECKPOINTS, ck_backend=False, checks=False, source_inputs=frozen.relative(INPUTS),
            source_input_sha256=inputs["sha256"], independent_budget=True, depends_on=[])
        seconds = old["timing"]["t_calibration"] + old["timing"]["t_L2_search"] * 1000/300 + old["timing"]["t_backend_offline"] + 45
        jobs.append(dict(id=name, kind="fault_rep", spec=spec, deps=[], result=f"results/{name}.json",
            argv=["{python}","-u","-B","e2e/hi_closeout_a.py","child","--spec","{spec}","--out","{out}"],
            estimated_seconds=seconds, peak_rss_gb=max(1.2, 1.5*old["peak_rss_gb"]), priority=20))
    manifest = dict(status="authorized", jobs=jobs, actual_searches=5, maximum_backend_samples=50,
        checkpoint_backend_samples=0, estimated_process_hours=sum(j["estimated_seconds"] for j in jobs)/3600,
        estimated_parallel_minutes=max(j["estimated_seconds"] for j in jobs)/60)
    write_json(MANIFEST, manifest)
    return {k:v for k,v in manifest.items() if k != "jobs"}


@contextmanager
def engine_context(spec, out):
    inputs = frozen.read_json(spec["source_inputs"])
    assert inputs["sha256"] == spec["source_input_sha256"]
    assert all(sha(frozen.path(inputs["paths"][k])) == value for k,value in inputs["sha256"].items())
    work = Path(out) / "a_budget"
    # The diagnostic has its own saved selection and physical fault description.
    for key, folder in (("selection", "selection"), ("faults", "faults")):
        frozen.write_json(work / folder / (spec["config_id"] + ".json"), frozen.read_json(inputs["paths"][key]))
    with hi_fault.model_adapter(spec, work) as protocol:
        yield work, inputs, protocol


def child(spec, out):
    assert spec["steps"] == 1000 and spec["seed"] in SEEDS
    assert spec["checkpoints"] == CHECKPOINTS and spec["ck_backend"] is False
    assert "previous_stop" not in spec and spec["backend_seeds"] == list(range(1, 11))
    started = time.perf_counter()
    with engine_context(spec, out) as (work, inputs, protocol):
        # Bypass fg_fault.child's stop-propagation branch; engine_job runs child_rep directly.
        result = frozen.engine_job(spec, work)
        raw = frozen.read_pickle(result["sink"])
        search = raw["search"]
        assert raw["spec"]["checks"] is False and raw["spec"]["ck_backend"] is False
        assert [r["step"] for r in search["rows"]] == list(range(1, 1001))
        assert raw["proposals"]["steps"] == 1000 and search["check_revert"]["n"] == 0
        assert [r["step"] for r in search["checkpoints"]] == CHECKPOINTS
        assert set(raw["record"]["backend"]) == set(range(1, 11)) and "checkpoint_backend" not in raw
        base = frozen.read_pickle(spec["l1_sink"])
        assert raw["T_L1_check"] == base["T_L1"] == search["T_start"]
        assert not raw["record"]["fault_violations"]
        checkpoints = []
        for ck in search["checkpoints"]:
            filename = work / "checkpoints" / (spec["job_id"] + f"__s{ck['step']}.pkl.gz")
            frozen.write_pickle(filename, dict(spec=spec, checkpoint=ck, source_L1=spec["l1_sink"]))
            checkpoints.append({k:v for k,v in ck.items() if k != "snap"} | dict(
                repair_seed=spec["seed"], total_budget=1000,
                accepted_proposals=sum(r["accepted"] for r in search["rows"][:ck["step"]]),
                calibration_seconds=search["t_calibration"], snapshot=frozen.relative(filename), snapshot_sha256=sha(filename)))
        result.update(job_id=spec["job_id"], kind=spec["kind"], status="ok", spec=spec,
            wall_seconds=time.perf_counter()-started, source_input_sha256=inputs["sha256"], protocol=protocol,
            actual_search_proposals=len(search["rows"]), accepted_proposals=sum(r["accepted"] for r in search["rows"]),
            start_T_L1=search["T_start"], start_identity_matches=True, checkpoints=checkpoints,
            backend_calls=10, checkpoint_backend_calls=0, finish_reason="independent_1000_step_budget_completed",
            original_stop300_inherited=False, best_step=search["best_step"], best_E=search["best_E"])
        import e2e_pilot_peak
        result["peak_rss_gb"] = e2e_pilot_peak.peak()/2**30
    frozen.write_json(Path(out)/"results"/(spec["job_id"]+".json"), result)
    return result


class SmokeComplete(Exception):
    pass


def smoke(out):
    """Eight actual proposals, checkpoint on/off; no backend call or quality sample."""
    spec = copy.deepcopy(frozen.read_json(MANIFEST)["jobs"][0]["spec"])
    spec.update(job_id="A_smoke_checkpoint_equivalence", steps=8, checkpoints=[3, 5, 8])
    began = time.perf_counter()
    evidence = {}
    with engine_context(spec, out) as (work, inputs, protocol):
        import eir
        import numpy as np
        original_rep, original_cal = eir.child_rep, eir.CAL_PROPOSALS
        eir.CAL_PROPOSALS = 3
        def inspect_engine(eng, inner, sink):
            base = eir.read_sink(inner["l1_sink"])
            runs = []
            for checkpoints in (None, [3, 5, 8]):
                eng.load(base["L1_snap"])
                result = eir.search(eng, "L2Gsync", 8, 123, base["seed_nodes"], False, checkpoints)
                runs.append((result, eng.fingerprint(), random.getstate(), pickle.dumps(np.random.get_state())))
            a, b = runs
            assert a[0]["rows"] == b[0]["rows"] and a[1:] == b[1:]
            assert all(a[0][k] == b[0][k] for k in ("Tmax", "Tmin", "best_E", "best_T", "best_step", "T_start"))
            assert [c["step"] for c in b[0]["checkpoints"]] == [3, 5, 8]
            assert b[0]["checkpoints"][-1]["best_E"] == b[0]["best_E"]
            evidence.update(status="passed", actual_searches=2, proposals_per_search=8,
                calibration_proposals_per_search=3, same_trajectory=True, same_final_state=True,
                same_python_and_numpy_random_state=True, checkpoint_steps=[3,5,8], backend_calls=0,
                source_input_sha256=inputs["sha256"], protocol=protocol,
                formal_budget=1000, formal_calibration_proposals=original_cal,
                smoke_not_used_as_quality_result=True)
            raise SmokeComplete()
        eir.child_rep = inspect_engine
        try:
            frozen.engine_job(spec, work)
        except SmokeComplete:
            pass
        finally:
            eir.child_rep, eir.CAL_PROPOSALS = original_rep, original_cal
    evidence["wall_seconds"] = time.perf_counter() - began
    write_json(BASE / "evidence/a_smoke.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "child", "smoke"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(), indent=2))
    elif args.command == "smoke":
        print(json.dumps(smoke(args.out), indent=2))
    else:
        child(frozen.read_json(args.spec), args.out)
