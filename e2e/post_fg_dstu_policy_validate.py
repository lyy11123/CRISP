"""One full fast-policy run, compared with a completed original-policy trace."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import post_fg_dstu as runner
from post_fg_dstu_policy_fast import MessagePolicy


def compare_issues(old_path, new_path):
    counts = [0, 0]
    hashes = [hashlib.sha256(), hashlib.sha256()]
    different = 0
    first = None
    with gzip.open(old_path, "rb") as old, gzip.open(new_path, "rb") as new:
        for index, pair in enumerate(itertools.zip_longest(old, new), 1):
            for i, line in enumerate(pair):
                if line is not None:
                    counts[i] += 1
                    hashes[i].update(line)
            a, b = pair
            if a != b and (a is None or b is None or json.loads(a) != json.loads(b)):
                different += 1
                if first is None:
                    first = dict(row=index, old=json.loads(a) if a else None, new=json.loads(b) if b else None)
    return dict(equal=different == 0, rows=counts, different_rows=different,
                first_difference=first, decompressed_sha256=[h.hexdigest() for h in hashes])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    baseline = runner.read_json(args.baseline_result)
    spec = dict(baseline["spec"],job_id=baseline["job_id"]+"_fast_validation",
                module="post_fg_dstu_policy_validate",trace=True)
    runner.MessagePolicy = MessagePolicy
    runner.child(spec,args.out)
    result_path = args.out/"results"/(spec["job_id"]+".json")
    current = runner.read_json(result_path)
    fields = ("completion","first_start","kv_window","release","simulated_total","unhidden_compute",
              "unhidden_communication","metrics","policy_counters","candidate_counters_available",
              "audit","evaluation_identity","chunk_stats")
    checks = {key:baseline[key] == current[key] for key in fields}
    issues = compare_issues(baseline["issue_trace"],current["issue_trace"])
    with gzip.open(baseline["message_trace"],"rt",encoding="utf-8") as file:
        old_messages = json.load(file)
    with gzip.open(current["message_trace"],"rt",encoding="utf-8") as file:
        new_messages = json.load(file)
    messages = {key:old_messages[key] == new_messages[key] for key in old_messages}
    passed = all(checks.values()) and issues["equal"] and all(messages.values())
    sources = {name:runner.sha(Path(__file__).with_name(name)) for name in
               ("post_fg_dstu_policy.py","post_fg_dstu_policy_fast.py","post_fg_dstu.py","fast_hop.py")}
    verification = dict(status="passed" if passed else "different",new_backend_calls=1,
                        baseline_result=str(args.baseline_result),result=str(result_path),fields=checks,
                        issues=issues,message_and_consumer_checks=messages,source_sha256=sources,
                        timing_context="Formal quality jobs were running concurrently; this is a semantic validation, not isolated speed measurement.",
                        original_simulation_seconds=baseline["simulation_seconds"],
                        new_simulation_seconds=current["simulation_seconds"],new_wall_seconds=current["wall_seconds"])
    runner.write_json(args.out/"verification.json",verification)
    print(json.dumps(verification),flush=True)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
