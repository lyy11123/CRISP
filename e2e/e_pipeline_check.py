"""Check E's analytic stage calculation and preserve the original G2 results."""

import argparse
import json
import math
from pathlib import Path

from e_report import pipeline_fields


def check(original_path, updated_path):
    original = json.loads(original_path.read_text(encoding="utf-8"))
    updated = json.loads(updated_path.read_text(encoding="utf-8"))
    key = lambda r: (r["fault"], r["scheme"], r["ln"], r["group"])
    new_rows = {key(r): r for r in updated["fault_model_table"]}
    for row in original["fault_model_table"]:
        assert all(new_rows[key(row)][field] == value for field, value in row.items()), key(row)
    for table in ("candidate_table", "operator_table", "layer_table", "model_table", "capacity_table", "timing_table", "terminal_table"):
        assert original[table] == updated[table], table
    assert len(new_rows) == len(updated["fault_model_table"]) == 570
    assert len(updated["pipeline_summary_table"]) == 30
    for row in updated["fault_model_table"]:
        if row["group"] == "G0":
            other = new_rows[row["fault"], row["scheme"], row["ln"], "G1"]
            for field, value in row.items():
                if field != "group" and not field.startswith("pipeline_bottleneck_groups_"):
                    assert value == other[field], (key(row), field)

    synthetic = []
    before = {"prefill": {"four": 40, "three": 30}, "decode": {"four": 4, "three": 3}}
    for damaged, factor, expected_stage_change, expected_throughput in (
        ("three", 1.0, 0, 0),
        ("three", 1.2, 0, 0),
        ("three", 1.35, 0.0125, 40 / 40.5 - 1),
        ("three", 1.3548, 0.0161, 40 / 40.644 - 1),
        ("four", 1.2, 0.2, 40 / 48 - 1),
    ):
        after = {phase: {group: value * (factor if group == damaged else 1) for group, value in values.items()}
                 for phase, values in before.items()}
        row = pipeline_fields({"before": before, "after": after})
        assert math.isclose(row["pipeline_stage_prefill_change"], expected_stage_change, abs_tol=1e-12)
        assert math.isclose(row["pipeline_ideal_throughput_prefill_change"], expected_throughput, abs_tol=1e-12)
        synthetic.append({"damaged_group": damaged, "factor": factor,
                          "stage_change": row["pipeline_stage_prefill_change"],
                          "throughput_change": row["pipeline_ideal_throughput_prefill_change"]})

    different_phase_leaders = {"prefill": {"a": 40, "b": 30}, "decode": {"a": 2, "b": 4}}
    row = pipeline_fields({"before": different_phase_leaders, "after": different_phase_leaders})
    assert row["pipeline_stage_request_128_before"] == 542
    assert row["pipeline_bottleneck_groups_request_128_before"] == "b"
    assert row["pipeline_stage_request_128_before"] != 40 + 128 * 4
    assert all(value == 0 for key, value in row.items() if key.endswith("_change"))
    return {"status": "passed", "original_G2_rows_unchanged": 190,
            "new_fault_rows": len(new_rows), "pipeline_summary_rows": 30,
            "other_tables_unchanged": 7, "G0_G1_values_equal": True,
            "synthetic_stage_cases": synthetic, "request_max_after_phase_sum": True,
            "new_simulations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("e2e/e/out/full_report/summary.json"))
    parser.add_argument("--updated", type=Path, default=Path("e2e/e/out/postprocess_report/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("e2e/e/out/pipeline_check.json"))
    args = parser.parse_args()
    result = check(args.original, args.updated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
