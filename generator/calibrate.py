"""
Reads output/historical_runs.csv and checks whether its aggregate on-time
percentage (and per-route-type spread) falls inside the real published
ranges recorded in data/calibration_targets.json.

This does NOT validate cause-level breakdowns -- see calibration_targets.json
for why (no official public breakdown at that granularity exists).
"""

import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "output")


def main():
    with open(os.path.join(DATA_DIR, "calibration_targets.json")) as f:
        targets = json.load(f)

    rows = []
    with open(os.path.join(OUT_DIR, "historical_runs.csv")) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Final-destination rows are the ones with no sched_dep (last station of route).
    final_rows = [r for r in rows if r["sched_dep"] == ""]

    on_time_overall = [1 if float(r["delay_min"]) <= 15 else 0 for r in final_rows]
    overall_pct = sum(on_time_overall) / len(on_time_overall)

    by_route_type = defaultdict(list)
    for r in final_rows:
        by_route_type[r["route_type"]].append(1 if float(r["delay_min"]) <= 15 else 0)

    by_train = defaultdict(list)
    for r in final_rows:
        by_train[r["train_no"]].append(1 if float(r["delay_min"]) <= 15 else 0)

    cause_counts = defaultdict(int)
    for r in rows:
        for c in r["causes"].split(";"):
            if c:
                cause_counts[c] += 1

    print("=== Calibration report ===")
    lo, hi = targets["overall_on_time_pct_range"]
    status = "PASS" if lo <= overall_pct <= hi else "OUT OF RANGE"
    print(f"Overall on-time %: {overall_pct:.3f}  (target range {lo}-{hi})  [{status}]")

    print("\nOn-time % by route type:")
    for rt, vals in by_route_type.items():
        pct = sum(vals) / len(vals)
        print(f"  {rt:22s} {pct:.3f}  (n={len(vals)})")

    print("\nOn-time % by train (division-level proxy):")
    dlo, dhi = targets["division_level_spread_pct_range"]
    for tn, vals in sorted(by_train.items()):
        pct = sum(vals) / len(vals)
        in_range = dlo - 0.15 <= pct <= dhi  # allow headroom since this is per-train not per-division
        print(f"  {tn}  {pct:.3f}  (n={len(vals)})")

    print("\nCause frequency (illustrative, not calibrated against an official breakdown):")
    total_causes = sum(cause_counts.values())
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        print(f"  {cause:30s} {count:5d}  ({count/total_causes:.1%})")

    print(f"\nTotal final-destination runs evaluated: {len(final_rows)}")


if __name__ == "__main__":
    main()
