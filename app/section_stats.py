"""
Computes average historical delay contribution per section, from
output/historical_runs.csv. This is the "historical average delay for that
section" input to the deterministic baseline calculator.

NOTE: historical_runs.csv is SIMULATED data (see BUILD_LOG.md) -- so these
averages are only as good as the simulator's calibration. Once
scripts/refresh_seed_from_ntes.py and a real outcomes archive exist, point
this at real data instead; the function signature doesn't need to change.
"""

import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUTPUT_DIR = os.path.join(HERE, "..", "output")


def compute_section_delay_stats(csv_path=None, seed_trains_path=None, max_date=None, exclude_dates=None):
    """Returns (section_stats, route_type_fallback_stats):
      section_stats: {(train_no, from_code, to_code): avg_delay_delta_min}
      route_type_fallback_stats: {route_type: avg_delay_delta_min}
    The fallback is used by baseline.py when a specific (train, from, to)
    triple isn't in the historical set -- e.g. if you add a new train to
    seed_trains.json that has no simulated history yet.

    max_date: if given (YYYY-MM-DD string), only rows with date < max_date
    are used.
    exclude_dates: if given (a set/collection of YYYY-MM-DD strings), rows
    with a date in this set are excluded. Used together with max_date=None
    for a STRATIFIED test split (e.g. "every 5th date") rather than a single
    chronological cutoff -- see build_training_table.py's docstring on why a
    pure chronological cutoff silently excluded monsoon season from the test
    set entirely."""
    csv_path = csv_path or os.path.join(OUTPUT_DIR, "historical_runs.csv")
    seed_trains_path = seed_trains_path or os.path.join(DATA_DIR, "seed_trains.json")
    exclude_dates = exclude_dates or set()

    with open(seed_trains_path) as f:
        seed = json.load(f)
    station_order = {t["train_no"]: [s["code"] for s in t["stations"]] for t in seed["trains"]}
    route_type_by_train = {t["train_no"]: t["route_type"] for t in seed["trains"]}

    delay_by_train_date = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if max_date is not None and row["date"] >= max_date:
                continue
            if row["date"] in exclude_dates:
                continue
            delay_by_train_date[(row["train_no"], row["date"])][row["station"]] = float(row["delay_min"])

    section_deltas = defaultdict(list)
    route_type_deltas = defaultdict(list)

    for (train_no, _date), station_delays in delay_by_train_date.items():
        order = station_order.get(train_no, [])
        route_type = route_type_by_train.get(train_no)
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            if a in station_delays and b in station_delays:
                delta = station_delays[b] - station_delays[a]
                section_deltas[(train_no, a, b)].append(delta)
                if route_type:
                    route_type_deltas[route_type].append(delta)

    section_stats = {k: sum(v) / len(v) for k, v in section_deltas.items()}
    route_type_fallback_stats = {k: sum(v) / len(v) for k, v in route_type_deltas.items()}

    return section_stats, route_type_fallback_stats


if __name__ == "__main__":
    section_stats, fallback = compute_section_delay_stats()
    print(f"Computed stats for {len(section_stats)} sections across "
          f"{len(set(k[0] for k in section_stats))} trains.\n")
    print("Route-type fallback averages (min added per section, on average):")
    for rt, v in fallback.items():
        print(f"  {rt:22s} {v:+.2f} min/section")
    print("\nSample section stats:")
    for k, v in list(section_stats.items())[:8]:
        print(f"  {k}: {v:+.2f} min")