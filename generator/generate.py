"""
Orchestrates the simulator across a synthetic calendar of dates, handles
connecting-train delay propagation, and writes the output training dataset.

Usage:
    python generate.py
Output:
    ../output/historical_runs.csv
"""

import json
import csv
import random
import os
from datetime import date, timedelta

from simulator import simulate_train_run, month_to_season, CONNECTION_DELAY_PROPAGATION_THRESHOLD_MIN, CONNECTION_DELAY_PROPAGATION_FRACTION

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "output")

SEED = 42                 # fixed seed -> reproducible dataset for the demo
N_SYNTHETIC_DAYS = 480    # ~2 synthetic "years" so every season appears many times
START_DATE = date(2024, 1, 1)


def load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def main():
    rng = random.Random(SEED)
    seed_data = load_json("seed_trains.json")
    network = load_json("network_attributes.json")
    trains = {t["train_no"]: t for t in seed_data["trains"]}
    connections = network.get("connecting_trains", [])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "historical_runs.csv")

    rows = []
    for day_offset in range(N_SYNTHETIC_DAYS):
        the_date = START_DATE + timedelta(days=day_offset)
        month = the_date.month

        # Run every train for this date first (no inherited delay yet).
        run_results = {}
        for train_no, train in trains.items():
            season = month_to_season(month, train.get("seasonal_risk"))
            records = simulate_train_run(train, network, season, rng)
            run_results[train_no] = (season, records)

        # Apply connecting-train propagation: if an arriving train was late
        # enough at the hub station, re-simulate the connecting train with
        # an inherited initial delay.
        for conn in connections:
            hub = conn["hub_station"]
            arriving = conn["arriving_train"]
            connecting = conn["connection"]
            if arriving not in run_results or connecting not in trains:
                continue
            _, arriving_records = run_results[arriving]
            hub_record = next((r for r in arriving_records if r["station"] == hub), None)
            if hub_record and hub_record["delay_min"] >= CONNECTION_DELAY_PROPAGATION_THRESHOLD_MIN:
                inherited = hub_record["delay_min"] * CONNECTION_DELAY_PROPAGATION_FRACTION
                season = month_to_season(month, trains[connecting].get("seasonal_risk"))
                new_records = simulate_train_run(
                    trains[connecting], network, season, rng,
                    inherited_initial_delay_min=inherited,
                )
                run_results[connecting] = (season, new_records)

        for train_no, (season, records) in run_results.items():
            train = trains[train_no]
            for rec in records:
                rows.append({
                    "train_no": train_no,
                    "train_name": train["name"],
                    "route_type": train["route_type"],
                    "date": the_date.isoformat(),
                    "season": season,
                    "station": rec["station"],
                    "sched_arr": rec["sched_arr"],
                    "sched_dep": rec["sched_dep"],
                    "actual_arr": rec["actual_arr"],
                    "actual_dep": rec["actual_dep"],
                    "delay_min": rec["delay_min"],
                    "causes": ";".join(rec["causes"]),
                })

    fieldnames = ["train_no", "train_name", "route_type", "date", "season",
                  "station", "sched_arr", "sched_dep", "actual_arr",
                  "actual_dep", "delay_min", "causes"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
