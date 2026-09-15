"""
Builds the training table for the residual model.

For every historical run (train_no, date) in output/historical_runs.csv, and
for every station i along that run treated as "the currently-known position"
(with its REAL recorded delay as current_delay_min), and every downstream
station j after it: one training row.

This directly mirrors how the live API will actually be called (at any point
mid-journey, "given we're currently this late at station i, predict stations
i+1..end") rather than only ever training on full-route-from-origin examples.

target = residual = actual_delay_at(j) - baseline_delay_predicted_for(j)

where the baseline is computed using ONLY train-period section_stats (see
app/section_stats.py's max_date parameter) to avoid leaking test-period
averages into the baseline the model is trained to correct.
"""

import os
import sys
import csv
import json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(HERE, "..", "app")
sys.path.insert(0, APP_DIR)

from app.baseline import compute_baseline
from app.features import build_feature_rows, FEATURE_NAMES
from app.section_stats import compute_section_delay_stats

DATA_DIR = os.path.join(HERE, "..", "data")
OUTPUT_DIR = os.path.join(HERE, "..", "output")


def month_of(date_str):
    return int(date_str.split("-")[1])


def season_of(date_str):
    from app.domain_constants import month_to_season
    return month_to_season(month_of(date_str))


def load_runs(csv_path, min_date=None, max_date=None, include_dates=None):
    """Returns {(train_no, date): {station_code: delay_min}}, restricted to
    the given date range (min_date inclusive, max_date exclusive), or to an
    explicit set of dates via include_dates (used for the stratified test
    split instead of a chronological range)."""
    runs = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            if include_dates is not None and d not in include_dates:
                continue
            if min_date is not None and d < min_date:
                continue
            if max_date is not None and d >= max_date:
                continue
            runs[(row["train_no"], d)][row["station"]] = float(row["delay_min"])
    return runs


def build_rows(runs, seed, network, section_stats, fallback_stats):
    """Returns (X, y, meta) where X is a list of feature-dicts, y is a list
    of residual targets, meta is a list of dicts with train_no/date/station
    for traceability (not fed to the model)."""
    trains_by_no = {t["train_no"]: t for t in seed["trains"]}
    X, y, meta = [], [], []

    for (train_no, date), station_delays in runs.items():
        train = trains_by_no.get(train_no)
        if train is None:
            continue
        stations = train["stations"]
        codes = [s["code"] for s in stations]
        season = season_of(date)

        for i in range(len(codes) - 1):
            current_code = codes[i]
            if current_code not in station_delays:
                continue
            current_delay = station_delays[current_code]

            baseline_preds = compute_baseline(
                train, section_stats, fallback_stats,
                current_delay_min=current_delay, from_station_code=current_code,
            )
            feature_rows = build_feature_rows(
                train, network, season,
                current_delay_min=current_delay, preceding_train_delay_min=None,
                from_station_code=current_code,
            )

            for b, f in zip(baseline_preds, feature_rows):
                station_code = b["station"]
                if station_code not in station_delays:
                    continue
                actual_delay = station_delays[station_code]
                residual = actual_delay - b["baseline_delay_min"]

                X.append({k: f[k] for k in FEATURE_NAMES})
                y.append(residual)
                meta.append({"train_no": train_no, "date": date,
                              "from_station": current_code, "to_station": station_code,
                              "actual_delay_min": actual_delay,
                              "baseline_delay_min": b["baseline_delay_min"],
                              "season": season})

    return X, y, meta


def build_train_test(csv_path=None, seed_trains_path=None, network_path=None,
                      test_stride=5):
    """Splits by a STRATIFIED date sample (every `test_stride`-th
    chronological date goes to test) rather than a single chronological
    cutoff. A plain cutoff was tried first and silently excluded the entire
    monsoon and festival_surge windows from the test set (see BUILD_LOG.md) --
    since the simulator has no day-to-day autocorrelation (each synthetic day
    is an independent draw), a stratified split is methodologically fine here
    and actually tests the seasonal-shift behavior this whole design exists
    to demonstrate."""
    csv_path = csv_path or os.path.join(OUTPUT_DIR, "historical_runs.csv")
    seed_trains_path = seed_trains_path or os.path.join(DATA_DIR, "seed_trains.json")
    network_path = network_path or os.path.join(DATA_DIR, "network_attributes.json")

    with open(seed_trains_path) as f:
        seed = json.load(f)
    with open(network_path) as f:
        network = json.load(f)

    all_dates = sorted(set(r["date"] for r in csv.DictReader(open(csv_path))))
    test_dates = set(all_dates[i] for i in range(0, len(all_dates), test_stride))
    train_dates = set(all_dates) - test_dates

    # Baseline stats computed from TRAIN dates only -- see section_stats.py's
    # exclude_dates docstring for why this matters.
    section_stats, fallback_stats = compute_section_delay_stats(csv_path, seed_trains_path, exclude_dates=test_dates)

    train_runs = load_runs(csv_path, include_dates=train_dates)
    test_runs = load_runs(csv_path, include_dates=test_dates)

    X_train, y_train, meta_train = build_rows(train_runs, seed, network, section_stats, fallback_stats)
    X_test, y_test, meta_test = build_rows(test_runs, seed, network, section_stats, fallback_stats)

    return {
        "X_train": X_train, "y_train": y_train, "meta_train": meta_train,
        "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
        "section_stats": section_stats, "fallback_stats": fallback_stats,
        "n_test_dates": len(test_dates), "n_train_dates": len(train_dates),
    }


if __name__ == "__main__":
    data = build_train_test()
    print(f"Train dates: {data['n_train_dates']}   Test dates: {data['n_test_dates']} (stratified, every 5th date)")
    print(f"Train rows: {len(data['X_train'])}   Test rows: {len(data['X_test'])}")
    print(f"\nSample train row:")
    print(f"  features: {data['X_train'][0]}")
    print(f"  target (residual): {data['y_train'][0]:.2f} min")
    print(f"  meta: {data['meta_train'][0]}")