"""
Deterministic baseline ETA calculator.

baseline_delay(station) = recovery-adjusted accumulation of historical
average per-section delay, starting from whatever delay is already known
(0 if predicting a full route cold, or a live current_delay_min if called
mid-journey).

Deliberately NO live weather/congestion signals here -- that's the whole
point of the design from earlier in this conversation: this baseline is the
"dumb but honest" estimate, and the ML residual model's job is to predict
what this formula gets wrong.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from domain_constants import ROUTE_RECOVERY_FRACTION, DEFAULT_RECOVERY_FRACTION


def _hhmm_to_minutes(hhmm):
    if not hhmm:
        return None
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_hhmm(total_minutes):
    total_minutes = int(round(total_minutes)) % (24 * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def compute_baseline(train, section_stats, route_type_fallback_stats,
                      current_delay_min=0.0, from_station_code=None):
    """
    train: one entry from data/seed_trains.json (dict with route_type, stations list)
    section_stats / route_type_fallback_stats: from app/section_stats.py
    from_station_code: if given, only stations AFTER this one are predicted
        (the ones before are assumed already happened); if None, predicts
        the whole route from origin.

    Returns a list of BaselinePrediction dicts (see SCHEMA.md).
    """
    stations = train["stations"]
    route_type = train["route_type"]
    recovery_fraction = ROUTE_RECOVERY_FRACTION.get(route_type, DEFAULT_RECOVERY_FRACTION)
    train_no = train["train_no"]

    start_idx = 0
    if from_station_code:
        codes = [s["code"] for s in stations]
        if from_station_code in codes:
            start_idx = codes.index(from_station_code)

    current_delay = current_delay_min
    results = []

    for i in range(start_idx, len(stations) - 1):
        s_from = stations[i]
        s_to = stations[i + 1]

        key = (train_no, s_from["code"], s_to["code"])
        avg_delta = section_stats.get(key)
        if avg_delta is None:
            avg_delta = route_type_fallback_stats.get(route_type, 0.0)

        recovered = current_delay * recovery_fraction
        current_delay = max(0.0, current_delay - recovered)
        current_delay = max(0.0, current_delay + avg_delta)

        sched_arr_min = _hhmm_to_minutes(s_to["sched_arr"])
        baseline_eta = _minutes_to_hhmm(sched_arr_min + current_delay) if sched_arr_min is not None else None

        results.append({
            "station": s_to["code"],
            "sched_arr": s_to["sched_arr"],
            "baseline_delay_min": round(current_delay, 1),
            "baseline_eta": baseline_eta,
        })

    return results


if __name__ == "__main__":
    import json
    from section_stats import compute_section_delay_stats

    DATA_DIR = os.path.join(HERE, "..", "data")
    with open(os.path.join(DATA_DIR, "seed_trains.json")) as f:
        seed = json.load(f)

    section_stats, fallback = compute_section_delay_stats()

    train = next(t for t in seed["trains"] if t["train_no"] == "10103")
    print(f"=== Baseline for {train['name']} ({train['train_no']}), cold start (no live delay) ===")
    for r in compute_baseline(train, section_stats, fallback):
        print(f"  {r['station']:6s} sched={r['sched_arr']:>5s}  "
              f"baseline_delay={r['baseline_delay_min']:+5.1f} min  baseline_eta={r['baseline_eta']}")

    print(f"\n=== Same train, mid-journey: currently 15 min late at ROHA ===")
    for r in compute_baseline(train, section_stats, fallback,
                               current_delay_min=15.0, from_station_code="ROHA"):
        print(f"  {r['station']:6s} sched={r['sched_arr']:>5s}  "
              f"baseline_delay={r['baseline_delay_min']:+5.1f} min  baseline_eta={r['baseline_eta']}")