"""
Builds the FeatureVector (see SCHEMA.md) for each downstream station of a
train's route. Used both to build the training table (offline, from
historical_runs.csv) and at live inference time.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RISK_LEVEL_SCORE = {None: 0, "low": 1, "medium": 2, "high": 3}


def _section_lookup(sections, train_no, from_code, to_code):
    for s in sections:
        if s["train_no"] == train_no and s["from"] == from_code and s["to"] == to_code:
            return s
    return {}


def build_feature_rows(train, network, season,
                        current_delay_min=0.0, preceding_train_delay_min=None,
                        from_station_code=None):
    """Returns a list of FeatureVector dicts, one per downstream station,
    in the same order/length as app.baseline.compute_baseline's output for
    the same arguments -- the two lists are meant to be zipped together."""
    stations = train["stations"]
    train_no = train["train_no"]
    sections = network["sections"]

    start_idx = 0
    if from_station_code:
        codes = [s["code"] for s in stations]
        if from_station_code in codes:
            start_idx = codes.index(from_station_code)

    cum_single_line = 0
    cum_monsoon_score = 0
    cum_fog_score = 0

    rows = []
    for i in range(start_idx, len(stations) - 1):
        s_from = stations[i]
        s_to = stations[i + 1]
        sec = _section_lookup(sections, train_no, s_from["code"], s_to["code"])

        if sec.get("single_line"):
            cum_single_line += 1
        cum_monsoon_score += RISK_LEVEL_SCORE.get(sec.get("monsoon_risk"), 0)
        cum_fog_score += RISK_LEVEL_SCORE.get(sec.get("fog_risk"), 0)

        rows.append({
            "station": s_to["code"],
            "cum_single_line_sections": cum_single_line,
            "cum_monsoon_risk_score": cum_monsoon_score,
            "cum_fog_risk_score": cum_fog_score,
            "distance_km": s_to["km"],
            "station_index": i + 1,
            "is_monsoon": 1 if season == "monsoon" else 0,
            "is_winter_fog": 1 if season == "winter_fog" else 0,
            "is_festival_surge": 1 if season == "festival_surge" else 0,
            "current_delay_min": current_delay_min,
            "preceding_train_delay_min": preceding_train_delay_min or 0.0,
        })
    return rows


FEATURE_NAMES = [
    "cum_single_line_sections", "cum_monsoon_risk_score", "cum_fog_risk_score",
    "distance_km", "station_index", "is_monsoon", "is_winter_fog",
    "is_festival_surge", "current_delay_min", "preceding_train_delay_min",
]

FEATURE_LABELS = {
    "cum_single_line_sections": "single-line section exposure so far",
    "cum_monsoon_risk_score": "monsoon-season conditions on the route so far",
    "cum_fog_risk_score": "winter fog exposure on the route so far",
    "distance_km": "distance covered",
    "station_index": "number of stops so far",
    "is_monsoon": "monsoon-season conditions",
    "is_winter_fog": "winter fog-season conditions",
    "is_festival_surge": "festival-season traffic surge",
    "current_delay_min": "already-known current delay",
    "preceding_train_delay_min": "delay of the preceding train on this section",
}


if __name__ == "__main__":
    import json
    DATA_DIR = os.path.join(HERE, "..", "data")
    with open(os.path.join(DATA_DIR, "seed_trains.json")) as f:
        seed = json.load(f)
    with open(os.path.join(DATA_DIR, "network_attributes.json")) as f:
        network = json.load(f)

    train = next(t for t in seed["trains"] if t["train_no"] == "10103")
    rows = build_feature_rows(train, network, season="monsoon")
    for r in rows:
        print(r)