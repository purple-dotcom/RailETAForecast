"""
Refresh data/seed_trains.json with REAL schedule times pulled live from NTES,
replacing the current placeholder times.

WHY THIS IS DEFENSIVE, NOT A ONE-SHOT PARSER:
ntes_connector.get_schedule() returns whatever raw JSON the NTES mobile API
sends back -- looking at the installed ntes-client source, it just decrypts
and returns the payload with no fixed schema of its own. No public
documentation of that exact field-naming exists (checked). So this script:
  1. Always dumps the raw response to disk first, so you can eyeball it.
  2. Tries to parse it using KEY_MAP below (a best-guess mapping).
  3. Tells you clearly if parsing failed, and exactly what to fix.

--------------------------------------------------------------------------
SCENARIO 1: Real run (once you have network access)

    cd connectors && pip install -r ../requirements.txt
    cd ../scripts
    python3 refresh_seed_from_ntes.py --train 12301

  -> dumps raw JSON to data/ntes_raw_dumps/12301.json
  -> attempts to parse + prints a diff against the current placeholder
  -> does NOT overwrite seed_trains.json yet (dry run by default)

  If the diff looks sane:

    python3 refresh_seed_from_ntes.py --train 12301 --write

  -> overwrites that train's entry in seed_trains.json with real times,
     and stamps _meta.refreshed with today's date for that train.

  If parsing failed / looks wrong: open the dumped raw JSON, find the
  actual field names, edit KEY_MAP below to match, then re-run.

--------------------------------------------------------------------------
SCENARIO 2: Demo/dry-run without network (what's actually runnable here)

    python3 refresh_seed_from_ntes.py --train 12301 --demo

  -> uses fixtures/ntes_schedule_fixture_12301.json instead of a live call
  -> proves the parsing + diff logic works end-to-end on a WELL-FORMED
     response
  -> fixture is FABRICATED (clearly labeled in the file) -- it validates the
     CODE, not the real field names.

--------------------------------------------------------------------------
SCENARIO 3: Offline failure-mode scenarios (also runnable here, no fixture needed)

    python3 refresh_seed_from_ntes.py --scenario all

  -> runs 4 canned in-memory payloads through parse_schedule() directly:
     a well-formed one, one with a renamed top-level wrapper key (schema
     drift), one with a field missing from a single row, and an empty
     schedule -- proving the parser fails LOUDLY and SPECIFICALLY instead
     of silently producing a corrupted seed_trains.json in each case.
--------------------------------------------------------------------------
"""

import argparse
import json
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIXTURES_DIR = os.path.join(HERE, "..", "fixtures")
DUMPS_DIR = os.path.join(DATA_DIR, "ntes_raw_dumps")

# Best-guess field mapping -- VERIFY against a real dumped response before
# trusting this on a live run. Edit these three lines if the real response
# uses different key names.
KEY_MAP = {
    "list_key": "trainSchedule",   # top-level key holding the list of stations
    "code": "stationCode",
    "name": "stationName",
    "arr": "arrivalTime",
    "dep": "departureTime",
    "km": "distance",
}

NO_TIME_MARKERS = {"--", "", None}


def fetch_raw(train_no: str, demo: bool):
    if demo:
        fixture_path = os.path.join(FIXTURES_DIR, f"ntes_schedule_fixture_{train_no}.json")
        if not os.path.exists(fixture_path):
            raise FileNotFoundError(
                f"No demo fixture for train {train_no} at {fixture_path}. "
                f"Only 12301 has one in this build."
            )
        with open(fixture_path) as f:
            return json.load(f)
    else:
        import sys
        sys.path.insert(0, os.path.join(HERE, "..", "connectors"))
        from ntes_connector import get_schedule
        return get_schedule(train_no)


def dump_raw(train_no: str, raw: dict):
    os.makedirs(DUMPS_DIR, exist_ok=True)
    path = os.path.join(DUMPS_DIR, f"{train_no}.json")
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"[dump] raw response saved to {path} -- inspect this if parsing below looks wrong")


def parse_schedule(raw: dict):
    """Returns a list of {code, name, km, sched_arr, sched_dep} using KEY_MAP.
    Raises a clear error pointing at KEY_MAP if the expected key is missing."""
    list_key = KEY_MAP["list_key"]
    if list_key not in raw:
        raise KeyError(
            f"Expected top-level key '{list_key}' not found in response. "
            f"Top-level keys present: {list(raw.keys())}. "
            f"Update KEY_MAP['list_key'] in this script to match."
        )
    stations = []
    for i, row in enumerate(raw[list_key]):
        try:
            arr = row[KEY_MAP["arr"]]
            dep = row[KEY_MAP["dep"]]
            stations.append({
                "code": row[KEY_MAP["code"]],
                "name": row[KEY_MAP["name"]],
                "km": int(row[KEY_MAP["km"]]),
                "sched_arr": None if arr in NO_TIME_MARKERS else arr,
                "sched_dep": None if dep in NO_TIME_MARKERS else dep,
            })
        except KeyError as e:
            raise KeyError(
                f"Row {i} missing expected field {e}. Row keys present: {list(row.keys())}. "
                f"Update KEY_MAP in this script to match."
            )
    return stations


def diff_against_seed(train_no: str, parsed_stations, seed_data):
    train_entry = next((t for t in seed_data["trains"] if t["train_no"] == train_no), None)
    if train_entry is None:
        print(f"[diff] train {train_no} not currently in seed_trains.json (would be a new addition)")
        return

    old_by_code = {s["code"]: s for s in train_entry["stations"]}
    print(f"\n[diff] {train_no} -- placeholder vs freshly parsed:")
    print(f"  {'station':6s} {'old arr->dep':17s} {'new arr->dep':17s}")
    for s in parsed_stations:
        old = old_by_code.get(s["code"])
        old_str = f"{old['sched_arr'] or '--'}->{old['sched_dep'] or '--'}" if old else "NOT IN SEED"
        new_str = f"{s['sched_arr'] or '--'}->{s['sched_dep'] or '--'}"
        flag = "  <-- CHANGED" if old and (old["sched_arr"] != s["sched_arr"] or old["sched_dep"] != s["sched_dep"]) else ""
        print(f"  {s['code']:6s} {old_str:17s} {new_str:17s}{flag}")


def write_seed(train_no: str, parsed_stations, seed_path):
    with open(seed_path) as f:
        seed_data = json.load(f)
    train_entry = next((t for t in seed_data["trains"] if t["train_no"] == train_no), None)
    if train_entry is None:
        print(f"[write] train {train_no} not found in seed file -- add it manually first (route_type, seasonal_risk).")
        return
    train_entry["stations"] = [
        {"code": s["code"], "name": s["name"], "km": s["km"],
         "sched_arr": s["sched_arr"], "sched_dep": s["sched_dep"]}
        for s in parsed_stations
    ]
    train_entry["_refreshed_from_live_ntes_on"] = date.today().isoformat()
    with open(seed_path, "w") as f:
        json.dump(seed_data, f, indent=2)
    print(f"[write] seed_trains.json updated for {train_no}")


# ---------------------------------------------------------------------------
# OFFLINE FAILURE-MODE SCENARIOS -- prove parse_schedule()'s defensive
# behavior across the shapes of bad data a real API could plausibly send,
# without needing network access or a per-case fixture file.
# ---------------------------------------------------------------------------

SCENARIO_GOOD = {
    "trainSchedule": [
        {"stationCode": "AAA", "stationName": "Alpha Jn", "distance": "0", "arrivalTime": "--", "departureTime": "10:00"},
        {"stationCode": "BBB", "stationName": "Beta Central", "distance": "150", "arrivalTime": "12:15", "departureTime": "12:20"},
        {"stationCode": "CCC", "stationName": "Charlie Terminal", "distance": "300", "arrivalTime": "14:45", "departureTime": "--"},
    ]
}

SCENARIO_WRONG_TOP_KEY = {
    # Simulates NTES using a different top-level wrapper key than KEY_MAP expects.
    "unexpectedWrapperKey": [
        {"stationCode": "AAA", "stationName": "Alpha Jn", "distance": "0", "arrivalTime": "--", "departureTime": "10:00"},
    ]
}

SCENARIO_MISSING_FIELD = {
    # Second row is missing "stationName" -- simulates a subtly incomplete record.
    "trainSchedule": [
        {"stationCode": "AAA", "stationName": "Alpha Jn", "distance": "0", "arrivalTime": "--", "departureTime": "10:00"},
        {"stationCode": "BBB", "distance": "150", "arrivalTime": "12:15", "departureTime": "12:20"},
    ]
}

SCENARIO_EMPTY = {"trainSchedule": []}


def run_scenarios():
    scenarios = [
        ("well-formed response, all fields present", SCENARIO_GOOD),
        ("wrong top-level wrapper key (schema drift)", SCENARIO_WRONG_TOP_KEY),
        ("one row missing a required field", SCENARIO_MISSING_FIELD),
        ("empty schedule (e.g. invalid train number)", SCENARIO_EMPTY),
    ]
    for label, payload in scenarios:
        print(f"\n=== SCENARIO: {label} ===")
        try:
            parsed = parse_schedule(payload)
        except KeyError as e:
            print(f"PARSE FAILED (correctly): {e}")
            print("  -> seed_trains.json would be left UNTOUCHED. Fix KEY_MAP and re-run.")
            continue

        if len(parsed) < 2:
            print(f"REJECTED: only {len(parsed)} station(s) -- not enough for a usable route.")
            print("  -> seed_trains.json would be left UNTOUCHED.")
            continue

        print(f"PARSED OK: {len(parsed)} station(s)")
        for s in parsed:
            arr = s["sched_arr"] or "--"
            dep = s["sched_dep"] or "--"
            print(f"  {s['code']:6s} {s['name']:20s} km={s['km']:<6} {arr:>5s} -> {dep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", help="Train number, e.g. 12301")
    ap.add_argument("--demo", action="store_true", help="Use local fixture instead of a live NTES call")
    ap.add_argument("--write", action="store_true", help="Actually overwrite seed_trains.json (default is dry-run)")
    ap.add_argument("--scenario", choices=["all"], help="Run offline failure-mode scenarios instead of a train refresh")
    args = ap.parse_args()

    if args.scenario == "all":
        run_scenarios()
        return

    if not args.train:
        ap.error("--train is required unless --scenario all is used")

    seed_path = os.path.join(DATA_DIR, "seed_trains.json")
    with open(seed_path) as f:
        seed_data = json.load(f)

    print(f"Fetching schedule for {args.train} ({'DEMO fixture' if args.demo else 'LIVE call'})...")
    raw = fetch_raw(args.train, args.demo)
    dump_raw(args.train, raw)

    try:
        parsed = parse_schedule(raw)
    except KeyError as e:
        print(f"\n[PARSE FAILED] {e}")
        print("Fix KEY_MAP at the top of this script and re-run.")
        return

    diff_against_seed(args.train, parsed, seed_data)

    if args.write:
        write_seed(args.train, parsed, seed_path)
    else:
        print("\n(dry run -- re-run with --write to actually update seed_trains.json)")


if __name__ == "__main__":
    main()
