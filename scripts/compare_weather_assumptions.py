"""
Pulls real historical weather for a station/date range and reports whether
it's broadly consistent with the season assumptions baked into
generator/simulator.py's SEASON_EFFECTS -- a sanity check, not a hard pass/
fail (there's no official "this precipitation level = this many minutes of
delay" conversion, so this stays advisory).

--------------------------------------------------------------------------
SCENARIO 1: Real run (once you have network access)

    python3 compare_weather_assumptions.py --station MAO --start 2024-07-10 --end 2024-07-11

  -> calls weather_connector.get_historical_weather() for Madgaon's real
     coordinates over a real monsoon-week date range
  -> prints avg precipitation/visibility and an advisory comparison against
     the monsoon multiplier currently set in simulator.py

--------------------------------------------------------------------------
SCENARIO 2: Demo/dry-run without network (what's actually runnable here)

    python3 compare_weather_assumptions.py --demo

  -> uses fixtures/open_meteo_fixture_monsoon_week.json instead of a live call
  -> the fixture's array SHAPE matches Open-Meteo's real, documented format;
     only the numeric values are made up (elevated to look like a real
     monsoon week) -- see the fixture's own _meta note

--------------------------------------------------------------------------
SCENARIO 3: Offline advisory-logic scenarios (also runnable here)

    python3 compare_weather_assumptions.py --scenario all

  -> runs 4 synthetic weather series (heavy monsoon, mild monsoon, heavy
     fog, clear) through summarize() + advisory_report() to show all the
     branches of the advisory logic, not just the one fixture happens to hit
--------------------------------------------------------------------------
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIXTURES_DIR = os.path.join(HERE, "..", "fixtures")

sys.path.insert(0, os.path.join(HERE, "..", "connectors"))
sys.path.insert(0, os.path.join(HERE, "..", "generator"))


def fetch_weather(station: str, start: str, end: str, demo: bool):
    if demo:
        path = os.path.join(FIXTURES_DIR, "open_meteo_fixture_monsoon_week.json")
        with open(path) as f:
            return json.load(f)
    from connectors.weather_connector import get_historical_weather, STATION_COORDS
    if station not in STATION_COORDS:
        raise ValueError(f"No coordinates for station {station} in weather_connector.STATION_COORDS")
    lat, lon = STATION_COORDS[station]
    return get_historical_weather(lat, lon, start, end)


def summarize(raw: dict):
    hourly = raw["hourly"]
    precip = hourly["precipitation"]
    vis = hourly["visibility"]
    avg_precip = sum(precip) / len(precip)
    avg_vis = sum(vis) / len(vis)
    hours_heavy_rain = sum(1 for p in precip if p > 10)
    hours_low_vis = sum(1 for v in vis if v < 3500)
    return {
        "avg_precip_mm": avg_precip,
        "avg_visibility_m": avg_vis,
        "hours_heavy_rain": hours_heavy_rain,
        "hours_low_vis": hours_low_vis,
        "n_hours": len(precip),
    }


def advisory_report(summary: dict):
    from generator.simulator import SEASON_EFFECTS

    print("\n=== Real weather summary ===")
    print(f"  Avg precipitation: {summary['avg_precip_mm']:.1f} mm/hr")
    print(f"  Avg visibility:    {summary['avg_visibility_m']:.0f} m")
    print(f"  Hours w/ heavy rain (>10mm/hr): {summary['hours_heavy_rain']} / {summary['n_hours']}")
    print(f"  Hours w/ low visibility (<3500m): {summary['hours_low_vis']} / {summary['n_hours']}")

    monsoon_mult = SEASON_EFFECTS["monsoon"]["event_prob_multiplier"]
    fog_mult = SEASON_EFFECTS["winter_fog"]["event_prob_multiplier"]

    print("\n=== Advisory comparison against simulator.py assumptions ===")
    heavy_rain_frac = summary["hours_heavy_rain"] / summary["n_hours"]
    if heavy_rain_frac > 0.4:
        print(f"  Heavy rain in {heavy_rain_frac:.0%} of hours -- this is a genuinely bad monsoon "
              f"stretch. Current monsoon event_prob_multiplier ({monsoon_mult}x) looks reasonable "
              f"or could arguably go slightly higher for weeks like this.")
    elif heavy_rain_frac > 0.1:
        print(f"  Heavy rain in {heavy_rain_frac:.0%} of hours -- moderate monsoon activity. "
              f"Current multiplier ({monsoon_mult}x) looks about right.")
    else:
        print(f"  Heavy rain in only {heavy_rain_frac:.0%} of hours -- this particular window wasn't "
              f"severe. Don't recalibrate SEASON_EFFECTS off one mild week; check a few more.")

    low_vis_frac = summary["hours_low_vis"] / summary["n_hours"]
    if low_vis_frac > 0.3:
        print(f"  Low visibility in {low_vis_frac:.0%} of hours -- consistent with fog-risk "
              f"assumptions (fog multiplier currently {fog_mult}x) if this were a winter reading.")

    print("\nThis is advisory, not a hard calibration target (unlike calibrate.py's on-time %% "
          "check, there's no official published mapping from mm of rain to minutes of delay).")


# ---------------------------------------------------------------------------
# OFFLINE ADVISORY-LOGIC SCENARIOS -- exercise every branch of
# advisory_report() with synthetic (clearly-labeled, in-memory-only) weather
# series, since one fixture file can only demonstrate one branch.
# ---------------------------------------------------------------------------

def _mock_hourly(n_hours, precip_range, vis_range, seed):
    rng = random.Random(seed)
    return {"hourly": {
        "time": [f"synthetic-h{i}" for i in range(n_hours)],
        "precipitation": [rng.uniform(*precip_range) for _ in range(n_hours)],
        "visibility": [rng.uniform(*vis_range) for _ in range(n_hours)],
        "temperature_2m": [26.0] * n_hours,
        "wind_speed_10m": [15.0] * n_hours,
    }}


def run_scenarios():
    n = 48  # 2 synthetic days, hourly
    scenarios = [
        ("Heavy monsoon stretch", _mock_hourly(n, (8, 25), (2000, 6000), seed=1)),
        ("Mild monsoon stretch", _mock_hourly(n, (0, 4), (4000, 9000), seed=2)),
        ("Heavy winter fog", _mock_hourly(n, (0, 0), (200, 2000), seed=3)),
        ("Clear conditions", _mock_hourly(n, (0, 0), (8000, 10000), seed=4)),
    ]
    for label, raw in scenarios:
        print(f"\n{'#' * 3} SCENARIO: {label} {'#' * 3}")
        summary = summarize(raw)
        advisory_report(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="MAO", help="Station code (must be in weather_connector.STATION_COORDS)")
    ap.add_argument("--start", default="2024-07-10", help="YYYY-MM-DD")
    ap.add_argument("--end", default="2024-07-11", help="YYYY-MM-DD")
    ap.add_argument("--demo", action="store_true", help="Use local fixture instead of a live Open-Meteo call")
    ap.add_argument("--scenario", choices=["all"], help="Run offline advisory-logic scenarios")
    args = ap.parse_args()

    if args.scenario == "all":
        run_scenarios()
        return

    print(f"Fetching historical weather for {args.station} {args.start}..{args.end} "
          f"({'DEMO fixture' if args.demo else 'LIVE call'})...")
    raw = fetch_weather(args.station, args.start, args.end, args.demo)
    summary = summarize(raw)
    advisory_report(summary)


if __name__ == "__main__":
    main()
