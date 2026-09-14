"""
Layered historical-outcome simulator.

Generates plausible "what actually happened" records on top of a real schedule skeleton (seed_trains.json) and 
real network annotations (network_attributes.json). This does NOT try to be a physically exact simulation of railway 
operations -- it exists to produce a training set whose AGGREGATE statistics (on-time %, cause mix) resemble real 
published numbers, so the downstream ML model has something realistic to learn a correction on top of.

Layers, applied per section (station i -> station i+1) in order:
  1. base runtime sampler   -- natural variability around scheduled duration
  2. event injector         -- discrete delay causes (weather/TSR/congestion/etc.)
  3. recovery               -- schedule padding claws back some of the backlog
  4. propagation            -- connecting trains inherit upstream lateness

A `season` parameter (normal / winter_fog / monsoon / festival_surge) shifts event probabilities and durations, so a 
periodic-retraining loop has a real distribution shift to adapt to in the demo.
"""

import random
import math
from datetime import datetime, timedelta

CAUSES = [
    "signal_halt",
    "temporary_speed_restriction",
    "congestion_preceding_train",
    "crew_changeover_delay",
    "weather",
    "chain_pulling_other",
]

# Base (per-section) probability of an event being injected at all, before
# any season/route modifiers are applied. Kept low because most sections on
# most days run cleanly -- the modifiers below are what create realistic
# clustering of bad days.
BASE_EVENT_PROB = 0.13

# Relative weight of each cause under "normal" conditions -- illustrative,
# not an official published breakdown (none was found publicly at
# cause-level granularity). Only the aggregate on-time % is calibrated
# against a real published figure; see calibrate.py.
BASE_CAUSE_WEIGHTS = {
    "signal_halt": 0.25,
    "temporary_speed_restriction": 0.20,
    "congestion_preceding_train": 0.20,
    "crew_changeover_delay": 0.10,
    "weather": 0.10,
    "chain_pulling_other": 0.15,
}

# How much a season shifts things. Multiplicative on event probability for
# affected sections, and additive weight boost for the relevant cause.
SEASON_EFFECTS = {
    "normal": {},
    "winter_fog": {
        "applies_to_flag": "fog_risk",
        "event_prob_multiplier": 2.2,
        "cause_boost": {"weather": 0.35},
        "duration_multiplier": 1.8,
    },
    "monsoon": {
        "applies_to_flag": "monsoon_risk",
        "event_prob_multiplier": 2.5,
        "cause_boost": {"weather": 0.30, "temporary_speed_restriction": 0.15},
        "duration_multiplier": 2.0,
    },
    "festival_surge": {
        "applies_to_flag": None,  # affects ALL sections, not flagged ones
        "event_prob_multiplier": 1.6,
        "cause_boost": {"congestion_preceding_train": 0.25},
        "duration_multiplier": 1.3,
    },
}

# Recovery: fraction of existing backlog schedule padding claws back per
# section, by route type (double-line trunk routes tend to have more padding
# built in than single-line coastal routes).
RECOVERY_FRACTION = {
    "double_line_trunk": 0.28,
    "single_line_coastal": 0.14,
}

# Base-runtime noise: lognormal multiplier applied to scheduled section
# duration to represent ordinary day-to-day variability (crew handling,
# minor signal timing, etc.) BEFORE any discrete event is injected.
NOISE_SIGMA = 0.025

CONNECTION_DELAY_PROPAGATION_THRESHOLD_MIN = 20
CONNECTION_DELAY_PROPAGATION_FRACTION = 0.5


def month_to_season(month: int, route_seasonal_risk: str) -> str:
    """Map a synthetic calendar month to a season label. This is a coarse,
    explicit mapping -- not climatology -- deliberately simple so it's easy
    to reason about in the retraining-loop demo."""
    if month in (12, 1):
        base = "winter_fog"
    elif month in (6, 7, 8, 9):
        base = "monsoon"
    elif month in (10, 11):
        base = "festival_surge"
    else:
        base = "normal"
    return base


def _hhmm_to_minutes(hhmm):
    if hhmm is None:
        return None
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_hhmm(total_minutes):
    total_minutes = int(round(total_minutes)) % (24 * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def section_lookup(network, train_no, from_code, to_code):
    for s in network["sections"]:
        if s["train_no"] == train_no and s["from"] == from_code and s["to"] == to_code:
            return s
    return {}


def sample_event(section_attrs, season, rng):
    """Decide whether a discrete delay event fires on this section, and if
    so, which cause and how many extra minutes it costs."""
    prob = BASE_EVENT_PROB
    weights = dict(BASE_CAUSE_WEIGHTS)

    if section_attrs.get("single_line"):
        prob *= 1.4
        weights["congestion_preceding_train"] += 0.15

    effect = SEASON_EFFECTS.get(season, {})
    if effect:
        applies_flag = effect.get("applies_to_flag")
        season_active_here = (applies_flag is None) or (
            section_attrs.get(applies_flag) in ("medium", "high")
        )
        if season_active_here:
            prob *= effect["event_prob_multiplier"]
            for cause, boost in effect.get("cause_boost", {}).items():
                weights[cause] = weights.get(cause, 0) + boost

    if rng.random() > prob:
        return None, 0.0

    causes, w = zip(*weights.items())
    cause = rng.choices(causes, weights=w, k=1)[0]

    # Base duration distribution per cause (minutes), then season duration
    # multiplier if this season is active on this section.
    base_duration_params = {
        "signal_halt": (3, 10),
        "temporary_speed_restriction": (5, 15),
        "congestion_preceding_train": (3, 18),
        "crew_changeover_delay": (3, 12),
        "weather": (6, 25),
        "chain_pulling_other": (2, 6),
    }
    lo, hi = base_duration_params[cause]
    duration = rng.uniform(lo, hi)

    if effect:
        applies_flag = effect.get("applies_to_flag")
        season_active_here = (applies_flag is None) or (
            section_attrs.get(applies_flag) in ("medium", "high")
        )
        if season_active_here:
            duration *= effect.get("duration_multiplier", 1.0)

    return cause, duration


def simulate_train_run(train, network, season, rng, inherited_initial_delay_min=0.0):
    """Simulate one historical run of one train. Returns a list of per-station
    outcome records."""
    stations = train["stations"]
    route_type = train["route_type"]
    recovery_fraction = RECOVERY_FRACTION.get(route_type, 0.12)

    current_delay = inherited_initial_delay_min
    records = []

    # First station: departure only, delay = inherited (e.g. from a
    # connecting-train propagation).
    first = stations[0]
    sched_dep0 = _hhmm_to_minutes(first["sched_dep"])
    records.append({
        "station": first["code"],
        "sched_arr": None,
        "sched_dep": first["sched_dep"],
        "actual_arr": None,
        "actual_dep": _minutes_to_hhmm(sched_dep0 + current_delay) if sched_dep0 is not None else None,
        "delay_min": round(current_delay, 1),
        "causes": [] if current_delay == 0 else ["preceding_train_connection"],
    })

    for i in range(len(stations) - 1):
        s_from = stations[i]
        s_to = stations[i + 1]
        sched_dep_from = _hhmm_to_minutes(s_from["sched_dep"])
        sched_arr_to = _hhmm_to_minutes(s_to["sched_arr"])

        scheduled_duration = (sched_arr_to - sched_dep_from) % (24 * 60)

        section_attrs = section_lookup(network, train["train_no"], s_from["code"], s_to["code"])

        noise_multiplier = rng.lognormvariate(0, NOISE_SIGMA)
        natural_duration = scheduled_duration * noise_multiplier

        cause, event_minutes = sample_event(section_attrs, season, rng)

        # Recovery claws back part of existing backlog using schedule padding.
        recovered = current_delay * recovery_fraction
        current_delay = max(0.0, current_delay - recovered)

        section_delay_delta = (natural_duration - scheduled_duration) + event_minutes
        current_delay = max(0.0, current_delay + section_delay_delta)

        causes_this_station = [cause] if cause else []

        actual_arr_min = sched_arr_to + current_delay
        # small extra halt if something went wrong, otherwise same as sched dep
        actual_dep = None
        if s_to["sched_dep"] is not None:
            sched_dep_to = _hhmm_to_minutes(s_to["sched_dep"])
            actual_dep = _minutes_to_hhmm(sched_dep_to + current_delay)

        records.append({
            "station": s_to["code"],
            "sched_arr": s_to["sched_arr"],
            "sched_dep": s_to["sched_dep"],
            "actual_arr": _minutes_to_hhmm(actual_arr_min),
            "actual_dep": actual_dep,
            "delay_min": round(current_delay, 1),
            "causes": causes_this_station,
        })

    return records
