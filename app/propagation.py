"""
Three propagation mechanisms, all deterministic graph-traversal / rule-based
(not learned) -- consistent with the earlier decision to model the network
as a graph data structure with rule-based traversal rather than a GNN.

  1. check_crew_changeover: if predicted delay arriving at a crew-changeover
     station exceeds a risk threshold, add a fixed extra wait for a fresh
     crew. This AFFECTS THE SAME TRAIN's own downstream stations -- the
     caller (api/main.py) must re-run baseline+model forward from the
     changeover station with the incremented delay. That re-propagation is
     what makes this "multi-hop": the extra delay isn't just a note, it
     actually changes every subsequent station's prediction.

  2. check_single_line_conflicts: for each single-line block train_no
     occupies, compares train_no's predicted (delayed) block-occupancy
     window against the opposing-direction train's SCHEDULED window (real
     live delay data on the opposing train isn't available in this call
     chain unless the caller supplies it). Returns two different kinds of
     result:
       - "single_line_hold_self": train_no itself is the one that has to
         wait -- same re-propagation requirement as crew changeover.
       - "single_line_hold_opposing": the OTHER train has to wait --
         informational only, doesn't change train_no's own predictions,
         but IS worth surfacing (a control room cares about both trains).

  3. check_hub_connections: unchanged from the original single-hop version
     -- arriving-train delay at a hub station propagating to a connecting
     train. api/main.py now chains this recursively into the connecting
     train's OWN prediction (including its own crew-changeover /
     single-line checks), which is the other half of "multi-hop".
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from domain_constants import (
    CONNECTION_DELAY_PROPAGATION_THRESHOLD_MIN,
    CONNECTION_DELAY_PROPAGATION_FRACTION,
    CREW_CHANGEOVER_RISK_THRESHOLD_MIN,
    CREW_CHANGEOVER_EXTRA_DELAY_MIN,
    SINGLE_LINE_BLOCK_CLEARANCE_BUFFER_MIN,
)
from baseline import _hhmm_to_minutes, _minutes_to_hhmm


def check_hub_connections(train_no, station_eta_rows, network):
    """Unchanged single-hop hub logic -- kept as its own function since
    api/main.py now calls it as one piece of a larger pipeline."""
    notes = []
    delay_by_station = {r["station"]: r["predicted_delay_min"] for r in station_eta_rows}

    for conn in network.get("connecting_trains", []):
        if conn["arriving_train"] != train_no:
            continue
        hub = conn["hub_station"]
        if hub not in delay_by_station:
            continue
        arriving_delay = delay_by_station[hub]
        if arriving_delay < CONNECTION_DELAY_PROPAGATION_THRESHOLD_MIN:
            continue

        inherited = arriving_delay * CONNECTION_DELAY_PROPAGATION_FRACTION
        notes.append({
            "hub_station": hub,
            "arriving_train": train_no,
            "connecting_train": conn["connection"],
            "arriving_delay_min": round(arriving_delay, 1),
            "inherited_delay_min": round(inherited, 1),
            "note": (f"{train_no} is {arriving_delay:.0f} min late into {hub}; "
                     f"{conn['connection']} may inherit ~{inherited:.0f} min at departure."),
        })

    return notes


def check_crew_changeover(train_no, delay_by_station, network):
    """delay_by_station: {station_code: delay_min}, must include the
    changeover station's ARRIVAL delay to be checked (skipped silently if
    not present -- e.g. the changeover station is behind the train's
    current position, already happened, nothing to check)."""
    triggers = []
    for sec in network["sections"]:
        if sec["train_no"] != train_no or not sec.get("crew_changeover"):
            continue
        to_code = sec["to"]
        if to_code not in delay_by_station:
            continue
        arrival_delay = delay_by_station[to_code]
        if arrival_delay < CREW_CHANGEOVER_RISK_THRESHOLD_MIN:
            continue

        triggers.append({
            "type": "crew_changeover_risk",
            "changeover_station": to_code,
            "arrival_delay_min": round(arrival_delay, 1),
            "extra_delay_min": CREW_CHANGEOVER_EXTRA_DELAY_MIN,
            "note": (f"{train_no} predicted {arrival_delay:.0f} min late into {to_code} -- "
                     f"exceeds the assumed relief-crew duty-time margin ({CREW_CHANGEOVER_RISK_THRESHOLD_MIN} min). "
                     f"Assume an additional {CREW_CHANGEOVER_EXTRA_DELAY_MIN} min wait for a fresh crew "
                     f"before departing {to_code}."),
        })
    return triggers


def check_single_line_conflicts(train_no, delay_by_station, seed, network, opposing_delay_min=0.0):
    """delay_by_station must include both boundary stations of a block for
    that block to be checked (same reasoning as crew changeover -- blocks
    behind the train's current position are silently skipped).

    opposing_delay_min: assumed live delay of the OTHER train in each block,
    if known. Defaults to 0 (assume it's running to schedule), since a
    single train's /eta call has no live information about a different
    train unless the caller looked it up separately."""
    trains_by_no = {t["train_no"]: t for t in seed["trains"]}
    train = trains_by_no.get(train_no)
    if train is None:
        return []
    stations = {s["code"]: s for s in train["stations"]}

    conflicts = []
    for block in network.get("single_line_blocks", []):
        occupant = next((o for o in block["occupants"] if o["train_no"] == train_no), None)
        if not occupant:
            continue
        opposing = next((o for o in block["occupants"] if o["train_no"] != train_no), None)
        if not opposing:
            continue

        a, b = occupant["direction"].split("->")
        if a not in stations or b not in stations:
            continue
        if a not in delay_by_station or b not in delay_by_station:
            continue

        entry_sched = _hhmm_to_minutes(stations[a]["sched_dep"] or stations[a]["sched_arr"])
        exit_sched = _hhmm_to_minutes(stations[b]["sched_arr"])
        entry_actual = entry_sched + delay_by_station[a]
        exit_actual = exit_sched + delay_by_station[b]

        opp_no = opposing["train_no"]
        opp_train = trains_by_no.get(opp_no)
        if opp_train is None:
            continue
        opp_stations = {s["code"]: s for s in opp_train["stations"]}
        oa, ob = opposing["direction"].split("->")
        if oa not in opp_stations or ob not in opp_stations:
            continue

        opp_entry_sched = _hhmm_to_minutes(opp_stations[oa]["sched_dep"] or opp_stations[oa]["sched_arr"])
        opp_exit_sched = _hhmm_to_minutes(opp_stations[ob]["sched_arr"])
        opp_entry_actual = opp_entry_sched + opposing_delay_min
        opp_exit_actual = opp_exit_sched + opposing_delay_min

        overlap = min(exit_actual, opp_exit_actual) - max(entry_actual, opp_entry_actual)
        if overlap + SINGLE_LINE_BLOCK_CLEARANCE_BUFFER_MIN <= 0:
            continue  # no conflict, comfortable buffer either way

        if entry_actual <= opp_entry_actual:
            # train_no is committed to the block first -> the OPPOSING train waits.
            hold_until = exit_actual + SINGLE_LINE_BLOCK_CLEARANCE_BUFFER_MIN
            wait_min = max(0.0, hold_until - opp_entry_actual)
            if wait_min <= 0:
                continue
            conflicts.append({
                "type": "single_line_hold_opposing",
                "block": [a, b],
                "holding_train": train_no,
                "held_train": opp_no,
                "wait_min": round(wait_min, 1),
                "note": (f"{train_no} occupies block {a}-{b} until ~{_minutes_to_hhmm(exit_actual)}; "
                         f"{opp_no} (opposing direction) may be held ~{wait_min:.0f} min at the loop."),
            })
        else:
            # the OPPOSING train is committed first -> train_no itself waits.
            hold_until = opp_exit_actual + SINGLE_LINE_BLOCK_CLEARANCE_BUFFER_MIN
            wait_min = max(0.0, hold_until - entry_actual)
            if wait_min <= 0:
                continue
            conflicts.append({
                "type": "single_line_hold_self",
                "block": [a, b],
                "holding_train": opp_no,
                "held_train": train_no,
                "hold_at_station": a,
                "wait_min": round(wait_min, 1),
                "note": (f"{opp_no} occupies block {a}-{b} until ~{_minutes_to_hhmm(opp_exit_actual)}; "
                         f"{train_no} may be held ~{wait_min:.0f} min before entering at {a}."),
            })

    return conflicts


if __name__ == "__main__":
    import json

    DATA_DIR = os.path.join(HERE, "..", "data")
    with open(os.path.join(DATA_DIR, "seed_trains.json")) as f:
        seed = json.load(f)
    with open(os.path.join(DATA_DIR, "network_attributes.json")) as f:
        network = json.load(f)

    print("=== Crew changeover: 12301 arrives GAYA fine, but 50 min late into DDU ===")
    delay_by_station = {"GAYA": 10.0, "DDU": 50.0}
    for t in check_crew_changeover("12301", delay_by_station, network):
        print(f"  {t['note']}")

    print("\n=== Crew changeover: same train, only 20 min late into DDU (below threshold) ===")
    triggers = check_crew_changeover("12301", {"GAYA": 5.0, "DDU": 20.0}, network)
    print(f"  triggers: {triggers} (expected empty)")

    print("\n=== Single-line: 10103 on time through ROHA-RN block (no conflict expected) ===")
    delay_by_station_ontime = {"ROHA": 0.0, "RN": 0.0}
    conflicts = check_single_line_conflicts("10103", delay_by_station_ontime, seed, network)
    print(f"  conflicts: {conflicts} (expected empty -- 10min real schedule buffer)")

    print("\n=== Single-line: 10103 running 25 min late through ROHA-RN block ===")
    delay_by_station_late = {"ROHA": 25.0, "RN": 25.0}
    conflicts = check_single_line_conflicts("10103", delay_by_station_late, seed, network)
    for c in conflicts:
        print(f"  [{c['type']}] {c['note']}")
    if not conflicts:
        print("  (no conflicts -- unexpected, investigate)")

    print("\n=== Single-line SELF-hold: 10104 badly delayed leaving MAO, "
          "10103 already committed to the same MAO-RN block ===")
    delay_by_station_self = {"MAO": 320.0, "RN": 320.0}
    conflicts_self = check_single_line_conflicts("10104", delay_by_station_self, seed, network)
    for c in conflicts_self:
        print(f"  [{c['type']}] {c['note']}")
        assert c["type"] == "single_line_hold_self", "expected the self-hold branch, not opposing"
    if not conflicts_self:
        print("  (no conflicts -- self-hold branch not exercised, investigate)")