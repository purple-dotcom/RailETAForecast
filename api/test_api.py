"""
FastAPI app. Single main endpoint (GET /eta/{train_no}) that runs the full
pipeline described in SCHEMA.md: LiveContext -> baseline -> features ->
residual quantile model -> uncertainty widening -> rootcause + propagation
-> StationETA list.

Models, section_stats, seed_trains, and network_attributes are all loaded
ONCE at startup (module-level), not per-request -- LightGBM Boosters and the
SHAP TreeExplainer both have real setup cost.
"""

import os
import sys
import json
import asyncio
from datetime import date
from typing import Optional

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(HERE, "..", "app")
DATA_DIR = os.path.join(HERE, "..", "data")
ARTIFACTS_DIR = os.path.join(HERE, "..", "models", "artifacts")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, HERE)

from app.baseline import compute_baseline, _hhmm_to_minutes, _minutes_to_hhmm
from features import build_feature_rows, FEATURE_NAMES
from app.uncertainty import widen_band
from app.rootcause import RootCauseExplainer
from app.propagation import check_hub_connections, check_crew_changeover, check_single_line_conflicts
from app.domain_constants import month_to_season, MAX_PROPAGATION_HOPS
import dispatch

# ---------------------------------------------------------------------------
# Startup: load everything once.
# ---------------------------------------------------------------------------

with open(os.path.join(DATA_DIR, "seed_trains.json")) as f:
    SEED = json.load(f)
with open(os.path.join(DATA_DIR, "network_attributes.json")) as f:
    NETWORK = json.load(f)
with open(os.path.join(ARTIFACTS_DIR, "section_stats.json")) as f:
    _stats_raw = json.load(f)
    SECTION_STATS = {tuple(k.split("|")): v for k, v in _stats_raw["section_stats"].items()}
    FALLBACK_STATS = _stats_raw["fallback_stats"]

MODELS = {
    name: lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, f"residual_{name}.txt"))
    for name in ("p10", "p50", "p90")
}
EXPLAINER = RootCauseExplainer(MODELS["p50"])

TRAINS_BY_NO = {t["train_no"]: t for t in SEED["trains"]}
STATION_ORDER_BY_TRAIN = {tno: [s["code"] for s in t["stations"]] for tno, t in TRAINS_BY_NO.items()}

app = FastAPI(title="Train ETA Prediction API", version="0.1.0")

# The frontend (frontend/) is served as static files from a different port
# than this API during local dev -- CORS has to be explicit for the browser
# to allow those cross-origin fetch() calls. Wide open (allow_origins=["*"])
# is fine for a local hackathon demo; a real deployment would restrict this
# to the actual frontend's origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def launch_background_dispatch():
    """Starts the real webhook dispatch loop -- see api/dispatch.py. This is
    what makes it an actual loop rather than something that only runs when
    a client happens to call /dispatch/tick: once the app is up, tracked
    trains get re-checked on their own, forever, until the process stops."""
    asyncio.create_task(dispatch.background_loop(predict_eta))


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _predict_segment(train, network, season, current_delay_min, from_station_code,
                      preceding_train_delay_min=None):
    """One baseline+model pass from a given position to the end of the
    route. Does NOT check propagation -- that's the caller's job, since
    triggering a re-propagation means calling this again from a later
    station, not something this function decides on its own."""
    baseline_preds = compute_baseline(
        train, SECTION_STATS, FALLBACK_STATS,
        current_delay_min=current_delay_min, from_station_code=from_station_code,
    )
    feature_rows = build_feature_rows(
        train, network, season,
        current_delay_min=current_delay_min, preceding_train_delay_min=preceding_train_delay_min,
        from_station_code=from_station_code,
    )
    if not baseline_preds:
        return []

    X = np.array([[f[k] for k in FEATURE_NAMES] for f in feature_rows])
    residuals = {name: model.predict(X) for name, model in MODELS.items()}

    station_etas = []
    for i, (b, f) in enumerate(zip(baseline_preds, feature_rows)):
        sched_arr_min = _hhmm_to_minutes(b["sched_arr"])
        p50_delay = b["baseline_delay_min"] + residuals["p50"][i]
        p10_delay = b["baseline_delay_min"] + residuals["p10"][i]
        p90_delay = b["baseline_delay_min"] + residuals["p90"][i]

        widened_p10_delay, widened_p90_delay, confidence_note = widen_band(p10_delay, p50_delay, p90_delay, f)
        top_reasons, cause_tags = EXPLAINER.explain(f)

        station_etas.append({
            "station": b["station"],
            "sched_arr": b["sched_arr"],
            "predicted_delay_min": round(float(p50_delay), 1),
            "eta_p10": _minutes_to_hhmm(sched_arr_min + widened_p10_delay) if sched_arr_min is not None else None,
            "eta_p50": _minutes_to_hhmm(sched_arr_min + p50_delay) if sched_arr_min is not None else None,
            "eta_p90": _minutes_to_hhmm(sched_arr_min + widened_p90_delay) if sched_arr_min is not None else None,
            "confidence_note": confidence_note,
            "top_reasons": top_reasons,
            "cause_tags": cause_tags,
        })
    return station_etas


def predict_eta(train_no, current_station=None, current_delay_min=0.0,
                 preceding_train_delay_min=None, on_date=None,
                 _visited_trains=None, _hop=0):
    """Multi-hop pipeline. Two kinds of hop:

    1. WITHIN one train's own route: a crew-changeover or single-line hold
       triggered partway down the route injects extra delay and re-runs
       _predict_segment from that station onward, so every subsequent
       station's prediction reflects it (not just a side note).

    2. ACROSS trains: a hub-station connecting-train note recursively calls
       predict_eta again for the connecting train, using the (possibly
       hold-adjusted) inherited delay -- so a delay that started as a crew-
       changeover hold on train A can flow into train B's own prediction,
       including train B's own crew-changeover/single-line checks.

    _visited_trains + _hop guard against cycles / runaway recursion (e.g. a
    data-entry mistake creating a connecting-train loop)."""
    _visited_trains = set(_visited_trains or set())

    train = TRAINS_BY_NO.get(train_no)
    if train is None:
        raise HTTPException(status_code=404, detail=f"Unknown train_no: {train_no}")
    if train_no in _visited_trains or _hop > MAX_PROPAGATION_HOPS:
        return None  # cycle/depth guard -- caller treats None as "stop chaining here"
    _visited_trains.add(train_no)

    on_date = on_date or date.today().isoformat()
    season = month_to_season(int(on_date.split("-")[1]))
    station_order = STATION_ORDER_BY_TRAIN[train_no]

    if current_station and current_station not in station_order:
        raise HTTPException(status_code=400, detail=f"Unknown station {current_station} for train {train_no}")

    segment_start_station = current_station
    segment_start_delay = current_delay_min
    all_station_etas = []
    applied_crew_holds = []
    applied_self_holds = []
    opposing_train_notes = []
    already_handled_stations = set()  # stations we've already injected a hold at -- don't re-trigger

    for _ in range(MAX_PROPAGATION_HOPS + 1):
        segment = _predict_segment(train, NETWORK, season, segment_start_delay,
                                    segment_start_station, preceding_train_delay_min)
        if not segment:
            if not all_station_etas:
                raise HTTPException(status_code=400,
                                     detail=f"No downstream stations for train_no={train_no}, "
                                            f"current_station={current_station} (already at/past destination?)")
            break

        all_station_etas.extend(segment)

        delay_by_station = {}
        if segment_start_station:
            delay_by_station[segment_start_station] = segment_start_delay
        for r in segment:
            delay_by_station[r["station"]] = r["predicted_delay_min"]

        crew_triggers = [t for t in check_crew_changeover(train_no, delay_by_station, NETWORK)
                          if t["changeover_station"] not in already_handled_stations]
        conflicts = check_single_line_conflicts(train_no, delay_by_station, SEED, NETWORK)
        self_holds = [c for c in conflicts if c["type"] == "single_line_hold_self"
                      and c["hold_at_station"] not in already_handled_stations]
        opposing_train_notes.extend(c for c in conflicts if c["type"] == "single_line_hold_opposing")

        candidates = []
        for t in crew_triggers:
            candidates.append((t["changeover_station"], t["extra_delay_min"], "crew", t))
        for c in self_holds:
            candidates.append((c["hold_at_station"], c["wait_min"], "single_line", c))

        if not candidates:
            break

        # earliest along the route wins -- resolve the first bottleneck before
        # considering anything further downstream, since fixing/re-simulating
        # from there naturally supersedes any later-station trigger found in
        # this same (now-stale) segment.
        candidates.sort(key=lambda c: station_order.index(c[0]) if c[0] in station_order else 999)
        hold_station, extra_delay, kind, info = candidates[0]

        already_handled_stations.add(hold_station)
        if kind == "crew":
            applied_crew_holds.append(info)
        else:
            applied_self_holds.append(info)

        idx_hold = station_order.index(hold_station)
        all_station_etas = [r for r in all_station_etas if station_order.index(r["station"]) <= idx_hold]

        new_delay_at_hold = delay_by_station[hold_station] + extra_delay
        segment_start_station = hold_station
        segment_start_delay = new_delay_at_hold

    hub_notes = check_hub_connections(train_no, all_station_etas, NETWORK)

    chained = []
    for note in hub_notes:
        chained_result = predict_eta(
            note["connecting_train"], current_station=None,
            current_delay_min=note["inherited_delay_min"], on_date=on_date,
            _visited_trains=_visited_trains, _hop=_hop + 1,
        )
        if chained_result is not None:
            chained.append({"via_hub": note["hub_station"], "connecting_train": note["connecting_train"],
                             "inherited_delay_min": note["inherited_delay_min"], "prediction": chained_result})

    return {
        "train_no": train_no,
        "train_name": train["name"],
        "season": season,
        "stations": all_station_etas,
        "hub_connection_notes": hub_notes,
        "crew_changeover_notes": applied_crew_holds,
        "single_line_self_hold_notes": applied_self_holds,
        "single_line_opposing_train_notes": opposing_train_notes,
        "chained_connections": chained,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/eta/{train_no}")
def get_eta(train_no: str, current_station: Optional[str] = None,
            current_delay_min: float = 0.0, preceding_train_delay_min: Optional[float] = None,
            on_date: Optional[str] = None):
    return predict_eta(train_no, current_station, current_delay_min,
                        preceding_train_delay_min, on_date)


class WebhookSubscription(BaseModel):
    url: str
    train_no: Optional[str] = None  # None = subscribe to all trains


@app.post("/webhook/subscribe")
def subscribe_webhook(sub: WebhookSubscription):
    n = dispatch.add_subscriber(sub.url, sub.train_no)
    return {"status": "subscribed", "n_subscribers": n}


class TrackRequest(BaseModel):
    train_no: str
    current_station: Optional[str] = None
    current_delay_min: float = 0.0
    on_date: Optional[str] = None


@app.post("/dispatch/track")
def track_train(req: TrackRequest):
    """Registers a train for the dispatch loop to watch, or updates its
    known live state if already tracked -- this is what a live-data poller
    would call every time fresh NTES data comes in (see dispatch.py's
    module docstring for why that polling itself isn't built here)."""
    if req.train_no not in TRAINS_BY_NO:
        raise HTTPException(status_code=404, detail=f"Unknown train_no: {req.train_no}")
    dispatch.track_train(req.train_no, req.current_station, req.current_delay_min, req.on_date)
    return {"status": "tracked", "train_no": req.train_no}


@app.delete("/dispatch/track/{train_no}")
def stop_tracking(train_no: str):
    removed = dispatch.untrack_train(train_no)
    return {"status": "untracked" if removed else "was_not_tracked", "train_no": train_no}


@app.post("/dispatch/tick")
async def dispatch_tick():
    """Runs one dispatch cycle immediately, rather than waiting for the
    background loop's next scheduled run. Legitimate for two reasons, not
    just a test hook: it makes behavior deterministic for testing/demo, and
    a push-triggered architecture (fire a tick when a live-data webhook from
    NTES itself arrives, instead of polling on a fixed interval) is a
    perfectly real production pattern too."""
    fired = await dispatch.run_dispatch_cycle(predict_eta)
    return {"events_fired": len(fired), "events": fired}


@app.get("/dispatch/status")
def dispatch_status():
    return dispatch.get_status()


class SimulateEtaChange(BaseModel):
    train_no: str
    station: str
    old_eta_p50: str
    new_eta_p50: str
    delta_min: float


@app.post("/simulate/eta_change")
def simulate_eta_change(evt: SimulateEtaChange):
    """Low-level preview helper: constructs and returns the exact
    ETAChangedEvent payload (see SCHEMA.md) for a HAND-SPECIFIED old/new
    value, without needing a tracked train or a real prediction change.
    Useful for previewing the payload shape or testing a subscriber
    endpoint in isolation. The REAL mechanism is /dispatch/track +
    /dispatch/tick (or the background loop) -- this endpoint does not run
    any prediction or diffing itself."""
    from datetime import datetime, timezone
    payload = {
        "event": "eta_changed", "train_no": evt.train_no, "station": evt.station,
        "old_eta_p50": evt.old_eta_p50, "new_eta_p50": evt.new_eta_p50,
        "delta_min": evt.delta_min,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    matching = [s for s in dispatch.WEBHOOK_SUBSCRIBERS if s["train_no"] in (None, evt.train_no)]
    return {"would_post_to": [s["url"] for s in matching], "payload": payload}


@app.get("/health")
def health():
    return {"status": "ok", "trains_loaded": list(TRAINS_BY_NO.keys())}


@app.get("/network")
def get_network():
    """Read-only: exposes the already-loaded network topology (station
    sequences, single-line blocks, hub connections) for the frontend's
    Network tab. No new logic -- just surfacing SEED/NETWORK, which were
    already loaded into memory at startup for the prediction pipeline."""
    return {
        "trains": [
            {"train_no": t["train_no"], "name": t["name"], "route_type": t["route_type"],
             "seasonal_risk": t.get("seasonal_risk"),
             "stations": [{"code": s["code"], "name": s["name"], "km": s["km"]} for s in t["stations"]]}
            for t in SEED["trains"]
        ],
        "single_line_blocks": NETWORK.get("single_line_blocks", []),
        "connecting_trains": NETWORK.get("connecting_trains", []),
        "sections": NETWORK.get("sections", []),
    }


@app.get("/model/eval_metrics")
def get_eval_metrics():
    """Read-only: the real, saved evaluation results from the last training
    run (models/train_residual_model.py), for the frontend's Trust tab.
    Not recomputed per request -- these are fixed until the model is
    retrained."""
    with open(os.path.join(ARTIFACTS_DIR, "eval_metrics.json")) as f:
        return json.load(f)