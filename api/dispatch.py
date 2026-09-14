"""
The actual webhook dispatch system -- the piece that was missing before
(api/main.py's /simulate/eta_change only ever constructed a payload by
hand; nothing watched for changes or sent anything on its own).

Deliberately has NO dependency on api/main.py (avoids a circular import,
since main.py needs to import this) -- callers pass in a `predict_fn`
(main.py's predict_eta) rather than this module importing it directly.

HOW "NEW DATA ARRIVED" IS REPRESENTED HONESTLY: there's no live GPS feed in
this environment. A real deployment would have a poller calling
ntes_connector.get_live_status() and feeding fresh current_delay_min values
into track_train() below. Here, track_train() is called explicitly (by a
client, or by the demo/test script) whenever new information is available --
the dispatch loop's actual job, and the part that's genuinely built and
tested here, is the DIFF-AND-FIRE logic once new data exists, which is
identical either way. This module doesn't pretend to solve "how do we get
live GPS" -- that's connectors/ntes_connector.py's job, already flagged in
BUILD_LOG.md as untested-live.
"""

import asyncio
from datetime import datetime, timezone

import httpx

from app.domain_constants import ETA_CHANGE_NOTIFY_THRESHOLD_MIN, DISPATCH_INTERVAL_SECONDS, WEBHOOK_POST_TIMEOUT_SECONDS

# ---------------------------------------------------------------------------
# In-memory state. A real deployment would persist both of these (a DB row
# per subscriber, a cache/DB row per tracked train) so they survive a
# restart -- kept in-memory here since this is a single-process demo.
# ---------------------------------------------------------------------------

WEBHOOK_SUBSCRIBERS = []   # [{"url": str, "train_no": str|None}]
TRACKED_TRAINS = {}        # train_no -> {"current_station", "current_delay_min", "on_date", "last_snapshot": {station: delay_min}}

_dispatch_log = []          # recent fired events, for /dispatch/status visibility -- not a durable audit log


def add_subscriber(url: str, train_no: str = None):
    WEBHOOK_SUBSCRIBERS.append({"url": url, "train_no": train_no})
    return len(WEBHOOK_SUBSCRIBERS)


def track_train(train_no: str, current_station: str = None, current_delay_min: float = 0.0, on_date: str = None):
    """(Re-)registers a train for tracking, or updates its known live state
    if already tracked. This is the function a live-data poller would call
    on every fresh NTES read; here it's called explicitly."""
    existing = TRACKED_TRAINS.get(train_no, {})
    TRACKED_TRAINS[train_no] = {
        "current_station": current_station,
        "current_delay_min": current_delay_min,
        "on_date": on_date,
        "last_snapshot": existing.get("last_snapshot", {}),
    }


def untrack_train(train_no: str):
    return TRACKED_TRAINS.pop(train_no, None) is not None


async def _fire_webhook(url: str, payload: dict) -> dict:
    """POSTs one event to one subscriber. Failures are caught and reported,
    not raised -- one unreachable subscriber must not stop the cycle or
    affect any other subscriber."""
    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_POST_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload)
        return {"url": url, "ok": resp.status_code < 400, "status_code": resp.status_code}
    except Exception as e:
        return {"url": url, "ok": False, "error": str(e)}


async def run_dispatch_cycle(predict_fn):
    """One full pass over every tracked train: recompute its prediction,
    diff each station's p50 delay against the last cycle's snapshot, and
    fire an ETAChangedEvent to every matching subscriber for any station
    whose predicted delay moved by more than the notify threshold.

    Returns a list of {"train_no", "station", "delta_min", "fired_to": [...]}
    for visibility/testing -- this is what proves the loop actually did
    something on a given tick, not just that it ran without error."""
    fired_events = []

    for train_no, state in list(TRACKED_TRAINS.items()):
        try:
            prediction = predict_fn(
                train_no,
                current_station=state["current_station"],
                current_delay_min=state["current_delay_min"],
                on_date=state["on_date"],
            )
        except Exception as e:
            _dispatch_log.append({"train_no": train_no, "error": f"predict_fn failed: {e}",
                                   "timestamp": datetime.now(timezone.utc).isoformat()})
            continue

        last_snapshot = state["last_snapshot"]
        new_snapshot = {}

        for row in prediction["stations"]:
            station = row["station"]
            new_delay = row["predicted_delay_min"]
            new_snapshot[station] = {"predicted_delay_min": new_delay, "eta_p50": row["eta_p50"]}

            old = last_snapshot.get(station)
            if old is None:
                continue  # first time seeing this station -- nothing to diff against yet

            delta = new_delay - old["predicted_delay_min"]
            if abs(delta) < ETA_CHANGE_NOTIFY_THRESHOLD_MIN:
                continue

            payload = {
                "event": "eta_changed", "train_no": train_no, "station": station,
                "old_eta_p50": old["eta_p50"], "new_eta_p50": row["eta_p50"],
                "delta_min": round(delta, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            matching = [s for s in WEBHOOK_SUBSCRIBERS if s["train_no"] in (None, train_no)]
            fire_results = await asyncio.gather(*[_fire_webhook(s["url"], payload) for s in matching])

            event_record = {"train_no": train_no, "station": station, "delta_min": round(delta, 1),
                             "payload": payload, "fired_to": fire_results}
            fired_events.append(event_record)
            _dispatch_log.append(event_record)

        state["last_snapshot"] = new_snapshot

    return fired_events


async def background_loop(predict_fn, interval_seconds=None):
    """Runs run_dispatch_cycle on a fixed interval, forever, until the
    surrounding process/task is cancelled (FastAPI cancels this on
    shutdown). Started once from api/main.py's startup event."""
    interval_seconds = interval_seconds or DISPATCH_INTERVAL_SECONDS
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await run_dispatch_cycle(predict_fn)
        except Exception as e:
            _dispatch_log.append({"error": f"background cycle failed: {e}",
                                   "timestamp": datetime.now(timezone.utc).isoformat()})


def get_status():
    return {
        "tracked_trains": TRACKED_TRAINS,
        "subscribers": WEBHOOK_SUBSCRIBERS,
        "recent_log": _dispatch_log[-20:],
    }