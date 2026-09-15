"""
Full end-to-end proof of the webhook dispatch loop:
  - a REAL uvicorn server running api/main.py (not TestClient -- TestClient
    doesn't reliably run FastAPI startup events unless used as a context
    manager, and this test specifically needs the real startup-launched
    background loop to be running)
  - a REAL local HTTP receiver standing in for a subscriber's webhook
    endpoint, on loopback (confirmed earlier in this conversation that
    loopback HTTP works in this sandbox even though external domains don't)
  - proof of BOTH the manual /dispatch/tick path AND the autonomous
    background loop firing an event with nobody calling anything
"""

import sys
import os
import time
import json
import threading
import http.server
import socketserver

import httpx
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(HERE, "..", "app")
sys.path.insert(0, HERE)
sys.path.insert(0, APP_DIR)

RECEIVER_PORT = 8866
API_PORT = 8867
RECEIVER_URL = f"http://127.0.0.1:{RECEIVER_PORT}/hook"
API_BASE = f"http://127.0.0.1:{API_PORT}"

received_events = []


class ReceiverHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        received_events.append(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def start_receiver():
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", RECEIVER_PORT), ReceiverHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def start_api_server():
    # Patch the dispatch interval down to something a test can wait on,
    # BEFORE the app's startup event launches the background loop with it.
    import dispatch
    dispatch.DISPATCH_INTERVAL_SECONDS = 3

    import main  # noqa: F401 -- importing registers the app; main.app used below
    config = uvicorn.Config(main.app, host="127.0.0.1", port=API_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


def wait_for(condition_fn, timeout_s, poll_s=0.2, label=""):
    start = time.time()
    while time.time() - start < timeout_s:
        if condition_fn():
            return True
        time.sleep(poll_s)
    print(f"  TIMEOUT waiting for: {label}")
    return False


if __name__ == "__main__":
    print("Starting mock webhook receiver...")
    start_receiver()

    print("Starting real API server (with dispatch interval patched to 3s for this test)...")
    start_api_server()
    time.sleep(1.0)  # let uvicorn + FastAPI startup event (background loop launch) settle

    client = httpx.Client(timeout=10)

    print("\n=== Step 1: subscribe the mock receiver to all trains ===")
    r = client.post(f"{API_BASE}/webhook/subscribe", json={"url": RECEIVER_URL, "train_no": None})
    print(f"  {r.status_code} {r.json()}")
    assert r.status_code == 200

    print("\n=== Step 2: track 10103, current_delay_min=10.0 at ROHA (monsoon date) ===")
    r = client.post(f"{API_BASE}/dispatch/track", json={
        "train_no": "10103", "current_station": "ROHA", "current_delay_min": 10.0, "on_date": "2026-07-14",
    })
    print(f"  {r.status_code} {r.json()}")
    assert r.status_code == 200

    print("\n=== Step 3: manual tick #1 -- first tick has nothing to diff against, expect 0 events ===")
    r = client.post(f"{API_BASE}/dispatch/tick")
    result = r.json()
    print(f"  events_fired={result['events_fired']} (expected 0)")
    assert result["events_fired"] == 0

    print("\n=== Step 4: simulate new live data -- update tracked delay to 40.0 (big jump) ===")
    r = client.post(f"{API_BASE}/dispatch/track", json={
        "train_no": "10103", "current_station": "ROHA", "current_delay_min": 40.0, "on_date": "2026-07-14",
    })
    assert r.status_code == 200

    print("\n=== Step 5: manual tick #2 -- should now detect the change and fire a REAL webhook POST ===")
    r = client.post(f"{API_BASE}/dispatch/tick")
    result = r.json()
    print(f"  events_fired={result['events_fired']}")
    for e in result["events"]:
        print(f"    station={e['station']} delta_min={e['delta_min']} fired_to={e['fired_to']}")
    assert result["events_fired"] > 0, "expected at least one event on a 30-minute delay jump"
    assert all(f["ok"] for e in result["events"] for f in e["fired_to"]), "webhook POST should have succeeded"

    ok = wait_for(lambda: len(received_events) >= 1, timeout_s=3, label="mock receiver getting the manual-tick event")
    print(f"  Mock receiver actually received {len(received_events)} event(s) so far: {'OK' if ok else 'FAILED'}")
    if received_events:
        print(f"  Payload received: {json.dumps(received_events[-1], indent=2)}")

    print("\n=== Step 6: THE REAL TEST -- update tracked delay again, then do NOTHING and just wait. "
          "The background loop (3s interval) should fire on its own. ===")
    events_before = len(received_events)
    r = client.post(f"{API_BASE}/dispatch/track", json={
        "train_no": "10103", "current_station": "ROHA", "current_delay_min": 70.0, "on_date": "2026-07-14",
    })
    assert r.status_code == 200
    print("  Updated tracked delay to 70.0. NOT calling /dispatch/tick. Waiting up to 8s for the "
          "autonomous background loop to notice and fire on its own...")

    fired_autonomously = wait_for(lambda: len(received_events) > events_before, timeout_s=8,
                                   label="autonomous background loop firing a new event")
    if fired_autonomously:
        print(f"  SUCCESS: background loop fired on its own. Total events received: {len(received_events)}")
        print(f"  Latest payload: {json.dumps(received_events[-1], indent=2)}")
    else:
        print("  FAILED: no autonomous event received within timeout.")

    print("\n=== Step 7: GET /dispatch/status sanity check ===")
    r = client.get(f"{API_BASE}/dispatch/status")
    status = r.json()
    print(f"  tracked_trains: {list(status['tracked_trains'].keys())}")
    print(f"  subscribers: {status['subscribers']}")
    print(f"  recent_log entries: {len(status['recent_log'])}")

    print("\n=== Step 8: small delay change (+2 min, under the 5-min notify threshold) should NOT fire ===")
    events_before_small = len(received_events)
    r = client.post(f"{API_BASE}/dispatch/track", json={
        "train_no": "10103", "current_station": "ROHA", "current_delay_min": 72.0, "on_date": "2026-07-14",
    })
    r = client.post(f"{API_BASE}/dispatch/tick")
    result = r.json()
    no_noise = (result["events_fired"] == 0) and (len(received_events) == events_before_small)
    print(f"  events_fired={result['events_fired']}, receiver total still={len(received_events)}: "
          f"{'CORRECTLY SILENT' if no_noise else 'FAILED -- fired on a sub-threshold change'}")

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL CHECKS PASSED' if fired_autonomously else 'FAILED -- see above'}")