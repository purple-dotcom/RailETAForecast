"""
End-to-end smoke test of api/main.py using FastAPI's TestClient (in-process,
no real network binding needed -- appropriate given this sandbox can't
expose a listening port to the outside world anyway, and this proves the
actual wiring, not just each module in isolation).
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def show(label, resp):
    print(f"\n{'=' * 3} {label} {'=' * 3}")
    print(f"status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    r = client.get("/health")
    show("GET /health", r)

    r = client.get("/eta/10103", params={"current_station": "ROHA", "current_delay_min": 15.0,
                                          "on_date": "2026-07-14"})
    show("GET /eta/10103 (mid-journey, 15 min late at ROHA, monsoon date)", r)

    r = client.get("/eta/12301", params={"on_date": "2026-01-05"})
    show("GET /eta/12301 (cold start, winter-fog date)", r)

    print("\n\n" + "#" * 20 + " MULTI-HOP PROPAGATION SCENARIOS " + "#" * 20)

    r = client.get("/eta/12301", params={"current_station": "GAYA", "current_delay_min": 55.0,
                                          "on_date": "2026-01-05"})
    show("GET /eta/12301, 55 min late at GAYA -> should trigger crew-changeover hold at DDU, "
         "AND should chain into 12013 via NDLS since the extra hold pushes it over threshold", r)

    r = client.get("/eta/12301", params={"current_station": "GAYA", "current_delay_min": 10.0,
                                          "on_date": "2026-01-05"})
    show("GET /eta/12301, only 10 min late at GAYA -> no crew-changeover trigger expected", r)

    r = client.get("/eta/10103", params={"current_station": "ROHA", "current_delay_min": 20.0,
                                          "on_date": "2026-07-14"})
    show("GET /eta/10103, 20 min late at ROHA -> should trigger single-line opposing-train "
         "note for 10104 on the ROHA-RN block (informational, doesn't change 10103's own ETA)", r)

    r = client.get("/eta/12301", params={"current_station": "GAYA", "current_delay_min": 90.0,
                                          "on_date": "2026-01-05"})
    show("GET /eta/12301, 90 min late at GAYA -> big enough to survive the recovery curve and "
         "actually trigger the crew-changeover hold at DDU (55min earlier test recovered to just "
         "under threshold and correctly did NOT trigger -- this one should)", r)

    r = client.get("/eta/10104", params={"current_station": "MAO", "current_delay_min": 320.0,
                                          "on_date": "2026-07-14"})
    show("GET /eta/10104, 320 min late leaving MAO -> should trigger the SINGLE-LINE SELF-HOLD "
         "branch (10103 already committed to the shared block) and re-propagate the extra wait "
         "through every remaining station, not just note it", r)

    r = client.post("/webhook/subscribe", json={"url": "https://example.com/hooks/eta", "train_no": None})
    show("POST /webhook/subscribe", r)

    r = client.post("/simulate/eta_change", json={
        "train_no": "10103", "station": "RN",
        "old_eta_p50": "13:20", "new_eta_p50": "13:41", "delta_min": 21.0,
    })
    show("POST /simulate/eta_change", r)

    r = client.get("/eta/99999")
    show("GET /eta/99999 (unknown train -- expect 404)", r)