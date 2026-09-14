"""
Domain constants for the live-inference app.

IMPORTANT: ROUTE_RECOVERY_FRACTION below is deliberately DUPLICATED from
generator/simulator.py's RECOVERY_FRACTION, not imported. Two reasons:
  1. The real app (app/, models/, api/) should not depend on the generator/
     package -- that's dev-only tooling for producing fake training data,
     and the real app must still run after generator/ is deleted post-hackathon.
  2. The value needs to stay consistent with what the simulator used to
     PRODUCE the historical data this app's models are trained on -- if you
     tune one, tune the other, or the residual model will be learning to
     correct for a baseline assumption that no longer matches how the
     training data was generated. If you retune generator/simulator.py's
     RECOVERY_FRACTION, update this dict to match (see BUILD_LOG.md Step 3
     for why this consistency mattered enough to design around).

Last synced with generator/simulator.py: see BUILD_LOG.md Step 5 (final
tuned values, iteration 5).
"""

ROUTE_RECOVERY_FRACTION = {
    "double_line_trunk": 0.28,
    "single_line_coastal": 0.14,
}
DEFAULT_RECOVERY_FRACTION = 0.20  # fallback for any route_type not listed above


def month_to_season(month: int) -> str:
    """Same coarse calendar mapping as generator/simulator.py's
    month_to_season -- duplicated for the same reason as above. Note the
    real version doesn't need the route_seasonal_risk parameter the
    simulator's version takes, since here we're labeling a live date's
    season generically, not deciding whether a specific section is affected
    (that's `app/features.py`'s job, using the section's own risk flags)."""
    if month in (12, 1):
        return "winter_fog"
    if month in (6, 7, 8, 9):
        return "monsoon"
    if month in (10, 11):
        return "festival_surge"
    return "normal"


CONNECTION_DELAY_PROPAGATION_THRESHOLD_MIN = 20
CONNECTION_DELAY_PROPAGATION_FRACTION = 0.5

# Crew changeover: if the delay entering a crew-changeover section exceeds
# this, the relief crew's duty-time margin is assumed exhausted, adding a
# fixed extra wait for a fresh crew. Both numbers are illustrative
# operational assumptions (no public data on real crew duty-time buffers),
# consistent with how the crew_changeover flag itself was already labeled
# in network_attributes.json.
CREW_CHANGEOVER_RISK_THRESHOLD_MIN = 45
CREW_CHANGEOVER_EXTRA_DELAY_MIN = 20

# Single-line block conflicts: small fixed buffer added on top of the raw
# schedule-overlap calculation, representing real signal/points reset time
# at the passing loop (a real block clearance isn't instantaneous the
# moment the first train's rear clears the section).
SINGLE_LINE_BLOCK_CLEARANCE_BUFFER_MIN = 5

# Recursion guard for multi-hop propagation (crew changeover / single-line
# holds re-triggering further down the SAME train's route, and connecting-
# train chains at hub stations) -- bounds how many hops the API will chase
# before stopping, so a data-entry mistake (e.g. a connecting-train cycle)
# can't cause infinite recursion.
MAX_PROPAGATION_HOPS = 4

# Webhook dispatch loop (see api/dispatch.py). A station's predicted p50
# delay has to move by at least this much between cycles to be worth
# notifying a subscriber about -- otherwise every 0.1-minute model jitter
# would fire an event, which is noise, not signal.
ETA_CHANGE_NOTIFY_THRESHOLD_MIN = 5.0

# How often the background loop re-checks tracked trains. 30s is a demo-
# scale value; a real deployment would tie this to how often live data
# actually refreshes (NTES updates roughly every 5-10 minutes per the
# earlier research in this conversation), not poll faster than the
# underlying data changes.
DISPATCH_INTERVAL_SECONDS = 30

WEBHOOK_POST_TIMEOUT_SECONDS = 5.0