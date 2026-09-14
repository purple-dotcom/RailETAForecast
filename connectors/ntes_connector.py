"""
Real connector for live train data via the `ntes-client` PyPI package.

STATUS: The method names/signatures below were verified by installing
ntes-client and inspecting the actual NTESClient class (pip install
ntes-client; inspect.signature(...)). They are NOT guessed. However, the
actual network calls to indianrail.gov.in have NOT been executed from this
sandbox (that domain isn't reachable here) -- run this file from a machine
with normal internet access to confirm live behaviour before the demo.

Verified NTESClient methods (from package introspection):
    search(query)
    live_status(train_no, start_date)
    schedule(train_no, start_date='')
    station_live(station_code, hours=2)
    train_info(train_no)
    trains_between(from_station, to_station, train_type='XXX')
    pnr_status(pnr)
    exceptions(train_no)

Only `schedule` and `live_status` are used for the ETA pipeline; the rest
are available if you want PNR/search features later.
"""

from ntes import NTESClient
import os

_client = NTESClient()


def get_schedule(train_no: str, start_date: str = ""):
    """Real static timetable: station sequence, scheduled arr/dep, distance.
    Use this to refresh data/seed_trains.json with real, live-pulled times
    instead of the illustrative placeholders currently in that file."""
    return _client.schedule(train_no, start_date)


def get_live_status(train_no: str, start_date: str):
    """Real current running status: last-reported station, delay, ETA at upcoming stations."""
    return _client.live_status(train_no, start_date)


def get_station_live(station_code: str, hours: int = 4):
    """All trains arriving/departing a station in the next `hours` -- useful as a live proxy for 'how many trains are converging on this section right now', i.e. a real-time congestion signal."""
    return _client.station_live(station_code, hours)


if __name__ == "__main__":
    # code added by me just for testing
    schedule_hwh = get_schedule('12301', "07-Sep-2026")
    schedule_mum_raj = get_schedule('12951', "07-Sep-2026")
    schedule_man_ex = get_schedule('10103', "07-Sep-2026")
    schedule_am_jsh = get_schedule('12301', "07-Sep-2026")

    print(schedule_hwh)
    print(schedule_man_ex)
    print(schedule_mum_raj)
    print(schedule_am_jsh)