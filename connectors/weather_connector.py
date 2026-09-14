"""
Real connector for Open-Meteo weather -- both live/forecast and historical
archive (ERA5 reanalysis, 1940-present, free, no API key required).

STATUS: written against Open-Meteo's documented REST interface. NOT executed
here (api.open-meteo.com / archive-api.open-meteo.com are not reachable from
this sandbox). Run from a machine with normal internet access.

Two separate base URLs are used deliberately:
  - api.open-meteo.com/v1/forecast       -> current conditions + forecast
  - archive-api.open-meteo.com/v1/archive -> historical (ERA5), for any past
    date, used to backfill real weather onto real historical train dates
    once you have a real outcomes archive (or to sanity-check the synthetic
    generator's weather assumptions against what actually happened).
"""

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = "temperature_2m,precipitation,visibility,wind_speed_10m"


def get_live_weather(lat: float, lon: float):
    """Current + short-range forecast weather for a station's coordinates."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "timezone": "auto",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_historical_weather(lat: float, lon: float, start_date: str, end_date: str):
    """Real historical weather (ERA5 reanalysis) for a station's coordinates
    between start_date and end_date (YYYY-MM-DD). Note ERA5 has a reporting
    lag of a few days, so very recent dates may not be available yet --
    use get_live_weather's forecast/past_days for the last week instead."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# Real station coordinates for the seed set, so the connectors above can be
# called directly once network access is available.
STATION_COORDS = {
    "HWH": (22.5839, 88.3428),   # Howrah Jn
    "NDLS": (28.6431, 77.2197),  # New Delhi
    "BCT": (18.9696, 72.8194),   # Mumbai Central
    "CSMT": (18.9402, 72.8355),  # Mumbai CSMT
    "MAO": (15.3809, 73.9581),   # Madgaon Jn
    "ASR": (31.6096, 74.8737),   # Amritsar Jn
}


if __name__ == "__main__":
    lat, lon = STATION_COORDS["NDLS"]
    print("Fetching live weather for New Delhi...")
    print(get_live_weather(lat, lon))
