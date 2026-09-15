"""
Verifies the frontend actually works against the real running API:
starts a real uvicorn server, serves the frontend as static files, loads it
in a real headless browser, checks for console errors, clicks through every
tab, and screenshots each one for visual review.
"""

import sys
import os
import time
import threading
import http.server
import socketserver
import functools

import uvicorn
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(HERE, "..", "api")
APP_DIR = os.path.join(HERE, "..", "app")
sys.path.insert(0, API_DIR)
sys.path.insert(0, APP_DIR)

FRONTEND_PORT = 8901
API_PORT = 8767  # must match window.API_BASE in index.html
SCREENSHOT_DIR = os.path.join(HERE, "screenshots")

console_errors = []


def start_frontend_server():
    socketserver.TCPServer.allow_reuse_address = True
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=HERE)
    httpd = socketserver.TCPServer(("127.0.0.1", FRONTEND_PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


def start_api_server():
    import main
    config = uvicorn.Config(main.app, host="127.0.0.1", port=API_PORT, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()


if __name__ == "__main__":
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    print("Starting real API server...", flush=True)
    start_api_server()
    print("Starting frontend static server...")
    start_frontend_server()
    time.sleep(1.5)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(f"PAGE ERROR: {exc}"))

        print(f"\nLoading http://127.0.0.1:{FRONTEND_PORT}/ ...")
        page.goto(f"http://127.0.0.1:{FRONTEND_PORT}/", wait_until="networkidle")
        time.sleep(2)  # let the initial data-load useEffect finish

        print("\n=== Fleet tab ===")
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "01_fleet.png"))
        fleet_rows = page.locator(".board tbody tr").count()
        print(f"  Fleet rows rendered: {fleet_rows} (expected 5)")
        assert fleet_rows == 5

        print("\n=== Train detail tab (click 12301's fleet row explicitly, not just 'first row' -- "
              "Object.keys() on numeric-string keys iterates numerically, so the visual first row "
              "is actually 10103, not 12301; found this the hard way, see BUILD_LOG.md) ===")
        page.click('[data-testid="fleet-row-12301"]')
        time.sleep(1)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "02_detail.png"))
        tl_stations = page.locator(".tl-station").count()
        print(f"  Timeline stations rendered: {tl_stations}")
        assert tl_stations > 0

        print("\n=== Scenario simulator: push 12301's delay to 90min at GAYA ===")
        page.select_option('[data-testid="train-select"]', "12301")
        page.select_option('[data-testid="station-select"]', "GAYA")
        delay_input = page.locator('input[type="number"]')
        delay_input.fill("90")
        page.locator("button", has_text="Apply").click()
        time.sleep(2)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "03_detail_after_simulate.png"))
        notices = page.locator(".notice").count()
        print(f"  Propagation notices rendered after 90min delay: {notices} (expected > 0 -- crew changeover / hub connection)")
        assert notices > 0

        print("\n=== Network tab ===")
        page.click(".rail-tab >> text=Network")
        time.sleep(1)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "04_network.png"))
        edges = page.locator(".net-edge").count()
        stations = page.locator(".net-station-dot").count()
        print(f"  Network edges: {edges}, station dots: {stations}")
        assert edges > 0 and stations > 0

        print("\n=== Live Feed tab ===")
        page.click(".rail-tab >> text=Live Feed")
        time.sleep(1)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "05_feed_before_tick.png"))
        page.click("button:has-text('Trigger dispatch tick now')")
        time.sleep(2)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "06_feed_after_tick.png"))
        feed_rows = page.locator(".feed-row").count()
        print(f"  Feed rows after manual tick: {feed_rows} (expected > 0 -- a baseline tick now runs on "
              f"page load, so this tick has a real prior value for 12301 to diff against, see BUILD_LOG.md)")
        assert feed_rows > 0

        print("\n=== Trust tab ===")
        page.click(".rail-tab >> text=Trust")
        time.sleep(1)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "07_trust.png"))
        stat_rows = page.locator(".stat-row").count()
        season_rows = page.locator(".season-table tbody tr").count()
        print(f"  Stat rows: {stat_rows}, season table rows: {season_rows} (expected 4)")
        assert season_rows == 4

        browser.close()

    print(f"\n{'='*60}")
    print(f"Console errors/warnings captured: {len(console_errors)}")
    for e in console_errors:
        print(f"  {e}")
    print(f"\nScreenshots saved to {SCREENSHOT_DIR}/")
    print("ALL CHECKS PASSED" if not console_errors else "CHECKS PASSED WITH CONSOLE ERRORS -- review above")