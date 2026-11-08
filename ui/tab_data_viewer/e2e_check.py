"""Headless-browser smoke test for tab_data_viewer (Issue #13).

Serves the repo over HTTP, loads demo/example_output.html in Chromium,
and asserts: React mounted, AlphaTab rendered SVGs, MIDI plays with
advancing position, tab plays with advancing tick, mute toggle works,
click-to-seek moves the playhead, zoom enlarges/resets the score.

Requires: playwright + a Chromium (playwright cache or Google Chrome).
Run:  python ui/tab_data_viewer/e2e_check.py [--url URL]

Exit code 0 = all green, 1 = failure (details printed).
"""
import functools
import http.server
import os
import socketserver
import sys
import threading

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main(url=None):
    from playwright.sync_api import sync_playwright

    server = None
    if url is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
        server = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/ui/tab_data_viewer/demo/example_output.html"
    print(f"page: {url}")

    errors, failed_req = [], []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome",
                args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
        except Exception:
            browser = pw.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("requestfailed", lambda r: failed_req.append(r.url[:120]))
        page.goto(url, wait_until="load", timeout=90000)
        page.wait_for_timeout(20000)

        check("React mounted",
              page.evaluate("() => !!document.querySelector('.ofv-root h3')"))
        check("no fallback banner",
              page.evaluate("() => !document.body.innerText.includes('Viewer bundle failed')"))
        n_svg = page.evaluate("() => document.querySelectorAll('.ofv-root svg').length")
        check("AlphaTab rendered notation", n_svg > 0, f"svg={n_svg}")
        check("MIDI custom elements registered",
              page.evaluate("() => !!customElements.get('midi-player') && !!customElements.get('midi-visualizer')"))
        check("tab play enabled",
              page.evaluate("() => ![...document.querySelectorAll('.ofv-root button')].find(b=>b.textContent.includes('▶ Tab')).disabled"))

        page.click("text=▶ MIDI input")
        page.wait_for_timeout(4000)
        t0 = page.evaluate("() => document.querySelector('midi-player').currentTime")
        page.wait_for_timeout(2000)
        t1 = page.evaluate("() => document.querySelector('midi-player').currentTime")
        check("MIDI plays + seeks position", bool(
            page.evaluate("() => document.querySelector('midi-player').playing")) and t1 > t0,
            f"{t0}->{t1}")

        page.click("button[title='Stop MIDI']")
        page.click("text=▶ Tab")
        page.wait_for_timeout(6000)
        tick = page.evaluate("""() => (document.querySelector('.ofv-root').innerText.match(/tick (\\d+)/) || [])[1] || null""")
        check("tab plays, tick advances", tick is not None and int(tick) > 0, f"tick={tick}")

        page.click("text=Mute tab")
        check("mute toggles",
              page.evaluate("() => document.body.innerText.includes('Unmute tab')"))

        # click-to-seek: stop playback, click mid-score, playhead must jump
        page.click("button[title='Stop tab']")
        page.wait_for_timeout(1000)
        box = page.evaluate(
            "() => { const s = document.querySelector('.ofv-root svg');"
            " if (!s) return null; const r = s.getBoundingClientRect();"
            " return {x: r.x, y: r.y, w: r.width, h: r.height}; }")
        if box:
            page.mouse.click(box["x"] + box["w"] * 0.7,
                             box["y"] + box["h"] * 0.3)
            page.wait_for_timeout(1500)
        tick1 = page.evaluate(
            "() => { const r = document.querySelector('.ofv-tab-render');"
            " return r && r._ofvApi ? r._ofvApi.tickPosition : -999; }")
        check("click seeks playhead", tick1 is not None and tick1 > 0,
              f"tick={tick1}")

        # zoom: score must get taller at 110%, reset restores 100%
        h0 = page.evaluate(
            "() => document.querySelector('.ofv-tab-render').getBoundingClientRect().height")
        page.click(".ofv-zoom >> text=A+")
        page.wait_for_timeout(4000)
        zh = page.evaluate(
            "() => ({h: document.querySelector('.ofv-tab-render').getBoundingClientRect().height,"
            " label: document.querySelector('.ofv-zoom-label').textContent})")
        check("zoom enlarges score", zh["h"] > h0 + 5 and zh["label"] == "110%",
              f"{h0:.0f}->{zh['h']:.0f} label={zh['label']}")
        page.click(".ofv-zoom-label")
        page.wait_for_timeout(3000)
        check("zoom resets",
              page.evaluate("() => document.querySelector('.ofv-zoom-label').textContent") == "100%")
        browser.close()

    real_errors = [e for e in errors if "404" not in e and "favicon" not in e.lower()]
    check("zero JS errors", len(real_errors) == 0, "; ".join(real_errors[:3]))
    check("zero failed requests", len(failed_req) == 0, "; ".join(failed_req[:3]))
    if server:
        server.shutdown()
    print("RESULT:", "PASS" if not FAILURES else f"FAIL ({FAILURES})")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    u = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else None
    sys.exit(main(u))
