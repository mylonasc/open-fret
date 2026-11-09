"""Headless-browser smoke test for tab_data_viewer (Issue #13).

Serves the repo over HTTP, loads demo/example_output.html in Chromium,
and asserts: React mounted, AlphaTab rendered SVGs, the single transport
plays (seek bar + roll playhead advance together), the slider seeks both
views, roll clicks seek the score, Predicted/Actual preserves position,
mute toggles, and zoom enlarges/resets the score.

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
        check("WaveRoll rendered piano roll",
              page.evaluate("() => document.querySelectorAll('.ofv-piano-roll canvas').length") == 1)
        check("no legacy MIDI player",
              page.evaluate("() => !document.querySelector('midi-player')"))
        check("transport play enabled",
              page.evaluate("() => ![...document.querySelectorAll('.ofv-root .ofv-transport button')].find(b=>b.textContent.includes('▶ Play')).disabled"))

        state = lambda: page.evaluate(
            "() => ({seek: document.querySelector('.ofv-seek').value,"
            " roll: document.querySelector('.ofv-piano-roll')._ofvRoll.getState().currentTime,"
            " tick: document.querySelector('.ofv-tab-render')._ofvApi.tickPosition,"
            " head: document.querySelector('.ofv-card-head h4').textContent})")

        page.click("text=▶ Play")
        page.wait_for_timeout(4000)
        s0 = state()
        check("single transport plays, tab + roll advance together",
              float(s0["seek"]) > 0.5 and abs(float(s0["roll"]) - float(s0["seek"])) < 0.8,
              f"{s0}")
        check("play button shows pause",
              page.evaluate("() => [...document.querySelectorAll('.ofv-transport button')].some(b=>b.textContent.includes('⏸ Pause'))"))
        check("green playback cursor tracks the score",
              page.evaluate("() => { const e = document.querySelector('.ofv-tab-render .at-cursor-bar');"
                           " return e && getComputedStyle(e).backgroundColor.includes('34, 197, 94'); }"))

        # Listen toggle: pause + seek to 25% first (deterministic), flip to the
        # input MIDI (same player, position kept, stays paused), flip back to
        # the tab, then flip mid-playback to cover resume.
        page.click("text=⏸ Pause")
        page.wait_for_timeout(500)
        target25 = page.evaluate("() => Number(document.querySelector('.ofv-seek').max) * 0.25")
        page.eval_on_selector(".ofv-seek", """(el, v) => {
          const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
          setter.call(el, String(v));
          el.dispatchEvent(new Event("input", {bubbles: true}));
        }""", target25)
        page.wait_for_timeout(800)
        page.evaluate("""() => { const b=[...document.querySelectorAll('.ofv-mode button')]
          .find(x=>x.textContent.trim()==='MIDI'); b&&b.click(); }""")
        page.wait_for_timeout(3000)
        sm = state()
        check("Listen MIDI shows the MIDI score",
              "Input MIDI" in sm["head"], f"{sm}")
        check("Listen flip preserves position while paused",
              abs(float(sm["seek"]) - target25) < 0.5 and sm["tick"] > 0, f"target={target25:.2f} {sm}")
        check("MIDI score still renders notation",
              page.evaluate("() => document.querySelectorAll('.ofv-root svg').length") > 0)
        page.evaluate("""() => { const b=[...document.querySelectorAll('.ofv-tab-controls .ofv-mode')[0].querySelectorAll('button')]
          .find(x=>x.textContent.trim()==='Tab'); b&&b.click(); }""")
        page.wait_for_timeout(3000)
        st = state()
        check("Listen flip back restores the tab score",
              "tab" in st["head"].lower() and abs(float(st["seek"]) - target25) < 0.5, f"{st}")
        page.click("text=▶ Play")
        page.wait_for_timeout(1500)
        page.evaluate("""() => { const b=[...document.querySelectorAll('.ofv-mode button')]
          .find(x=>x.textContent.trim()==='MIDI'); b&&b.click(); }""")
        page.wait_for_timeout(3000)
        sr = state()
        check("Listen flip mid-playback resumes in sync",
              abs(float(sr["roll"]) - float(sr["seek"])) < 0.8 and float(sr["seek"]) > target25,
              f"{sr}")
        page.evaluate("""() => { const b=[...document.querySelectorAll('.ofv-tab-controls .ofv-mode')[0].querySelectorAll('button')]
          .find(x=>x.textContent.trim()==='Tab'); b&&b.click(); }""")
        page.wait_for_timeout(2000)

        # slider seek: pause first for a deterministic assertion, use the
        # native value setter so React's tracker fires. Seek to 60% of the
        # score (never the exact end: hitting the final tick finishes
        # playback and resets to zero, like most players).
        page.click("text=⏸ Pause")
        page.wait_for_timeout(500)
        target = page.evaluate("() => Number(document.querySelector('.ofv-seek').max) * 0.6")
        page.eval_on_selector(".ofv-seek", """(el, v) => {
          const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
          setter.call(el, String(v));
          el.dispatchEvent(new Event("input", {bubbles: true}));
        }""", target)
        page.wait_for_timeout(1200)
        s1 = state()
        check("slider seeks score + roll",
              abs(float(s1["seek"]) - target) < 0.3 and abs(float(s1["roll"]) - target) < 0.4,
              f"target={target:.2f} {s1}")

        page.click("text=🔇 Mute")
        check("mute toggles",
              page.evaluate("() => document.body.innerText.includes('Unmute')"))
        page.click("text=🔈 Unmute")

        # score click-to-seek moves the same shared position
        page.click("button[title='Stop and return to the start']")
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
        check("score click seeks shared position", tick1 is not None and tick1 > 0,
              f"tick={tick1}")

        # piano-roll click seeks the score too
        roll_box = page.evaluate(
            "() => { const r = document.querySelector('.ofv-piano-roll');"
            " if (!r) return null; const b = r.getBoundingClientRect();"
            " return {x: b.x, y: b.y, w: b.width, h: b.height}; }")
        if roll_box:
            page.mouse.click(roll_box["x"] + roll_box["w"] * 0.5,
                             roll_box["y"] + roll_box["h"] * 0.5)
            page.wait_for_timeout(1200)
        s2 = state()
        check("roll click seeks score",
              float(s2["seek"]) > 0.5 and s2["tick"] > 0, f"{s2}")

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
