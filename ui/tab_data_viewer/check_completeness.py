"""Browser acceptance for fixture_pipeline.py's separate viewer (no audit edits)."""
import argparse
import functools
import http.server
import json
from pathlib import Path
import threading


def main(file):
    from playwright.sync_api import sync_playwright
    path = Path(file).resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(path.parent))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(channel='chrome', args=['--mute-audio', '--autoplay-policy=no-user-gesture-required'])
            except Exception:
                browser = pw.chromium.launch(args=['--mute-audio', '--autoplay-policy=no-user-gesture-required'])
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/{path.name}', wait_until='load', timeout=90000)
            page.wait_for_function("document.querySelector('.ofv-tab-render')?._ofvApi?.score", timeout=60000)
            result = page.evaluate("""() => {
                const api = document.querySelector('.ofv-tab-render')._ofvApi;
                const generator = new alphaTab.midi.MidiFileGenerator(api.score, api.settings, {});
                const actual = [];
                for (const t of api.score.tracks) for (const s of t.staves)
                  for (const b of s.bars) for (const v of b.voices)
                    for (const beat of v.beats) for (const n of beat.notes) {
                      if (n.isTieDestination) continue;
                      const d = generator.getNoteDuration(n, beat.playbackDuration);
                      actual.push([beat.absolutePlaybackStart/2,
                        (beat.absolutePlaybackStart+Math.max(d.untilTieOrSlideEnd,d.letRingEnd))/2,n.realValue]);
                    }
                const bundle = Object.values(window.__OFV_BUNDLES)[0];
                return {actual, expected:bundle.expected_notes.map(n=>[n.start,n.end,n.pitch])};
            }""")
            assert sorted(result['actual']) == sorted(result['expected']), result
            page.wait_for_timeout(4000)
            page.evaluate("document.querySelector('.ofv-tab-render')._ofvApi.tickPosition = 480")
            page.locator('.ofv-mode button', has_text='Actual').click()
            page.wait_for_timeout(1500)
            assert page.locator('.ofv-card-head h4').inner_text().startswith('Actual')
            page.locator('.ofv-mode button', has_text='Predicted').click()
            page.wait_for_timeout(1500)
            tick0 = page.evaluate("document.querySelector('.ofv-tab-render')._ofvApi.tickPosition")
            assert abs(tick0 - 480) < 10, tick0
            page.locator('.ofv-btn.primary', has_text='Tab').click()
            page.wait_for_timeout(1500)
            tick1 = page.evaluate("document.querySelector('.ofv-tab-render')._ofvApi.tickPosition")
            assert tick1 > tick0, (tick0, tick1)
            assert not errors, errors
            print(json.dumps({'status': 'passed', 'complete_notes': len(result['actual']),
                              'playback_ticks': [tick0, tick1], 'page_errors': errors}))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    main(ap.parse_args().file)
