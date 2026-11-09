"""Vendor the viewer's runtime JS/soundfont blobs into the repo.

Copies the pinned AlphaTab bundle from the local npm install and downloads
the pinned WaveRoll UMD + SONiVOX soundfont, so notebook viewers
(`open_fret.viewer.show(..., inline_deps=True)`) work with zero CDN access.
Versions mirror open_fret.viewer (ALPHATAB_JS / WAVE_ROLL_JS); rerun after
any viewer dependency bump and commit the result.

Licenses travel with the files (see ui/tab_data_viewer/THIRD_PARTY_NOTICES.md
and vendor/SOURCES.md).
"""

import hashlib
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.viewer import ALPHATAB_JS, SOUNDFONT_URL, WAVE_ROLL_JS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "ui", "tab_data_viewer", "vendor")

FILES = {
    "alphaTab.min.js": ("npm", os.path.join(
        ROOT, "tools", "alphatex_parser", "node_modules",
        "@coderline", "alphatab", "dist", "alphaTab.min.js")),
    "wave-roll.umd.js": ("url", WAVE_ROLL_JS),
    "sonivox.sf2": ("url", SOUNDFONT_URL),
    "Bravura.woff2": ("npm", os.path.join(
        ROOT, "tools", "alphatex_parser", "node_modules",
        "@coderline", "alphatab", "dist", "font", "Bravura.woff2")),
    "Bravura-OFL.txt": ("npm", os.path.join(
        ROOT, "tools", "alphatex_parser", "node_modules",
        "@coderline", "alphatab", "dist", "font", "Bravura-OFL.txt")),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(VENDOR, exist_ok=True)
    lines = ["# Vendored viewer runtime blobs", "",
             "Regenerate with `python scripts/vendor_viewer_js.py`.", ""]
    for name, (kind, src) in FILES.items():
        dest = os.path.join(VENDOR, name)
        if kind == "npm":
            if not os.path.isfile(src):
                raise SystemExit(f"Missing local npm file: {src}\n"
                                 "Run `npm ci --prefix tools/alphatex_parser` first.")
            shutil.copyfile(src, dest)
            origin = f"npm:@coderline/alphatab (local install): {src}"
        else:
            print(f"Downloading {src} ...")
            with urllib.request.urlopen(src, timeout=300) as r, \
                    open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            origin = src
        digest = sha256(dest)
        size = os.path.getsize(dest)
        print(f"{name}: {size} bytes sha256:{digest[:16]}...")
        lines += [f"## {name}", f"- source: {origin}",
                  f"- sha256: {digest}", f"- bytes: {size}", ""]
    with open(os.path.join(VENDOR, "SOURCES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(FILES)} files to {VENDOR}")


if __name__ == "__main__":
    main()
