# Vendored viewer runtime blobs

Regenerate with `python scripts/vendor_viewer_js.py`.

## alphaTab.min.js
- source: npm:@coderline/alphatab (local install): /home/charilaos/Workspace/open-fret/tools/alphatex_parser/node_modules/@coderline/alphatab/dist/alphaTab.min.js
- sha256: 93ad885df118bc0265ed032c47cc3470e4911a538a119e8e6709bfd501f546c2
- bytes: 712852

## wave-roll.umd.js
- source: https://cdn.jsdelivr.net/npm/wave-roll@0.4.0/dist/wave-roll.umd.js
- sha256: f7878a55232ab42bc08082d6bc47ec90f7015b6afe550af37f26141e82a9e510
- bytes: 1218260

## sonivox.sf2
- source: https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/soundfont/sonivox.sf2
- sha256: 7384ea9fe3bddfc048145e004752ef0a2e32a1d4d17757480f0c8b42da9f5c8b
- bytes: 1289890

## Bravura.woff2
- source: npm:@coderline/alphatab (local install): /home/charilaos/Workspace/open-fret/tools/alphatex_parser/node_modules/@coderline/alphatab/dist/font/Bravura.woff2
- sha256: f0c8e9d9462d5c2cddf93145ed54deb00b8db4c2f7162f4da55db6aebc61b6fd
- bytes: 189444

## Bravura-OFL.txt
- source: npm:@coderline/alphatab (local install): /home/charilaos/Workspace/open-fret/tools/alphatex_parser/node_modules/@coderline/alphatab/dist/font/Bravura-OFL.txt
- sha256: 8e76ea7651b265bd047d192a4a796c25bbc904759a2fa9ee7bb9dba958a61158
- bytes: 4514

Bravura.woff2 is SIL Open Font License 1.1 (see Bravura-OFL.txt in this
directory); the viewer registers it as the 'alphaTab' music family so
notation renders with no network.
