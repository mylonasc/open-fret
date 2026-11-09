# Bundled components

The tab-data-viewer UMD bundle includes React, React DOM and Scheduler.
Their MIT notice is reproduced below. Open-Fret's root LICENSE applies to
Open-Fret's original contributions, not as a replacement for this notice.

MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Runtime dependencies (loaded externally, not bundled here)

- AlphaTab 1.3.0 (notation, synthesizer, fonts and soundfont):
  https://github.com/CoderLine/alphaTab/tree/v1.3.0
- WaveRoll 0.4.0 (piano-roll rendering):
  https://github.com/crescent-stdio/wave-roll (MIT)
  - PixiJS 8 (renderer): https://github.com/pixijs/pixijs (MIT)
  - Tone.js 15 (bundled audio engine): https://github.com/Tonejs/Tone.js (MIT)
  - @tonejs/midi (MIDI parsing): https://github.com/Tonejs/Midi (MIT)
  - d3-scale (time/pitch scales): https://github.com/d3/d3-scale (ISC)
- html-midi-player 1.5.0 (audio controls and Magenta player):
  https://github.com/cifkao/html-midi-player (BSD-2-Clause)
- Tone.js 14.7.77 (html-midi-player audio):
  https://github.com/Tonejs/Tone.js (MIT)
- Magenta Music 1.23.0 (html-midi-player backend):
  https://github.com/magenta/magenta-js/tree/master/music (Apache-2.0)
- focus-visible:
  https://github.com/WICG/focus-visible (W3C BSD-3-Clause)

Each external component and its audio assets remain subject to its own terms.
