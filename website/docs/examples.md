---
sidebar_position: 6
---

# Examples

import useBaseUrl from '@docusaurus/useBaseUrl';

Five fixed validation windows with **real model predictions** from the
curated run (`full-curated-3ep-20261109`, latest sampled step): input MIDI on
the left, predicted tab on the right. Press play, drag the seek bar, or click
either view; flip **Listen** between Tab and MIDI, and **Predicted / Actual**
where a reference is available.

<div className="ofv-provenance">
Windows from <code>vldsavelyev/guitar_tab</code> — training-data rights are
not established by the published material (see <a href={useBaseUrl('/docs/dataset')}>Dataset</a>).
Included here with the maintainer's explicit go-ahead, one window per song,
predictions by the curated-run checkpoint. Two references below cannot even
be rendered: they assign two frets to one string in a single attack — the
exact defect the cache builder masks during training.
</div>

## 12 Stones — Crash

Driving eighth-note riff at 130 BPM. Its reference carries a same-onset /
same-string conflict, so only the prediction is shown — compare it against
the MIDI instead.

<iframe className="ofv-player" title="12 Stones — Crash" src={useBaseUrl('/examples/crash/')} />

## 12 Stones — The Way I Feel

Mid-tempo 80 BPM verse window with mixed single notes and dyads.

<iframe className="ofv-player" title="12 Stones — The Way I Feel" src={useBaseUrl('/examples/the-way-i-feel/')} />

## 1974 AD — Pahilo Junema

Sparse 72 BPM melodic line with open strings. Like Crash, its reference is
unrenderable (same-onset conflict); the viewer still A/Bs the prediction
against the input MIDI.

<iframe className="ofv-player" title="1974 AD — Pahilo Junema" src={useBaseUrl('/examples/pahilo-junema/')} />

## 1 God — Glory

Dense 133 BPM strummed chords — a workout for same-onset string assignment.

<iframe className="ofv-player" title="1 God — Glory" src={useBaseUrl('/examples/glory/')} />

## 23 P.H. — End Games

120 BPM window balancing melody against accompaniment figures.

<iframe className="ofv-player" title="23 P.H. — End Games" src={useBaseUrl('/examples/end-games/')} />
