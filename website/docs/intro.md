---
sidebar_position: 1
---

# Introduction

Open-Fret turns **MIDI note streams into playable guitar tablature**. The
core question is not "which notes" — the input already fixes the pitches and
timing — but **where to play them**: which string, which fret, and when to
shift position, so the result sounds right and a human can actually play it.

## Motivation

Automatic tablature is a *fingering* problem, not a transcription problem.
The same pitch exists in up to six places on the fretboard, and the choice
interacts: an open string rings differently from a fretted note, a stretch
across five frets may be unplayable at tempo, and the previous and next notes
constrain where the hand can be. Heuristic systems (shortest path, fixed
position boxes) produce legal but often unmusical fingerings.

A sequence model can learn these preferences from real tablatures — **if**
the training pipeline preserves attacks, timing and note identity exactly,
and the decoder cannot silently invent notes. That "if" is most of this
project: strict parsing, anchored decoding, and losses that understand that
two fingerings can be pitch-equivalent.

## Methodology in one page

1. **Corpus.** GuitarPro tablatures (`vldsavelyev/guitar_tab`) are parsed
   with AlphaTab 1.3.0 into attack/release/pitch/string/fret events, then
   windowed, tokenized and split by song family so no song leaks across
   train/validation/test. See [Dataset](dataset.md).
2. **Model.** A small T5 encoder-decoder (d_model 128, 3+3 layers, 4 heads)
   trained **from scratch** maps an input token stream
   (`CAPO_*`, `TUNING_*`, `NO_*`/`NF_*` on/off events, `TS_*` time shifts) to
   an output stream (`TAB_<string>_<fret>` attacks, `TS_*` gaps, techniques).
   See [Training](training.md).
3. **Losses.** Plain cross-entropy plus opt-in domain terms: a pitch
   marginal over same-pitch fingerings, a conditional variant, a
   macro-averaged event-type term, and equivalence smoothing. See
   [Losses](losses.md).
4. **Decoding.** Anchored generation emits exactly one fingering slot per
   input attack, masks out pitch-inequivalent tokens, enforces distinct
   strings per onset, and reports unsupported input instead of fabricating
   notes.
5. **Selection.** Model choice uses validation-set metrics only — a frozen
   five-window panel with anchored generation, playable viewers and human
   listening. Test scores are never used for tuning.

## Condensed examples

Each example below pairs the **input MIDI** (piano roll, left) with the
**predicted tab** (notation + tab, right) on one transport: press play, drag
the seek bar, or click either view. Flip **Listen** between Tab and MIDI to
A/B the fingering against the input through the same player, and switch
**Predicted / Actual** to compare fingerings.

- *12 Stones — Crash* (130 BPM): driving eighth-note riff, model prediction
  vs. reference.
- *1974 AD — Pahilo Junema* (72 BPM): sparse melodic line, open strings.
- *1 God — Glory* (133 BPM): dense strummed chords.

Full interactive versions with downloads live on the [examples page](examples.md).
