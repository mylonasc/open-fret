---
sidebar_position: 2
---

# Dataset and preprocessing

## Source

Training data comes from **`vldsavelyev/guitar_tab`**: GuitarPro 3/4/5
tablatures converted to AlphaTex, downloaded with `scripts/00_download.py`
(pinned revision `e6a6dbbccf7cad2e057770cae5ebf8257ec36382`).

:::caution Provenance
The published card and loader declare **no license**, and the upstream
converter repository carries none either — so training-data rights are not
established by the public material. Public access is not a rights grant.
The full evidence trail lives in `docs/dataset_licensing.md`, and clearer-
licensed alternatives are tracked in issue #22. The song excerpts on the
[examples page](examples.md) are included with the maintainer's explicit
go-ahead, with per-song source attribution.
:::

## Pipeline

Each stage is a separate script; caches are versioned and never mixed:

| Stage | Script | What happens |
|---|---|---|
| Download | `00_download.py` | Fetch the HF snapshot, atomic traversal-safe extract |
| Parse | `01_process.py` | AlphaTab 1.3.0 parses each track to attack/release events (pitch, string, fret, timing); strict subset, rejects malformed syntax visibly |
| Vocabulary | `01.5_generate_vocab.py` | Collect tokens; `--structural-lists` adds `CAPO_*`/`TUNING_*`/`NO_*`/`NF_*`/`TS_*` from all splits (conditioning, not leakage — `TAB_*` targets stay train-only) |
| Tokenize | `02_tokenize.py` | Train a WordLevel tokenizer on the rebuilt corpus |
| Split | `03_split.py` | Family/fingerprint-disjoint train/validation/test lists; frozen five-window validation panel |
| Cache | `04_cache_dataset.py` | Numerical cache + manifest; same-onset/same-string reference conflicts masked at group level (labels → `-100`) |
| Augment | `05_augment_train.py` | **Train-only** deterministic capo augmentation; validation/test stay capo 0 |

Final corpus (`loss-v4-curated`): **843,886 train / 105,394 validation /
104,920 test** windows, vocab **6,631**, zero `[UNK]` labels, 441,322 masked
windows (23.6M masked note tokens) where the reference assigned two frets to
one string in a single attack.

## Worked example: raw events to tokens

Take a real window (`1974_AD_-_Pinjada_Ko_Suga.tex`, track 3, standard tuning,
capo 0). After parsing, each note is an event with absolute ticks and a
validated fingering (`pitch == tuning[string-1] + capo + fret`):

| note | start | end | pitch | string | fret | check |
|---|---|---|---|---|---|---|
| 1 | 0 | 120 | 78 | 1 | 14 | 64 + 0 + 14 = 78 ✓ |
| 2 | 120 | 240 | 76 | 1 | 12 | 64 + 0 + 12 = 76 ✓ |
| 3 | 240 | 360 | 79 | 1 | 15 | 64 + 0 + 15 = 79 ✓ |

The **model input** (`midi` tokens) keeps on/off events and timing, with the
conditioning prefix first:

```
CAPO_0 TUNING_64_59_55_50_45_40
NF_79 NO_78 TS_120 NF_78 NO_76 TS_120 NF_76 NO_79 TS_120 ...
```

`NO_<p>` / `NF_<p>` are note-on/off for pitch `p`; `TS_<n>` advances the clock
`n` ticks (480 ticks = one beat); a leading `NF_79` closes a note already
ringing when the window starts. Off-before-on ordering at equal ticks is
enforced by `notes_to_tokens`.

The **target** (`tab` tokens) keeps only attacks and gaps — durations are
recovered at render time, and ties are re-derived from note identities:

```
TAB_1_14 TS_120 TAB_1_12 TS_120 TAB_1_15 TS_120 TAB_1_14 TS_120 ...
```

`TAB_<string>_<fret>` (with optional `_<technique>` suffix) marks one attack;
chords are consecutive `TAB_*` tokens with no `TS_*` between them. The final
`[EOS]` (not shown) terminates the sequence.

## Token vocabulary at a glance

- `CAPO_<n>` — capo position, always first; added exactly once downstream.
- `TUNING_<p…>` — uncapoed concert-MIDI open pitches, high string first;
  `-1` marks an unavailable string (never a sounding pitch).
- `NO_<p>` / `NF_<p>` — input note on/off, `p` in 0–127.
- `TS_<n>` — time shift in ticks.
- `TAB_<s>_<f>[_<tech>]` — target attack on string `s`, fret `f`
  (0–24), optional technique (`b`/`s`/`v`).
- `[EOS]` — end of sequence; one fingering slot per input attack is enforced
  at decode time, so attacks can never be invented or dropped silently.

Reproduce the tokenized corpus with the commands in
`docs/domain_aware_losses.md` (rebuild timing section).
