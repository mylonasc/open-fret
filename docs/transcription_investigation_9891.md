# Investigation: invented chords, timing, losses and split integrity

## Bottom line

The observed failure is real and measurable. **Preserving input attacks and
rhythm is a more urgent objective than improving pitch-equivalence loss.** There
are also independent rendering defects and confirmed train/validation duplicates.
None is cured by penalizing fretboard movement alone.

This was an investigation, not a training change: the running experiment was
left alone and completed during the audit. Final checkpoint: **9891**, three
epochs; best validation **combined objective** 2.561883. This is not a held-out
test result. No model evaluation on test music was performed.

## Evidence and reproducibility

Run: `models/full-combined-loss-3ep-20261108/`.
Primary evidence: `samples/step-00009891/`; earlier checkpoints/validation
snapshots were compared too. Machine-readable results are in
`analysis/checkpoint-9891/`:

- `saved_samples.json`: all 45 saved song/step comparisons.
- `decode_ablation.json`: controlled generation and teacher-forced diagnostics.
- `render_audit.json`: independent AlphaTab import/completeness checks.
- `browser_render_audit.json`, `latest-viewer.png`: actual Chromium verification.
- `split_audit.json`: group overlaps, duplicate confirmation and data profiles.
- `event_comparison.html`: input/prediction attack timeline with extra notes and
  same-string conflicts highlighted; independent of the broken score renderer.

Reproduction commands (Python with the project's ML dependencies for ablations):

```bash
python scripts/investigate_transcription.py \
  --run-dir models/full-combined-loss-3ep-20261108 --checkpoint-step 9891 \
  --out-dir models/full-combined-loss-3ep-20261108/analysis/checkpoint-9891
# Same arguments, adding --decode-ablation for the controlled CPU decoding probe.
# Same arguments, adding --render-audit for the independent AlphaTab probe.
```

Five fixed examples are a **diagnostic panel**, not an unbiased estimate of
overall accuracy. They are alphabetically selected first windows, include two
12 Stones songs, and should not be the sole model-selection set. Matching below
uses cumulative TS ticks at 480 PPQ, source tuning/capo, duplicate-aware one-to-one
pitch/onset matching, and a predeclared tolerance of 30 ticks. That tolerance is
not constant milliseconds across different tempos. Notes sustaining across an
onset are **not** counted as newly attacked chord notes.

## 1. The raw model invents simultaneous attacks

All five references exactly match their input MIDI note-on pitches and ticks.
Thus the false chords below are present in generated tokens, not introduced by
the reference parser or a neighbor-search correction. Samples are raw decoded
predictions; post-processing was not used for these comparisons.

| Final-checkpoint example | Input notes | Predicted notes | Input chord attacks / all attacks | Predicted chord attacks / all attacks | Same-string-conflict attacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Crash | 194 | 409 | 75 / 119 | 94 / 102 | 50 |
| The Way I Feel | 207 | 295 | 72 / 98 | 57 / 57 | 33 |
| Pahilo Junima | 165 | 317 | **1 / 164** | **106 / 196** | **40** |
| Glory | 191 | 375 | 30 / 125 | 119 / 137 | 43 |
| End Games | 198 | 413 | 83 / 115 | 93 / 100 | 68 |

Across this panel: **955 input notes become 1,809 predictions**. Chord-attack
rate rises from **42.0% to 79.2%**. There are **234 attacks with multiple notes
assigned to the same string**. End Games includes an 18-note simultaneous group
despite its input containing at most two simultaneous attacks.

The almost-monophonic Pahilo example is decisive. At tick 360 the input attacks
one B3, MIDI pitch 59; the model emits `TAB_4_9 TAB_4_9` at that tick. Both notes
have the correct pitch, but they duplicate the same string and are not the
single intended attack. A pitch-distance penalty cannot detect this error.
Later the model also invents pitches and drifts off the input time grid.

Panel timed pitch/onset F1 is only **12.3%**, despite all individual predicted
string/fret values being within the supported ranges. Range validity is not
musical validity. This number is a panel diagnostic, not a test-set score.

### Directly measurable errors to report

1. Excess/missing attacks per **input onset**, including unmatched predicted
   onsets rather than just comparing chord histograms.
2. False-chord rate conditional on an input **single-note attack**.
3. Duplicate-string occupancy at an onset; sustained occupancy is a separate
   check requiring note ends and identities.
4. One-to-one timed pitch precision/recall/F1, including duplicates, insertions,
   omissions and output after the input ends.
5. Duration/time-grid error, EOS termination, premature completion, output-cap
   stops and temporal coverage.

Do not use an unconditional "polyphony is bad" loss: real chords and overlapping
arpeggio sustains must remain legal when present in the input.

## 2. The renderer silently truncates some scores

`open_fret/render.py` emits dotted duration prefixes such as `:8.`. AlphaTab
1.3.0 expects a supported dotted **beat effect**, e.g. `:16 0.1{d}` for a dotted
sixteenth, not that prefix syntax. Its importer can return a partial/default
score without a JavaScript error.

Confirmed in both the independent Node importer and the **actual final HTML
page in Chromium**, with zero page errors:

| Example | Notes in saved prediction | Notes actually imported into displayed score |
| --- | ---: | ---: |
| Crash | 409 | 409 |
| The Way I Feel | 295 | 295 |
| Pahilo Junima | **317** | **15** |
| Glory | **375** | **0** |
| End Games | 413 | 413 |

Rendering the references through the same path also drops Pahilo from 165 notes
to 6 and Glory from 191 to zero. An SVG existing and a transport advancing are
therefore insufficient viewer tests; assert imported note/onset completeness.

There is a second bug in the dotted map: 180 ticks maps to `:8.`, but a dotted
eighth is 360 ticks; 180 is a dotted sixteenth. Other dotted entries have the same
factor-of-two error. Unsupported gaps are also silently approximated or replaced
with quarters, and leading silence is discarded. These can distort playback
timing independently of the model.

**Required fix, not a loss change:** correct duration values and AlphaTex syntax;
test full-score note counts/onsets through the independent importer, including
dots, tuplets, rests and long gaps. Preserve input note ends when available;
TAB-onset tokens alone cannot reconstruct every sustained duration.

No existing training code, rendered pages or samples were overwritten as part
of this investigation. The diagnostic event table bypasses this rendering path.

## 3. Decoding is hiding a deeper collapse

### Blanket three-token repetition bans conflict with real music

The valid target prefix
`TAB_2_0 TS_120 TAB_2_0 TS_120 TAB_2_0`
is banned at its **fifth token** by `no_repeat_ngram_size=3`.
All five saved references contain forbidden repetitions. In a separate,
source-balanced, capo-zero profile, this happens in **503/512 training windows**
and **507/512 validation windows**. Musical riffs are supposed to repeat.

Simply disabling the ban is nevertheless **not** a fix. A controlled ablation
on checkpoint 9891, same five inputs and CPU float32, beam width 4 and cap 512:

| Input EOS explicitly appended | N-gram ban | Total predicted notes | Timed pitch/onset F1 | EOS stops / 5 |
| --- | ---: | ---: | ---: | ---: |
| No, current export path | 3 | 1,714 | 12.06% | 2 |
| No | 0 | 2,048 | 1.60% | 0 |
| Yes, matches training preprocessing | 3 | 1,700 | 9.87% | 2 |
| Yes | 0 | 2,048 | 1.60% | 0 |

Without the guard, outputs collapse into repetitive notes or note/time patterns;
several become enormous same-time groups. The global ban makes the collapse less
obvious, while forbidding legitimate solutions. These controlled CPU generations
are not bit-for-bit replays of the saved CUDA exports; beam rankings can diverge.

The exporter also omits the explicit input EOS that cache/training append.
That mismatch should be fixed, but adding EOS alone did **not** cure this panel.
Replace the repetition workaround with musically meaningful constraints, rather
than disabling it in production without a replacement.

### Teacher-forced probabilities expose weak boundary/stop supervision

With the **correct reference history** and training-style EOS input, checkpoint
9891 assigns the following probabilities on these five examples:

| Expected class | Positions | Mean probability of correct class |
| --- | ---: | ---: |
| TAB note | 955 | 90.27% |
| TS time advance | 634 | 50.17% |
| EOS | 5 | **0.138%** |

At positions where a time shift is required, mean probability mass on TAB tokens
is **49.49%**. The decoder often prefers another note to advancing time; because
consecutive TAB tokens mean a chord, this directly produces the reported symptom.
Poor free-running history then compounds it.

## 4. Better objectives, prioritized

### P0/P1: preserve what MIDI already tells us

**Best next baseline:** input-anchored constrained decoding. MIDI already gives
note identities, pitches, attack cardinality, starts and ends. Let the network
choose fingerings, not compose a new note sequence:

- One fingering slot per input note; copy the input onset/time structure.
- Mask candidate string/fret pairs to that input pitch/tuning/capo.
- Enforce distinct strings for simultaneous notes, with lookahead/backtracking
  so a locally chosen fingering cannot make the remaining chord infeasible.
- Add sustained-string occupancy using input note ends/IDs.
- Allow EOS only when the input slots are consumed; report unplayable inputs.
- Include a lowest-fret/DP fingering baseline using the same input structure.

This prevents invented attacks by construction for supported inputs, without a
new full training run. That is a structural guarantee, **not** evidence that
its fingerings will be ergonomic or match the reference. Those need evaluation.
The existing neighbor-search helper is not sufficient: it consumes pitches
without timing, appends missing notes and does not preserve onset identity.

### P1: separate event type, timing and fingering supervision

Current loss is `0.3 * smoothed_CE + pitch_marginal`, with the pitch term averaged
over note positions but CE averaged over all non-padding positions. A note gets
both objectives; TS/EOS get only downweighted CE. The approximate coefficient
ratio per note versus structural position is
`1 + N_nonpad / (0.3 * N_notes)`, roughly **6–8×** in these examples.
This is coefficient scaling, not a measurement of equal gradient magnitudes.

Further, `-log P(correct-pitch TAB set)` includes another note-type preference:
`-log P(valid TAB) - log P(correct pitch | valid TAB)`.

Test a decomposed objective instead:

`L = lambda_type * CE(event_type) + lambda_time * CE(delta | TS)`
`  + lambda_fingering * CE(fingering | TAB) + lambda_pitch * pitch_loss(TAB-conditional)`

Keep explicit type supervision; conditional pitch loss **alone** would allow
near-zero probability of emitting a note. Normalize/log losses separately by
eligible class and monitor EOS calibration. Tune bounded TS/EOS weights against
coverage and premature-stop rates, not just teacher-forced loss. Do not blindly
give rare EOS unlimited weight: that can replace endless generation with early
termination. Compare with a same-data CE=1 control before blaming CE=0.3 alone;
the earlier CE-only prototype also had greedy-generation degeneration.
Insufficient training or the early LR decay could also contribute; validation
loss was still improving when the three-epoch budget ended. This investigation
does not isolate their causal effects. More epochs would not, however, make a
decoder rule that forbids valid reference riffs correct.

### P1/P2: onset-aware cardinality and conflict risk

Given input onset groups `G`, define an event-level risk with terms for
`sum_g abs(predicted_count(g) - input_count(g))`, unmatched predicted/input
attacks, and `sum_g,s max(count(g,s)-1, 0)` for duplicate strings. Include pitch,
timing, missing coverage and stopping penalties; do not align only by note order
or ignore unmatched onsets. Normalize by input event count and document tolerance.

These are useful metrics immediately. Hard decoded counts/argmax are **not
differentiable** training losses. To train on them, use candidate reranking or
minimum-risk/sequence-level training over sampled/beam candidates with gradients
through sequence log probabilities. A soft cardinality/type surrogate on
teacher-forced slots is cheaper but is not equivalent to free-running counts.

Domain-specific unlikelihood can penalize genuinely invalid continuations such
as another note on an already occupied string at the same onset, or extra slots
beyond the input chord size. **Do not** mark ordinary repeated riffs as negative
examples. Context-aware fingering smoothing must also exclude chord/occupancy
conflicts; independent same-pitch alternatives are not jointly guaranteed legal.

### Lower priority for this failure

Focal loss on rare frets, generic fret-span penalties, and merely splitting
string/fret heads do not enforce attack count/time. Duplicating an open string
has zero fret span and can still be wrong. Return to these improvements after
onset/cardinality preservation is reliable. An encoder/infill model predicting
one fingering at each known MIDI note is a sensible later architecture ablation.

## 5. Better splits and training-data balance

### Confirmed duplicate leakage despite disjoint source IDs

- Exact source-filename group intersections are zero across all splits.
- Normalizing case/punctuation and a terminal numbered version suffix identifies
  **1,314 candidate version families crossing splits**, including **669 across
  train/validation**. These names alone are candidates, not proof of duplicates.
- Among unaugmented introductory windows in train/validation candidate families,
  requiring at least 16 notes, **337 validation windows have exactly identical
  timed MIDI events in training**. **325 have identical complete input tokens
  (including conditioning) and TAB target tokens**.
  Matches occur across **125 families**. This is a lower-bound audit, not a
  full-corpus duplicate percentage or proof that entire arrangements are identical.
- Example: `Aerosmith_-_Dream_On.tex` in training and
  `Aerosmith_-_Dream_On_(3).tex` in validation share an identical 188-note,
  31,200-tick first-window input and target.

**Recommendation:** establish canonical song/version families and exact musical
fingerprints before splitting; keep all tracks, versions, windows and capo
derivatives together. Add near-duplicate/transpose-aware fingerprints where
justified, without merging every common musical motif. Freeze/version the split
and retrain: an existing model must not be tested on reshuffled music it trained
on. More honest splitting may lower measured scores, not cure generation errors.

### The broad attack distributions are similar, but domain curation matters

Profile: seed 42, 512 randomly selected source groups per train/validation split,
one random capo-zero window per source; no test music read.

| Profile | Train | Validation |
| --- | ---: | ---: |
| Pure monophonic windows, all string counts | 93/512 | 88/512 |
| Chord attacks among all attacks | 35.37% | 35.90% |
| Six-string windows | 400/512 | 415/512 |
| Pure monophonic six-string windows | 38/400 | 40/415 |
| Four-string windows | 101/512 | 82/512 |

There is no evidence here of a large train/validation attack-polyphony mismatch
causing the model's near-doubling of notes. However, the corpus includes bass:
for example the sampled four-string `Faria,_Pedro_-_Young_Jazz` has tuning
`[43,38,33,28]` and source program 33. Several sampled Sum 41 tracks similarly
have bass tunings/programs. A nylon playback sound does not make this an
acoustic/classical-guitar training corpus.

Recommended protocol:

1. Curate/label instrument family and intended guitar domain; separate bass and
   other instruments unless mixed-instrument support is intentional. Preserve
   source quality/technique/voice-role metadata rather than silently pooling it.
2. After family deduplication, use group-level stratification by mono/near-mono/
   chord content, rhythm density, tuning, instrument and duration. Never split
   tracks/windows independently to balance strata.
3. Use source-balanced training sampling and a mild monophonic-guitar curriculum
   or mixed sampler; validate its benefit, rather than assuming all chords are
   bad or eliminating them from training.
4. Main validation/test: unaugmented, frozen source-family groups. Apply random
   capo augmentation to training; retain a separately declared conditioned
   challenge set if desired. Currently exactly half of val/test windows are
   synthetic capo variants, so window counts overstate independent examples.
   This correlation is not itself cross-split leakage.
5. Add a fixed source-balanced generation panel and report metrics by mono/riff,
   arpeggio with sustain, genuine chords and rhythmic complexity. Bootstrap by
   song family, not correlated windows. Keep the five demo songs qualitative.
6. Optionally report an artist-disjoint OOD challenge. 3,340/3,426 validation
   source groups share a top-level artist-directory proxy with training; this
   is expected for a song-disjoint split and is **not itself leakage**. Nested
   collection directories are not reliable artist metadata.

## Follow-up implemented after this investigation

The recommendations were implemented as separate, auditable changes; the v3
artifacts above were not modified.

- **Anchored decoding** is integrated into inference/export/sample callbacks. It
  builds one fingering slot per input attack and time shift, masks candidates to
  pitch-equivalent fingerings, enforces distinct strings per onset with
  lookahead feasibility, and gates EOS on consumed slots. Real-model tests cover
  cached beam search, chords, techniques, repeated riffs, budgets and explicit
  unsupported-input errors. On the 24 decodable windows of the frozen 512-window
  panel, anchored decoding preserved **exactly 3,874 input notes/attacks with 0
  same-string conflicts and timed F1 1.0** (4/24 exact reference fingerings).
  Legacy decoding produced 8,324 notes, 1,183 conflicts and F1 ~0.06.
- **Sustained-string occupancy across separate attacks remains unenforced** and
  must not be claimed: the onset representation has no independent source note
  ends.
- **Supervision audit:** 0 invalid labels; 388 validation `[UNK]` labels caused
  by train-only structural vocabulary; 46.4% of sampled train and 39.4% of
  sampled validation windows contain at least one same-onset/same-string
  reference group. `open_fret.curation` now masks those groups at onset level,
  and `scripts/04_cache_dataset.py` records counts and rejects OOV tokens.
- **Loss audit:** `scripts/audit_losses.py` independently reproduced the
  vectorized CE/pitch/type objective on real validation rows to `~1e-7` with
  finite gradients.
- **Curated rerun:** `models/full-curated-3ep-20261109/` trains from scratch with
  the structural-token vocabulary, curated cache, event-type marginal and
  conditional pitch. Its five fixed samples are chosen only from decodable
  validation windows. Results are reported in
  [the run protocol](full_combined_run.md); no result is claimed until measured.

## Proposed next integration order

1. Fix renderer completeness/durations and input EOS preprocessing; add exact
   importer note/onset assertions. No retraining needed for those plumbing fixes.
2. Evaluate input-anchored/pitch-legal/unique-string constrained decoding and a
   deterministic fingering baseline on a frozen validation panel. Replace the
   lexical repetition ban only with a tested structural alternative.
3. Deduplicate/version song-family splits, curate instrument domains, and add
   generation metrics/stratified panels before another large model-selection run.
4. Train matched same-data controls: CE-only, current loss, decomposed type/time
   objective, then cardinality/conflict sequence risk. Select by generated
   fidelity and playability, not scalar CE/composite loss alone.

The current lower validation loss is not enough to declare transcription good.
Do not evaluate the held-out test repeatedly while choosing these changes.

## Research informing the recommendations

- [Shen et al., Minimum Risk Training for Neural Machine Translation (ACL 2016)](https://aclanthology.org/P16-1159/):
  a principled way to optimize non-differentiable sequence-level task risk through
  candidate sequence probabilities. Adaptation to musical event risk is proposed,
  not a demonstrated improvement here.
- [Welleck et al., Neural Text Generation with Unlikelihood Training (2019)](https://arxiv.org/abs/1908.04319):
  supports explicit negative-continuation objectives; music needs context-aware
  invalidity definitions rather than the paper's text-repetition assumptions.
- [Hamberger et al., Fretting-Transformer (2025), §§3.2–3.6, 4.1–4.6](https://arxiv.org/html/2506.14223v1):
  curated acoustic tracks, duplicate cleanup, 20-note inference chunks and
  pitch-preserving post-processing are material parts of its protocol. Matching
  the tiny T5 dimensions alone does not reproduce that study. Its masked-note
  alternative and fingering baselines also motivate preserving known MIDI events.
