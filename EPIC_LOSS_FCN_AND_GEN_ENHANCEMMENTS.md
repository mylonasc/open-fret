# Epic: domain-aware loss functions and generation enhancements

Status: partially implemented. The two S2 loss engines (pitch marginal and
equivalence smoothing), required token metadata, shared Trainer/HPO integration
and fixture tests are implemented. CE remains the default. Quality ablations,
the full S0 evaluation contract and S1 generation work are still pending.
Parent roadmap: [#1](https://github.com/mylonasc/open-fret/issues/1).
GitHub parent issue: [#14](https://github.com/mylonasc/open-fret/issues/14).

Implementation guide: [domain-aware losses](docs/domain_aware_losses.md). A
confirmed parser duration-inheritance bug was fixed first; old processed data
must be rebuilt before these new loss experiments. This is implementation
progress, not evidence that either method passed its quality promotion gate.

Execution update: at the user's explicit request, the combined objective
`0.3 * equivalence-smoothed CE (epsilon 0.1) + 1.0 * pitch marginal` is enabled
for a fresh three-epoch full run after an extended AlphaTab-parser rebuild.
See [the run protocol and inspection paths](docs/full_combined_run.md). The
combined run is not an isolated ablation and does not establish promotion-gate
success; the original separate comparisons below remain necessary.

## Goal and recommendation

Improve **generated musical correctness and playable fingering**, not merely
teacher-forced token accuracy. Retain ordinary T5 cross-entropy (CE) as the
reproducible baseline and integrate opt-in improvements in independently
measurable stages. Do not replace the tokenizer or combine all losses upfront.

Recommended order:

1. Establish trustworthy event-level evaluation and conditioning metadata.
2. Unify decoding and test musical constraints on existing weights.
3. Add differentiable pitch-equivalence supervision; separately ablate
   equivalence-aware label smoothing.
4. Add time-aware playability reranking, then experimental training penalties.
5. Trial focal loss only if frequency-stratified errors justify it.
6. Consider factorized output heads only after simpler approaches plateau.

Expected improvements below are **hypotheses, not measured gains or promises**.
Numeric promotion thresholds are proposed experiment gates, not forecasts.

## Current evidence and constraints

- The model is a T5 with atomic `TAB_<string>_<fret>[_<technique>]` targets,
  interleaved `TS_<ticks>` tokens and EOS; input prefixes encode capo/tuning.
  The current cache has 4,402 tokenizer entries. Targets do not encode note ends.
- Training uses standard teacher-forced CE. The completed three-epoch run has
  best validation loss 1.6436; this is not a musical quality or test-set score.
- A previous 100-window validation probe reported 81% low-diversity outputs and
  0% EOS with greedy decoding; beam search plus a no-repeat guard reported
  0%/100%. The loop heuristic was only `unique_tokens <= 3`: valid repeated
  music can trigger it, and more complex degeneration can evade it. Re-measure
  with better diagnostics before treating these rates as robust baselines.
- Sample export uses beams=4 and no-repeat-ngram=3; normal inference still
  defaults to greedy. A global n-gram ban can suppress legitimate musical
  repetition. EOS alone can also be improved by prematurely ending a song.
- Current post-processing can insert/refinger notes; its results must not be
  mistaken for improvements in raw model predictions.
- Independent issues remain in preprocessing, window boundaries, stitching,
  rendering and evaluation. A custom loss cannot fix corrupted supervision,
  missing time alignment or incorrect decoding plumbing.
- Held-out test data stays untouched during this epic's model/decoder selection.
  Resolve #5's augmentation/duplicate audit before a confirmatory benchmark;
  version any corrected split and rebuild affected artifacts consistently.

## Review of the five suggestions

### 1. Pitch-aware loss — recommended first training experiment (P1)

The motivation is sound: CE has no explicit knowledge that two different TAB
tokens can sound identical. However, CE penalties depend on predicted
probabilities, not simply on whether a mistake is a wrong string or wrong pitch.

**Do not compute a training penalty from argmax/decoded tokens.** Argmax and
Python token parsing do not provide useful gradients to the token logits.
Likewise, squared error on the *mean predicted pitch* is unsafe: equal mass on
two incorrect pitches can average to the correct pitch.

Build a vocabulary metadata table, then derive pitch per example:

`pitch(v, example) = tuning[string(v)-1] + capo + fret(v)`

Use concert MIDI integers, not AlphaTex octave-name assumptions. Define whether
tuning already includes capo and test that capo is applied exactly once. Invalid
strings/frets and non-note tokens have no pitch, not pitch zero. Pitch metadata
for a technique token represents its base fretted pitch, not an entire bend.

For a note target `y`, let `E(y,x)` be valid TAB tokens with the same base pitch
under input `x`. First test a stable log-sum-exp marginal:

`L_pitch = -log(sum(v in E(y,x)) softmax(logits)[v])`

`L = L_CE + alpha * mean_note_positions(L_pitch)`

This rewards total probability on equivalent pitches without giving non-note
outputs a free pass. It approaches zero only when equivalent-note mass approaches
one; predicting a same-pitch alternative does **not** remove the original CE
penalty. CE still supervises exact fingering, timing, technique and EOS.

Later, compare expected bounded pitch distance, not mean-pitch MSE:
`sum(v in valid TAB) p(v) * min(abs(pitch(v)-pitch(y)), d_max) / d_max`,
plus an explicit penalty on probability outside valid TAB tokens. This gives
partial credit to near misses but risks accepting small pitch errors; compare
against the exact pitch marginal rather than assuming distance is better.

Expected outcome: higher raw pitch/onset F1 and fewer wrong-pitch substitutions;
exact source-fingering agreement might fall. Does not ensure chord feasibility,
rhythm, sustained occupancy or correct sequence alignment.

### 2. Biomechanical/playability loss — rerank first (P1), train later (P2)

Consecutive decoder tokens are not necessarily consecutive played notes: notes
in one chord are serialized, and TS tokens separate time positions. A penalty
on adjacent token frets is therefore not a valid general biomechanics model.

Use timed onset groups and a configurable player profile. Separate chord hand
span, position shifts per available time, and sustained string occupancy. Treat
open strings separately from fretted-hand span. A four/five-fret limit is a
heuristic, not a universal physical law; stretches depend on neck position,
finger assignment, barre shapes and player ability. Same-string simultaneous
notes are a firmer constraint than a universal maximum fret jump.

First rerank pitch-valid beam candidates with transparent penalties. Follow
with optional differentiable surrogate costs on note distributions at *aligned
reference onsets*. Teacher forcing supplies reference history, not free-running
predicted history: measured generation gains are required before promotion.
Note ends must come from an explicit alignment to input MIDI or a versioned
duration representation; TAB onsets alone cannot establish sustained occupancy.

Expected outcome: fewer same-string conflicts, extreme stretches and rapid
position shifts, potentially at some latency cost. Avoid making every passage
low-position or flattening intended technique. This is not a guarantee of human
playability; report unresolved constraints and the profile used.

### 3. Domain-aware label smoothing — recommended alternative (P1)

Use the same context-dependent equivalence table. For note targets, assign
`q(y)=1-epsilon` and distribute `epsilon` across valid same-pitch alternatives
excluding `y`; use a normalized uniform distribution initially. If no valid
alternative exists, keep a hard target. Do not smooth onto TS, EOS, invalid
frets, or incompatible technique variants. Start by leaving technique-bearing
targets hard until equivalence of technique semantics is defined.

`L_smooth = -sum(v) q(v) * log_softmax(logits)[v]`

Unlike an additive pitch term, this directly relaxes exact-fingering supervision.
Use masks for known chord occupancy where available; independently valid
alternatives are not necessarily jointly valid in a chord. Without a full chord
assignment, explicitly call this token-local smoothing, not chord validity.

Expected outcome: less overconfidence in one annotator's fingering and improved
same-pitch alternatives/calibration. It can reduce exact TAB accuracy or introduce
chord conflicts. Compare CE+pitch versus smoothed CE separately before testing a
combination; both encourage pitch equivalence and may be redundant.

### 4. Decoupled multi-task loss — defer architecture changes (P3)

Independent string/fret heads can share statistical strength, but correct fret
on the wrong string can produce a very wrong note. Giving that prediction
partial credit is not intrinsically musically correct. Independent heads may
also compose combinations absent from the training data or invalid for a tuning.

First test auxiliary event-type/string/fret heads on decoder hidden states while
retaining the atomic output vocabulary and original generation interface.
Mask string/fret losses to note targets; keep TS/EOS/technique supervision in the
main head. Compare adding a pitch head and a consistency constraint as ablations.

Only if that helps, prototype `p(type) * p(string | type, history) *
p(fret | string, type, history)` with joint pitch/legal-pair constraints.
Define timing and techniques explicitly; do not accidentally change sequence
length and context budget without measuring the effect. A factorized decoder
requires versioned checkpoints/tokenization and updated inference/rendering.

Expected outcome: possible better data efficiency and rare-combination coverage;
highest integration cost and uncertain generated-quality benefit. Full
factorization is a separate architecture experiment, not a loss-function swap.

### 5. Focal loss — conditional experiment, not default (P2)

First measure token/fret/string frequencies on **training only**, then validation
errors by support and difficulty. Low frets/lower strings being dominant is a
hypothesis to audit, not an established property of this corpus. Focal loss
emphasizes low-confidence examples, which need not be rare: noisy or ambiguous
labels can receive the strongest emphasis.

Test note-only `L_focal = -(1-p_y)^gamma * log(p_y)` while retaining ordinary CE
on TS/EOS. Compare `gamma=0` (CE control), 1 and 2. Keep class weighting as a
separate bounded ablation; if used, derive weights from training counts only.
Do not change smoothing, focal weighting and sampling simultaneously.

Expected outcome: possible higher high-fret/rare-position recall, with risks to
common passages, calibration and convergence. Proceed only if rare-position
errors remain material after the earlier stages. Pitch-equivalent lower-fret
alternatives must not be scored as pitch failures merely for avoiding high frets.

## Priorities and staged integration

| Stage / priority | Integration and dependencies | Expected benefit | Cost / risk | Promotion evidence |
| --- | --- | --- | --- | --- |
| S0 / P0 | Event metadata and evaluation; #2, #4, #5, #6, #8, #9 | Makes gains attributable; catches tuning/timing confounds | Medium; prerequisite | Hand-specified fixtures pass; immutable validation manifest and CE baseline reports |
| S1 / P0 | Shared decoding config in training samples, export and inference; #7 | Most immediate leverage on loops, termination and missing/extra notes without retraining | Low–medium; constraints can remove valid music | Coverage-aware EOS/loop metrics improve without lost repetitions or onset recall |
| S2 / P1 | Pitch marginal loss and equivalence smoothing as separate opt-in runs | Better raw pitch preservation / alternative fingerings | Medium; alignment and loss-scale risk | Pitch gain against same-decoder CE control, stable gradients, no structural regression |
| S3a / P1 | Timed playability beam reranker; #7, #9 | Fewer string/chord conflicts and implausible shifts | Medium; reranking latency and heuristic bias | Fewer violations at matched pitch/onset quality; report unresolvable inputs |
| S3b / P2 | Differentiable playability regularizer after S3a | Learn positional consistency rather than repair afterward | High; surrogate may not transfer to generation | Generated gains beyond reranking-only control |
| S4 / P2 | Note-only focal / bounded class weighting after frequency audit | Improve rare-position support where justified | Low–medium; noise amplification | Rare-bin recall improves with bounded overall-quality regression |
| S5 / P3 | Auxiliary heads, then conditional factorization if warranted | Potential sample efficiency and unseen-combination generalization | High; checkpoint/decoder changes | Gain beyond best simpler method at matched data/compute |
| S6 / P0 release gate | Freeze selected model and decoder; #11, #12 | Trustworthy final quality and reproducible release | Medium | One final held-out evaluation, reload smoke and model card |

### S0 — establish the event and evaluation contract

- Planned `open_fret/token_metadata.py`: vocabulary classification, string/fret/
  technique IDs, valid-pair masks and tuning/capo-aware pitch lookup. Version and
  hash the mapping with tokenizer/config/split identities. Retain per-window
  conditioning and onset IDs through the collator; do not assume standard tuning.
- Extend #6's evaluator with one-to-one timed note matching, explicit onset
  tolerance in ticks and seconds, pitch/onset precision/recall/F1, timing error,
  malformed tokens, valid string/fret rate, missing/extra notes, premature EOS,
  cap stops, temporal coverage, same-string conflicts and profile-based spans/
  shifts. Handle empty predictions; report per-song and support-weighted results.
- Separate exact reference fingering, same-pitch musical correctness and
  playability. Stratify by high fret (15–24), rare training tokens, chord density,
  tuning/capo, length and repeated motifs. Predicted/reference fret bins serve
  different purposes; report denominators to avoid misleading rare-position gains.
- Score raw tokens, constrained output and corrected output independently.
  Rendering/listening are extra checks, not substitutes for token/event metrics.
- Validate loss metadata for all six strings, drop-D/custom tuning, nonzero capo,
  techniques, padding `-100`, TS, EOS, invalid IDs and no-note batches.

### S1 — repair and unify generation before attributing gains to loss

- Planned `open_fret/generation.py`: one serializable configuration and generation
  entry point reused by `scripts/train_model.py`, `scripts/export_samples.py`
  and `scripts/inference.py`; record beams, budgets, penalties and stop reason.
- Benchmark existing greedy and beam+no-repeat settings on the same frozen
  checkpoint; add beam without repetition bans and a grammar-constrained variant.
- Start constraints with token grammar, valid strings/frets and legal TS values.
  Preserve simultaneous notes and repeated motifs; mask impossible same-string
  assignments only where onset state is known. Add tuning/capo-aware pitch or
  coverage constraints only after input/output event alignment is reliable.
- Test optional input-note-count/time-coverage stopping guards, not arbitrary
  minimum lengths. Unsupported/unplayable input must produce explicit reports,
  not an all-masked distribution or forced fabricated notes. Specify fallback.
- Repair EOS budgeting, shared preprocessing and absolute-time window stitching
  through #4/#7. Do not let a new decoding policy conceal window-boundary loss.
- Acceptance fixtures: repeated ostinato, silence/rests, chords, long multi-window
  passage, unsupported pitch and early/late EOS. No silent dropped/duplicated
  boundary onsets; deterministic reproducibility; latency measured per note.

### S2–S5 — loss/architecture experiments behind configuration flags

- Planned `open_fret/losses.py`: pure, unit-testable loss functions. A shared
  Trainer adapter exposes a loss configuration to both normal training and HPO.
  Preserve model serialization and normal T5 generation for S2–S4.
- Keep the current CE behavior as the default. Mask `-100` positions and compute
  each auxiliary mean over eligible note positions, not padded token count.
  Log CE, every auxiliary component, eligible counts, weights and gradient norms.
  Return a finite zero auxiliary loss for no-note batches.
- Use stable log-softmax/log-sum-exp, float32 reductions under mixed precision,
  and device-side lookup buffers. Avoid a dense vocabulary-by-vocabulary distance
  matrix or per-batch `[batch, length, vocab, vocab]` tensors. Benchmark actual
  peak memory and tokens/sec on the available 24 GB GPU before longer trials.
- Proposed small initial search: pitch `alpha` in {0.1, 0.3, 1.0}; smoothing
  `epsilon` in {0.05, 0.1, 0.2}. Include zero-weight controls. Normalize component
  scales before interpreting these weights; optionally ramp auxiliary weights
  after CE warmup. Log the exact schedule and use identical treatment in controls.
- Begin each pitch/smoothing/focal experiment from the same seed/init or identical
  CE checkpoint and reset optimizer state consistently. A fine-tuning pilot is
  not evidence that from-scratch training improves; confirm promising choices
  with matched from-scratch runs before selecting that deployment protocol.
- Test nonzero finite gradients, CE equivalence with features disabled, lower
  pitch loss for same-pitch distributions than wrong-pitch distributions, no
  mean-pitch cancellation loophole, normalized smoothing, and reload/resume.
- S3 introduces event grouping/occupancy and a configurable cost profile before
  a differentiable training surrogate. S5 introduces a checkpoint-format version
  only if a new output representation is actually adopted.

## Experiment budget, comparison matrix and decision gates

Use the existing song-disjoint 5% development protocol after required split/data
audits; never silently compare old and rebuilt caches. Keep a versioned fixed
validation panel including hard cases, plus the five fixed-song browser/listening
exports at every validation. Those five examples are qualitative, not the score.

1. **Decoder pilot:** no training; CE checkpoint with greedy, beam, guarded beam,
   grammar constraints and optional reranking. Select and freeze a decoder
   control before loss comparisons; also retain a fixed greedy diagnostic.
2. **Training pilot:** CE control, CE+pitch, equivalence smoothing, then the better
   pitch method+reranker. Add focal and playability training only if gated in.
   Change one variable per comparison. Factorization waits for these results.
3. **Budget:** initially at most four training configurations × three seeds,
   five dev epochs each, with the existing 30–45 minute trial cap as a backstop.
   Same effective batch, examples, optimizer and schedule; record both examples
   and wall time when a cap binds. Report extra decoding/training cost explicitly.
   Tune small loss-weight grids under a separately declared #11 budget, not an
   uncontrolled Cartesian product of architecture, decoder and loss settings.
4. **Confirm:** repeat finalists on a larger frozen development evaluation and
   matched full-data training budget. Paired bootstrap comparisons resample
   source songs, not correlated windows; report seed variation and uncertainty.

Proposed promotion gates (approve/freeze after S0 establishes baseline variance):

- **Pitch methods:** at least +2 absolute percentage points raw timed pitch/onset
  F1, with a positive song-bootstrap confidence interval; no >1 point drop in
  onset recall or structural validity. Exact-fingering accuracy is diagnostic,
  not a veto on valid alternative fingerings.
- **Playability:** at least 20% relative reduction in the declared primary
  violation rate, with absolute counts/rates reported, and no >1 point loss in
  timed pitch/onset F1. If baseline violations are near zero, reconsider the
  objective rather than claiming a meaningful relative gain.
- **Focal:** at least +3 points recall in the predeclared rare-position bin and
  no >1 point overall timed-F1 drop, with adequate support and uncertainty shown.
- **Generation:** improve degeneration/premature-stop rates without lowering
  timed F1 or legitimate repetition recall. Report length, coverage and cap stops;
  a higher EOS rate alone is not a pass.
- **Resources:** target <=10% training-throughput loss for S2/S4 and <=2× decode
  latency versus the selected beam baseline for reranking. These are engineering
  budgets, not guaranteed costs. Larger costs require an explicit quality tradeoff.
- **Stop/rollback:** reject NaNs, loss-scale domination, broken EOS/grammar,
  systematic tuning/capo bias or gains that appear only after corrective insertion.
  Preserve the CE checkpoint/config as an immediate fallback.

## Final acceptance and GitHub work breakdown

This epic coordinates existing issues rather than closing their unfinished
criteria. Suggested future child work packages (not yet separate GitHub issues):

- [ ] S0 metadata/evaluation contract — depends on #2/#4/#5/#6/#8/#9.
- [ ] S1 unified generation and boundary-safe decoding — extends #7/#4.
- [ ] S2 pitch marginal and domain-aware smoothing ablations — integrates #11.
- [ ] S3a timed playability reranker and constraint reports — extends #9/#7.
- [ ] S3b optional differentiable playability loss after reranking evidence.
- [ ] S4 conditional focal-loss experiment after frequency/error audit.
- [ ] S5 optional auxiliary heads / separately gated factorized decoder.
- [ ] S6 frozen final evaluation, documentation and release — #6/#11/#12.

Done means tested opt-in components with backward-compatible CE defaults,
machine-readable matched-budget ablations and full model/decoder/data provenance;
generation quality gains meet predeclared gates, not just lower custom loss.
Publish raw and corrected metrics, limitations and cost. Freeze all selection
choices before the single final held-out evaluation under #12's protocol.
If a method fails its gate, document and defer it rather than enabling it anyway.
