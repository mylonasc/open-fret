---
sidebar_position: 3
---

# Loss functions

All objectives are implemented in `open_fret/losses.py` (`TabLoss`, config
`LossConfig`, version 2). The model architecture, tokenizer and inference
interface never change — losses only reshape supervision. Without any flag,
training delegates to Hugging Face's original cross-entropy.

## Notation

At teacher-forced position $(b, t)$ let $\mathbf{l}_{bt}$ be the logits,
$p_{bt} = \mathrm{softmax}(\mathbf{l}_{bt})$, $y_{bt}$ the label, and $A$ the
set of active (non-`-100`) positions. For a TAB target, its sounding pitch is

$$
\pi = \mathrm{tuning}[\mathrm{string}] + \mathrm{capo} + \mathrm{fret},
$$

using the example's concert-MIDI tuning (no standard-tuning fallback) with the
capo added exactly once. Let $C(\pi)$ be all vocabulary TAB tokens sounding
pitch $\pi$ under the example's tuning/capo.

## The composite objective

$$
L = w_{\mathrm{ce}}\,(\mathrm{CE} + \Delta_{\mathrm{smooth}})
  + \alpha_{\mathrm{pitch}}\,\bar{P}
  + \alpha_{\mathrm{type}}\,T
$$

with the components below. TS/EOS/technique positions always keep exact CE.

### 1. Cross-entropy (baseline, always on)

$$
\mathrm{CE} = \frac{1}{|A|}\sum_{(b,t)\in A} -\log p_{bt}(y_{bt})
$$

Ordinary next-token loss. Every comparison in this project is measured
against a same-decoder CE control — and note the warning in the training
logs: **scalar values of different objectives are not comparable as
quality**; model selection uses generated-note metrics on the frozen
validation panel, never the composite loss.

### 2. Pitch marginal

For each TAB position, sum the full-vocabulary probability of *every*
fingering that sounds the target pitch:

$$
P_b = -\log\sum_{c \in C(\pi_b)} p_{bt}(c),
\qquad
\bar{P} = \frac{1}{N_{\mathrm{notes}}}\sum_{b} P_b
$$

**Motivation.** Exact-fingering CE treats a musically identical alternative
fingering as fully wrong. The marginal rewards *any* correct-pitch fingering
while CE still supervises the exact reference technique and placement. The
sum runs over the full softmax, so emitting TS/EOS is never a free pitch
match. It does not model bend contours and does not align free-running
outputs — see the roadmap.

*Current weight: $\alpha_{\mathrm{pitch}} = 1.0$.*

### 3. Conditional pitch marginal

Adds the playable-TAB normalizer, maximizing pitch probability
*conditioned* on emitting a TAB token:

$$
\bar{P}_{\mathrm{cond}} = \bar{P}
  + \frac{1}{N_{\mathrm{notes}}}\sum_{b}
  \log\sum_{c \in \mathrm{playable}} p_{bt}(c)
$$

**Motivation.** The plain marginal also pressures the model to emit *more*
TAB tokens (raising $P(\pi)$ by raising $P(\mathrm{TAB})$), which distorts
the TAB/TS/EOS balance and can inflate note counts. Conditioning removes
that pressure; exact CE still decides *when* notes occur.
Requires `--pitch-loss-alpha > 0` to have an effect.

### 4. Event-type marginal

Macro-average over three observed classes — melodic TAB note, time shift,
EOS — each normalized by its own count before averaging:

$$
T = \frac{1}{3}\sum_{k}
  \Bigl[-\,\mathbb{E}_{(b,t)\in A_k}\,
  \log\sum_{c \in \mathrm{class}_k} p_{bt}(c)\Bigr]
$$

where $k$ ranges over $\mathrm{TAB}$, $\mathrm{TS}$, $\mathrm{EOS}$.

**Motivation.** Rare-but-critical EOS tokens are drowned by thousands of
note and time tokens under flat CE, which is the cheapest explanation for
runaway or truncated generations. Per-class normalization gives termination
its own gradient without touching anything else. All other structural and
technique tokens keep exact CE.

*Current weight: $\alpha_{\mathrm{type}} = 0.3$.*

### 5. Equivalence smoothing

At technique-free TAB targets that *have* vocabulary alternatives, replace
the hard target with $q = (1-\varepsilon)$ on the reference plus
$\varepsilon$ spread uniformly over the other plain same-pitch tokens.
Implemented as CE plus a delta (no second softmax — it reuses CE's
log-probabilities):

$$
\Delta_{\mathrm{smooth}} = \varepsilon\,
  \frac{1}{|A|}\sum_{\mathrm{eligible}}
  \Bigl(\log p_{bt}(y_{bt}) - \mathrm{mean}_{c \in \mathrm{Alt}} \log p_{bt}(c)\Bigr)
$$

**Motivation.** The reference is one valid fingering, not the only one;
hard CE makes the model overconfident about an arbitrary choice. Smoothing
is token-local only — alternatives need not be jointly playable in a chord.

*Current setting: $\varepsilon = 0.1$ with $w_{\mathrm{ce}} = 0.3$.*

### 6. Positional distance

Mean absolute error between the expected fret and the reference fret, over
fretted targets (open strings excluded, as in Noise2Fret):

$$
\bar{P}_{\mathrm{pos}} = \frac{1}{N}\sum_{b} \bigl|\mathbb{E}[f_b] - f^{\mathrm{target}}_b\bigr|,
\qquad
\mathbb{E}[f_b] = \sum_{f} f \cdot P(\mathrm{fret}=f \mid \mathrm{string}_b).
$$

The expectation runs over the same candidate tables as the pitch marginal —
softmax over same-string TAB tokens, no argmax anywhere.

**Motivation.** Pitch-only supervision accepts any correct pitch *anywhere*
on the neck, including unplayable high positions for simple passages. This
term prices the *where*: it rewards putting the hand where the reference
put it, while the pitch marginal still allows equivalent fingerings to share
probability mass.

*Current setting: $\alpha_{\mathrm{pos}} = 0$ (opt-in).*

### 7. Expected hand-span hinge

Per onset group (TABs sounding at the same tick), the expected span is the
max minus the min expected fret over fretted target strings (opens excluded),
hinged at a margin $M$:

$$
\bar{S} = \mathrm{mean}_{\mathrm{groups}}\,
  \max\bigl(0,\ (\max_s \mathbb{E}[f_s] - \min_s \mathbb{E}[f_s]) - M\bigr).
$$

This is one quantity with two literature names: the **stretch** metric of
midi-to-tab (max fret distance within simultaneous onsets, Table 1) and the
**hand-span feasibility loss** of Noise2Fret (their hinge with $M=6$, the max
span observed in GOAT). On this corpus, 98.9% of onsets span ≤ 4 frets and
spans ≥ 5 are the top ~1% outliers — so the default $M = 4$ (a standard
four-fret hand box) fires only on genuinely wide shapes.

*Current setting: $\alpha_{\mathrm{span}} = 0$, $M = 4$ (both opt-in).*

### 8. Neighbourhood-deviation stretch

The trainable form of midi-to-tab's §4.4 relocation rule: for each fretted
note, compare its expected fret against the mean of the ±5 neighbouring
notes' reference frets (an 11-note run, opens excluded), hinged at a margin:

$$
\bar{D} = \mathrm{mean}_{b}\,
  \max\bigl(0,\ |\mathbb{E}[f_b] - \mathrm{mean}_{\pm5}(f^{\mathrm{target}})| - M_d\bigr),
\qquad M_d = 5\ \text{(their MAX\\_DEVIATION)}.
$$

Where the span hinge prices chord *width*, this prices *jumps*: a lone note
far up the neck between open-position phrases. Same expected-fret machinery,
different axis — both are HPO-tunable separately.

*Current setting: $\alpha_{\mathrm{stretch}} = 0$, $M_d = 5$ (both opt-in).*

### Decoding-side relocation (no training)

midi-to-tab §4.4 as a post-pass (`open_fret.postprocess.relocate_deviating_notes`,
opt-in `--relocate-deviating` in `scripts/inference.py`): for each note, if
its fret exceeds the max or deviates from the ±5-note local mean by more
than the threshold, move it to a free open string, else to the free string
whose required fret is closest to the mean. Pitch is preserved exactly by
construction; fully occupied neighbourhoods are left untouched and reported.
On their data this touched only 0.53% of notes — it is a safety net for
outliers, not a voicing engine.

## Worked micro-example

Target `TAB_1_14` at 100 BPM, standard tuning, capo 0: pitch
$64 + 14 = 78$. Genuine same-pitch alternatives satisfy
$\mathrm{tuning}[s-1] + \mathrm{fret} = 78$ within the 24-fret limit, e.g.
`TAB_2_19` ($59 + 19 = 78$) or `TAB_3_23` ($55 + 23 = 78$). The pitch
marginal sums probability over exactly these equivalent tokens; smoothing
splits $\varepsilon$ uniformly across the plain (technique-free) ones.

## Playability weight screen — blind-listening result (2026-11-10)

Eight studies (one weight combination each, 2 trials each, 45-min cap) on the
loss-v4 corpus, all on the shared base (`w_ce=0.3`, pitch 1.0, smoothing 0.1,
type 0.3, conditional). Each trial exported the same fixed 5-song panel with a
playable viewer; selection was by ear only, never by `eval_loss`.

**Method.** Blind A/B in `helper_apps/pair_judge` (sides randomized, trial
identities hidden until voting). Pool: all unordered trial pairs per shared
song (600 pairs). Judged 2026-11-10 20:41–21:14 UTC, single judge (`anon`):
46 judgments — 27 decided (14 left / 12 right / 1 tie) + 19 skipped
(unrenderable or unjudgeable). Elo, base 1500 / K=24, recomputed in vote order
from `helper_apps/pair_judge/votes.db`.

**Winner: `play-pos05-stretch05`** (`--position-alpha 0.5 --stretch-alpha 0.5`).
Study aggregate 7 wins / 8 decided (tie-free win rate 0.88); both its trials sit
in the top 3 by Elo:

| trial | Elo | decided | wins |
|---|---|---|---|
| pos05-stretch05/trial_0 | 1541.8 | 6 | 5 |
| stretch05/trial_1 | 1526.5 | 6 | 4 |
| pos05-stretch05/trial_1 | 1524.0 | 2 | 2 |
| base/trial_0 | 1523.5 | 6 | 4 |
| span05/trial_1 | 1453.3 | 4 | 0 |
| span05-stretch05/trial_1 | 1455.5 | 4 | 0 |

Study win rates: pos05-stretch05 0.88, stretch05 0.71, pos05 0.67, base / all05
/ pos05-span05 ~0.50, span05 0.22, span05-stretch05 0.00 (only 5 decided —
sparse-coverage outlier, treat with caution).

**Core insight.** The stretch term (neighbourhood-deviation hinge, margin 5)
prices what judges actually hear: lone-note *jumps* far from the ±5-note local
mean — stray high-position notes between open-position phrases. The span term
(per-onset max−min hinge, margin 4) prices chord *width*, but on this corpus
98.9% of onsets already span ≤ 4, so it fires only on the top ~1% shapes and
adds almost no audible signal at a 45-minute training budget — span-only and
span-combination studies sink to the bottom. Position alone is neutral (≈ base);
it helps only combined with stretch. Caveats: sparse coverage (46/600 pairs),
one judge, time-capped (directional, not converged) trials.

## Roadmap

Implemented (S2 slice): pitch marginal, conditional variant, event-type
marginal, equivalence smoothing, **positional distance, expected-span hinge,
neighbourhood-deviation stretch, and the midi-to-tab §4.4 relocation
pass**, strict parsing, onset-level curation, anchored decoding.

Related work this builds on:

- [TabCNN](https://github.com/andywiggins/tab-cnn) (Wiggins & Kim, ISMIR
  2019) — the CNN baseline and the standard tab metrics (pitch/tab
  precision/recall/F-measure, tab disambiguation rate) this project
  compares against.
- [Noise2Fret](https://arxiv.org/abs/2608.30854) ([code](https://github.com/RiccardoVib/Noise2Fret)) —
  five auxiliary losses (pitch-class, circle-of-fifths, positional,
  string-similarity, hand-span) with ablations; our positional/span terms
  follow their soft-distribution formulation.
- [MIDI-to-Tab](https://arxiv.org/pdf/2408.05024) — stretch metric, user
  study on playability, and the §4.4 neighbourhood relocation our decoding
  pass implements.

Still open, in the epic's recommended order:

1. **Isolated ablations with promotion gates, now including playability.**
   The running full models combine everything; each term — including the new
   positional/span/stretch weights — still needs a same-decoder CE-control
   comparison on generated-note metrics (pitch preservation, exact
   fingering agreement, termination/loops, plus span/proximity proxies)
   before it earns its place. Per the loss-study protocol, run one Optuna
   study per weight combination (weights are fixed per study, not searched)
   and compare on the frozen panel, never on scalar `eval_loss`.
2. **Held-out test evaluation.** Test scores are reported, never selected
   on; the S0 evaluation contract (frozen manifest, immutable panel) governs
   this.
3. **Timing/occupancy regularizers beyond onsets.** The span hinge prices
   simultaneous width; sustained-string occupancy across onsets and position
   *shifts* (hand travel between onsets) are still unpriced. Deferred with
   the Viterbi/Yazawa-style transition decoding and the data-driven pairwise
   feasibility model — deliberately not implemented here.
4. **Bend and articulation contours.** The marginal matches base pitch only;
   bends, slides and vibrato need continuous treatment.
5. **Free-running alignment.** All marginals are teacher-forced; aligning
   decoded outputs (not just forced paths) is future work.
6. **Objective discipline.** Different $(w_{\mathrm{ce}}, \alpha, \varepsilon)$
   settings produce incomparable scalars — keep selecting on validation
   generation metrics, not on `eval_loss`.
