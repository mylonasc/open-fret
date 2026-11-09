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

## Worked micro-example

Target `TAB_1_14` at 100 BPM, standard tuning, capo 0: pitch
$64 + 14 = 78$. Genuine same-pitch alternatives satisfy
$\mathrm{tuning}[s-1] + \mathrm{fret} = 78$ within the 24-fret limit, e.g.
`TAB_2_19` ($59 + 19 = 78$) or `TAB_3_23` ($55 + 23 = 78$). The pitch
marginal sums probability over exactly these equivalent tokens; smoothing
splits $\varepsilon$ uniformly across the plain (technique-free) ones.

## Roadmap

Implemented (S2 slice): pitch marginal, conditional variant, event-type
marginal, equivalence smoothing, strict parsing, onset-level curation,
anchored decoding.

Still open, in the epic's recommended order:

1. **Isolated ablations with promotion gates.** The running full models
   combine everything; each term still needs a same-decoder CE-control
   comparison on generated-note metrics (pitch preservation, exact
   fingering agreement, termination/loops) before it earns its place.
2. **Held-out test evaluation.** Test scores are reported, never selected
   on; the S0 evaluation contract (frozen manifest, immutable panel) governs
   this.
3. **Timing/occupancy/playability regularizer.** Current equivalence is
   token-local: alternatives need not be jointly playable, and nothing prices
   hand position, stretches, or string occupancy. This is the biggest known
   modeling gap.
4. **Bend and articulation contours.** The marginal matches base pitch only;
   bends, slides and vibrato need continuous treatment.
5. **Free-running alignment.** All marginals are teacher-forced; aligning
   decoded outputs (not just forced paths) is future work.
6. **Objective discipline.** Different $(w_{\mathrm{ce}}, \alpha, \varepsilon)$
   settings produce incomparable scalars — keep selecting on validation
   generation metrics, not on `eval_loss`.
