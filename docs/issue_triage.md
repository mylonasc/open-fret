# Baseline acceptance and remaining issues

**Update:** research ablations are deferred by maintainer direction; #12 must
remain untouched. The identity/duration, full-song, strict encoding and pipeline
acceptance implementation is documented in [engineering acceptance](engineering_acceptance.md).
The original ledger below records the initial audit, not the latest closure list;
GitHub closure comments record each tested batch.

The current trained model is the accepted development starting point, per the
maintainer's direction. This is not a held-out benchmark, proof of a loss
improvement, or a public model release. Do not interrupt training, rewrite its
artifacts, reshuffle its evaluation data, or tune on test results to close issues.

## First closure batch

- **#17 — MIDI-anchored decoding:** implemented in `open_fret/generation.py`,
  shared by inference, exports and training previews. Real T5 tests cover cached
  beam scoring, lookahead chord feasibility, unisons, capo, repeated riffs,
  determinism, input/output budgets, unsupported pitches and EOS. Guarantees
  concern input attacks/onset times and distinct strings **at each onset**;
  sustained occupancy and complete-song boundary policy remain #7/#9/#4 work.
- **#3 — fresh training / resume:** explicit resume now fails before creating
  output metadata if the latest/requested checkpoint is absent or incomplete.
  Weights-only initialization uses `--init-from` into a new run, not resume.
  A CPU CLI integration test trains step 1, resumes to step 2, verifies Adafactor
  step counters and scheduler state, and reloads the saved model/tokenizer.
  Existing nonempty output directories cannot be reused for fresh training.

Verification (no production training or test-set evaluation):

```bash
OMP_NUM_THREADS=2 python -m pytest tests/test_training.py tests/test_loss_cli.py tests/test_generation.py -q
```

The resume test explicitly hides CUDA and uses synthetic data. Training currently
auto-selects CUDA with fp16 when available, otherwise CPU/full precision. There
is no universal GPU-memory or runtime guarantee; changing precision/device CLI
policy is separate from verifying the existing supported paths.

## Remaining acceptance ledger

These issues stay open. Existing code is not sufficient evidence of every
criterion; the following are closure gates, not claims that all listed work is
absent.

| Issues | Remaining gate / next action |
| --- | --- |
| #2 parser | Audit extended-parser syntax rejection/counts and representative corpus results against independent fixtures; synchronize supported-technique claims. |
| #4 aligned windows | Prove cached/uncached equivalence, absolute onset coverage and explicit sustained-note boundary policy, not just equal counts. |
| #5 / #19 splits | Canonical version-family and timing/multiplicity fingerprints, training-only augmentation, deterministic manifests and leakage assertions. Do not call evaluation on a reshuffled split unseen by existing weights. |
| #6 / #18 evaluation | Complete frozen-panel strata, failure denominators, source/family uncertainty and baseline/provenance reports. Anchored F1 measures constrained input preservation, not learned fingering quality. |
| #7 inference | End-to-end multi-window MIDI tests, note identities/absolute timing at boundaries, sustained-note and multi-instrument policy, documented errors and artifact loading. |
| #8 / #16 rendering | Exact onset round trips exist; verify browser-imported completeness against expected events and cover source note ends/overlapping sustains explicitly. Preserve original audit previews. |
| #9 correction | Timed identity matching, sustained occupancy and explicit unresolved-violation reporting; avoid unconditional pitch/playability claims. Anchored decoding does not complete this broader contract. |
| #10 pipeline | CPU fixture pipeline that actually trains/resumes/reloads/infers, pinned/tested dependency strategy and CI; fail-fast stage manifests, partial/corrupt/stale artifact tests. |
| #11 Optuna | Persistent two-trial fixture with reporting/pruning/resume/export and provenance; distinguish HPO engineering acceptance from confirmatory retraining and final test reporting. |
| #12 release | Distribution permission, chosen versioned artifact, model card, clean-environment download/reload/inference and frozen final evaluation. Local weights are not a release. |
| #20 losses | Matched-data/budget CE controls and decomposed-loss/risk ablations with generated metrics. Finite gradients and satisfactory baseline training alone do not establish improvement. |
| #21 domains | Declare supported instrument policy, preserve metadata, report rejection/projection counts and evaluate source-balanced sampling without hiding validation failures. |
| #1 / #14 / #15 epics | Keep open until their child engineering/release gates pass or the maintainer explicitly defers research criteria. |

## Decisions to triage

1. **Research versus baseline release:** recommend keeping #14/#20's expensive
   loss/architecture ablations as follow-up research, not prerequisites for this
   baseline. That requires an explicit scope update, not closing them as done.
2. **New split protocol:** #19 requires separate versioned data and retraining.
   Preserve the current run as a historical baseline; don't invalidate or
   overwrite it in place. Schedule retraining separately after split acceptance.
3. **Musical scope:** recommend retaining exact attack/onset preservation as the
   current guarantee. End-to-end sustained durations/occupied strings require
   input-note identity alignment and must remain tracked under #4/#7/#8/#9.
4. **Release destination/licensing:** confirm dataset/model redistribution
   permissions and desired hosting before publishing weights or choosing the
   authorized-training-workflow alternative in #12.

Next engineering priority: #10's clean CPU fixture/CI and dependency workflow,
then full-song boundary and render-completeness acceptance tests. These provide
the evidence needed to close older broad issues without requiring another
production training experiment.
