# Improvement checkpoint

Execution epic: #15, under loss/generation epic #14; sub-issues #16–#21.

Included work:
- Domain-aware losses, shared training/HPO integration and component logging.
- Extended AlphaTab parsing and reproducible full-run orchestration.
- Final-checkpoint investigation and read-only diagnostic tooling.
- Initial exact-duration rendering repair and independent round-trip tests (#16).
- Initial MIDI-anchored generation module (#17).

Validation at this checkpoint: **114 tests passed** (six existing fork warnings).
The new `open_fret/generation.py` is **untested work in progress**, not integrated
into inference, sample export or training callbacks. Its cache handling and
constraint/chunking behavior still require real-model tests. Sustained-string
occupancy across separate attacks is not enforced.

Rendering still needs saved-checkpoint completeness and browser verification of
separately regenerated previews. Original audit artifacts must remain unchanged.
The onset representation cannot recover independent source note durations.

No new training run has been launched for these improvements. Models, datasets,
local databases and generated audit artifacts remain ignored and are not part
of the Git checkpoint.
