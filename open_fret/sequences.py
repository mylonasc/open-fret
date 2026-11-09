"""MIDI/tab sequence building with musically aligned windowing.

Issue #4: the old pipeline independently truncated input (with note-off
tokens) and target (onsets only) to 512 tokens, so equal token counts
covered different musical spans. This module builds a shared timed event
list first, then windows by musical time/onsets while respecting both
input and target budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .alphatex import ParsedTrack


@dataclass
class FlatNote:
    pitch: int
    string: int
    fret: int
    tech: str | None = None
    note_id: str = ""
    start: int = 0
    end: int = 0


@dataclass
class TimedStep:
    time: int
    ons: list[FlatNote] = field(default_factory=list)
    offs: list[int] = field(default_factory=list)  # pitches


@dataclass
class SequenceExample:
    midi_tokens: list[str]
    tab_tokens: list[str]
    source: str = ""
    track_index: int = 0
    window_index: int = 0
    start_tick: int = 0
    end_tick: int = 0
    capo: int = 0
    tuning: list[int] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    carry_notes: list[dict] = field(default_factory=list)
    window_version: str = "aligned-v2-identities"

    def to_text_pair(self, eos_token: str = "[EOS]") -> tuple[str, str]:
        midi_text = " ".join(self.midi_tokens) + (" " + eos_token if eos_token else "")
        tab_text = " ".join(self.tab_tokens) + (" " + eos_token if eos_token else "")
        return midi_text, tab_text


class SequenceBuilder:
    """Build shared timed steps + flat token streams for one track."""

    def __init__(self, ticks_per_quarter: int = 480):
        self.ticks_per_quarter = ticks_per_quarter

    def build_steps(self, track: ParsedTrack, capo: int = 0) -> tuple[list[TimedStep], list[int]]:
        tuning = track.tuning
        # Map absolute time -> step
        by_time: dict[int, TimedStep] = {}
        identity = 0
        for event in track.events:
            start = event.start
            duration = event.duration
            if start < 0 or duration <= 0:
                raise ValueError("Invalid source note timing")
            for note in event.notes:
                if note.string < 1 or note.string > len(tuning):
                    raise ValueError("Invalid source string")
                if note.fret < 0:
                    raise ValueError("Invalid source fret")
                pitch = tuning[note.string - 1] + note.fret + capo
                step_on = by_time.setdefault(start, TimedStep(time=start))
                step_on.ons.append(FlatNote(pitch=pitch, string=note.string,
                                            fret=note.fret, tech=note.tech,
                                            note_id=str(identity), start=start, end=start + duration))
                identity += 1
                end = start + duration
                step_off = by_time.setdefault(end, TimedStep(time=end))
                step_off.offs.append(pitch)
        steps = [by_time[t] for t in sorted(by_time)]
        return steps, list(tuning)

    def steps_to_tokens(self, steps: list[TimedStep], tuning: list[int],
                        capo: int = 0) -> tuple[list[str], list[str]]:
        tuning_str = "_".join(map(str, tuning))
        midi_seq = [f"CAPO_{capo}", f"TUNING_{tuning_str}"]
        tab_seq: list[str] = []
        last_time = 0
        for step in steps:
            delta = step.time - last_time
            if delta < 0:
                raise ValueError("Timed steps must be sorted")
            if delta > 0:
                midi_seq.append(f"TS_{delta}")
                tab_seq.append(f"TS_{delta}")
            last_time = step.time
            # OFF before ON at same tick is handled by step ordering:
            # emit OFFs first for determinism.
            for pitch in sorted(step.offs):
                midi_seq.append(f"NF_{pitch}")
            for note in sorted(step.ons, key=lambda n: n.pitch):
                midi_seq.append(f"NO_{note.pitch}")
                tech_sfx = f"_{note.tech}" if note.tech else ""
                tab_seq.append(f"TAB_{note.string}_{note.fret}{tech_sfx}")
        return midi_seq, tab_seq

    def build_example(self, track: ParsedTrack, capo: int = 0, source: str = "",
                      track_index: int = 0, window_index: int = 0) -> SequenceExample:
        steps, tuning = self.build_steps(track, capo=capo)
        midi_tokens, tab_tokens = self.steps_to_tokens(steps, tuning, capo=capo)
        start_tick = steps[0].time if steps else 0
        end_tick = steps[-1].time if steps else 0
        return SequenceExample(
            midi_tokens=midi_tokens, tab_tokens=tab_tokens, source=source,
            track_index=track_index, window_index=window_index,
            start_tick=start_tick, end_tick=end_tick, capo=capo, tuning=list(tuning),
        )


class AlignedWindower:
    """Split timed steps into windows covering identical musical spans.

    Windows are cut only at step boundaries. TS deltas are recomputed
    relative to each window start so every window is self-contained.
    """

    def __init__(self, max_input_tokens: int = 512, max_target_tokens: int = 512,
                 max_onsets_per_window: int | None = None,
                 include_prefix_in_input: bool = True):
        self.max_input_tokens = max_input_tokens
        self.max_target_tokens = max_target_tokens
        self.max_onsets_per_window = max_onsets_per_window
        self.include_prefix_in_input = include_prefix_in_input

    @staticmethod
    def _window_cost(n_steps_with_gap: int, n_ons: int, n_offs: int,
                     prefix_len: int) -> tuple[int, int]:
        # Input: prefix + TS gaps + ON + OFF. Target: TS gaps + TAB(ON).
        # n_steps_with_gap = number of steps after the first that need TS.
        in_len = prefix_len + n_steps_with_gap + n_ons + n_offs
        tgt_len = n_steps_with_gap + n_ons
        return in_len, tgt_len

    def window(self, steps: list[TimedStep], tuning: list[int], capo: int = 0,
               source: str = "", track_index: int = 0) -> list[SequenceExample]:
        if not steps:
            return []
        prefix_len = 2 if self.include_prefix_in_input else 0
        tuning_str = "_".join(map(str, tuning))
        prefix = [f"CAPO_{capo}", f"TUNING_{tuning_str}"] if self.include_prefix_in_input else []

        windows: list[SequenceExample] = []
        all_notes = [{"id": f"{source}:{track_index}:{n.note_id}", "start": n.start,
                      "end": n.end, "pitch": n.pitch, "string": n.string,
                      "fret": n.fret, "tech": n.tech}
                     for step in steps for n in step.ons]
        cur: list[TimedStep] = []
        cur_ons = 0
        cur_offs = 0
        cur_gaps = 0  # TS tokens needed (steps after first with time advance)

        def flush(cur_steps: list[TimedStep], window_index: int) -> None:
            if not cur_steps:
                return
            # Re-base times to window start for TS computation.
            t0 = 0 if window_index == 0 else cur_steps[0].time
            rebased = [TimedStep(time=s.time - t0,
                                 ons=list(s.ons), offs=list(s.offs)) for s in cur_steps]
            builder = SequenceBuilder()
            midi_tokens, tab_tokens = builder.steps_to_tokens(
                rebased, tuning, capo=capo)
            # steps_to_tokens always adds prefix; strip if not wanted.
            if not self.include_prefix_in_input:
                midi_tokens = midi_tokens[2:]
            # Ensure prefix correctness (steps_to_tokens uses same scheme)
            if self.include_prefix_in_input:
                midi_tokens = prefix + midi_tokens[2:]
            windows.append(SequenceExample(
                midi_tokens=midi_tokens, tab_tokens=tab_tokens, source=source,
                track_index=track_index, window_index=window_index,
                start_tick=t0, end_tick=cur_steps[-1].time,
                capo=capo, tuning=list(tuning),
                notes=[n for n in all_notes if t0 <= n['start'] <= cur_steps[-1].time],
                carry_notes=[n for n in all_notes if n['start'] < t0 < n['end']]))

        window_index = 0
        for i, step in enumerate(steps):
            add_ons = len(step.ons)
            add_offs = len(step.offs)
            add_gap = 0
            if not cur and window_index == 0 and step.time > 0:
                add_gap = 1
            if cur:
                # Gap needed if time advanced from previous step in window.
                if step.time != cur[-1].time:
                    add_gap = 1
                else:
                    add_gap = 0
            # Onset limit check (count ON notes, not steps)
            if (self.max_onsets_per_window is not None
                    and cur_ons + add_ons > self.max_onsets_per_window
                    and cur):
                flush(cur, window_index)
                window_index += 1
                cur, cur_ons, cur_offs, cur_gaps = [], 0, 0, 0
                add_gap = 0
            new_ons = cur_ons + add_ons
            new_offs = cur_offs + add_offs
            new_gaps = cur_gaps + add_gap
            in_len, tgt_len = self._window_cost(new_gaps, new_ons, new_offs, prefix_len)
            # Reserve 1 slot for EOS in each sequence.
            if cur and (in_len + 1 > self.max_input_tokens or tgt_len + 1 > self.max_target_tokens):
                flush(cur, window_index)
                window_index += 1
                cur, cur_ons, cur_offs, cur_gaps = [], 0, 0, 0
                in_len, tgt_len = self._window_cost(0, add_ons, add_offs, prefix_len)
                add_gap = 0
            if in_len + 1 > self.max_input_tokens or tgt_len + 1 > self.max_target_tokens:
                raise ValueError(f"Single timed step at {step.time} exceeds window budget")
            cur.append(step)
            cur_ons += add_ons
            cur_offs += add_offs
            cur_gaps += add_gap
        flush(cur, window_index)
        return windows

    @staticmethod
    def _tokens_without_prefix(steps: list[TimedStep], tuning: list[int],
                               capo: int) -> tuple[list[str], list[str]]:
        midi_seq: list[str] = []
        tab_seq: list[str] = []
        last_time: int | None = None
        for step in steps:
            if last_time is None:
                last_time = step.time
            else:
                delta = step.time - last_time
                if delta > 0:
                    midi_seq.append(f"TS_{delta}")
                    tab_seq.append(f"TS_{delta}")
                last_time = step.time
            for pitch in sorted(step.offs):
                midi_seq.append(f"NF_{pitch}")
            for note in sorted(step.ons, key=lambda n: n.pitch):
                midi_seq.append(f"NO_{note.pitch}")
                tech_sfx = f"_{note.tech}" if note.tech else ""
                tab_seq.append(f"TAB_{note.string}_{note.fret}{tech_sfx}")
        return midi_seq, tab_seq
