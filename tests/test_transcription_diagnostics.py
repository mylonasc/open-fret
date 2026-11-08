import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.transcription_diagnostics import (compare_streams, event_stream, family_key,
                                                repetition_conflicts, structure, timed_match)

PREFIX = ["CAPO_0", "TUNING_64_59_55_50_45_40"]


def test_arpeggio_collapsed_to_chord_and_duplicate_strings_are_measured():
    midi = PREFIX + "NO_64 TS_120 NF_64 NO_67 TS_120 NF_67 NO_69 TS_120 NF_69".split()
    ref = "TAB_1_0 TS_120 TAB_2_8 TS_120 TAB_1_5 TS_120".split()
    pred = "TAB_1_0 TAB_2_8 TAB_1_5 TS_360".split()
    result = compare_streams(midi, ref, pred)
    assert result["input_reference_onsets_identical"]
    assert result["input"]["chord_onsets"] == 0
    assert result["prediction"]["chord_onsets"] == 1
    assert result["prediction"]["same_string_conflict_onsets"] == 1
    assert result["false_chord_at_mono_onset_exact_tick"] == 1
    assert result["timed_pitch"]["matched_notes"] == 1


def test_genuine_chord_is_not_a_false_chord():
    midi = PREFIX + "NO_64 NO_67 TS_480 NF_64 NF_67".split()
    ref = "TAB_1_0 TAB_2_8 TS_480".split()
    result = compare_streams(midi, ref, ref)
    assert result["input"]["chord_onsets"] == result["prediction"]["chord_onsets"] == 1
    assert result["false_chord_at_mono_onset_exact_tick"] == 0
    assert result["timed_pitch"]["f1"] == 1


def test_sustained_overlap_is_not_simultaneous_attack():
    notes, end = event_stream("NO_64 TS_120 NO_67 TS_120 NF_64 NF_67", "midi")
    assert end == 240
    assert structure(notes)["chord_onsets"] == 0
    assert [n["tick"] for n in notes] == [0, 120]


def test_duplicates_cannot_match_one_reference_note_multiple_times():
    source = [{"tick": 0, "pitch": 64}]
    result = timed_match(source, source * 3)
    assert result["matched_notes"] == 1
    assert result["precision"] == 1 / 3


def test_untimed_correct_pitches_do_not_hide_timing_error():
    source = [{"tick": 0, "pitch": 64}, {"tick": 120, "pitch": 67}]
    predicted = [{"tick": 60, "pitch": 64}, {"tick": 180, "pitch": 67}]
    assert timed_match(source, predicted, tolerance=30)["f1"] == 0
    assert timed_match(source, predicted, tolerance=60)["f1"] == 1


def test_repetition_ban_conflicts_with_valid_riff_at_fifth_token():
    result = repetition_conflicts("TAB_2_0 TS_120 TAB_2_0 TS_120 TAB_2_0".split())
    assert result["first_forbidden_token_1based"] == 5


def test_custom_tuning_and_capo_are_used():
    midi = "CAPO_2 TUNING_64_59_55_50_45_38 NO_42 TS_480 NF_42".split()
    ref = "TAB_6_2 TS_480".split()
    result = compare_streams(midi, ref, ref)
    assert result["input_reference_onsets_identical"]
    assert result["timed_pitch"]["f1"] == 1


def test_song_family_normalization_is_only_a_version_candidate():
    assert family_key("Hoobastank__Hoobastank_-_Crawling_In_The_Dark_(2).tex") == family_key(
        "Hoobastank__Hoobastank_-_crawling_in_the_dark.tex")
