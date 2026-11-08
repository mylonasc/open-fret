"""Target curation: same-onset/same-string conflicts and cache masking."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from open_fret.curation import conflict_positions, conflicts_for_window
from open_fret.alphatex import PARSER_VERSION


def test_conflict_positions_masks_whole_group_only():
    tokens = ["TAB_2_0", "TAB_2_3", "TS_480", "TAB_1_0", "TAB_2_1", "TS_480",
              "TAB_3_2"]
    assert conflict_positions(tokens) == [0, 1]
    assert conflict_positions(["TAB_1_0", "TAB_2_1"]) == []
    assert conflict_positions(["TAB_1_0", "TAB_1_0"]) == [0, 1]
    with pytest.raises(ValueError):
        conflict_positions(["TAB_1"])


def test_conflict_positions_crosses_time_shifts():
    # A repeated string is only contradictory inside the same attack group.
    assert conflict_positions(["TAB_2_0", "TS_480", "TAB_2_3"]) == []


def test_conflicts_for_window_finds_unassignable_chord():
    tuning = [64, 59, 55, 50, 45, 40]
    # 40 and 41 are both reachable only on the lowest string.
    assert conflicts_for_window(["NO_40", "NO_41"], tuning) == [40, 41]
    assert conflicts_for_window(["NO_40", "NO_45"], tuning) == []
    assert conflicts_for_window(["NO_40"], tuning) == []
    # Out-of-range pitch cannot be placed at all.
    assert conflicts_for_window(["NO_20", "NO_45"], tuning) == [20, 45]


def load_script(filename):
    spec = importlib.util.spec_from_file_location("fixture_" + filename,
                                                  ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_masks_conflicting_labels_and_reports_counts(tmp_path):
    data = tmp_path / "individual"
    data.mkdir()
    midi = ["CAPO_0", "TUNING_64_59_55_50_45_40", "NO_64", "NO_64", "TS_480",
            "NO_62"]
    tab = ["TAB_2_0", "TAB_2_3", "TS_480", "TAB_1_0"]
    name = "conflict__c0.json"
    (data / name).write_text(json.dumps({"midi": midi, "tab": tab,
                                        "parser_version": PARSER_VERSION}))
    tokens = sorted(set(midi + tab))
    (tmp_path / "vocab.json").write_text(json.dumps(tokens))
    split = tmp_path / "files.txt"
    split.write_text(name)
    load_script("02_tokenize.py").build_tokenizer(str(tmp_path / "vocab.json"),
                                                  str(tmp_path / "tokenizer"))
    load_script("04_cache_dataset.py").create_cache(
        tokenizer_dir=str(tmp_path / "tokenizer"), data_dir=str(data),
        train_list=str(split), val_list=str(split),
        test_list=str(tmp_path / "no-test.txt"),
        cache_dir=str(tmp_path / "cache"), max_length=32)

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "tokenizer")
    ids = tokenizer.convert_tokens_to_ids(tab)
    labels = load_from_disk(str(tmp_path / "cache"))["train"][0]["labels"]
    assert labels[0:2] == [-100, -100]
    assert labels[2] != -100  # time shift keeps supervision
    assert labels[3] == ids[3]
    assert labels[4] == tokenizer.eos_token_id

    manifest = json.loads((tmp_path / "cache" / "cache_manifest.json").read_text())
    assert manifest["mask_conflicts"] is True
    assert manifest["curation"]["totals"] == {
        "scanned_windows": 2, "masked_windows": 2, "masked_notes": 4}
    assert manifest["curation"]["splits"]["train"]["masked_notes"] == 2
