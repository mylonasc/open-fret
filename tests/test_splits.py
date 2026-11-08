import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.splits import split_by_group, assert_disjoint, group_id_from_filename


def test_group_split_no_leakage():
    files = [f"song{i//2}__item_{i%2}.json" for i in range(20)]
    groups = [group_id_from_filename(f) for f in files]
    train, val, test = split_by_group(files, groups, seed=0)
    assert_disjoint(train, val, test)
    # All derivatives of one song stay together.
    for song in {f.split("__item_")[0] for f in files}:
        parts = [f for f in files if f.startswith(song + "__item_")]
        holders = []
        for p in parts:
            if p in train:
                holders.append("train")
            elif p in val:
                holders.append("val")
            else:
                holders.append("test")
        assert len(set(holders)) == 1, f"{song} split across {holders}"
