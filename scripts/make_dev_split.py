"""Build a small group-disjoint dev split for HPO (default 5% of windows).

Samples whole source-song groups (never splits a song across the dev
subsets), preserving the train/val separation of the parent split.
Writes lists + manifest to dataset/processed/dev-hpo-split-v1/.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.splits import split_by_group  # noqa: F401 (re-export check)
from open_fret.splits import group_id_from_filename


def sample_groups(files, frac, seed):
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(group_id_from_filename(f), []).append(f)
    names = sorted(groups)
    rnd = random.Random(seed)
    rnd.shuffle(names)
    target = int(len(files) * frac)
    picked: list[str] = []
    for name in names:
        if len(picked) >= target:
            break
        picked.extend(groups[name])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="dev-hpo-split-v1")
    ap.add_argument("--processed", default="dataset/processed")
    a = ap.parse_args()

    outdir = os.path.join(a.processed, a.name)
    os.makedirs(outdir, exist_ok=True)
    subsets = {}
    list_names = {"train": "train_files.txt", "validation": "val_files.txt"}
    for split in ("train", "validation"):
        with open(os.path.join(a.processed, list_names[split]), encoding="utf-8") as f:
            files = [ln.strip() for ln in f if ln.strip()]
        picked = sample_groups(files, a.frac, a.seed + (0 if split == "train" else 1))
        with open(os.path.join(outdir, f"{split}_files.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(picked)))
        subsets[split] = {"windows": len(picked),
                          "groups": len({group_id_from_filename(x) for x in picked}),
                          "parent_windows": len(files)}
        print(f"{split}: {len(picked)}/{len(files)} windows "
              f"({subsets[split]['groups']} groups)", flush=True)
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"name": a.name, "frac": a.frac, "seed": a.seed,
                   "subsets": subsets}, f, indent=2)
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
