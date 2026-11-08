"""Split processed windows by source song (Issue #5)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.splits import split_by_group, assert_disjoint, group_id_from_filename


def _group_for_file(input_dir: str, filename: str) -> str:
    # Fast path: derive from filename (exact for names written by 01_process).
    # Only read the file if the name doesn't match known schemes.
    gid = group_id_from_filename(filename)
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    if gid != stem:
        return gid
    try:
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        src = data.get("source") or data.get("file")
        if src:
            return str(src)
    except Exception:
        pass
    return gid


def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))
    if not files:
        print("No processed JSON files found.")
        return
    groups = [_group_for_file(input_dir, f) for f in files]
    train_files, val_files, test_files = split_by_group(
        files, groups, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    assert_disjoint(train_files, val_files, test_files,
                    group_of=lambda f: groups[files.index(f)] if f in files else f)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_files.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(output_dir, "val_files.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_files))
    with open(os.path.join(output_dir, "test_files.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_files))
    # Persist manifest for reproducibility.
    manifest = {
        "seed": seed, "train_ratio": train_ratio, "val_ratio": val_ratio,
        "n_groups": len(set(groups)),
        "n_train": len(train_files), "n_val": len(val_files), "n_test": len(test_files),
    }
    with open(os.path.join(output_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Split complete: {len(train_files)} train, {len(val_files)} val, "
          f"{len(test_files)} test windows ({len(set(groups))} source groups).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="dataset/processed/individual")
    ap.add_argument("--output-dir", default="dataset/processed")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    split_dataset(a.input_dir, a.output_dir, seed=a.seed)
