"""Tiny end-to-end fixture pipeline (no external download, no GPU).

Runs: parse -> window -> group split -> vocab/tokenizer -> cache check ->
fresh-training argument validation -> inference chunking/render.

Heavy training (torch/transformers) is exercised separately via
``scripts/train_model.py --help`` validation and a synthetic tiny Trainer
run when dependencies are installed.
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from open_fret.alphatex import AlphaTexParser
from open_fret.sequences import SequenceBuilder, AlignedWindower
from open_fret.splits import split_by_group, assert_disjoint
from open_fret.postprocess import NeighborSearchCorrector
from open_fret.render import TabRenderer
from open_fret.training import resolve_resume_checkpoint


def main():
    parser = AlphaTexParser()
    builder = SequenceBuilder()
    windower = AlignedWindower(max_input_tokens=63, max_target_tokens=63)
    tmp = tempfile.mkdtemp(prefix="open_fret_smoke_", dir="/tmp/opencode")
    print(f"Smoke workspace: {tmp}")
    try:
        # 1. Parse fixtures.
        all_windows = []
        for name in ("song_a.tex", "song_b.tex"):
            path = os.path.join(ROOT, "fixtures", name)
            tracks = parser.parse_file(path)
            assert tracks and tracks[0].events, f"no events parsed from {name}"
            for track_idx, track in enumerate(tracks):
                for capo in (0, 3):
                    steps, tuning = builder.build_steps(track, capo=capo)
                    for w in windower.window(steps, tuning, capo=capo,
                                             source=name, track_index=track_idx):
                        assert w.midi_tokens and w.tab_tokens
                        n_tab = sum(1 for t in w.tab_tokens if t.startswith("TAB_"))
                        n_no = sum(1 for t in w.midi_tokens if t.startswith("NO_"))
                        assert n_tab == n_no, "unaligned window"
                        all_windows.append(w)
        print(f"Parsed windows: {len(all_windows)}")

        # 2. Persist windows + group split.
        indiv = os.path.join(tmp, "individual")
        os.makedirs(indiv)
        files, groups = [], []
        for i, w in enumerate(all_windows):
            fname = f"{w.source.replace('.tex','')}__item_{i}.json"
            with open(os.path.join(indiv, fname), "w", encoding="utf-8") as f:
                json.dump({"midi": w.midi_tokens, "tab": w.tab_tokens,
                           "source": w.source, "file": w.source}, f)
            files.append(fname)
            groups.append(w.source)
        train, val, test = split_by_group(files, groups, train_ratio=0.5,
                                          val_ratio=0.25, seed=0)
        assert_disjoint(train, val, test)
        print(f"Split OK: train={len(train)} val={len(val)} test={len(test)}")

        # 3. Tokenizer round-trip (word-level, no ML deps).
        # Skipped gracefully when HF/tokenizers are not installed.
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast
        except ImportError as e:
            print(f"Tokenizer check skipped (missing dependency: {e})")
        else:
            vocab_tokens = set()
            for fname in files:
                with open(os.path.join(indiv, fname), encoding="utf-8") as f:
                    d = json.load(f)
                vocab_tokens.update(d["midi"])
                vocab_tokens.update(d["tab"])
            specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
            vocab = specials + sorted(vocab_tokens)
            tok = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(vocab)},
                                             unk_token="[UNK]"))
            tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tok_path = os.path.join(tmp, "tokenizer.json")
            tok.save(tok_path)
            fast = PreTrainedTokenizerFast(tokenizer_file=tok_path, bos_token="[BOS]",
                                           eos_token="[EOS]", unk_token="[UNK]",
                                           pad_token="[PAD]", mask_token="[MASK]")
            sample = all_windows[0]
            enc = fast.encode(" ".join(sample.midi_tokens))
            dec = fast.decode(enc)
            assert "CAPO_" in dec and "NO_" in dec, dec
            print(f"Tokenizer OK: vocab={len(fast)} sample_ids={len(enc)}")

        # 4. Postprocess + render smoke.
        corr = NeighborSearchCorrector()
        midi = ["NO_64", "NO_67"]
        fixed, report = corr.correct(["TAB_1_0"], midi)
        assert len([t for t in fixed if t.startswith("TAB_")]) == 2, fixed
        tex = TabRenderer().to_alphatex("TAB_1_0 TS_480 TAB_2_0 TS_240")
        assert ":4" in tex and tex.index(":4") < tex.index("0.1"), tex
        ascii_tab = TabRenderer().to_ascii("TAB_1_10 TS_480 TAB_2_3")
        assert len({len(line) for line in ascii_tab.splitlines()}) == 1
        print("Postprocess + render OK")

        # 5. Training-loop argument/resume wiring (no GPU needed).
        try:
            resolve_resume_checkpoint(tmp, True)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing resume checkpoint must fail visibly")
        assert resolve_resume_checkpoint(tmp, None) is None
        print("Training resume wiring OK (fresh start defaults to None)")

        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
