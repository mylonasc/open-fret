"""Full-song MIDI inference with shared preprocessing (Issues #6/#7).

- Deterministic decoding by default; sampling is opt-in.
- Long inputs are chunked at TS boundaries and stitched back together.
- Conditioning matches training (CAPO_0 + standard tuning by default).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

DEFAULT_MAX_INPUT = 512
DEFAULT_MAX_OUTPUT = 512


def midi_to_tokens(midi_path, target_res=480):
    from open_fret.midi_io import midi_file_to_tokens
    return midi_file_to_tokens(midi_path, target_resolution=target_res)


def chunk_tokens(tokens, budget=DEFAULT_MAX_INPUT - 2):
    """Split NO/NF/TS stream at TS boundaries to fit the input budget."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        if tok.startswith("TS_") and len(cur) + 1 > budget and cur:
            chunks.append(cur)
            cur = []
        cur.append(tok)
        if len(cur) >= budget:
            # Prefer to cut at next TS; for now cut here and continue.
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def run_inference(midi_path, model_dir, max_input_tokens=DEFAULT_MAX_INPUT,
                  max_new_tokens=DEFAULT_MAX_OUTPUT, deterministic=True,
                  capo=0, tuning=None):
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    from open_fret.midi_io import add_conditioning

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    midi_tokens = midi_to_tokens(midi_path)
    if not midi_tokens:
        raise ValueError(f"No notes found in MIDI: {midi_path}")
    print(f"Input MIDI Tokens: {' '.join(midi_tokens[:30])}... ({len(midi_tokens)} total)")

    chunks = chunk_tokens(midi_tokens, budget=max_input_tokens - 2)
    if len(chunks) > 1:
        print(f"Long input: split into {len(chunks)} windows for full-song coverage.")
    decoded_all: list[str] = []
    for chunk in chunks:
        full_input = " ".join(add_conditioning(chunk, capo=capo, tuning=tuning))
        inputs = tokenizer(full_input, return_tensors="pt",
                           truncation=True, max_length=max_input_tokens).to(device)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "early_stopping": True}
        if deterministic:
            gen_kwargs.update({"do_sample": False, "num_beams": 1})
        else:
            gen_kwargs.update({"do_sample": True, "top_p": 0.9, "temperature": 0.7})
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        decoded_all.extend(tokenizer.decode(outputs[0], skip_special_tokens=True).split())

    try:
        from post_process import neighbor_search
        final_tokens = neighbor_search(decoded_all, midi_tokens)
        return " ".join(final_tokens)
    except Exception as e:  # noqa: BLE001
        print(f"Post-processing failed ({e}); returning raw predictions.")
        return " ".join(decoded_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, required=True, help="Path to input MIDI file")
    parser.add_argument("--input", type=str, default=None,
                        help="Alias for --midi (backwards compat with README).")
    parser.add_argument("--model", type=str, default="models/tiny-tab-v1/final")
    parser.add_argument("--out", type=str, default=None, help="Write token output to file.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use sampling instead of deterministic decoding.")
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_OUTPUT)
    args = parser.parse_args()
    midi_path = args.midi or args.input
    if not midi_path:
        parser.error("Provide --midi /path/to/song.mid")

    result = run_inference(midi_path, args.model,
                           max_input_tokens=args.max_input_tokens,
                           max_new_tokens=args.max_new_tokens,
                           deterministic=not args.stochastic)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Wrote tokens to {args.out}")

    print("\n--- AlphaTex Output (.tex) ---")
    try:
        from tokens_to_tab import tokens_to_alphatex
        print(tokens_to_alphatex(result))
    except Exception as e:  # noqa: BLE001
        print(f"Error generating AlphaTex: {e}")

    print("\n--- ASCII Tablature (Preview) ---")
    try:
        from tokens_to_tab import tokens_to_ascii
        ascii_tab = tokens_to_ascii(result)
        print("\n".join(line[:200] for line in ascii_tab.split("\n")))
    except Exception:  # noqa: BLE001
        pass
    print("--------------------------------------")


if __name__ == "__main__":
    main()
