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
    from open_fret.generation import chunk_complete_onsets
    return chunk_complete_onsets(tokens, budget)


def run_inference(midi_path, model_dir, max_input_tokens=DEFAULT_MAX_INPUT,
                   max_new_tokens=DEFAULT_MAX_OUTPUT, deterministic=True,
                    capo=0, tuning=None, decoding_mode="anchored", beams=4,
                    instrument=None, return_report=False, relocate_outliers=False):
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    from open_fret.midi_io import add_conditioning
    from open_fret.generation import generate_tab

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # model_dir may be a local path or a Hugging Face hub id (local fine-tune
    # or pretrained endpoint). from_pretrained handles both; surface a clear
    # error instead of a bare stack trace when neither resolves.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not load tokenizer from {model_dir!r}: {e}") from e
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not load model from {model_dir!r}: {e}") from e
    model.eval()

    from open_fret.midi_io import read_midi_notes
    from open_fret.inference import transcribe_notes
    notes, metadata = read_midi_notes(midi_path, instrument=instrument)
    result = transcribe_notes(model, tokenizer, notes, tuning=tuning, capo=capo,
                              decoding_mode=decoding_mode, beams=beams,
                              max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
                              stochastic=not deterministic)
    result['midi'] = metadata
    if relocate_outliers:
        # midi-to-tab §4.4: move outlier fingerings toward the local hand
        # position after identity-preserving correction. Pitch is preserved
        # exactly; only string/fret change, so token structure is untouched.
        from open_fret.postprocess import notes_to_tab_tokens, relocate_deviating_notes
        corrected = [dict(n) for n in result['correction']['corrected_notes']]
        moved, rel_report = relocate_deviating_notes(
            corrected, tuning=result['tuning'], capo=result['capo'])
        result['corrected_tokens'] = notes_to_tab_tokens(moved)
        result['relocation'] = {"relocated": rel_report.relocated,
                                "checked": rel_report.checked,
                                "out_of_range": rel_report.out_of_range,
                                "unavailable": rel_report.unavailable,
                                "details": rel_report.details}
    return result if return_report else ' '.join(result['corrected_tokens'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, help="Path to input MIDI file")
    parser.add_argument("--input", type=str, default=None,
                        help="Alias for --midi (backwards compat with README).")
    parser.add_argument("--model", type=str, default="models/tiny-tab-v1/final",
                        help="Local model dir or HF hub id (e.g. 't5-small' for a "
                             "pretrained baseline, though it won't know TAB tokens).")
    parser.add_argument("--out", type=str, default=None, help="Write token output to file.")
    parser.add_argument("--raw-out", help="Write uncorrected model tokens separately")
    parser.add_argument("--report", help="Write source identities, corrections and raw/corrected metrics as JSON")
    parser.add_argument("--tex-out", help="Write timing/duration-preserving AlphaTex")
    parser.add_argument("--ascii-out", help="Write aligned ASCII (onset preview, not duration notation)")
    parser.add_argument("--instrument", type=int, help="MIDI instrument index; required for multiple pitched tracks")
    parser.add_argument("--capo", type=int, default=0)
    parser.add_argument("--tuning", type=int, nargs='+', help="Concert MIDI pitches, highest string first")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use sampling instead of deterministic decoding.")
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_OUTPUT)
    parser.add_argument("--decoding-mode", choices=["anchored", "legacy"], default="anchored")
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--relocate-outliers", action="store_true",
                        help="midi-to-tab §4.4: relocate outlier fingerings toward "
                        "the local hand position after correction (pitch-preserving).")
    args = parser.parse_args()
    midi_path = args.midi or args.input
    if not midi_path:
        parser.error("Provide --midi /path/to/song.mid")

    result = run_inference(midi_path, args.model,
                           max_input_tokens=args.max_input_tokens,
                           max_new_tokens=args.max_new_tokens,
                            deterministic=not args.stochastic,
                             decoding_mode=args.decoding_mode, beams=args.beams,
                             capo=args.capo, tuning=args.tuning, instrument=args.instrument,
                             return_report=True, relocate_outliers=args.relocate_outliers)
    import json
    from open_fret.viewer import renderer_for_input
    from open_fret.midi_io import add_conditioning
    renderer = renderer_for_input(add_conditioning([], result['capo'], result['tuning']),
                                  tempo=result['midi']['tempo'])
    # Render before writing any success artifacts; unsupported rhythms fail.
    tex = renderer.to_alphatex_notes(result['correction']['corrected_notes'])
    ascii_tab = renderer.to_ascii(result['corrected_tokens'])
    text = ' '.join(result['corrected_tokens'])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote tokens to {args.out}")

    for path, content in ((args.raw_out, ' '.join(result['raw_tokens'])),
                          (args.report, json.dumps(result, indent=2)),
                          (args.tex_out, tex), (args.ascii_out, ascii_tab)):
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
    print(f"Windows: {len(result['windows'])}; corrections: "
          f"{result['correction']['refingered']} refingered, "
          f"{result['correction']['inserted']} inserted, {result['correction']['dropped_extra']} dropped")
    if "relocation" in result:
        print(f"Relocation: {result['relocation']['relocated']} moved "
              f"({result['relocation']['checked']} checked)")
    print("\n--- AlphaTex Output (.tex) ---\n" + tex)
    print("\n--- ASCII Tablature (onset preview) ---\n" + ascii_tab)
    print("--------------------------------------")


if __name__ == "__main__":
    main()
