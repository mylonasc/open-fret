"""Backwards-compatible rendering wrappers (Issue #7)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.render import TabRenderer, STANDARD_TUNING_STR  # noqa: F401


def tokens_to_alphatex(tokens_str, tempo=120, tuning=None):
    renderer = TabRenderer(tempo=tempo, tuning_str=tuning or STANDARD_TUNING_STR)
    return renderer.to_alphatex(tokens_str)


def format_chord(note_strs):
    return TabRenderer.format_chord(note_strs)


def tokens_to_ascii(tokens_str):
    return TabRenderer().to_ascii(tokens_str)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('tokens', nargs='*')
    ap.add_argument('--input', help='Token file (or use positional tokens/stdin)')
    ap.add_argument('--output', help='Output file; otherwise stdout')
    ap.add_argument('--format', choices=['tex', 'ascii'], default='ascii')
    ap.add_argument('--tex', action='store_true', help='Alias for --format tex')
    ap.add_argument('--notes-json', help='Identity/string/fret/start/end sidecar for exact durations')
    ap.add_argument('--tempo', type=float, default=120)
    ap.add_argument('--tuning', default=STANDARD_TUNING_STR, help='Written convention, e.g. E5 B4 G4 D4 A3 E3')
    args = ap.parse_args()
    if args.input and args.tokens:
        ap.error('Use --input or positional tokens, not both')
    renderer = TabRenderer(tempo=args.tempo, tuning_str=args.tuning)
    if args.notes_json:
        if not (args.tex or args.format == 'tex'):
            ap.error('--notes-json requires --format tex')
        result = renderer.to_alphatex_notes(json.loads(Path(args.notes_json).read_text()))
    else:
        if args.input:
            content = Path(args.input).read_text()
        elif len(args.tokens) == 1 and Path(args.tokens[0]).is_file():
            content = Path(args.tokens[0]).read_text()
        else:
            content = ' '.join(args.tokens) if args.tokens else sys.stdin.read()
        result = renderer.to_alphatex(content) if args.tex or args.format == 'tex' else renderer.to_ascii(content)
    if args.output:
        Path(args.output).write_text(result)
    else:
        print(result)
