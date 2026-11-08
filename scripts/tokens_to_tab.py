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
    import sys as _sys
    content = _sys.argv[1:]
    if "--tex" in content:
        content.remove("--tex")
        print(tokens_to_alphatex(" ".join(content)))
    elif content:
        # File-based usage documented in README was missing; support both.
        path = content[0] if len(content) == 1 else None
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                print(tokens_to_ascii(f.read()))
        else:
            print(tokens_to_ascii(" ".join(content)))
    else:
        print(tokens_to_ascii(_sys.stdin.read()))
