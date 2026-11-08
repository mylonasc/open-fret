"""Training helpers: resume handling and model/tokenizer wiring (Issue #3)."""

from __future__ import annotations

import os


def find_latest_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, full))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resolve_resume_checkpoint(output_dir: str, resume: str | bool | None) -> str | bool | None:
    """Resolve what to pass as ``resume_from_checkpoint``.

    - ``False``/``None`` -> start fresh (None).
    - ``True`` -> latest checkpoint in output_dir, or None if none exists
      (fresh start instead of crashing).
    - str path -> that path (validated by caller/Trainer).
    """
    if resume is True:
        return find_latest_checkpoint(output_dir)
    if resume in (False, None):
        return None
    return resume


def training_layout(output_dir: str) -> dict[str, str]:
    return {
        "output_dir": output_dir,
        "final_dir": os.path.join(output_dir, "final"),
    }
