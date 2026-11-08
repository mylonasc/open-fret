"""Song-group aware dataset splitting (Issue #5)."""

from __future__ import annotations

import random


import re

_NEW_SCHEME_RE = re.compile(r"^(.*)__t\d+__c\d+__w\d+\.json$")


def group_id_from_filename(filename: str) -> str:
    """Derive source-song group from a processed JSON filename.

    New scheme: ``<relpath with __>__t<track>__c<capo>__w<window>.json``.
    Legacy scheme: ``<name>__item_<k>.json``.
    """
    base = filename.rsplit("/", 1)[-1]
    m = _NEW_SCHEME_RE.match(base)
    if m:
        return m.group(1)
    if "__item_" in base:
        return base.split("__item_")[0]
    # Fall back to stem without extension.
    if "." in base:
        return base.rsplit(".", 1)[0]
    return base


def split_by_group(items: list[str], group_ids: list[str],
                   train_ratio: float = 0.8, val_ratio: float = 0.1,
                   seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    """Split items so groups never cross split boundaries."""
    assert len(items) == len(group_ids)
    groups: dict[str, list[str]] = {}
    for item, gid in zip(items, group_ids):
        groups.setdefault(gid, []).append(item)
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    total = len(group_keys)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_keys = set(group_keys[:train_end])
    val_keys = set(group_keys[train_end:val_end])
    train, val, test = [], [], []
    for item, gid in zip(items, group_ids):
        if gid in train_keys:
            train.append(item)
        elif gid in val_keys:
            val.append(item)
        else:
            test.append(item)
    return train, val, test


def assert_disjoint(train: list[str], val: list[str], test: list[str],
                    group_of=None) -> None:
    group_of = group_of or group_id_from_filename
    tg = {group_of(x) for x in train}
    vg = {group_of(x) for x in val}
    eg = {group_of(x) for x in test}
    overlap = (tg & vg) | (tg & eg) | (vg & eg)
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(f"Group leakage across splits: {sample}")
